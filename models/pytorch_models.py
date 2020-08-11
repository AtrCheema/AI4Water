__all__ = ["HARHNModel", "IMVLSTMModel"]

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

from .global_variables import torch
nn = torch.nn
DataLoader = torch.utils.data.DataLoader
TensorDataset = torch.utils.data.TensorDataset

from .HARHN import HARHN
from .imv_networks import IMVTensorLSTM
from main import Model


class HARHNModel(Model):

    def __init__(self, **kwargs):
        self.api = 'pytorch'
        self.pt_model = None
        self.opt = None
        self.epoch_scheduler = None
        self.min_max = None
        self.saved_model = None

        super(HARHNModel, self).__init__(**kwargs)


    def build_nn(self):
        config = self.nn_config['HARHN_config']

        self.pt_model = HARHN(config['n_conv_lyrs'],
                              self.lookback, self.ins, self.outs,
                              n_units_enc=config['enc_units'],
                              n_units_dec=config['dec_units']).cuda()
        self.opt = torch.optim.Adam(self.pt_model.parameters(), lr=self.nn_config['lr'])

        self.epoch_scheduler = torch.optim.lr_scheduler.StepLR(self.opt, 20, gamma=0.9)

        self.loss = nn.MSELoss()

        return

    def train_nn(self, st=0, en=None, indices=None, **callbacks):

        x, y_his, target = self.prepare_batches(self.data[st:en],  self.data_config['outputs'][0])

        x_tr, x_val, y_his_tr, y_his_val, target_tr, target_val = train_test_split(x, y_his, target,
                                                                                   test_size=self.data_config['val_fraction'])

        self.min_max = {
            'x_max': x_tr.max(axis=0),
            'x_min': x_tr.min(axis=0),
            'y_his_max': y_his_tr.max(axis=0),
            'y_his_min': y_his_tr.min(axis=0),
            'target_max': target_tr.max(axis=0),
            'target_min': target_tr.min(axis=0)
        }

        x_train = (x_tr - self.min_max['x_min']) / (self.min_max['x_max'] - self.min_max['x_min'])
        x_val = (x_val - self.min_max['x_min']) / (self.min_max['x_max'] - self.min_max['x_min'])

        y_his_train = (y_his_tr - self.min_max['y_his_min']) / (self.min_max['y_his_max'] - self.min_max['y_his_min'])
        y_his_val = (y_his_val - self.min_max['y_his_min']) / (self.min_max['y_his_max'] - self.min_max['y_his_min'])

        target_train = (target_tr - self.min_max['target_min']) / (self.min_max['target_max'] - self.min_max['target_min'])
        target_val = (target_val - self.min_max['target_min']) / (self.min_max['target_max'] - self.min_max['target_min'])

        x_train_t = to_torch_tensor(x_train)
        x_val_t = to_torch_tensor(x_val)

        y_his_train_t = to_torch_tensor(y_his_train)
        y_his_val_t = to_torch_tensor(y_his_val)

        target_train_t = to_torch_tensor(target_train)
        target_val_t = to_torch_tensor(target_val)

        data_train_loader = DataLoader(TensorDataset(x_train_t, y_his_train_t, target_train_t), shuffle=True,
                                       batch_size=self.data_config['batch_size'])
        data_val_loader = DataLoader(TensorDataset(x_val_t, y_his_val_t, target_val_t), shuffle=False,
                                     batch_size=self.data_config['batch_size'])

        min_val_loss = self.nn_config['HARHN_config']['min_val_loss']
        counter = 0
        losses = {'train_loss': [],
                  'val_loss': []}

        for i in range(self.nn_config['epochs']):
            mse_train = 0
            for batch_x, batch_y_h, batch_y in data_train_loader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                batch_y_h = batch_y_h.cuda()
                self.opt.zero_grad()
                y_pred = self.pt_model(batch_x, batch_y_h)
                y_pred = y_pred.squeeze(1)
                l = self.loss(y_pred, batch_y)
                l.backward()
                mse_train += l.item() * batch_x.shape[0]
                self.opt.step()
            self.epoch_scheduler.step()
            with torch.no_grad():
                mse_val = 0
                preds = []
                true = []
                for batch_x, batch_y_h, batch_y in data_val_loader:
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                    batch_y_h = batch_y_h.cuda()
                    output = self.pt_model(batch_x, batch_y_h)
                    output = output.squeeze(1)
                    preds.append(output.detach().cpu().numpy())
                    true.append(batch_y.detach().cpu().numpy())
                    mse_val += self.loss(output, batch_y).item() * batch_x.shape[0]
            preds = np.concatenate(preds)
            true = np.concatenate(true)

            if min_val_loss > mse_val ** 0.5:
                min_val_loss = mse_val ** 0.5
                print("Saving...")
                self.saved_model = os.path.join(self.path, "harhn_nasdaq.pt")
                torch.save(self.pt_model.state_dict(), self.saved_model)
                counter = 0
            else:
                counter += 1

            if counter == self.nn_config['HARHN_config']['patience']:
                break
            train_loss = (mse_train / len(x_train_t)) ** 0.5
            val_loss = (mse_val / len(x_val_t)) ** 0.5
            losses['train_loss'].append(train_loss)
            losses['val_loss'].append(val_loss)
            print("Iter: ", i, "train: ", train_loss, "val: ", val_loss)
            if i % 10 == 0:
                preds = preds * (self.min_max['target_max'] - self.min_max['target_min']) + self.min_max['target_min']
                true = true * (self.min_max['target_max'] - self.min_max['target_min']) + self.min_max['target_min']

                self.process_results(true, preds, 'validation_' + str(i))

        return losses

    def predict(self, st=0, ende=None, indices=None):
        x_test, y_his_test, target_test = self.prepare_batches(self.data[st:ende],  self.data_config['outputs'][0])
        x_test = (x_test - self.min_max['x_min']) / (self.min_max['x_max'] - self.min_max['x_min'])
        y_his_test = (y_his_test - self.min_max['y_his_min']) / (self.min_max['y_his_max'] - self.min_max['y_his_min'])
        target_test = (target_test - self.min_max['target_min']) / (self.min_max['target_max'] - self.min_max['target_min'])

        x_test_t = to_torch_tensor(x_test)
        y_his_test_t = to_torch_tensor(y_his_test)
        target_test_t = to_torch_tensor(target_test)

        data_test_loader = DataLoader(TensorDataset(x_test_t, y_his_test_t, target_test_t), shuffle=False,
                                      batch_size=self.data_config['batch_size'])

        self.pt_model.load_state_dict(torch.load(self.saved_model))

        with torch.no_grad():
            mse_val = 0
            preds = []
            true = []
            for batch_x, batch_y_h, batch_y in data_test_loader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                batch_y_h = batch_y_h.cuda()
                output = self.pt_model(batch_x, batch_y_h)
                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())
                mse_val += self.loss(output, batch_y).item()*batch_x.shape[0]
        preds = np.concatenate(preds)
        true = np.concatenate(true)

        preds = preds*(self.min_max['target_max'] - self.min_max['target_min']) + self.min_max['target_min']
        true = true*(self.min_max['target_max'] - self.min_max['target_min']) + self.min_max['target_min']

        self.process_results(true, preds, 'validation_')

        return

class IMVLSTMModel(HARHNModel):

    def __init__(self, **kwargs):
        self.alphas = None
        self.betas = None
        super(IMVLSTMModel, self).__init__(**kwargs)

    def prepare_batches(self, data, target):
        x = np.zeros((len(data), self.lookback, data.shape[1]))

        for i, name in enumerate(list(data.columns)):
            # print(name)
            for j in range(self.lookback):
                x[:, j, i] = data[name].shift(self.lookback - j - 1).fillna(method="bfill")

        prediction_horizon = 1
        target = data[target].shift(-prediction_horizon).fillna(method="ffill").values

        x = x[self.lookback:]
        target = target[self.lookback:]

        return x, target

    def build_nn(self):

        self.pt_model = IMVTensorLSTM(self.ins + self.outs, self.outs, self.data_config['batch_size']).cuda()

        self.opt = torch.optim.Adam(self.pt_model.parameters(), lr=self.nn_config['lr'])

        self.epoch_scheduler = torch.optim.lr_scheduler.StepLR(self.opt, 20, gamma=0.9)

        self.loss = nn.MSELoss()

        return

    def train_nn(self, st=0, en=None, indices=None, **callbacks):

        x, target = self.prepare_batches(self.data[st:en], self.data_config['outputs'][0])

        x_train, x_val, target_train, target_val = train_test_split(x, target, test_size=self.data_config['val_fraction'])

        self.min_max = {
            'x_max': x_train.max(axis=0),
            'x_min': x_train.min(axis=0),
            'target_max': target_train.max(axis=0),
            'target_min': target_train.min(axis=0)
        }

        x_train = (x_train - self.min_max['x_min']) / (self.min_max['x_max'] - self.min_max['x_min'])
        target_train = (target_train - self.min_max['target_min']) / (self.min_max['target_max'] - self.min_max['target_min'])
        x_val = (x_val - self.min_max['x_min']) / (self.min_max['x_max'] - self.min_max['x_min'])
        target_val = (target_val - self.min_max['target_min']) / (self.min_max['target_max'] - self.min_max['target_min'])

        x_train_t = to_torch_tensor(x_train)
        target_train_t = to_torch_tensor(target_train)
        x_val_t = to_torch_tensor(x_val)
        target_val_t = to_torch_tensor(target_val)

        data_train_loader = DataLoader(TensorDataset(x_train_t, target_train_t),
                                       shuffle=True, batch_size=self.data_config['batch_size'])
        data_val_loader = DataLoader(TensorDataset(x_val_t, target_val_t),
                                     shuffle=False, batch_size=self.data_config['batch_size'])

        min_val_loss = self.nn_config['min_val_loss']
        counter = 0
        losses = {'train_loss': [],
                  'val_loss': []}
        for i in range(self.nn_config['epochs']):
            mse_train = 0
            for batch_x, batch_y in data_train_loader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                self.opt.zero_grad()
                y_pred, alphas, betas = self.pt_model(batch_x)
                y_pred = y_pred.squeeze(1)
                l = self.loss(y_pred, batch_y)
                l.backward()
                mse_train += l.item() * batch_x.shape[0]
                self.opt.step()
            self.epoch_scheduler.step()
            with torch.no_grad():
                mse_val = 0
                preds = []
                true = []
                for batch_x, batch_y in data_val_loader:
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                    output, alphas, betas = self.pt_model(batch_x)
                    output = output.squeeze(1)
                    preds.append(output.detach().cpu().numpy())
                    true.append(batch_y.detach().cpu().numpy())
                    mse_val += self.loss(output, batch_y).item() * batch_x.shape[0]
            pred = np.concatenate(preds)
            true = np.concatenate(true)

            if min_val_loss > mse_val ** 0.5:
                min_val_loss = mse_val ** 0.5
                print("Saving...")
                self.saved_model = os.path.join(self.path, "imv_tensor_lstm_nasdaq.pt")
                torch.save(self.pt_model.state_dict(), self.saved_model)
                counter = 0
            else:
                counter += 1

            if counter == self.nn_config['patience']:
                break
            train_loss = (mse_train / len(x_train_t)) ** 0.5
            val_loss = (mse_val / len(x_val_t)) ** 0.5
            losses['train_loss'].append(train_loss)
            losses['val_loss'].append(val_loss)
            print("Iter: ", i, "train: ", train_loss, "val: ", val_loss)
            if i % 10 == 0:
                pred = pred * (self.min_max['target_max'] - self.min_max['target_min']) + self.min_max['target_min']
                true = true * (self.min_max['target_max'] - self.min_max['target_min']) + self.min_max['target_min']

                self.process_results(true, pred, str(i))

        return

    def predict(self, st=0, ende=None, indices=None):
        self.pt_model.load_state_dict(torch.load(self.saved_model))

        x_test, target_test = self.prepare_batches(self.data[st:ende], self.data_config['outputs'][0])

        x_test = (x_test - self.min_max['x_min']) / (self.min_max['x_max'] - self.min_max['x_min'])
        target_test = (target_test - self.min_max['target_min']) / (self.min_max['target_max'] - self.min_max['target_min'])

        x_test_t = to_torch_tensor(x_test)
        target_test_t = to_torch_tensor(target_test)

        data_test_loader = DataLoader(TensorDataset(x_test_t, target_test_t),
                                      shuffle=False, batch_size=self.data_config['batch_size'])

        with torch.no_grad():
            mse_val = 0
            preds = []
            true = []
            alphas = []
            betas = []
            for batch_x, batch_y in data_test_loader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                output, a, b = self.pt_model(batch_x)
                output = output.squeeze(1)
                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())
                alphas.append(a.detach().cpu().numpy())
                betas.append(b.detach().cpu().numpy())
                mse_val += self.loss(output, batch_y).item()*batch_x.shape[0]
        preds = np.concatenate(preds)
        true = np.concatenate(true)

        preds = preds*(self.min_max['target_max'] - self.min_max['target_min']) + self.min_max['target_min']
        true = true*(self.min_max['target_max'] - self.min_max['target_min']) + self.min_max['target_min']

        self.process_results(true, preds, 'validation')

        return preds, true, alphas, betas

    def plot_activations(self):
        all_cols = self.data_config['inputs'] + self.data_config['outputs']
        plt.close('all')
        fig, ax = plt.subplots()
        # fig.set_figwidth(12)
        # fig.set_figheight(14)
        plt.figure(dpi=400)
        _ = ax.imshow(self.alphas)
        ax.set_xticks(np.arange(self.lookback))
        ax.set_yticks(np.arange(len(all_cols)))
        ax.set_xticklabels(["t-"+str(i) for i in np.arange(self.lookback, -1, -1)])
        ax.set_yticklabels(list(all_cols))
        for i in range(len(all_cols)):
            for j in range(self.lookback):
                _ = ax.text(j, i, round(self.alphas[i, j], 3),
                               ha="center", va="center", color="w")
        ax.set_title("Importance of features and timesteps")

        plt.show()

        plt.figure()
        plt.title("Feature importance")
        plt.bar(range(self.ins + self.outs), self.betas)
        plt.xticks(ticks=range(len(all_cols)), labels=list(all_cols), rotation=90)

def to_torch_tensor(array):
    return torch.Tensor(array)
