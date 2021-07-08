import os

import numpy as np

from .backend import torch
from .utils.torch_utils import to_torch_dataset, TorchMetrics

DataLoader = torch.utils.data.DataLoader

F = {
    'mse': [np.nanmin, np.less],
    'nse': [np.nanmax, np.greater],
    'r2': [np.nanmax, np.greater],
    'pbias': [np.nanmin, np.less],
    'mape': [np.nanmin, np.less],
    'rmse': [np.nanmin, np.less],
    'nrmse': [np.nanmin, np.less],
    'kge': [np.nanmax, np.greater],
}

class AttributeContainer(object):

    def __init__(self, num_epochs, to_monitor=None):
        self.to_monitor = get_metrics_to_monitor(to_monitor)
        self.num_epochs = num_epochs

        self.epoch = 0
        self.val_loader = None
        self.train_loader = None
        self.criterion = None
        self.optimizer = None
        self.val_epoch_losses = {}
        self.train_epoch_losses = None
        self.train_metrics = {metric: np.full(num_epochs, np.nan) for metric in self.to_monitor}
        self.val_metrics = {f'val_{metric}': np.full(num_epochs, np.nan) for metric in self.to_monitor}
        self.best_epoch = 0   # todo,
        self.use_cuda = torch.cuda.is_available()

    @property
    def use_cuda(self):
        return self._use_cuda

    @use_cuda.setter
    def use_cuda(self, x):
        self._use_cuda = x

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, x):
        self._optimizer = x

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, x):
        self._loss = x


class Learner(AttributeContainer):
    """Trains the pytorch model. Motivated from fastai"""

    def __init__(self,
                 model,
                 batch_size:int = 32,
                 num_epochs: int = 14,
                 patience: int = 100,
                 shuffle: bool = True,
                 to_monitor:list=None
                 ):
        """
        Arguments:
            model : a pytorch model having following attributes and methods
                - num_outs
                - w_path
                - `loss`
                - `get_optimizer`
            batch_size : batch size
            num_epochs : Number of epochs for which to train the model
            patience : how many epochs to wait before stopping the training in
                case `to_monitor` does not improve.
            shuffle :
            to_monitor : list of metrics to monitor
        """
        super().__init__(num_epochs, to_monitor)

        self.model = model
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.patience = patience

    def fit(self, x, **kwargs):
        """Runs the training loop for pytorch model"""
        self.on_train_begin(x, **kwargs)

        for epoch in range(self.num_epochs):

            self.epoch = epoch

            self.train_for_epoch()
            self.validate_for_epoch()

            self.on_epoch_end()

            if epoch - self.best_epoch > self.patience:
                print(f"Stopping early because improvment did not happen since {self.best_epoch}th epoch")
                break

        return self.on_train_end()

    def train_for_epoch(self):
        """Trains pytorch model for one complete epoch"""

        epoch_losses = {metric: np.full(len(self.train_loader), np.nan) for metric in self.to_monitor}

        num_outs = self.model.num_outs

        for i, (batch_x, batch_y) in enumerate(self.train_loader):
            pred_y = self.model(batch_x.float())

            loss = self.criterion(batch_y.float().view(-1, num_outs), pred_y.view(-1, num_outs))
            loss = loss.float()
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            # calculate metrics for each mini-batch
            er = TorchMetrics(batch_y, pred_y)

            for k, v in epoch_losses.items():
                v[i] = getattr(er, k)().detach().item()

        # take the mean for all mini-batches without considering infinite values
        self.train_epoch_losses = {k: round(float(np.mean(v[np.isfinite(v)])), 4) for k, v in epoch_losses.items()}

        return

    def validate_for_epoch(self):
        """If validation data is available, then it performs the validation """

        if self.val_loader is not None:

            epoch_losses = {metric: np.full(len(self.val_loader), np.nan) for metric in self.to_monitor}

            for i, (batch_x, batch_y) in enumerate(self.val_loader):

                pred_y = self.model(batch_x.float())

                # calculate metrics for each mini-batch
                er = TorchMetrics(batch_y, pred_y)

                for k,v in epoch_losses.items():
                    v[i] = getattr(er, k)().detach().item()

            # take the mean for all mini-batches
            self.val_epoch_losses = {f'val_{k}': round(float(np.mean(v)), 4) for k, v in epoch_losses.items()}

            for k, v in self.val_epoch_losses.items():
                metric = k.split('_')[1]
                f1 = F[metric][0]
                f2 = F[metric][1]

                if f2(v , f1(self.val_metrics[k])):

                    torch.save(self.model.state_dict(), self._weight_fname(self.epoch, v))
                    self.best_epoch = self.epoch
                    break  # weights are saved for this epoch so no need to check other metrics

        return

    def _weight_fname(self, epoch, loss):
        return os.path.join(self.model.w_path, f"weights_{epoch}_{loss}")

    def _get_train_val_loaders(self, x, **kwargs):

        if isinstance(x, list) and len(x) == 1:
            x = x[0]
            if isinstance(x, torch.utils.data.Dataset):
                train_dataset = x
            else:
                train_dataset = to_torch_dataset(x, kwargs['y'])
        elif isinstance(x, torch.utils.data.Dataset):
            train_dataset = x

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=self.shuffle
                                  )
        val_dataset = None
        if 'validation_data' in kwargs:
            if isinstance(kwargs['validation_data'], torch.utils.data.Dataset):
                val_dataset = kwargs['validation_data']

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        return train_loader, val_loader

    def on_train_begin(self, x, **kwargs):
        print("{}{}{}".format('*'*25, 'Training Started', '*'*25))
        print("{:<7} {:<15} {:<15} {:<15} {:<15}".format('Epoch: ',
                                                         *self.train_metrics.keys(),
                                                         *self.train_metrics.keys()))
        print("{}".format('*' * 70))
        self.criterion = getattr(self.model, 'loss', self.loss)()
        self.optimizer = getattr(self.model, 'optimizer', self.optimizer)()

        self.train_loader, self.val_loader = self._get_train_val_loaders(x, **kwargs)

        return

    def on_train_end(self):
        self.train_metrics['loss'] = self.train_metrics.pop('mse')
        self.val_metrics['val_loss'] = self.val_metrics.pop('val_mse')
        
        class History(object):
            history = {}
            history.update(self.train_metrics)
            history.update(self.val_metrics)

        return History()

    def update_metrics(self):

        for k, v in self.train_metrics.items():
            v[self.epoch] = self.train_epoch_losses[k]

        if self.val_loader is not None:
            for k, v in self.val_metrics.items():
                v[self.epoch] = self.val_epoch_losses[k]

        return

    def on_epoch_begin(self):
        return

    def on_epoch_end(self):
        formatter = "{:<7}" + "{:<15.5f} " * (len(self.val_epoch_losses) + len(self.train_epoch_losses))

        if self.val_loader is None:

            for k, v in self.train_epoch_losses.items():
                f1 = F[k][0]
                f2 = F[k][1]

                if f2(v , f1(self.train_metrics[k])):

                    torch.save(self.model.state_dict(), self._weight_fname(self.epoch, v))
                    self.best_epoch = self.epoch
                    break

        print(formatter.format(self.epoch, *self.train_epoch_losses.values(), *self.val_epoch_losses.values()))

        self.update_metrics()

        return


def get_metrics_to_monitor(metrics):
    if metrics is None:
        _metrics = ['mse']
    elif isinstance(metrics, list):

        _metrics = metrics + ['mse']
    else:
        assert isinstance(metrics, str)
        _metrics = ['mse', metrics]

    return list(set(_metrics))
