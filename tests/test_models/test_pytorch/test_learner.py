import unittest

import os
import time
import site
dirname = os.path.dirname
ai4_dir = dirname(dirname(dirname(dirname(os.path.abspath(__file__)))))
site.addsitedir(ai4_dir)

import torch
import numpy as np
import torch.nn as nn
from torch import sigmoid
from torch.utils.data import IterableDataset

import matplotlib.pyplot as plt

from ai4water.models._torch import Learner


class Net(nn.Module):

    # Constructor
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        # hidden layer
        self.linear1 = nn.Linear(D_in, H)
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(H, D_out)

    # Prediction
    def forward(self, x):
        l1 = self.linear1(x)
        a1 = sigmoid(l1)
        a1 = self.dropout(a1)
        yhat = sigmoid(self.linear2(a1))
        return yhat


class NetArgs(Net):
    def forward(self, *args):
        tensors = []
        for t in args:
            tensors.append(t)

        # Ensure all shapes align
        base = tensors[0]
        # Sum all (broadcast-safe only if same shape)
        acc = torch.zeros_like(base)
        for t in tensors:
            acc = acc + t

        h = torch.relu(self.linear1(acc))
        h = self.dropout(h)
        return torch.sigmoid(self.linear2(h))


class Netkwargs(Net):
    def forward(self, **kwargs):
        tensors = []
        for key in kwargs:
            tensors.append(kwargs[key])

        # Ensure all shapes align
        base = tensors[0]
        # Sum all (broadcast-safe only if same shape)
        acc = torch.zeros_like(base)
        for t in tensors:
            acc = acc + t

        h = torch.relu(self.linear1(acc))
        h = self.dropout(h)
        return torch.sigmoid(self.linear2(h))


def PlotStuff(model, train_data, epoch, **kwargs):

    x, y = [], []
    for _x, _y in train_data:
        x.append(_x)
        y.append(_y)
    x = torch.stack(x)

    pred_y = model(x)
    x = x.detach().view(-1,)

    plt.close('all')
    plt.plot(x, pred_y.detach().view(-1,), label=('epoch ' + str(epoch)))
    plt.plot(x, torch.stack(y).view(-1,), 'r')
    plt.xlabel('x')

    plt.legend()
    #plt.show()

def criterion_cross(labels, outputs):
    out = -1 * torch.mean(labels * torch.log(outputs) + (1 - labels) * torch.log(1 - outputs))
    return out


def make_learner(epochs=10, use_cuda=False, in_features=1,
                 net_type='default', **kwargs):
    if net_type == 'args':
        model = NetArgs(in_features, 2, 1)
    elif net_type == 'dict':
        model = Netkwargs(in_features, 2, 1)
    else:
        model = Net(in_features, 2, 1)
    learner = Learner(model=model,
                      num_epochs=epochs,
                      patience=50,
                      batch_size=1,
                      shuffle=False,
                      use_cuda=use_cuda,
                    **kwargs
                      )

    learner.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    learner.loss = criterion_cross
    return learner


def get_xy(in_features=1):
    X = torch.arange(0, 40*in_features, 1).view(-1, in_features).type(torch.FloatTensor)
    # let Y be random 0 and 1
    Y = torch.zeros(X.shape[0])
    Y[torch.rand(X.shape[0]) > 0.5] = 1.0
    return X, Y


class DatasetArgs(torch.utils.data.Dataset):
    def __init__(self, in_features=1):
        self.in_features = in_features
        self.X = torch.arange(0, 40*in_features, 1).view(-1, in_features).type(torch.FloatTensor)
        self.Y = torch.zeros(self.X.shape[0])
        self.Y[(self.X[:, 0] > -4) & (self.X[:, 0] < 4)] = 1.0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.X[idx]*0.5], self.Y[idx]


class DatasetDict(torch.utils.data.Dataset):
    def __init__(self, in_features=1):
        self.in_features = in_features
        self.X = torch.arange(0, 40*in_features, 1).view(-1, in_features).type(torch.FloatTensor)
        self.Y = torch.zeros(self.X.shape[0])
        self.Y[(self.X[:, 0] > -4) & (self.X[:, 0] < 4)] = 1.0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {'var1': self.X[idx], 'var2': self.X[idx]*0.5}, self.Y[idx]


class IterDataset(IterableDataset):
    def __init__(self, generator, **kwargs):
        self.generator = generator
        self.kwargs = kwargs

    def __iter__(self):
        return self.generator(**self.kwargs)


def get_iterdataset(in_features=1):
    def generator():
        for i in range(10):
            yield np.random.random((1, in_features)), np.random.random((1, 1))

    ds = IterDataset(generator)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)


class TestLearner(unittest.TestCase):

    def test_docstring(self):
        time.sleep(1)  # to ensure different timestamp for weight files
        learner = make_learner()
        X, Y = get_xy()

        learner.fit(x=X,
                        y=Y,
                        callbacks = [{'after_epochs': 3, 'func': PlotStuff}]
                        )
        m = learner.evaluate(X, y=Y, metrics=['r2', 'nse', 'mape'])
        assert len(m) == 3
        p = learner.predict(X, y=Y, name='training')
        assert isinstance(p, np.ndarray)

        return

    def test_multi_ins(self):
        learner = make_learner(in_features=14, epochs=4)
        X, Y = get_xy(in_features=14)
        learner.fit(x=X, y=Y)
        return
    
    def test_IterDataset(self):
        learner = make_learner(in_features=2, epochs=4)
        ds = get_iterdataset(in_features=2)
        h = learner.fit(ds)
        return
    
    def test_IterDataset_for_val(self):
        learner = make_learner(in_features=2, epochs=4)
        ds = get_iterdataset(in_features=2)
        h = learner.fit(ds, validation_data=ds)
        return

    def test_wandb(self):
        try:
            import wandb
        except ImportError:
            return
        learner = make_learner(in_features=2, epochs=4,
                               verbosity=0,
                               wandb_config=dict(project='test_ai4water'))
        X, Y = get_xy(in_features=2)
        _ = learner.fit(x=X, y=Y)
        return
    
    def test_max_time(self):
        time.sleep(1)  # to ensure different timestamp for weight files
        learner = make_learner(in_features=2, epochs=100, 
                               max_time=0.00027 # roughly 1 seconds
                               )
        X, Y = get_xy(in_features=2)
        _ = learner.fit(x=X, y=Y)
        assert learner.stopped_early_ == 2, learner.stopped_early_
        return
    
    def test_avg_func(self):
        learner = make_learner(in_features=2, epochs=2,)
        X, Y = get_xy(in_features=2)
        learner.avg_fn = np.nanmedian
        _ = learner.fit(x=X, y=Y)        
        return

    def test_w_path(self):
        learner = make_learner(in_features=2, epochs=2,)
        assert os.path.exists(learner.w_path)
        return

    def test_train_for_single_epoch(self):
        time.sleep(1)  # to ensure different timestamp for weight files
        # weights should be saved when when model is trained even for single epoch
        learner = make_learner(in_features=2, epochs=1,)
        X, Y = get_xy(in_features=2)
        learner.fit(x=X, y=Y)
        assert len(os.listdir(learner.w_path)) > 0
        return

    def train_for_two_epochs(self):
        # two weights should be saved when when model is trained for two epochs
        # this can not always be true if metrics in second epoch is not better
        # than metrics in first epoch, however, here it should be true!
        learner = make_learner(in_features=2, epochs=2,)
        X, Y = get_xy(in_features=2)
        learner.fit(x=X, y=Y)
        assert len(os.listdir(learner.w_path)) > 1
        return

    def test_train_for_single_epoch_with_val_data(self):       
        # weights should be saved when when model is trained even for single epoch
        # wait for 1 second to ensure different timestamp
        time.sleep(1)
        learner = make_learner(in_features=2, epochs=1,)
        X, Y = get_xy(in_features=2)
        learner.fit(x=X, y=Y, validation_data=(X, Y))
        assert len(os.listdir(learner.w_path)) > 0
        return

    def train_for_two_epochs_with_val_data(self):
        learner = make_learner(in_features=2, epochs=2)
        X, Y = get_xy(in_features=2)
        learner.fit(x=X, y=Y, validation_data=(X, Y))
        assert len(os.listdir(learner.w_path)) > 1
        return

    def test_use_cuda(self):
        import torch
        use_cuda = False
    
        if torch.cuda.is_available():
            use_cuda = True
            print("CUDA is available. Testing on CUDA device.")
        learner = make_learner(epochs=2, use_cuda=use_cuda)
    
        if torch.cuda.is_available():
            assert next(learner.model.parameters()).is_cuda
    
        X, Y = get_xy()
    
        learner.fit(x=X, y=Y)
        return

    def test_loader_with_list_of_args(self):
        # test when the loader yields a list of inputs

        learner = make_learner(in_features=3, epochs=2, net_type='args')
        dataset = DatasetArgs(in_features=3)
        learner.fit(dataset)
        p1 = learner.predict(dataset)
        p2 = learner.predict(dataset)
        self.assertEqual(p1.shape, p2.shape)
        self.assertTrue(np.allclose(p1, p2), "Predictions differ on repeated calls with list input.")
        return

    def test_loader_with_dict(self):
        # test when the loader yields a dict of inputs
        learner = make_learner(in_features=3, epochs=2, net_type='dict')
        dataset = DatasetDict(in_features=3)
        learner.fit(dataset)
        p1 = learner.predict(dataset)
        p2 = learner.predict(dataset)
        self.assertEqual(p1.shape, p2.shape)
        self.assertTrue(np.allclose(p1, p2), "Predictions differ on repeated calls with dict input.")
        return


    def test_predict_consistency_after_training_single_tensor(self):
        learner = make_learner(in_features=3, epochs=2)
        X, Y = get_xy(in_features=3)
        learner.fit(x=X, y=Y)
        p1 = learner.predict(X, y=Y)
        p2 = learner.predict(X, y=Y)
        self.assertTrue(np.allclose(p1, p2), "Predictions differ on repeated calls with single tensor.")
        return


if __name__ == "__main__":
    unittest.main()