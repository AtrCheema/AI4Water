import unittest

import os
import sys
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
        self.linear2 = nn.Linear(H, D_out)

    # Prediction
    def forward(self, x):
        l1 = self.linear1(x)
        a1 = sigmoid(l1)
        yhat = sigmoid(self.linear2(a1))
        return yhat

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


def make_learner(epochs=501, use_cuda=False, in_features=1, **kwargs):
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
    Y = torch.zeros(X.shape[0])
    Y[(X[:, 0] > -4) & (X[:, 0] < 4)] = 1.0
    return X, Y


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
        learner = make_learner()
        X, Y = get_xy()


        learner.fit(x=X,
                        y=Y,
                        callbacks = [{'after_epochs': 300, 'func': PlotStuff}]
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
                               wandb_config=dict(project='test', entity='ai4water'))
        X, Y = get_xy(in_features=2)
        h = learner.fit(x=X, y=Y)
        return

    # def test_use_cuda(self):
    #     import torch
    #     use_cuda = False
    #
    #     if torch.cuda.is_available():
    #         use_cuda = True
    #     learner = make_learner(epochs=5, use_cuda=use_cuda)
    #
    #     if torch.cuda.is_available():
    #         assert next(learner.model.parameters()).is_cuda
    #
    #     X, Y = get_xy()
    #
    #     learner.fit(x=X, y=Y)
    #     return

if __name__ == "__main__":
    unittest.main()