

import torch
import torch.nn as nn
from torch import sigmoid
import matplotlib.pyplot as plt

from AI4Water.pytorch_training import Learner


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

    plt.plot(x, pred_y.detach().view(-1,), label=('epoch ' + str(epoch)))
    plt.plot(x, torch.stack(y).view(-1,), 'r')
    plt.xlabel('x')

    plt.legend()
    plt.show()

def criterion_cross(labels, outputs):
    out = -1 * torch.mean(labels * torch.log(outputs) + (1 - labels) * torch.log(1 - outputs))
    return out


model = Net(1, 2, 1)
learner = Learner(model=model,
                  num_epochs=301,
                  batch_size=1,
                  shuffle=False)
learner.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
learner.loss = criterion_cross

X = torch.arange(-20, 20, 1).view(-1, 1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0])
Y[(X[:, 0] > -4) & (X[:, 0] < 4)] = 1.0


h = learner.fit(x=X,
                y=Y,
                callbacks = [{'after_epochs': 300, 'func': PlotStuff}]
                )
m = learner.evaluate(X, y=Y, metrics=['r2', 'nse', 'mape'])
assert len(m) == 3
