import numpy as np

import torch
from torch.utils.data import Dataset


class TorchDataset(Dataset):

    def __init__(self, x, y=None):
        self.x_is_list = False

        if isinstance(x, list):
            self.x_is_list = True

        self.x = x
        self.y = y

        if y is None:
            self.y = torch.tensor(data=np.full(shape=(self.__len__(), 0), fill_value=np.nan))

    def __len__(self):

        if self.x_is_list:
            return len(self.x[0])

        return len(self.x)

    def __getitem__(self, item):

        if self.x_is_list:
            x = [self.x[i][item] for i in range(len(self.x))]
        else:
            x = self.x[item]

        return x, self.y[item]


def to_torch_dataset(x, y=None):
    return TorchDataset(x, y=y)


class TorchMetrics(object):

    def __init__(self, true, predicted):

        self.true = true.view(-1, )
        self.predicted = predicted.view(-1, )

    def r2(self):
        # https://stackoverflow.com/a/66992970/5982232
        target_mean = torch.mean(self.predicted)
        ss_tot = torch.sum((self.predicted - target_mean) ** 2)
        ss_res = torch.sum((self.predicted - self.true) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2

    def mape(self):
        return (self.predicted - self.true).abs() / (self.true.abs() + 1e-8)

    def nse(self):
        _nse = 1 - sum((self.predicted - self.true) ** 2) / sum((self.true - torch.mean(self.true)) ** 2)
        return _nse

    def pbias(self):
        return 100.0 * sum(self.predicted - self.true) / sum(self.true)

    def mse(self):
        return torch.mean((self.true - self.predicted) ** 2)

    def rmse(self):
        return torch.sqrt(torch.mean(self.true - self.predicted)**2)
