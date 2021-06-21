import os

import numpy as np

from .backend import torch
from .utils.torch_utils import to_torch_dataset

DataLoader = torch.utils.data.DataLoader


class AttributeContainer(object):

    def __init__(self, num_epochs, to_monitor=None):
        self.to_monitor = get_metrics_to_monitor(to_monitor)
        self.num_epochs = num_epochs

        self.epoch = 0
        self.val_loader = None
        self.train_loader = None
        self.criterion = None
        self.optimizer = None
        self.val_epoch_loss = None
        self.train_epoch_loss = None
        self.train_metrics = {metric: np.full(num_epochs, np.nan) for metric in self.to_monitor}
        self.val_metrics = {f'val_{metric}': np.full(num_epochs, np.nan) for metric in self.to_monitor}
        self.best_epoch = 0   # todo,


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

        epoch_loss = np.full(len(self.train_loader), np.nan)

        for i, (batch_x, batch_y) in enumerate(self.train_loader):
            pred_y = self.model(batch_x.float())

            loss = self.criterion(batch_y.float(), pred_y)
            loss = loss.float()
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            epoch_loss[i] = loss.item()

        self.train_epoch_loss = round(float(np.average(epoch_loss)), 4)

        return

    def validate_for_epoch(self):
        """If validation data is available, then it performs the validation """
        if self.val_loader is not None:

            epoch_loss = np.full(len(self.val_loader), np.nan)

            for i, (batch_x, batch_y) in enumerate(self.val_loader):
                pred_y = self.model(batch_x.float())

                loss = self.criterion(batch_y.float(), pred_y)

                epoch_loss[i] = loss.item()

            self.val_epoch_loss = round(float(np.mean(epoch_loss)), 4)

            if self.val_epoch_loss<np.nanmin(self.val_metrics['val_loss']):
                torch.save(self.model.state_dict(), self._weight_fname(self.epoch, self.val_epoch_loss))
                self.best_epoch = self.epoch

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

        self.criterion = self.model.loss()
        self.optimizer = self.model.get_optimizer()

        self.train_loader, self.val_loader = self._get_train_val_loaders(x, **kwargs)

        return

    def on_train_end(self):

        class History(object):
            history = {}
            history.update(self.train_metrics)
            history.update(self.val_metrics)

        return History()

    def update_metrics(self):

        self.train_metrics['loss'][self.epoch] = self.train_epoch_loss
        self.val_metrics['val_loss'][self.epoch] = self.val_epoch_loss

        return

    def on_epoch_begin(self):
        return

    def on_epoch_end(self):

        if self.val_loader is None:
            if self.train_epoch_loss < np.nanmin(self.train_metrics['loss']):
                torch.save(self.model.state_dict(), self._weight_fname(self.epoch, self.train_epoch_loss))

                self.best_epoch = self.epoch

        print(f' epoch: {self.epoch} loss: {self.train_epoch_loss} val_loss: {self.val_epoch_loss}')

        self.update_metrics()

        return


def get_metrics_to_monitor(metrics):
    if metrics is None:
        _metrics = ['loss']
    elif isinstance(metrics, list):
        if 'loss' in metrics:
            _metrics = metrics
        else:
            _metrics = metrics + ['loss']
    else:
        assert isinstance(metrics, str)
        _metrics = ['loss', metrics]

    return _metrics
