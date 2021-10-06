import os
from typing import Union

import numpy as np
import matplotlib.pyplot as plt

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

from .backend import torch

# only so that docs can be built without having torch to be installed
try:
    from .utils.torch_utils import to_torch_dataset, TorchMetrics
except ModuleNotFoundError:
    to_torch_dataset, TorchMetrics = None, None

if torch is not None:
    from .pytorch_attributes import LOSSES
else:
    LOSSES = {}

from .utils.utils import dateandtime_now, find_best_weight
from ai4water.post_processing.SeqMetrics import RegressionMetrics
from .utils.visualizations import regplot

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

    def __init__(self, num_epochs, to_monitor=None, use_cuda=None, path=None):
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
        self.best_epoch = 0  # todo,
        self.use_cuda = use_cuda if use_cuda is not None else torch.cuda.is_available()

        def_path = path if path is not None else os.path.join(os.getcwd(), 'results', dateandtime_now())
        if not os.path.exists(def_path):
            if not os.path.isdir(def_path):
                os.makedirs(def_path)
            else:
                os.mkdir(def_path)
        self.path = def_path

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
        if isinstance(x, str):
            x = LOSSES[x.upper()]()
        self._loss = x

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, x):
        self._path = x

    def _device(self):
        if self.use_cuda:
            return torch.device("cuda")
        else:
            return torch.device("cpu")

class Learner(AttributeContainer):
    """Trains the pytorch model. Motivated from fastai"""

    def __init__(self,
                 model,  # torch.nn.Module,
                 batch_size: int = 32,
                 num_epochs: int = 14,
                 patience: int = 100,
                 shuffle: bool = True,
                 to_monitor: list = None,
                 use_cuda:bool = False,
                 path: str = None,
                 wandb_config:dict = None,
                 verbosity=1,
                 **kwargs
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
            use_cuda : whether to use cuda or not
            to_monitor : list of metrics to monitor
            path : path to save results/weights
            wandb_config : config for wandb

        Example
        --------
        ```python
        >>>from torch import nn
        >>>class Net(nn.Module):
        >>>    def __init__(self, D_in, H, D_out):
        ...        super(Net, self).__init__()
        ...        # hidden layer
        ...        self.linear1 = nn.Linear(D_in, H)
        ...        self.linear2 = nn.Linear(H, D_out)
        >>>    def forward(self, x):
        ...        l1 = self.linear1(x)
        ...        a1 = sigmoid(l1)
        ...        yhat = sigmoid(self.linear2(a1))
        ...        return yhat
        ...
        >>>learner = Learner(model=Net(1, 2, 1),
        ...                      num_epochs=501,
        ...                      patience=50,
        ...                      batch_size=1,
        ...                      shuffle=False)
        ...
        >>>learner.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>>def criterion_cross(labels, outputs):
        ...    out = -1 * torch.mean(labels * torch.log(outputs) + (1 - labels) * torch.log(1 - outputs))
        ...    return out
        >>>learner.loss = criterion_cross
        ...
        >>>X = torch.arange(-20, 20, 1).view(-1, 1).type(torch.FloatTensor)
        >>>Y = torch.zeros(X.shape[0])
        >>>Y[(X[:, 0] > -4) & (X[:, 0] < 4)] = 1.0
        ...
        >>>learner.fit(X, Y)
        >>>metrics = learner.evaluate(X, y=Y, metrics=['r2', 'nse', 'mape'])
        >>>t = learner.predict(X, y=Y, name='training')
        ```
        """
        super().__init__(num_epochs, to_monitor, path=path, use_cuda=use_cuda)

        if self.use_cuda:
            model = model.to(self._device())

        self.model = model
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.patience = patience
        self.wandb_config = wandb_config
        self.verbosity = verbosity

    def fit(self, x, y=None, validation_data=None, **kwargs):
        """Runs the training loop for pytorch model.

        Arguments:
            x : Can be one of following

                - an instance of torch.Dataset, y will be ignored
                - an instance of torch.DataLoader, y will be ignored
                - a torch tensor containing input data for each example
                - a numpy array
                - a list of torch tensors or numpy arrays
            y : if `x` is torch tensor, then `y` is the label/target for
                each corresponding example.
            validation_data : can be one of following:
                - an instance of torch.Dataset
                - an instance of torch.DataLoader
                - a tuple of x,y pairs where x and y are tensors
                Default is None, which means no validation is performed.
            kwargs : can be `callbacks` For example to use a callable
                as callback use following
                ```python
                callbacks = [{'after_epochs': 300, 'func': PlotStuff}]
                ```
                where `PlotStuff` is a callable.
                Each `callable` is provided with following keyword arguments
                    - epoch : the current epoch at which callable is called.
                    - model : the model
                    - train_data : training data_loader
                    - val_data : validation data_loader

        """
        self.on_train_begin(x, y=y, validation_data=validation_data, **kwargs)

        for epoch in range(self.num_epochs):

            self.epoch = epoch

            self.train_for_epoch()
            self.validate_for_epoch()

            self.on_epoch_end()

            if epoch - self.best_epoch > self.patience:
                if self.verbosity > 0:
                    print(f"Stopping early because improvment in loss did not happen since {self.best_epoch}th epoch")
                break

        return self.on_train_end()

    def predict(self,
                x,
                y=None,
                batch_size: int = None,
                reg_plot: bool = True,
                name: str = None,
                **kwargs) -> np.ndarray:
        """Makes prediction on the given data
        Arguments:
            x : data on which to evalute. It can be

                - a torch.utils.data.Dataset
                - a torch.utils.data.DataLoader
                - a torch.Tensor
                - a numpy array
                - a list of torch tensors numpy arrays
            y : only relevent if `x` is torch.Tensor. It comprises labels for
                correspoing x.
            batch_size : None means make prediction on whole data in one go
            reg_plot : whether to plot regression line or not
            name : string to be used for title and name of saved plot
        Returns:
            predicted output as numpy array
        """
        true, pred = self._eval(x=x, y=y, batch_size=batch_size)

        if reg_plot and true.size > 0.0:
            regplot(true, pred, name=name)
            plt.savefig(os.path.join(self.path, f'{name}_regplot.png'))

        return pred

    def _eval(self, x, y=None, batch_size=None):
        loader, _ = self._get_loader(x=x, y=y, batch_size=batch_size, shuffle=False)

        true, pred = [], []

        for i, (batch_x, batch_y) in enumerate(loader):

            batch_y, pred_y = self.eval(batch_x, batch_y)

            true.append(batch_y.detach().cpu().numpy())
            pred.append(pred_y.detach().cpu().numpy())

        true = np.concatenate(true)
        pred = np.concatenate(pred)

        return true, pred

    def eval(self, batch_x, batch_y):
        """Calls the model with x and y data and returns trues and preds"""
        batch_x = batch_x if isinstance(batch_x, list) else [batch_x]

        batch_x = [tensor.float() for tensor in batch_x]

        if self.use_cuda:
            batch_x = [tensor.cuda() for tensor in batch_x]
            batch_y = batch_y.cuda()

        pred_y = self.model(*batch_x)

        return batch_y, pred_y

    def evaluate(self,
                 x,
                 y=None,
                 batch_size: int = None,
                 metrics: Union[str, list] = 'r2',
                 **kwargs
                 ):
        """Evaluates the `model` on the given data.

        Arguments:
            x : data on which to evalute. It can be

                - a torch.utils.data.Dataset
                - a torch.utils.data.DataLoader
                - a torch.Tensor
                - a numpy.ndarray
                - a list of torch tensors numpy arrays
            y : only relevent if `x` is torch.Tensor. It comprises labels for
                correspoing x.
            batch_size : None means make prediction on whole data in one go
            metrics : name of performance metric to measure. It can be a single metric
                or a list of metrics. Allowed metrics are anyone from
                `ai4water.post_processing.SeqMetrics.RegressionMetrics`
            kwargs :
            """
        if isinstance(metrics, str):
            metrics = [metrics]

        assert isinstance(metrics, list)

        true, pred = self._eval(x=x, y=y, batch_size=batch_size)

        evaluator = RegressionMetrics(true, pred)

        errors = {}

        for m in metrics:
            errors[m] = getattr(evaluator, m)()

        return errors

    def train_for_epoch(self):
        """Trains pytorch model for one complete epoch"""

        epoch_losses = {metric: np.full(len(self.train_loader), np.nan) for metric in self.to_monitor}

        # todo, it would be better to avoid reshaping/view at all
        if hasattr(self.model, 'num_outs'):
            num_outs = self.model.num_outs
        else:
            num_outs = self.num_outs

        for i, (batch_x, batch_y) in enumerate(self.train_loader):

            batch_y, pred_y = self.eval(batch_x, batch_y)

            if num_outs:
                batch_y = batch_y.float().view(len(batch_y), num_outs)
                pred_y = pred_y.view(len(pred_y), num_outs)

            loss = self.criterion(batch_y, pred_y)
            loss = loss.float()
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            # calculate metrics for each mini-batch
            er = TorchMetrics(batch_y, pred_y)

            for k, v in epoch_losses.items():
                v[i] = getattr(er, k)().detach().item()
            #epoch_losses['mse'][i] = loss.detach()

        # take the mean for all mini-batches without considering infinite values
        self.train_epoch_losses = {k: round(float(np.mean(v[np.isfinite(v)])), 4) for k, v in epoch_losses.items()}

        if self.wandb_config is not None:
            wandb.log(self.train_epoch_losses, step=self.epoch)

        if self.use_cuda:
            torch.cuda.empty_cache()
        return

    def validate_for_epoch(self):
        """If validation data is available, then it performs the validation """

        if self.val_loader is not None:

            epoch_losses = {metric: np.full(len(self.val_loader), np.nan) for metric in self.to_monitor}

            for i, (batch_x, batch_y) in enumerate(self.val_loader):

                batch_y, pred_y = self.eval(batch_x, batch_y)

                # calculate metrics for each mini-batch
                er = TorchMetrics(batch_y, pred_y)

                for k, v in epoch_losses.items():
                    v[i] = getattr(er, k)().detach().item()

            # take the mean for all mini-batches
            self.val_epoch_losses = {f'val_{k}': round(float(np.mean(v)), 4) for k, v in epoch_losses.items()}

            if self.wandb_config is not None:
                wandb.log(self.val_epoch_losses, step=self.epoch)

            for k, v in self.val_epoch_losses.items():
                metric = k.split('_')[1]
                f1 = F[metric][0]
                f2 = F[metric][1]

                if f2(v, f1(self.val_metrics[k])):
                    torch.save(self.model.state_dict(), self._weight_fname(self.epoch, v))
                    self.best_epoch = self.epoch
                    break  # weights are saved for this epoch so no need to check other metrics

        return

    def _weight_fname(self, epoch, loss):

        return os.path.join(getattr(self.model, 'w_path', self.path), f"weights_{epoch}_{loss}")

    def _get_train_val_loaders(self, x, y=None, validation_data=None):

        train_loader, self.num_outs = self._get_loader(x=x, y=y, batch_size=self.batch_size, shuffle=self.shuffle)
        val_loader, _ = self._get_loader(x=validation_data, batch_size=self.batch_size, shuffle=self.shuffle)

        return train_loader, val_loader

    def on_train_begin(self, x, y=None, validation_data=None, **kwargs):

        self.cbs = kwargs.get('callbacks', [])  # no callback by default

        if self.verbosity > 0:
            print("{}{}{}".format('*' * 25, 'Training Started', '*' * 25))
            formatter = "{:<7}" + " {:<15}" * (len(self.train_metrics) + len(self.val_metrics))
            print(formatter.format('Epoch: ',
                                   *self.train_metrics.keys(),
                                   *self.train_metrics.keys()))

            print("{}".format('*' * 70))
        if hasattr(self.model, 'loss'):
            self.criterion = self.model.loss()
        else:
            self.criterion = self.loss

        if hasattr(self.model, 'get_optimizer'):
            self.optimizer = self.model.get_optimizer()
        else:
            self.optimizer = self.optimizer

        self.train_loader, self.val_loader = self._get_train_val_loaders(x, y=y, validation_data=validation_data)

        if self.wandb_config is not None:
            assert wandb is not None
            assert isinstance(self.wandb_config, dict)

            wandb.init(name=os.path.basename(self.path),
                       project=self.wandb_config.get('probject', 'test_project'),
                       notes='This is Learner from AI4Water test run',
                       tags=['ai4water', 'pytorch'],
                       entity=self.wandb_config.get('entity', ''))
        return

    def on_train_end(self):

        self.update_weights()

        self.train_metrics['loss'] = self.train_metrics.pop('mse')
        self.val_metrics['val_loss'] = self.val_metrics.pop('val_mse')

        class History(object):
            history = {}
            history.update(self.train_metrics)
            history.update(self.val_metrics)

        setattr(self, 'history', History())

        if self.wandb_config is not None:
            wandb.finish()

        return History()

    def update_weights(self, weight_file_path: str = None):
        """If `weight_file_path` is not given then it finds the best weights
        and updates the model with best wieghts.

        Arguments:
            weight_file_path : complete path of weights which are to be loaded
        """

        if weight_file_path:
            assert os.path.exists(weight_file_path)
            best_weights = os.path.basename(weight_file_path)
        else:
            w_path = getattr(self.model, 'w_path', self.path)
            best_weights = find_best_weight(w_path, epoch_identifier=self.best_epoch)
            if best_weights is not None:
                weight_file_path = os.path.join(w_path, best_weights)

        if best_weights is not None:
            fpath = os.path.splitext(weight_file_path)[0]  # we are not saving the whole model but only state_dict
            self.model.load_state_dict(torch.load(fpath))
            if self.verbosity > 0:
                print("{} Successfully loaded weights from {} file {}".format('*' * 10, best_weights, '*' * 10))
        return

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
        formatter = "{:<7}" + "{:<15.7f} " * (len(self.val_epoch_losses) + len(self.train_epoch_losses))

        if self.val_loader is None:  # otherwise model is already saved based upon validation performance

            for k, v in self.train_epoch_losses.items():
                f1 = F[k][0]
                f2 = F[k][1]

                if f2(v, f1(self.train_metrics[k])):
                    torch.save(self.model.state_dict(), self._weight_fname(self.epoch, v))
                    self.best_epoch = self.epoch
                    break

        if self.verbosity > 0:
            print(formatter.format(self.epoch, *self.train_epoch_losses.values(), *self.val_epoch_losses.values()))

        for cb in self.cbs:
            if self.epoch % cb['after_epochs'] == 0:
                cb['func'](epoch=self.epoch,
                           model=self.model,
                           train_data=self.train_loader,
                           val_data=self.val_loader
                           )

        self.update_metrics()

        return

    def _get_loader(self, x, y=None, batch_size=None, shuffle=True):

        data_loader = None
        num_outs = None

        if x is None:
            return None, None

        if isinstance(x, list):
            if len(x) == 1:
                x = x[0]
                if isinstance(x, torch.utils.data.Dataset):
                    dataset = x
                else:
                    dataset = to_torch_dataset(x, y)
            else:
                dataset = to_torch_dataset(x, y)

        elif isinstance(x, np.ndarray):
            if y is not None:
                assert isinstance(y, np.ndarray)
            dataset = to_torch_dataset(x, y)

        elif isinstance(x, torch.utils.data.Dataset):
            dataset = x

            if len(x[0][1].shape) == 1:
                num_outs = 1
            else:
                num_outs = x[0][1].shape[-1]

        elif isinstance(x, torch.utils.data.DataLoader):
            data_loader = x

        elif isinstance(x, torch.Tensor):
            if len(x.shape) == 1:
                num_outs = 1
            else:
                num_outs = x.shape[-1]

            dataset = to_torch_dataset(x=x, y=y)

        elif isinstance(x, tuple):  # x is tuple of x,y pairs
            assert len(x) == 2
            dataset = to_torch_dataset(x=x[0], y=x[1])

        else:
            raise NotImplementedError(f'unrecognized data of type {x.__class__.__name__} given')

        if data_loader is None:

            if batch_size is None:
                batch_size = len(dataset)

            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle
            )

        return data_loader, num_outs

    def plot_model_using_tensorboard(self,
                                     x=None,
                                     path='tensorboard/tensorboard'
                                     ):
        """Plots the neural network on tensorboard
        Arguments:
            x : torch.Tensor
                input to the model
            path : str
                path to save tensorboard graph
        """
        from torch.utils.tensorboard import SummaryWriter

        # default `log_dir` is "runs" - we'll be more specific here
        writer = SummaryWriter(path)
        if x is None:
            x, _ = iter(self.train_loader).next()
        writer.add_graph(self.model, x)
        writer.close()
        return

    def plot_model(self, y=None):
        """Helper function to plot dot diagram of model using torchviz module.
        Arguments:
            y : torch.Tensor
                output tensor
            """
        try:
            from torchviz import make_dot
        except ModuleNotFoundError:
            print("You must install torchviz to plot model."
                  "see https://github.com/szagoruyko/pytorchviz#installation for installation")
            return

        if y is None:
            x, _ = iter(self.train_loader).next()
            y = self.model(x)

        fname = os.path.join(self.path, 'model.png')
        dot = make_dot(y, dict(self.model.named_parameters()),
                       show_attrs=True,
                       show_saved=True)

        dot.render(fname)

        return dot


def get_metrics_to_monitor(metrics):
    if metrics is None:
        _metrics = ['mse']
    elif isinstance(metrics, list):

        _metrics = metrics + ['mse']
    else:
        assert isinstance(metrics, str)
        _metrics = ['mse', metrics]

    return list(set(_metrics))
