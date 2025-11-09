
import gc
import time
from typing import Union, Tuple, List

from SeqMetrics import RegressionMetrics

from ai4water.backend import os, np, torch, pd
from ai4water.backend import wandb
from ai4water.postprocessing import ProcessPredictions
from ai4water.utils.utils import dateandtime_now, find_best_weight

# only so that docs can be built without having torch to be installed
try:
    from .utils import to_torch_dataset
except ModuleNotFoundError:
    to_torch_dataset = None

from .pytorch_attributes import LOSSES
from SeqMetrics.utils import METRIC_TYPES


METRIC_TYPES.update({'loss': 'min'})
F = {}
for k,v in METRIC_TYPES.items():
    if v == "max":
        F[k] = [np.nanmax, np.greater]
    elif v == "min":
        F[k] = [np.nanmin, np.less]
    else:
        raise ValueError(f"unknown metric type {v}")


class AttributeContainer(object):

    def __init__(
            self, 
            num_epochs, 
            to_monitor=None, 
            use_cuda=None,
            path=None, 
            verbosity=1
            ):
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
        self.verbosity = verbosity

        def_path = path if path is not None else os.path.join(os.getcwd(), 'results', dateandtime_now())
        if not os.path.exists(def_path) and verbosity >= 0:
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

    def __init__(
            self,
            model,  # torch.nn.Module,
            batch_size: int = 32,
            num_epochs: int = 14,
            patience: int = 100,
            shuffle: bool = True,
            to_monitor: List[str] = None,
            use_cuda:bool = False,
            mode: str = 'regression',
            path: str = None,
            wandb_config:dict = None,
            max_time: int = 100,
            verbosity=1,
            **kwargs
    ):
        """
        Initializes the Learner class

        Arguments:
            model : torch.nn.Module
                a pytorch model
            batch_size : int
                batch size
            num_epochs : Number of epochs for which to train the model
            patience : how many epochs to wait before stopping the training in
                case `to_monitor` does not improve.
            shuffle :
            use_cuda : whether to use cuda or not
            mode : str
                mode of the model. It can be one of following
                    - 'regression'
                    - 'classification'
                It is only used for postprocessing of predictions.
            to_monitor : List[str]
                list of metrics to monitor. It can be any performance metric from
                SeqMetrics
            wandb_config : dict
                config for wandb. If given, it must contain at least ``project`` key.
                The results will be logged to wandb however this requires wandb to be installed.
            max_time : int
                maximum time in hours for which to train the model. The training loop
                will stop after this time. This is useful when you want to stop the training
                after certain time.
            path : 
                path to save results/weights
            verbosity : int
                - 0 means nothing will be printed, 
                - 1 means metrics' values after each epoch will be printed.
                - 2 means loss values after each batch will be printed.

        Example
        -------
            >>> from torch import nn
            >>> import torch
            >>> from ai4water.models._torch import Learner
            ...
            >>> class Net(nn.Module):
            >>>    def __init__(self, D_in, H, D_out):
            ...        super(Net, self).__init__()
            ...        # hidden layer
            ...        self.linear1 = nn.Linear(D_in, H)
            ...        self.linear2 = nn.Linear(H, D_out)
            >>>    def forward(self, x):
            ...        l1 = self.linear1(x)
            ...        a1 = torch.sigmoid(l1)
            ...        yhat = torch.sigmoid(self.linear2(a1))
            ...        return yhat
            ...
            >>> learner = Learner(model=Net(1, 2, 1),
            ...                      num_epochs=501,
            ...                      patience=50,
            ...                      batch_size=1,
            ...                      shuffle=False)
            ...
            >>> learner.optimizer = torch.optim.SGD(learner.model.parameters(), lr=0.1)
            >>> def criterion_cross(labels, outputs):
            ...    out = -1 * torch.mean(labels * torch.log(outputs) + (1 - labels) * torch.log(1 - outputs))
            ...    return out
            >>> learner.loss = criterion_cross
            ...
            >>> X = torch.arange(-20, 20, 1).view(-1, 1).type(torch.FloatTensor)
            >>> Y = torch.zeros(X.shape[0])
            >>> Y[(X[:, 0] > -4) & (X[:, 0] < 4)] = 1.0
            ...
            >>> learner.fit(X, Y)
            >>> metrics = learner.evaluate(X, y=Y, metrics=['r2', 'nse', 'mape'])
            >>> t = learner.predict(X, y=Y, name='training')
        """
        super().__init__(
            num_epochs, 
            to_monitor, 
            path=path,
            use_cuda=use_cuda,
            verbosity=verbosity
            )

        if self.use_cuda:
            model = model.to(self._device())

        self.model = model
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.patience = patience
        self.mode = mode
        self.wandb_config = wandb_config
        self.use_wb = self._use_wb()

        self.max_time = max_time
        self.start_time = time.time()

        self.agg_fn = np.nanmean  # method to aggregate across batches for each epoch

    def _use_wb(self):
        return self.wandb_config is not None and wandb is not None

    @property
    def w_path(self)->Union[str, os.PathLike]:
        weight_path = getattr(self.model, 'w_path', None)
        if weight_path is None:
            weight_path = os.path.join(self.path, 'weights')
            if not os.path.exists(weight_path):
                os.makedirs(weight_path)
        return weight_path

    def fit(
            self,
            x,
            y=None,
            validation_data=None,
            **kwargs
    ):
        """Runs the training loop for pytorch model.

        Arguments
        ---------
            x :
                Can be one of following

                - an instance of torch.Dataset, y will be ignored
                - an instance of torch.DataLoader, y will be ignored
                - a torch tensor containing input data for each example
                - a numpy array or pandas DataFrame
                - a list of torch tensors or numpy arrays
            y :
                if `x` is torch tensor, then `y` is the label/target for
                each corresponding example.
            validation_data :
                can be one of following:
                - an instance of torch.Dataset
                - an instance of torch.DataLoader
                - a tuple of x,y pairs where x and y are tensors
                Default is None, which means no validation is performed.
            kwargs :
                can be `callbacks` For example to use a callable
                as callback use following

                >>> callbacks = [{'after_epochs': 300, 'func': PlotStuff}]

                where `PlotStuff` is a callable.
                Each `callable` is provided with following keyword arguments

                - epoch : the current epoch at which callable is called.
                - model : the model
                - train_data : training data_loader
                - val_data : validation data_loader

        """
        self.on_train_begin(x, y=y, validation_data=validation_data, **kwargs)

        for epoch in range(self.num_epochs):

            self.on_epoch_begin(epoch)

            self.epoch = epoch

            self.train_for_epoch()
            self.validate_for_epoch()

            self.on_epoch_end()

            if epoch - self.best_epoch > self.patience:
                if self.verbosity > 0:
                    print(f"Stopping early because improvment in loss did not happen since {self.best_epoch}th epoch")
                self.stopped_early_ = 1
                break
            
            time_in_hours = (time.time() - self.start_time) / 3600
            if time_in_hours > self.max_time:
                if self.verbosity > 0:
                    print(f"Stopping early because max_time of {self.max_time} hours is reached")
                self.stopped_early_ = 2
                break

        return self.on_train_end()

    def predict(
            self,
            x,
            y=None,
            batch_size: int = None,
            plots: List[str] = None,
            return_true:bool = False,
            **kwargs
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
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
            plots : The type of plots to draw. One or more from following
                are acceptable. For more see ai4water.postprocessing.ProcessPredictions

                    - ``regression``
                    - ``residual``
                    - ``prediction``
                    - ``edf``
            return_true : bool
                if set to True, then a tuple is returned wholse first element is
                true array and second array

        Returns:
            predicted output as numpy array. If return_true is True, then a tuple
            is returned whose first element is true array and second element is predicted
            array
        """
        true, pred = self._eval(x=x, y=y, batch_size=batch_size)

        if len(true) >0 and plots is not None and pred.size > 0.0:
            pp = ProcessPredictions(
                mode=self.mode,
                path=self.path,
                forecast_len=1, # todo, what if this is not satisfied
                output_features=None,
                plots=plots,
                show=bool(self.verbosity),
            )
            pp(true, pred, model=self)

            if self.use_wb:
                self.wb_run_.log_predict(true, pred, mode=self.mode, prefix=dateandtime_now())

        #if self.use_cuda:
        torch.cuda.empty_cache()
        gc.collect()

        if return_true:
            return true, pred

        return pred

    def _eval(
            self,
            x,
            y=None,
            batch_size=None
    )->Tuple[np.ndarray, np.ndarray]:
        """
        prepares the loader from x,y and iterate over batches present
        in loader. The results are concatenated as numpy array and returned as tuple
        
        This method is only called by `evaluate` and `predict` methods i.e. not
        during training.
        """
        loader, _ = self._get_loader(x=x, y=y, batch_size=batch_size, shuffle=False)

        true, pred = [], []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):

                batch_y, pred_y = self.eval(batch_x, batch_y)

                true.append(batch_y.detach().cpu().numpy())
                pred.append(pred_y.detach().cpu().numpy())

        true = np.concatenate(true)
        pred = np.concatenate(pred)

        del loader
        del batch_x
        del batch_y
        gc.collect()

        return true, pred

    def eval(self, batch_x, batch_y):
        """Calls the model with x and y data and returns trues and preds.

        Supports:
          - batch_x: Tensor | list[Tensor] | tuple[Tensor] | dict[str, Tensor]
          - Moves tensors to correct device
          - Casts only floating-point inputs to float32, preserves non-float dtypes
          - Preserves dtype for batch_y
        """
        device = self._device() if self.use_cuda else torch.device("cpu")

        def _to_device_and_cast(obj, cast_inputs=False):
            # Recursively move to device; cast only floating-point inputs if requested
            if torch.is_tensor(obj):
                t = obj.to(device)
                if cast_inputs and t.is_floating_point():
                    t = t.float()
                return t
            if isinstance(obj, (list, tuple)):
                items = [_to_device_and_cast(o, cast_inputs) for o in obj]
                return type(obj)(items) if isinstance(obj, tuple) else items
            if isinstance(obj, dict):
                return {k: _to_device_and_cast(v, cast_inputs) for k, v in obj.items()}
            return obj  # leave non-tensors as-is

        # Inputs: cast floating tensors to float32; Targets: preserve dtype
        batch_x = _to_device_and_cast(batch_x, cast_inputs=True)
        batch_y = _to_device_and_cast(batch_y, cast_inputs=False)

        # Dispatch call based on input type
        if isinstance(batch_x, dict):
            pred_y = self.model(**batch_x)
        elif isinstance(batch_x, (list, tuple)):
            pred_y = self.model(*batch_x)
        else:
            pred_y = self.model(batch_x)

        del batch_x

        return batch_y, pred_y

    def evaluate(
            self,
            x,
            y,
            batch_size: int = None,
            metrics: Union[str, list] = 'r2',
            **kwargs
    ):
        """
        Evaluates the `model` on the given data.

        Arguments:
            x : data on which to evalute. It can be

                - a torch.utils.data.Dataset
                - a torch.utils.data.DataLoader
                - a torch.Tensor
                - a numpy.ndarray
                - a list of torch tensors numpy arrays
            y : It comprises labels for
                correspoing x.
            batch_size : None means make prediction on whole data in one go
            metrics : name of performance metric to measure. It can be a single metric
                or a list of metrics. Allowed metrics are anyone from
                `ai4water.post_processing.SeqMetrics.RegressionMetrics`
            kwargs :

        Returns:
            if metrics is string the returned value is float otherwise
            it will be a dictionary
        """
        # todo y->pred is only converting tensor into numpy array
        true, pred = self._eval(x=x, y=y, batch_size=batch_size)

        evaluator = RegressionMetrics(true, pred)

        errors = {}

        if isinstance(metrics, str):
            errors = getattr(evaluator, metrics)()
        else:
            assert isinstance(metrics, list)
            for m in metrics:
                errors[m] = getattr(evaluator, m)()

        return errors

    def train_for_epoch(self):
        """Trains pytorch model for one complete epoch"""

        # empty list instead of np.full(len(self.train_loader), np.nan) 
        # because we can't determine len of generator datasets
        epoch_losses = {metric: [] for metric in self.to_monitor}

        # todo, it would be better to avoid reshaping/view at all
        if hasattr(self.model, 'num_outs'):
            num_outs = self.model.num_outs
        else:
            num_outs = self.num_outs
        
        batch_loss = 0.0

        for i, (batch_x, batch_y) in enumerate(self.train_loader):

            self.optimizer.zero_grad()
            
            # todo, feeding batch_y to eval is only putting it on right device
            # can we do it before?
            batch_y, pred_y = self.eval(batch_x, batch_y)

            if num_outs:
                batch_y = batch_y.float().view(len(batch_y), num_outs)
                pred_y = pred_y.view(len(pred_y), num_outs)

            loss = self.criterion(batch_y, pred_y)
            loss = loss.float()
            loss.backward()

            batch_loss += loss.detach().item()

            self.log_after_batch(i, batch_loss)

            self.optimizer.step()

            # calculate metrics for each mini-batch
            er = RegressionMetrics(batch_y.detach().cpu().numpy(), pred_y.detach().cpu().numpy())

            for metric in epoch_losses.keys():
                if metric == 'loss':
                    epoch_losses[metric].append(float(loss.detach().item()))
                else:
                    epoch_losses[metric].append(getattr(er, metric)())

        # take the mean/median for all mini-batches without considering infinite values
        self.train_epoch_losses = {k: round(float(self.agg_fn(np.array(v)[np.isfinite(v)])), 4) for k, v in epoch_losses.items()}

        return

    def log_after_batch(self, batch:int, batch_loss):
        if self.verbosity>1:
            print(f"\rEpoch: {self.epoch}, batch: {batch}, loss: {round(batch_loss/(batch+1),3)}", end='', flush=True)
        return

    def validate_for_epoch(self):
        """If validation data is available, then it performs the validation """

        if self.val_loader is not None:

            epoch_losses = {metric: [] for metric in self.to_monitor}

            for _, (batch_x, batch_y) in enumerate(self.val_loader):

                batch_y, pred_y = self.eval(batch_x, batch_y)

                # calculate metrics for each mini-batch  # todo : is detach.numpy expensive?
                er = RegressionMetrics(batch_y.detach().cpu().numpy(), pred_y.detach().cpu().numpy())

                for metric in epoch_losses.keys(): ###
                    if metric == 'loss':
                        val_loss = self.criterion(batch_y, pred_y)
                        epoch_losses[metric].append(val_loss.detach().item())
                    else:
                        epoch_losses[metric].append(getattr(er, metric)())

            # take the mean for all mini-batches
            self.val_epoch_losses = {f'val_{k}': round(float(self.agg_fn(v)), 4) for k, v in epoch_losses.items()}

            for k, v in self.val_epoch_losses.items():
                metric = k.split('_')[1]
                f1 = F[metric][0]
                f2 = F[metric][1]

                # for first epoch, the weights must be saved no matter what
                # the value of v is w.r.t its previous values!
                if self.epoch == 0:
                    torch.save(self.model.state_dict(), self._weight_fname(self.epoch, v))
                    self.best_epoch = self.epoch
                    break

                if f2(v, f1(self.val_metrics[k])):
                    torch.save(self.model.state_dict(), self._weight_fname(self.epoch, v))
                    self.best_epoch = self.epoch
                    break  # weights are saved for this epoch so no need to check other metrics

        return

    def _weight_fname(self, epoch, loss):

        return os.path.join(self.w_path, f"weights_{epoch}_{loss}")

    def _get_train_val_loaders(self, x, y=None, validation_data=None):

        train_loader, self.num_outs = self._get_loader(x=x,
                                                       y=y,
                                                       batch_size=self.batch_size,
                                                       shuffle=self.shuffle)
        val_loader, _ = self._get_loader(x=validation_data,
                                         batch_size=self.batch_size,
                                         shuffle=self.shuffle)

        return train_loader, val_loader

    def on_train_begin(self, x, y=None, validation_data=None, **kwargs):

        self.cbs = kwargs.get('callbacks', [])  # no callback by default

        if self.verbosity > 0:
            print("{}{}{}".format('*' * 25, 'Training Started', '*' * 25))
            formatter = "{:<7}" + " {:<15}" * (len(self.train_metrics) + len(self.val_metrics))

            print(formatter.format('Epoch: ',
                                   *self.train_metrics.keys(),
                                   *self.val_metrics.keys()))

            print("{}".format('*' * 70))
        if hasattr(self.model, 'loss'):
            self.criterion = self.model.loss()  # todo : should we initialize the loss or not
        else:
            self.criterion = self.loss

        if hasattr(self.model, 'get_optimizer'):
            self.optimizer = self.model.get_optimizer()
        else:
            self.optimizer = self.optimizer

        self.train_loader, self.val_loader = self._get_train_val_loaders(
            x,
            y=y,
            validation_data=validation_data)

        self.wb_run_ = self._maybe_init_wandb()

        return
    
    def _maybe_init_wandb(self):
        """initializes the wandb and creates a run for the model if wandb_config is not None.
        """
        wb_run = None
        if self.use_wb:

            assert isinstance(self.wandb_config, dict)

            from ..._wb import WB

            iconfig = dict(
                name=os.path.basename(self.path),
                project=self.wandb_config.get('probject', 'test_project'),
                notes='This is Learner from AI4Water',
                tags=['ai4water', 'pytorch', 'learner'],
                )
            
            iconfig.update(self.wandb_config)
            wb_run = WB(self.model, iconfig)

        return wb_run

    def on_train_end(self):

        self.update_weights()

        class History(object):
            history = {}
            history.update(self.train_metrics)
            history.update(self.val_metrics)

        setattr(self, 'history', History())

        if self.use_wb:
            self.wb_run_.log_loss_curve(History().history)
            #self.wb_run_.finish()

        return History()

    def update_weights(self, weight_file_path: str = None):
        """If `weight_file_path` is not given then it finds the best weights
        and updates the model with best wieghts.

        Arguments:
            weight_file_path : complete path of weights which are to be loaded
        """

        if weight_file_path:
            assert os.path.exists(weight_file_path), f"{weight_file_path} does not exist"
            best_weights = os.path.basename(weight_file_path)
        else:
            best_weights = find_best_weight(self.w_path, epoch_identifier=self.best_epoch)

            if best_weights is not None:

                if best_weights.endswith(".hdf5"):  # todo, find_best_weight should not add .hdf5
                    best_weights = best_weights.split(".hdf5")[0]

                weight_file_path = os.path.join(self.w_path, best_weights)

        if best_weights is not None:
            # fpath = os.path.splitext(weight_file_path)[0]  # we are not saving the whole model but only state_dict
            kwargs = {'weights_only': True}
            if not self.use_cuda:  # if the saved model was trained with cuda but we want to load it on cpu
                kwargs['map_location'] = torch.device('cpu')            
            self.model.load_state_dict(torch.load(weight_file_path, **kwargs))
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

    def on_epoch_begin(self, epoch:int):
        """This function is called at the beginning of each epoch"""
        return

    def on_epoch_end(self):
        formatter = "{:<7}" + "{:<15.7f} " * (len(self.val_epoch_losses) + len(self.train_epoch_losses))

        if self.val_loader is None:  # otherwise model is already saved based upon validation performance

            for k, v in self.train_epoch_losses.items():
                f1 = F[k][0]
                f2 = F[k][1]

                # if it is the first epoch, the weights must be saved no matter what
                # the value of v is w.r.t its previous values!
                if self.epoch == 0:
                    torch.save(self.model.state_dict(), self._weight_fname(self.epoch, v))
                    self.best_epoch = self.epoch
                    break

                if f2(v, f1(self.train_metrics[k])):
                    torch.save(self.model.state_dict(), self._weight_fname(self.epoch, v))
                    self.best_epoch = self.epoch
                    break

        if self.verbosity > 0:
            if self.verbosity > 1:
                print('')  # 
            print(formatter.format(self.epoch, *self.train_epoch_losses.values(), *self.val_epoch_losses.values()))

        for cb in self.cbs:
            if self.epoch % cb['after_epochs'] == 0:
                cb['func'](epoch=self.epoch,
                           model=self.model,
                           train_data=self.train_loader,
                           val_data=self.val_loader
                           )

        # should be done after saving the model because when saving the weights we want to compare the current
        # metrics' values with the previous values.
        self.update_metrics()

        if getattr(self, 'scheduler', None) is not None:
            self.scheduler.step()

        if self.use_wb:
            self.wb_run_.on_epoch_end(self.epoch, self.train_epoch_losses, self.val_epoch_losses)
        if self.use_cuda:
            torch.cuda.empty_cache()
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

        elif isinstance(x, (np.ndarray, pd.DataFrame)):
            if y is not None:

                # if x is numpy array or DataFrame, so should y
                assert isinstance(y, (np.ndarray, pd.DataFrame, pd.Series))

                # if it is DataFrame or Series
                if hasattr(y, 'values'):
                    y = y.values

                if len(y.shape) == 1:
                    num_outs = 1
                else:
                    num_outs = y.shape[-1]

            if isinstance(x, pd.DataFrame):
                x = x.values

            dataset = to_torch_dataset(x, y)

        elif isinstance(x, torch.utils.data.Dataset):
            dataset = x

        elif isinstance(x, torch.utils.data.DataLoader):
            data_loader = x

        elif isinstance(x, torch.Tensor):

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

    def plot_model_using_tensorboard(
            self,
            x=None,
            path='tensorboard/tensorboard'
    ):
        """Plots the neural network on tensorboard

        Arguments
        ---------
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

        Arguments
        ---------
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
        _metrics = ['loss']
    elif isinstance(metrics, list):

        _metrics = ['loss'] + metrics
    else:
        assert isinstance(metrics, str)
        _metrics = ['loss', metrics]

    return list(set(_metrics))
