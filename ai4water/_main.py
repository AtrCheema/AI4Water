import os
import json
import time
import random
import warnings
from typing import Union, Callable, Tuple
from types import MethodType

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

import joblib
import matplotlib  # for version info
import numpy as np
import pandas as pd

try:
    from scipy.stats import median_abs_deviation as mad
except ImportError:
    from scipy.stats import median_absolute_deviation as mad

from ai4water.nn_tools import NN
from ai4water.backend import sklearn_models
from ai4water.utils.plotting_tools import Plots
from ai4water.utils.visualizations import PlotResults
from ai4water.utils.utils import ts_features, make_model
from ai4water.preprocessing.datahandler import DataHandler
from ai4water.utils.utils import maybe_three_outputs, get_version_info
from ai4water.models.tensorflow.custom_training import train_step, test_step
from ai4water.utils.utils import find_best_weight, reset_seed, update_model_config
from ai4water.utils.utils import maybe_create_path, dict_to_file, dateandtime_now
from .backend import tf, keras, torch, catboost_models, xgboost_models, lightgbm_models
import ai4water.backend as K

if K.BACKEND == 'tensorflow' and tf is not None:
    from ai4water.tf_attributes import LOSSES, OPTIMIZERS

elif K.BACKEND == 'pytorch' and torch is not None:
    from ai4water.models.torch import LOSSES, OPTIMIZERS

try:
    from wandb.keras import WandbCallback
    import wandb
except ModuleNotFoundError:
    WandbCallback = None
    wandb = None


class BaseModel(NN, Plots):

    """ Model class that implements logic of AI4Water. """

    def __init__(self,
                 model: Union[dict, str] = None,
                 data=None,
                 lr: float = 0.001,
                 optimizer='Adam',
                 loss: Union[str, Callable] = 'mse',
                 quantiles=None,
                 epochs: int = 14,
                 min_val_loss: float = 0.0001,
                 patience: int = 100,
                 save_model: bool = True,
                 metrics: Union[str, list] = None,
                 val_metric: str = 'mse',
                 cross_validator: dict = None,
                 wandb_config: dict = None,
                 seed: int = 313,
                 prefix: str = None,
                 path: str = None,
                 verbosity: int = 1,
                 accept_additional_args: bool = False,
                 **kwargs):
        """
        The Model class can take a large number of possible arguments depending
        upon the machine learning model/algorithm used. Not all the arguments
        are applicable in each case. The user must define only the relevant/applicable
        parameters and leave the others as it is.

        Arguments:
            model :
                a dictionary defining machine learning model.
                If you are building a non-tensorflow model
                then this dictionary must consist of name of name of model as key
                and the keyword arguments to that model as dictionary. For example
                to build a decision forest based model
                ```python
                model = {'DecisionTreeRegressor': {"max_depth": 3, "criterion": "mae"}}
                ```
                The key 'DecisionTreeRegressor' should exactly match the name of
                the model from following libraries
                            -scikit-learn
                            -xgboost
                            -catboost
                            -lightgbm
                The value {"max_depth": 3, "criterion": "mae"} is another dictionary
                which can be any keyword argument which the `model` (DecisionTreeRegressor
                in this case) accepts. The user must refer to the documentation
                of the underlying library (scikit-learn for DecisionTreeRegressor)
                to find out complete keyword arguments applicable for a particular model.
                If You are building a Deep Learning model using tensorflow, then the key
                must be 'layers' and the value must itself be a dictionary defining layers
                of neural networks. For example we can build an MLP as following
                ```python
                model = {'layers': {
                            "Dense_0": {'units': 64, 'activation': 'relu'},
                             "Flatten": {},
                             "Dense_3": {'units': 1}
                            }}
                ```
                The MLP in this case consists of dense, and flatten layers. The user
                can define any keyword arguments which is accepted by that layer in
                TensorFlow. For example the `Dense` layer in TensorFlow can accept
                `units` and `activation` keyword argument among others. For details
                on how to buld neural networks using such layered API
                [see](https://ai4water.readthedocs.io/en/latest/build_dl_models/)
            lr  float:, default 0.001.
                learning rate,
            optimizer str/keras.optimizers like:
                the optimizer to be used for neural network training. Default is 'Adam'
            loss str/callable:  Default is `mse`.
                the cost/loss function to be used for training neural networks.
            quantiles list: Default is None
                quantiles to be used when the problem is quantile regression.
            epochs int:  Default is 14
                number of epochs to be used.
            min_val_loss float:  Default is 0.0001.
                minimum value of validatin loss/error to be used for early stopping.
            patience int:
                number of epochs to wait before early stopping. Set this value to None
                if you don't want to use EarlyStopping.
            save_model bool:,
                whether to save the model or not. For neural networks, the model will
                be saved only an improvement in training/validation loss is observed.
                Otherwise model is not saved.
            subsequences int: Default is 3.
                The number of sub-sequences. Relevent for building CNN-LSTM based models.
            metrics str/list:
                metrics to be monitored. e.g. ['nse', 'pbias']
            val_metric str:
                performance metric to be used for validation/cross_validation.
                This metric will be used for hyper-parameter optimizationa and
                experiment comparison
            cross_validator dict:
                selects the type of cross validation to be applied. It can be any
                cross validator from sklear.model_selection. Default is None, which
                means validation will be done using `validation_data`. To use
                kfold cross validation,
                ```python
                cross_validator = {'kfold': {'n_splits': 5}}
                ```
            batches str:
                either `2d` or 3d`.
            wandb_config dict:
                Only valid if wandb package is installed.  Default value is None,
                which means, wandb will not be utilized. For simplest case, pass
                a dictionary with at least two keys namely `project` and `entity`.
                Otherwise use a dictionary of all the
                arugments for wandb.init, wandb.log and WandbCallback. For
                `training_data` and `validation_data` in `WandbCallback`, pass
                `True` instead of providing a tuple as shown below
                ```python
                wandb_config = {'entity': 'entity_name', 'project': 'project_name',
                                'training_data':True, 'validation_data': True}
                ```
            seed int:
                random seed for reproducibility. This can be set to None. The seed
                is set to `np`, `os`, `tf`, `torch` and `random` modules simultaneously.
            prefix str:
                prefix to be used for the folder in which the results are saved.
                default is None, which means within
                ./results/model_path
            path str/path like:
                if not given, new model_path path will not be created.
            verbosity int: default is 1.
                determines the amount of information being printed. 0 means no
                print information. Can be between 0 and 3. Setting this value to 0
                will also reqult in not showing some plots such as loss curve or
                regression plot. These plots will only be saved in self.path.
            accept_additional_args bool:  Default is False
                If you want to pass any additional argument, then this argument
                must be set to True, otherwise an error will be raise.
            kwargs :
                keyword arguments for [`DataHandler`][ai4water.preprocessing.datahandler.DataHandler.__init__] class

        Example:
            >>>from ai4water import Model
            >>>from ai4water.datasets import arg_beach
            >>>df = arg_beach()
            >>>model = Model(data=df,
            ...              batch_size=16,
            ...           model={'layers': {'LSTM': 64}},
            ...)
            >>>history = model.fit()
            >>>y, obs = model.predict()
        """
        if self._go_up:
            maker = make_model(
                model=model,
                prefix=prefix,
                path=path,
                verbosity=verbosity,
                lr=lr,
                optimizer=optimizer,
                loss=loss,
                quantiles=quantiles,
                epochs=epochs,
                min_val_loss=min_val_loss,
                patience=patience,
                save_model=save_model,
                metrics=metrics or ['nse'],
                val_metric=val_metric,
                cross_validator=cross_validator,
                accept_additional_args=accept_additional_args,
                seed=seed,
                wandb_config=wandb_config,
                **kwargs
            )

            # data_config, model_config = config['data_config'], config['model_config']
            reset_seed(maker.config['seed'], os=os, random=random, tf=tf, torch=torch)
            if tf is not None:
                # graph should be cleared everytime we build new `Model` otherwise, if two `Models` are prepared in same
                # file, they may share same graph.
                tf.keras.backend.clear_session()

            if data.__class__.__name__ in ["DataHanlder", "SiteDistributedDataHandler"]:
                self.dh = data
            else:
                self.dh = DataHandler(data=data, **maker.data_config)

            self.opt_paras = maker.opt_paras
            self._original_model_config = maker.orig_model

            # if DataHanlder defines input and output features, we must put it back in config
            # so that it can be accessed as .config['input_features'] etc.
            maker.config['input_features'] = self.dh.input_features
            maker.config['output_features'] = self.dh.output_features

            NN.__init__(self, config=maker.config)

            self.path = maybe_create_path(path=path, prefix=prefix)
            self.config['path'] = self.path
            self.verbosity = verbosity
            self.category = self.config['category']
            self.mode = self.config['mode']
            self.info = {}

            Plots.__init__(self, self.path, self.mode, self.category,
                           config=maker.config)

    def __getattr__(self, item):
        """instead of doing model.dh.num_ins do model.num_ins"""
        if item in [
            'data',
            'test_indices', 'train_indices',
            'num_outs', 'forecast_step', 'num_ins',
                    ]:
            return getattr(self.dh, item)
        else:
            raise AttributeError(f'BaseModel has no attribute named {item}')

    # because __getattr__ does not work with pytorch, we explicitly get attributes from
    # DataHandler and assign them to Model
    @property
    def forecast_len(self):
        return self.dh.forecast_len

    @property
    def num_outs(self):
        return self.dh.num_outs

    @property
    def num_ins(self):
        return self.dh.num_ins

    @property
    def is_binary(self):
        return self.dh.is_binary

    @property
    def is_multiclass(self):
        return self.dh.is_multiclass

    @property
    def output_features(self):
        return self.dh.output_features

    @property
    def classes(self):
        return self.dh.classes

    @property
    def num_classes(self):
        return self.dh.num_classes

    @property
    def is_multilabel(self):
        if self.dh.data is None:
            return None
        return self.dh.is_multilabel

    @property
    def quantiles(self):
        return self.config['quantiles']

    @property
    def act_path(self):
        return os.path.join(self.path, 'activations')

    @property
    def w_path(self):
        return os.path.join(self.path, 'weights')

    @property
    def data_path(self):
        return os.path.join(self.path, 'data')

    # because __getattr__ does not work with pytorch, we explicitly get attributes from
    # DataHandler and assign them to Model
    def training_data(self, *args, **kwargs):

        return self.dh.training_data(*args, **kwargs)

    def validation_data(self, *args, **kwargs):
        return self.dh.validation_data(*args, **kwargs)

    def test_data(self, *args, **kwargs):
        return self.dh.test_data(*args, **kwargs)

    def nn_layers(self):
        if hasattr(self, 'layers'):
            return self.layers
        elif hasattr(self._model, 'layers'):
            return self._model.layers
        else:
            return None

    @property
    def ai4w_outputs(self):
        """alias for keras.MOdel.outputs!"""
        if hasattr(self, 'outputs'):
            return self.outputs
        elif hasattr(self._model, 'outputs'):
            return self._model.outputs
        else:
            return None

    def trainable_parameters(self) -> int:
        """Calculates trainable parameters in the model

        for more [see](https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9)
        """
        if self.config['backend'] == 'pytorch':
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            if hasattr(self, 'count_params'):
                return int(self.count_params())
            else:
                return int(self._model.count_params())

    def loss(self):
        # overwrite this function for a customized loss function.
        # this function should return something which can be accepted as 'loss' by the keras Model.
        # It can be a string or callable.
        if callable(self.config['loss']):
            return self.config['loss']

        if self.config['backend'] == 'pytorch':
            return LOSSES[self.config['loss']]()

        return LOSSES[self.config['loss']]

    @property
    def fit_fn(self):
        # this points to the Keras's fit method
        return NotImplementedError

    @property
    def evaluate_fn(self):
        # this points to the Keras's evaluate method
        return NotImplementedError

    @property
    def predict_fn(self, *args, **kwargs):
        return NotImplementedError

    @property
    def api(self):
        return NotImplementedError

    @property
    def input_layer_names(self):
        return NotImplementedError

    @property
    def num_input_layers(self):
        return NotImplementedError

    @property
    def layer_names(self):
        return NotImplementedError

    @property
    def dl_model(self):
        if self.api == "subclassing":
            return self
        else:
            return self._model

    def first_layer_shape(self):
        return NotImplementedError

    def get_callbacks(self, val_data, callbacks=None):

        if self.config['backend'] == 'pytorch':
            return self.cbs_for_pytorch(val_data, callbacks)
        else:
            return self.cbs_for_tf(val_data, callbacks)

    def cbs_for_pytorch(self, *args, **kwargs):
        """Callbacks for pytorch training."""
        return []

    def cbs_for_tf(self, val_data, callbacks=None):

        if callbacks is None:
            callbacks = {}
        # container to hold all callbacks
        _callbacks = list()

        _monitor = 'val_loss' if val_data else 'loss'
        fname = "{val_loss:.5f}.hdf5" if val_data else "{loss:.5f}.hdf5"

        if int(''.join(tf.__version__.split('.')[0:2])) <= 115:
            for lyr_name in self.layer_names:
                if 'HA_weighted_input' in lyr_name or 'SeqWeightedAttention_weights' in lyr_name:
                    self.config['save_model'] = False

                    warnings.warn("Can not save Heirarchical model with tf<= 1.15")

        if self.config['save_model']:
            _callbacks.append(keras.callbacks.ModelCheckpoint(
                filepath=self.w_path + f"{os.sep}weights_" + "{epoch:03d}_" + fname,
                save_weights_only=True,
                monitor=_monitor,
                mode='min',
                save_best_only=True))

        if self.config['patience']:
            _callbacks.append(keras.callbacks.EarlyStopping(
                monitor=_monitor, min_delta=self.config['min_val_loss'],
                patience=self.config['patience'], verbose=0, mode='auto'
            ))

        if 'tensorboard' in callbacks:
            tb_kwargs = callbacks['tensorboard']
            if 'log_dir' not in tb_kwargs: tb_kwargs['log_dir'] = self.path
            _callbacks.append(keras.callbacks.TensorBoard(**tb_kwargs))
            callbacks.pop('tensorboard')

        for val in callbacks.values():
            _callbacks.append(val)

        return _callbacks

    def get_val_data(self, val_data):
        """Finds out if there is val_data or not"""
        if isinstance(val_data, tuple):
            if val_data[0] is None and val_data[1] is None:
                return None
            # val_data was probably available in kwargs, so use them as it is
            return val_data

        validation_data = None

        if val_data is not None:
            if isinstance(val_data, tuple):
                x = val_data[0]
                if x is not None:
                    if isinstance(x, np.ndarray):
                        if x.size > 0:
                            validation_data = val_data
                    elif isinstance(x, dict):  # x may be a dictionary
                        for v in x.values():
                            if v.size > 0:
                                validation_data = val_data
                                break
                    elif isinstance(x, list):
                        for v in x:
                            if v.size > 0:
                                validation_data = val_data
                                break
                    else:
                        raise ValueError(f'Unrecognizable validattion data {val_data.__class__.__name__}')

        return validation_data

    def DO_fit(self, x, **kwargs):
        """
        Some preprocessing before calling actual fit

        If nans are present in y, then tf.keras.model.fit is called as it
        is otherwise it is called with custom train_step and test_step which
        avoids calculating loss at points containing nans."""
        if kwargs.pop('nans_in_y_exist'):  # todo, for model-subclassing?
            if not isinstance(x, tf.data.Dataset):  # when x is tf.Dataset, we don't have y in kwargs
                y = kwargs['y']
                assert np.isnan(y).sum() > 0
                kwargs['y'] = np.nan_to_num(y)  # In graph mode, masking of nans does not work
            self._model.train_step = MethodType(train_step, self._model)
            self._model.test_step = MethodType(test_step, self._model)

        return self.fit_fn(x, **kwargs)

    def _FIT(self, inputs, outputs, validation_data, validation_steps=None, callbacks=None, **kwargs):

        nans_in_y_exist = False
        if isinstance(outputs, np.ndarray):
            if np.isnan(outputs).sum() > 0:
                nans_in_y_exist = True
        elif isinstance(outputs, list):
            for out_array in outputs:
                if np.isnan(out_array).sum() > 0:
                    nans_in_y_exist = True
        elif isinstance(outputs, dict):
            for out_array in outputs.values():
                if np.isnan(out_array).sum() > 0:
                    nans_in_y_exist = True

        validation_data = self.get_val_data(validation_data)

        outputs = get_values(outputs)

        if validation_data is not None:
            val_outs = validation_data[-1]
            val_outs = get_values(val_outs)
            validation_data = (validation_data[0], val_outs)

        if K.BACKEND == 'tensorflow':
            callbacks = self.get_wandb_cb(callbacks, train_data=(inputs, outputs),
                                          validation_data=validation_data,
                                          )

        callbacks = self.get_callbacks(validation_data, callbacks=callbacks)

        st = time.time()

        # .fit was called with epochs, so we must put that in config as well!
        if 'epochs' in kwargs:
            self.config['epochs'] = kwargs.pop('epochs')

        batch_siz = None if inputs.__class__.__name__ in ['TorchDataset', 'BatchDataset'] else self.config['batch_size']

        self.DO_fit(
            x=inputs,
            y=None if inputs.__class__.__name__ in ['TorchDataset', 'BatchDataset'] else outputs,
            epochs=self.config['epochs'],
            batch_size=batch_siz,
            validation_data=validation_data,
            callbacks=callbacks,
            shuffle=self.config['shuffle'],
            steps_per_epoch=self.config['steps_per_epoch'],
            verbose=max(self.verbosity, 0),
            nans_in_y_exist=nans_in_y_exist,
            validation_steps=validation_steps,
            **kwargs,
        )

        self.info['training_time_in_minutes'] = round(float(time.time() - st) / 60.0, 2)

        return self.post_fit()

    def get_wandb_cb(self, callback, train_data, validation_data) -> dict:
        """Makes WandbCallback and add it in callback"""
        if callback is None:
            callback = {}

        self.use_wandb = False
        if wandb is not None:
            wandb_config: dict = self.config['wandb_config']
            if wandb_config is not None:
                self.use_wandb = True
                for key in ['project', 'entity']:
                    assert key in wandb_config, f"wandb_config must have {key} key in it"
                wandb.init(name=os.path.basename(self.path),
                           project=wandb_config.pop('project'),
                           notes=wandb_config.get('notes', f"{self.mode} with {self.config['backend']}"),
                           tags=['ai4water', self.api, self.category, self.problem],
                           entity=wandb_config.pop('entity'))

                monitor = self.config.get('monitor', 'val_loss')
                if 'monitor' in wandb_config:
                    monitor = wandb_config.pop('monitor')

                add_train_data = False
                if 'training_data' in wandb_config:
                    add_train_data = wandb_config.pop('training_data')

                add_val_data = False
                if 'validation_data' in wandb_config:
                    add_val_data = wandb_config.pop('validation_data')

                assert callable(WandbCallback)

                callback['wandb_callback'] = WandbCallback(monitor=monitor,
                                                           training_data=train_data if add_train_data else None,
                                                           validation_data=validation_data if add_val_data else None,
                                                           **wandb_config)
        return callback

    def post_fit_wandb(self):
        """does some stuff related to wandb at the end of training."""
        if K.BACKEND == 'tensorflow' and self.use_wandb:
            getattr(wandb, 'finish')()

        return

    def post_fit(self):
        """Does some stuff after Keras model.fit has been called"""
        if K.BACKEND == 'pytorch':
            history = self.torch_learner.history

        elif hasattr(self, 'history'):
            history = self.history
        else:
            history = self._model.history

        self.save_config(history.history)

        if self.verbosity >= 0:
            # save all the losses or performance metrics
            df = pd.DataFrame.from_dict(history.history)
            df.to_csv(os.path.join(self.path, "losses.csv"))

        return history

    def maybe_not_3d_data(self, true, predicted, forecast_len):

        if true.ndim < 3:
            assert forecast_len == 1, f'{forecast_len}'
            axis = 2 if true.ndim == 2 else (1, 2)
            true = np.expand_dims(true, axis=axis)

        if predicted.ndim < 3:
            assert forecast_len == 1
            axis = 2 if predicted.ndim == 2 else (1, 2)
            predicted = np.expand_dims(predicted, axis=axis)

        return true, predicted

    def process_class_results(self,
                              true: np.ndarray,
                              predicted: np.ndarray,
                              metrics="minimal",
                              prefix=None,
                              index=None,
                              user_defined_data: bool = False
                              ):
        """post-processes classification results."""
        if user_defined_data:
            return

        from ai4water.postprocessing.SeqMetrics import ClassificationMetrics

        if self.is_multiclass:
            pred_labels = [f"pred_{i}" for i in range(predicted.shape[1])]
            true_labels = [f"true_{i}" for i in range(true.shape[1])]
            fname = os.path.join(self.path, f"{prefix}_prediction.csv")
            pd.DataFrame(np.concatenate([true, predicted], axis=1),
                         columns=true_labels + pred_labels, index=index).to_csv(fname)
            class_metrics = ClassificationMetrics(true, predicted, multiclass=True)

            dict_to_file(self.path,
                         errors=class_metrics.calculate_all(),
                         name=f"{prefix}_{dateandtime_now()}.json")
        else:
            if predicted.ndim == 1:
                predicted = predicted.reshape(-1, 1)
            for idx, _class in enumerate(self.out_cols):
                _true = true[:, idx]
                _pred = predicted[:, idx]

                fpath = os.path.join(self.path, _class)
                if not os.path.exists(fpath):
                    os.makedirs(fpath)

                class_metrics = ClassificationMetrics(_true, _pred, multiclass=False)
                dict_to_file(fpath,
                             errors=getattr(class_metrics, f"calculate_{metrics}")(),
                             name=f"{prefix}_{_class}_{dateandtime_now()}.json"
                             )

                fname = os.path.join(fpath, f"{prefix}_{_class}.csv")
                array = np.concatenate([_true.reshape(-1, 1), _pred.reshape(-1, 1)], axis=1)
                pd.DataFrame(array, columns=['true', 'predicted'],  index=index).to_csv(fname)

        return

    def process_regres_results(
            self,
            true: np.ndarray,
            predicted: np.ndarray,
            metrics="minimal",
            prefix=None,
            index=None,
            remove_nans=True,
            user_defined_data: bool = False,
            annotate_with="r2",
    ):
        """
        predicted, true are arrays of shape (examples, outs, forecast_len).
        annotate_with : which value to write on regression plot
        """
        from ai4water.postprocessing.SeqMetrics import RegressionMetrics

        metric_names = {'r2': "$R^2$"}

        visualizer = PlotResults(path=self.path)

        if user_defined_data:
            # when data is user_defined, we don't know what out_cols, and forecast_len are
            if predicted.ndim == 1:
                out_cols = ['output']
            else:
                out_cols = [f'output_{i}' for i in range(predicted.shape[-1])]
            forecast_len = 1
            true, predicted = self.maybe_not_3d_data(true, predicted, forecast_len)
        else:
            # for cases if they are 2D/1D, add the third dimension.
            true, predicted = self.maybe_not_3d_data(true, predicted, self.forecast_len)

            forecast_len = self.forecast_len
            if isinstance(forecast_len, dict):
                forecast_len = np.unique(list(forecast_len.values())).item()

            out_cols = list(self.out_cols.values())[0] if isinstance(self.out_cols, dict) else self.out_cols

        for idx, out in enumerate(out_cols):

            horizon_errors = {metric_name: [] for metric_name in ['nse', 'rmse']}
            for h in range(forecast_len):

                errs = dict()

                fpath = os.path.join(self.path, out)
                if not os.path.exists(fpath):
                    os.makedirs(fpath)

                t = pd.DataFrame(true[:, idx, h], index=index, columns=['true_' + out])
                p = pd.DataFrame(predicted[:, idx, h], index=index, columns=['pred_' + out])

                if wandb is not None and self.config['wandb_config'] is not None:
                    self._wandb_scatter(t.values, p.values, out)

                df = pd.concat([t, p], axis=1)
                df = df.sort_index()
                fname = prefix + out + '_' + str(h) + dateandtime_now() + ".csv"
                df.to_csv(os.path.join(fpath, fname), index_label='index')

                annotation_val = getattr(RegressionMetrics(t, p), annotate_with)()
                visualizer.plot_results(t,
                                        p,
                                        name=prefix + out + '_' + str(h),
                                        where=out,
                                        annotation_key=metric_names.get(annotate_with, annotate_with),
                                        annotation_val=annotation_val,
                                        show=self.verbosity)

                if remove_nans:
                    nan_idx = np.isnan(t)
                    t = t.values[~nan_idx]
                    p = p.values[~nan_idx]

                errors = RegressionMetrics(t, p)
                errs[out + '_errors_' + str(h)] = getattr(errors, f'calculate_{metrics}')()
                errs[out + 'true_stats_' + str(h)] = ts_features(t)
                errs[out + 'predicted_stats_' + str(h)] = ts_features(p)

                dict_to_file(fpath, errors=errs, name=prefix)

                for p in horizon_errors.keys():
                    horizon_errors[p].append(getattr(errors, p)())

            if forecast_len > 1:
                visualizer.horizon_plots(horizon_errors, f'{prefix}_{out}_horizons.png')
        return

    def _wandb_scatter(self, true: np.ndarray, predicted: np.ndarray, name: str) -> None:
        """Adds a scatter plot on wandb."""
        data = [[x, y] for (x, y) in zip(true.reshape(-1,), predicted.reshape(-1,))]
        table = wandb.Table(data=data, columns=["true", "predicted"])
        wandb.log({
            "scatter_plot": wandb.plot.scatter(table, "true", "predicted",
                                               title=name)
                   })
        return

    def build_ml_model(self):
        """Builds ml models

        Models that follow sklearn api such as xgboost,
        catboost, lightgbm and obviously sklearn.
        """

        ml_models = {**sklearn_models, **xgboost_models, **catboost_models, **lightgbm_models}
        estimator = list(self.config['model'].keys())[0]

        kwargs = list(self.config['model'].values())[0]

        if estimator in ['HistGradientBoostingRegressor', 'SGDRegressor', 'MLPRegressor']:
            if self.config['val_fraction'] > 0.0:
                kwargs.update({'validation_fraction': self.config['val_fraction']})
            elif self.config['test_fraction'] > 0.0:
                kwargs.update({'validation_fraction': self.config['test_fraction']})

        # some algorithms allow detailed output during training, this is allowed when self.verbosity is > 1
        if estimator in ['OneClassSVM']:
            kwargs.update({'verbose': True if self.verbosity > 1 else False})

        if estimator in ["CatBoostRegressor", "CatBoostClassifier"]:
            # https://stackoverflow.com/a/52921608/5982232
            if not any([arg in kwargs for arg in ['verbose', 'silent', 'logging_level']]):
                if self.verbosity == 0:
                    kwargs['logging_level'] = 'Silent'
                elif self.verbosity == 1:
                    kwargs['logging_level'] = 'Verbose'
                else:
                    kwargs['logging_level'] = 'Info'
            if 'random_seed' not in kwargs:
                kwargs['random_seed'] = self.config['seed']

        if estimator in ["XGBRegressor", "XGBClassifier"]:
            if 'seed' not in kwargs:
                kwargs['random_state'] = self.config['seed']

        # following sklearn based models accept random_state argument
        if estimator in [
            "AdaBoostRegressor",
            "BaggingClassifier", "BaggingRegressor",
            "DecisionTreeClassifier", "DecisionTreeRegressor",
            "ExtraTreeClassifier", "ExtraTreeRegressor",
            "ExtraTreesClassifier", "ExtraTreesRegressor",
            "ElasticNet", "ElasticNetCV",
            "GradientBoostingClassifier", "GradientBoostingRegressor",
            "GaussianProcessRegressor",
            "HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
            "LogisticRegression",
            "Lars",
            "Lasso",
            "LassoCV",
            "LassoLars",
            "LinearSVR",
            "MLPClassifier", "MLPRegressor",
            "PassiveAggressiveClassifier", "PassiveAggressiveRegressor",
            "RandomForestClassifier", "RandomForestRegressor",
            "RANSACRegressor", "RidgeClassifier",
            "SGDClassifier", "SGDRegressor",
            "TheilSenRegressor",
        ]:
            if 'random_state' not in kwargs:
                kwargs['random_state'] = self.config['seed']

        # in sklearn version >1.0 precompute automatically becomes to True
        # which can raise error
        if estimator in ["ElasticNetCV", "LassoCV"] and 'precompute' not in kwargs:
            kwargs['precompute'] = False

        self.residual_threshold_not_set = False
        if estimator == "RANSACRegressor" and 'residual_threshold' not in kwargs:
            self.residual_threshold_not_set = True

        if estimator in ["LGBMRegressor", 'LGBMClassifier']:
            if 'random_state' not in kwargs:
                kwargs['random_state'] = self.config['seed']

        # initiate the estimator/model class
        if estimator in ml_models:
            model = ml_models[estimator](**kwargs)
        else:
            from .backend import sklearn, lightgbm, catboost, xgboost
            version_info = get_version_info(sklearn=sklearn, lightgbm=lightgbm, catboost=catboost,
                                            xgboost=xgboost)
            if estimator in ['TweedieRegressor', 'PoissonRegressor', 'LGBMRegressor', 'LGBMClassifier',
                             'GammaRegressor']:
                sk_maj_ver = int(sklearn.__version__.split('.')[0])
                sk_min_ver = int(sklearn.__version__.split('.')[1])

                if sk_maj_ver < 1 and sk_min_ver < 23:
                    raise ValueError(
                        f"{estimator} is available with sklearn version >= 0.23 but you have {version_info['sklearn']}")
            raise ValueError(f"model {estimator} not found. {version_info}")

        self._model = model

        return

    def fit(
            self,
            data: str = 'training',
            callbacks: dict = None,
            **kwargs
    ):
        """
        Trains the model with data which is taken from data accoring to `data` arguments.

        Arguments:
            data :
                data to use for model training. Default is 'training`.
            callbacks:
                Any callback compatible with keras. If you want to log the output
                to tensorboard, then just use `callbacks={'tensorboard':{}}` or
                to provide additional arguments
                ```python
                callbacks={'tensorboard': {'histogram_freq': 1}}
                ```
            kwargs :
                Any keyword argument for the `fit` method of the underlying algorithm.
                if 'x' is present in kwargs, that will take precedent over `data`.
        Returns:
            A keras history object in case of deep learning model with tensorflow
            as backend or anything returned by `fit` method of underlying model.
        """

        if isinstance(data, str):
            assert data in ['training', 'test', 'validation']

        return self.call_fit(data=data, callbacks=callbacks, **kwargs)

    def call_fit(self,
                 data='training',
                 callbacks=None,
                 **kwargs):

        visualizer = PlotResults(path=self.path)
        self.is_training = True

        if isinstance(data, np.ndarray):  # .fit(x,...)
            assert 'x' not in kwargs
            kwargs['x'] = data

        if isinstance(callbacks, np.ndarray):  # .fit(x,y)
            assert 'y' not in kwargs
            kwargs['y'] = callbacks
            callbacks = None

        if 'x' not in kwargs:
            train_data = getattr(self, f'{data}_data')()
            inputs, outputs = maybe_three_outputs(train_data, self.dh.teacher_forcing)
        else:
            outputs = None
            if 'y' in kwargs:
                outputs = kwargs['y']
                kwargs.pop('y')
            inputs = kwargs.pop('x')

        if isinstance(outputs, np.ndarray) and self.category == "DL":
            if isinstance(self.ai4w_outputs, list):
                assert len(self.ai4w_outputs) == 1
                model_output_shape = tuple(self.ai4w_outputs[0].shape.as_list()[1:])

                if getattr(self, 'quantiles', None) is not None:
                    assert model_output_shape[0] == len(self.quantiles) * self.num_outs

                # todo, it is assumed that there is softmax as the last layer
                elif self.mode == 'classification':
                    # todo, don't know why it is working
                    assert model_output_shape[0] == self.num_classes, f"""inferred number of classes are 
                            {self.num_classes} while model's output has {model_output_shape[0]} nodes """
                    assert model_output_shape[0] == outputs.shape[1]
                else:
                    assert model_output_shape == outputs.shape[1:], f"""
    ShapeMismatchError: Shape of model's output is {model_output_shape}
    while the targets in prepared have shape {outputs.shape[1:]}."""

        self.info['training_start'] = dateandtime_now()

        if self.category == "DL":
            if 'validation_data' not in kwargs:
                val_data = self.validation_data()
                val_x, val_y = maybe_three_outputs(val_data, self.dh.teacher_forcing)
                val_data = (val_x, val_y)
                kwargs['validation_data'] = val_data
            history = self._FIT(inputs, outputs, callbacks=callbacks, **kwargs)

            if self.verbosity >= 0:
                visualizer.plot_loss(history.history, show=self.verbosity)

            self.load_best_weights()
        else:
            history = self.fit_ml_models(inputs, outputs)

        self.info['training_end'] = dateandtime_now()
        self.save_config()

        if self.verbosity >= 0:
            dict_to_file(os.path.join(self.path, 'info.json'), others=self.info)

        self.is_training = False
        return history

    def load_best_weights(self) -> None:
        if self.config['backend'] != 'pytorch':
            # load the best weights so that the best weights can be used during model.predict calls
            best_weights = find_best_weight(os.path.join(self.path, 'weights'))
            if best_weights is None:
                warnings.warn("best weights could not be found and are not loaded", UserWarning)
            else:
                self.allow_weight_loading = True
                self.update_weights(os.path.join(self.w_path, best_weights))
        return

    def fit_ml_models(self, inputs, outputs):
        # following arguments are strictly about nn so we don't need to save them in config file
        # so that it does not confuse the reader.
        for arg in ["composite", "optimizer", "lr", "epochs", "subsequences"]:
            if arg in self.config:
                self.config.pop(arg)

        if self.dh.is_multiclass:
            outputs = outputs
        else:
            outputs = outputs.reshape(-1, )

        self._maybe_change_residual_threshold(outputs)

        history = self._model.fit(inputs, outputs)

        self._save_ml_model()

        return history

    def _save_ml_model(self):
        """Saves the non-NN/ML models in the disk."""
        model_name = list(self.config['model'].keys())[0]
        fname = os.path.join(self.w_path, self.category + '_' + self.mode + '_' + model_name)

        if "tpot" not in model_name:
            joblib.dump(self._model, fname)

        return

    def cross_val_score(
            self,
            scoring: str = None,
            data=None,
    ) -> float:
        """computes cross validation score

        Arguments:
            scoring:
                performance metric to use for cross validation.
                If None, it will be taken from config['val_metric']
            data:
                a tuple of x,y parirs
                If given, it will override the data prepared by Model class.

        Returns:
            cross validation score

        Note
        ----
            Currently not working for deep learning models.

        """
        from ai4water.postprocessing.SeqMetrics import RegressionMetrics, ClassificationMetrics

        if scoring is None:
            scoring = self.config['val_metric']

        scores = []

        if self.config['cross_validator'] is None:
            raise ValueError("Provide the `cross_validator` argument to the `Model` class upon initiation")

        cross_validator = list(self.config['cross_validator'].keys())[0]
        cross_validator_args = self.config['cross_validator'][cross_validator]

        if data is None:

            if self.num_outs > 1:  # we may not be able to check num_outs with custom data
                raise ValueError

            if callable(cross_validator):
                splits = cross_validator(**cross_validator_args)
            else:
                splits = getattr(self.dh, f'{cross_validator}_splits')(**cross_validator_args)
        else:
            from .utils.utils import TrainTestSplit
            splitter = TrainTestSplit(*data, test_fraction=self.config['test_fraction'])
            splits = splitter.KFold_splits(**cross_validator_args)

        for fold, ((train_x, train_y), (test_x, test_y)) in enumerate(splits):

            verbosity = self.verbosity
            self.verbosity = 0
            # make a new classifier/regressor at every fold
            self.build(self._get_dummy_input_shape())

            self.verbosity = verbosity

            self._maybe_change_residual_threshold(train_y)

            self._model.fit(train_x, y=train_y.reshape(-1, ))

            pred = self._model.predict(test_x)

            metrics = RegressionMetrics(test_y.reshape(-1, 1), pred)
            val_score = getattr(metrics, scoring)()

            scores.append(val_score)

            if self.verbosity > 0:
                print(f'fold: {fold} val_score: {val_score}')

        # save all the scores as json in model path
        cv_name = str(cross_validator)
        fname = os.path.join(self.path, f'{cv_name}_{scoring}.json')
        with open(fname, 'w') as fp:
            json.dump(scores, fp)

        # set it as class attribute so that it can be used
        setattr(self, f'cross_val_{scoring}', scores)

        # if we do not run .fit(), then we should still have model saved in the disk
        # so that it can be used.
        self._save_ml_model()

        return np.mean(scores).item()

    def _maybe_change_residual_threshold(self, outputs) -> None:
        # https://stackoverflow.com/a/64396757/5982232
        if self.residual_threshold_not_set:
            old_value = self._model.residual_threshold or mad(outputs.reshape(-1, ).tolist())
            if np.isnan(old_value) or old_value < 0.001:
                self._model.set_params(residual_threshold=0.001)
                if self.verbosity > 0:
                    print(f"changing residual_threshold from {old_value} to {self._model.residual_threshold}")
        return

    def evaluate(
            self,
            data='test',
            metrics=None,
            **kwargs
    ):
        """
        Evalutes the performance of the model on a given data.
        calls the `evaluate` method of underlying `model`. If the `evaluate`
        method is not available in underlying `model`, then `predict` is called.

        Arguments:
            data:
                which data type to use, valid values are `training`, `test`
                and `validation`. You can also provide your own x,y values as keyword
                arguments. In such a case, this argument will have no meaning.
            metrics:
                the metrics to evaluate. It can a string indicating the metric to
                evaluate. It can also be a list of metrics to evaluate. Any metric
                name from [RegressionMetrics][ai4water.postprocessing.SeqMetrics.RegressionMetrics]
                or [ClassificationMetrics][ai4water.postprocessing.SeqMetrics.ClassificationMetrics]
                can be given. It can also be name of group of metrics to evaluate.
                Following groups are available

                    - `minimal`
                    - `all`
                    - `hydro_metrics`
                If this argument is given, the `evaluate` function of the underlying class
                is not called. Rather the model is evaluated for given metrics.
            kwargs:
                any keyword argument for the `evaluate` method of the underlying
                model.
        Returns:
            If `metrics` is not given then this method returns whatever is returned
            by `evaluate` method of underlying model. Otherwise the model is evaluated
            for given metric or group of metrics and the result is returned

        Example:
            >>> from ai4water import Model
            >>> from ai4water.datasets import arg_beach
            >>> model = Model(model={"layers": {"Dense": 1}}, data=arg_beach())
            >>> model.fit()

            for evaluation on test data
            >>> model.evaluate()

            for evaluation on training data
            >>> model.evaluate('training')

            evaluate on any metric from [Metrics][ai4water.postprocessing.SeqMetrics.RegressionMetrics] module
            >>> model.evaluate(metrics='pbias')

            to evaluate on custom data, the user can provide its own x and y
            >>> x = np.random.random((10, 13))
            >>> y = np.random.random((10, 1, 1))
            >>> model.evaluate(x, y)

            # backward compatability
            Since the ai4water's Model is supposed to behave same as Keras' Model
            the following expressions are equally valid.
            >>> model.evaluate(x, y=y)
            >>> model.evaluate(x=x, y=y)
        """
        return self.call_evaluate(data, **kwargs)

    def call_evaluate(self, data='test', metrics=None, **kwargs):

        if data.__class__.__name__ in ["ndarray", "Dataset"]:  # .fit(x,y,...)
            assert 'x' not in kwargs
            x = data
            if 'y' in kwargs:  # (.evaluate(x, y=y)
                y = kwargs.pop('y')
            else:
                y = metrics  # (.evaluate(x, y,...)
                metrics = None

        elif 'x' in kwargs:  # .fit(x=x, y=y)
            assert data in ['training', 'test', 'validation']
            x = kwargs.pop('x')
            y = kwargs.pop('y')

        elif isinstance(data, str):
            assert data in ['training', 'test', 'validation']

            # get the relevant data
            data = getattr(self, f'{data}_data')()
            x, y = maybe_three_outputs(data, self.dh.teacher_forcing)

        else:
            raise ValueError

        # dont' make call to underlying evaluate function rather manually
        # evaluate the given metrics
        if metrics is not None:
            return self._manual_eval(x, y, metrics)

        if hasattr(self._model, 'evaluate'):
            return self._model.evaluate(x, y, **kwargs)

        return self.evaluate_fn(x, y, **kwargs)

    def evalute_ml_models(self, x, y, metrics=None):
        if metrics is None:
            metrics = self.config['val_metric']
        return self._manual_eval(x, y, metrics)

    def _manual_eval(self, x, y, metrics):
        """manual evaluation"""
        p = self.predict(x=x, return_true=False, process_results=False)

        if self.problem == "regression":
            from ai4water.postprocessing.SeqMetrics import RegressionMetrics
            errs = RegressionMetrics(y, p)
        else:
            from ai4water.postprocessing.SeqMetrics import ClassificationMetrics
            errs = ClassificationMetrics(y, p)

        if isinstance(metrics, str):

            if metrics in ['minimal', 'all', 'hydro_metrics']:
                results = getattr(errs, f"calculate_{metrics}")()
            else:
                results = getattr(errs, metrics)()

        elif isinstance(metrics, list):
            results = {}
            for m in metrics:
                results[m] = getattr(errs, m)()

        elif callable(metrics):
            results = metrics(x, y)
        else:
            raise ValueError(f"unknown metrics type {metrics}")

        return results

    def predict(
            self,
            data: str = 'test',
            x=None,
            y=None,
            process_results: bool = True,
            metrics: str = "minimal",
            return_true: bool = False,
            **kwargs
    ):
        """
        Makes prediction from the trained model.

        Arguments:
            data: str
                which data to use. Possible values are

                - `training`
                - `test`
                - `validation`.
                By default, `test` data is used for predictions.
            x:
                if given, it will override `data`
            y:
                Used for pos-processing etc. if given it will overrite `data`
            process_results: bool
                post processing of results
            metrics: str
                only valid if process_results is True. The metrics to calculate.
                Valid values are 'minimal', 'all', 'hydro_metrics'
            return_true: bool
                whether to return the true values along with predicted values
                or not. Default is False, so that this method behaves sklearn type.
            kwargs : any keyword argument for `fit` method.
        Returns:
            An numpy array of predicted values.
            If return_true is True then a tuple of arrays. The first is true
            and the second is predicted. If `x` is given but `y` is not given,
            then, first array which is returned is None.
        """
        if isinstance(data, str):
            assert data in ['training', 'test', 'validation']

        assert metrics in ("minimal", "all", "hydro_metrics")

        return self.call_predict(data=data, x=x, y=y, process_results=process_results,
                                 metrics=metrics,
                                 return_true=return_true,
                                 **kwargs)

    def call_predict(self,
                     data='test',
                     x=None,
                     y=None,
                     process_results=True,
                     metrics="minimal",
                     return_true: bool = False,
                     **kwargs):

        transformation_key = None

        if data.__class__.__name__ in ["ndarray", "Dataset", "list"]: # .predict(x,y,...)
            # the predict method is called like .predict(x)
            assert 'x' not in kwargs
            inputs = data
            user_defined_data = True
            true_outputs = None
            prefix = 'x'
        else:
            transformation_key = '5'

            if x is None:  # .predict('training')
                if self.dh.data is None:
                    raise ValueError("You must specify the data on which to make prediction")
                user_defined_data = False
                prefix = data
                data = getattr(self, f'{data}_data')(key=transformation_key)
                inputs, true_outputs = maybe_three_outputs(data, self.dh.teacher_forcing)
            else:  # .predict(x=x,...)
                user_defined_data = True
                prefix = 'x'
                inputs = x
                true_outputs = y

        if 'verbose' in kwargs:
            verbosity = kwargs.pop('verbose')
        else:
            verbosity = self.verbosity

        batch_size = self.config['batch_size']
        if 'batch_size' in kwargs:
            batch_size = kwargs.pop('batch_size')

        if self.category == 'DL':
            predicted = self.predict_fn(x=inputs,
                                        batch_size=batch_size,
                                        verbose=verbosity,
                                        **kwargs)
        else:
            predicted = self.predict_ml_models(inputs, **kwargs)

        if true_outputs is None:
            if return_true:
                return true_outputs, predicted
            return predicted

        dt_index = np.arange(len(true_outputs))  # dummy/default index when data is user defined
        if not user_defined_data:
            true_outputs, predicted = self.inverse_transform(true_outputs, predicted, transformation_key)

            true_outputs, dt_index = self.dh.deindexify(true_outputs, key=transformation_key)

        if isinstance(true_outputs, np.ndarray) and true_outputs.dtype.name == 'object':
            true_outputs = true_outputs.astype(predicted.dtype)

        if true_outputs is None:
            process_results = False

        if self.quantiles is None:

            # it true_outputs and predicted are dictionary of len(1) then just get the values
            true_outputs = get_values(true_outputs)
            predicted = get_values(predicted)

            if process_results:
                if self.mode == 'regression':
                    self.process_regres_results(true_outputs, predicted,
                                                metrics=metrics,
                                                prefix=prefix + '_', index=dt_index,
                                                user_defined_data=user_defined_data)
                else:
                    self.process_class_results(true_outputs,
                                               predicted,
                                               metrics=metrics,
                                               prefix=prefix,
                                               index=dt_index,
                                               user_defined_data=user_defined_data)

        else:
            assert self.num_outs == 1
            self.plot_quantiles1(true_outputs, predicted)
            self.plot_quantiles2(true_outputs, predicted)
            self.plot_all_qs(true_outputs, predicted)

        if return_true:
            return true_outputs, predicted
        return predicted

    def predict_ml_models(self, inputs, **kwargs):
        """So that it can be overwritten easily for ML models."""
        return self.predict_fn(inputs, **kwargs)

    def inverse_transform(self,
                          true: Union[np.ndarray, dict],
                          predicted: Union[np.ndarray, dict],
                          key: str
                          ) -> Tuple[np.ndarray, np.ndarray]:

        if self.dh.source_is_dict or self.dh.source_is_list:
            true = self.dh.inverse_transform(true, key=key)
            if isinstance(predicted, np.ndarray):
                assert len(true) == 1
                predicted = {list(true.keys())[0]: predicted}
            predicted = self.dh.inverse_transform(predicted, key=key)

        else:
            true_shape, pred_shape = true.shape, predicted.shape
            if isinstance(true, np.ndarray) and self.forecast_len == 1 and isinstance(self.num_outs, int):
                true = pd.DataFrame(true.reshape(len(true), self.num_outs), columns=self.out_cols)
                predicted = pd.DataFrame(predicted.reshape(len(predicted), self.num_outs), columns=self.out_cols)

            true = self.dh.inverse_transform(true, key=key)
            predicted = self.dh.inverse_transform(predicted, key=key)

            if predicted.__class__.__name__ in ['DataFrame', 'Series']:
                predicted = predicted.values
            if true.__class__.__name__ in ['DataFrame', 'Series']:
                true = true.values

            true = true.reshape(true_shape)
            predicted = predicted.reshape(pred_shape)

        return true, predicted

    def plot_model(self, nn_model) -> None:
        kwargs = {}
        if int(tf.__version__.split('.')[1]) > 14:
            kwargs['dpi'] = 300

        try:
            keras.utils.plot_model(nn_model, to_file=os.path.join(self.path, "model.png"), show_shapes=True, **kwargs)
        except (AssertionError, ImportError) as e:
            print(f"dot plot of model could not be plotted due to {e}")
        return

    def get_opt_args(self) -> dict:
        """get input arguments for an optimizer.

        It is being explicitly defined here so that it can be overwritten
        in sub-classes
        """
        kwargs = {'lr': self.config['lr']}

        if self.config['backend'] == 'tensorflow' and int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) >= 250:
            kwargs['learning_rate'] = kwargs.pop('lr')

        if self.config['backend'] == 'pytorch':
            kwargs.update({'params': self.parameters()})  # parameters from pytorch model
        return kwargs

    def get_metrics(self) -> list:
        """Returns the performance metrics specified."""
        _metrics = self.config['metrics']

        metrics = None
        if _metrics is not None:
            if not isinstance(_metrics, list):
                assert isinstance(_metrics, str)
                _metrics = [_metrics]

            from ai4water.utils.tf_losses import nse, kge, pbias, tf_r2

            metrics_with_names = {'nse': nse,
                                  'kge': kge,
                                  "r2": tf_r2,
                                  'pbias': pbias}

            metrics = []
            for m in _metrics:
                if m in metrics_with_names.keys():
                    metrics.append(metrics_with_names[m])
                else:
                    metrics.append(m)
        return metrics

    def view(
            self,
            layer_name: Union[list, str] = None,
            data: str = 'training',
            x=None,
            y=None,
            examples_to_view=None,
            show=False
    ):
        """shows all activations, weights and gradients of the model.

        Arguments:
            layer_name:
                the layer to view. If not given, all the layers will be viewed.
                This argument is only required when the model consists of layers of neural
                networks.
            data:
                the data to use when making calls to model for activation calculation
                or for gradient calculation. It can either 'training', 'validation' or
                'test'.
            x:
                input, alternative to data. If given it will override `data` argument.
            y:
                target/observed/label, alternative to data. If given it will
                override `data` argument.
            examples_to_view:
                the examples to view.
            show:
                whether to show the plot or not!

        Returns:
            An isntance of [Visualize][ai4water.postprocessing.visualize.Visualize] class.
        """
        from ai4water.postprocessing.visualize import Visualize

        visualizer = Visualize(model=self)

        visualizer(layer_name,
                   data=data,
                   x=x,
                   y=y,
                   examples_to_use=examples_to_view,
                   show=show)

        return visualizer

    def interpret(
            self,
            **kwargs
    ):
        """
        Interprets the underlying model. Call it after training.

        Returns:
            An instance of [Interpret][ai4water.postprocessing.interpret.Interpret] class

        Example:
            >>> model.fit()
            >>> model.interpret()
        ```
        """
        # importing ealier will try to import np types as well again
        from ai4water.postprocessing import Interpret

        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        if 'layers' not in self.config['model']:

            if self.mode.lower().startswith("cl"):

                data = self.test_data()
                x, y = maybe_three_outputs(data)
                self.confusion_matrx(x=x, y=y)
                self.precision_recall_curve(x=x, y=y)
                self.roc_curve(x=x, y=y)

        return Interpret(self)

    def explain(self, *args, **kwargs):
        """Calls the [explain_model][ai4water.postprocessing.explain.explain_model] function
         to explain the model.
         """
        from ai4water.postprocessing.explain import explain_model
        return explain_model(self, *args, **kwargs)

    def save_indices(self):
        indices = {}
        for idx in ['train_indices', 'test_indices']:
            if hasattr(self, idx):
                idx_val = getattr(self, idx)
                if idx_val is not None and not isinstance(idx_val, str):
                    idx_val = np.array(idx_val, dtype=int).tolist()
            else:
                idx_val = None

            indices[idx] = idx_val
        dict_to_file(indices=indices, path=self.path)
        return

    def save_config(self, history: dict = None):

        self.save_indices()

        config = dict()
        if history is not None:
            config['min_loss'] = None
            config['min_val_loss'] = None
            min_loss_array = history.get('min_loss_array', None)
            val_loss_array = history.get('val_loss', None)

            if val_loss_array is not None and not all(np.isnan(val_loss_array)):
                config['min_val_loss'] = np.nanmin(val_loss_array)
            if min_loss_array is not None and not all(np.isnan(min_loss_array)):
                config['min_loss'] = np.nanmin(min_loss_array)

        config['config'] = self.config.copy()
        config['method'] = self.method

        if 'path' in config['config']:  # we don't want our saved config to have 'path' key in it
            config['config'].pop('path')

        if self.category == "DL":
            config['loss'] = self.loss_name()

        dict_to_file(config=config, path=self.path)
        return config

    @classmethod
    def from_config(
            cls,
            config: dict,
            data=None,
            make_new_path: bool = False,
            **kwargs
    ):
        """Loads the model from config dictionary i.e. model.config

        Arguments:
            config: dict
                dictionary containing model's parameters i.e. model.config
            data:
                the data
            make_new_path:
                whether to make new path or not?
            kwargs:
                any additional keyword arguments to Model class.
        Returns:
            an instalnce of Model
        """
        config, path = cls._get_config_and_path(cls, config=config, make_new_path=make_new_path)

        return cls(**config,
                   data=data,
                   path=path,
                   **kwargs)

    @classmethod
    def from_config_file(
            cls,
            config_path: str,
            data=None,
            make_new_path: bool = False,
            **kwargs) -> "BaseModel":
        """
        Loads the model from a config file.

        Arguments:
            config_path:
                complete path of config file
            data :
                data for Model
            make_new_path:
                If true, then it means we want to use the config
                file, only to build the model and a new path will be made. We
                would not normally update the weights in such a case.
            kwargs :
                any additional keyword arguments for the `Model`
        Return:
            an instance of `Model` class
        """
        config, path = cls._get_config_and_path(cls, config_path=config_path, make_new_path=make_new_path)

        return cls(**config,
                   data=data,
                   path=path,
                   **kwargs)

    @staticmethod
    def _get_config_and_path(cls, config_path: str = None, config=None, make_new_path=False):
        """Sets some attributes of the cls so that it can be built from config.

        Also fetches config and path which are used to initiate cls."""
        if config is not None and config_path is not None:
            raise ValueError

        if config is None:
            assert config_path is not None
            with open(config_path, 'r') as fp:
                config = json.load(fp)
                config = config['config']
                idx_file = os.path.join(os.path.dirname(config_path), 'indices.json')
                path = os.path.dirname(config_path)
        else:
            assert isinstance(config, dict), f"config must be dictionary but it is of type {config.__class__.__name__}"
            path = config['path']
            idx_file = os.path.join(path, 'indices.json')

        # todo
        # shouldn't we remove 'path' from Model's init? we just need prefix
        # path is needed in clsas methods only?
        if 'path' in config:
            config.pop('path')

        with open(idx_file, 'r') as fp:
            indices = json.load(fp)

        cls.from_check_point = True

        # These paras neet to be set here because they are not withing init method
        cls.test_indices = indices["test_indices"]
        cls.train_indices = indices["train_indices"]

        if make_new_path:
            cls.allow_weight_loading = False
            path = None
        else:
            cls.allow_weight_loading = True

        return config, path

    def update_weights(self, weight_file: str = None):
        """
        Updates the weights of the underlying model.

        Arguments:
            weight_file:
                complete path of weight file. If not given, the
                weights are updated from model.w_path directory. For neural
                network based models, the best weights are updated if more
                than one weight file is present in model.w_path.
        Returns:
            None
        """
        if weight_file is None:
            weight_file = find_best_weight(self.w_path)
            weight_file_path = os.path.join(self.w_path, weight_file)
        else:
            assert os.path.isfile(weight_file), f'weight_file must be complete path of weight file'
            weight_file_path = weight_file
            weight_file = os.path.basename(weight_file)  # for printing

        if not self.allow_weight_loading:
            raise ValueError(f"Weights loading not allowed because allow_weight_loading is {self.allow_weight_loading}"
                             f"and model path is {self.path}")

        if self.category == "ML":
            self._model = joblib.load(weight_file_path)
        else:
            # loads the weights of keras model from weight file `w_file`.
            if self.api == 'functional' and self.config['backend'] == 'tensorflow':
                self._model.load_weights(weight_file_path)
            else:
                self.load_weights(weight_file_path)
        if self.verbosity > 0:
            print("{} Successfully loaded weights from {} file {}".format('*' * 10, weight_file, '*' * 10))
        return

    def write_cache(self, _fname, input_x, input_y, label_y):
        fname = os.path.join(self.path, _fname)
        if h5py is not None:
            h5 = h5py.File(fname, 'w')
            h5.create_dataset('input_X', data=input_x)
            h5.create_dataset('input_Y', data=input_y)
            h5.create_dataset('label_Y', data=label_y)
            h5.close()
        return

    def eda(self, freq: str = None, cols=None):
        """Performs comprehensive Exploratory Data Analysis.

        Arguments:
            freq:
                if specified, small chunks of data will be plotted instead of
                whole data at once. The data will NOT be resampled. This is valid
                only `plot_data` and `box_plot`. Possible values are `yearly`,
                    weekly`, and  `monthly`.
            cols:

        Returns:
            an instance of [EDA][ai4water.eda.EDA] class
        """
        # importing EDA earlier will import numpy etc as well
        from ai4water.eda import EDA

        # todo, Uniform Manifold Approximation and Projection (UMAP) of input data
        if self.data is None:
            print("data is None so eda can not be performed.")
            return
        # todo, radial heatmap to show temporal trends http://holoviews.org/reference/elements/bokeh/RadialHeatMap.html
        eda = EDA(data=self.data, path=self.path, in_cols=self.in_cols, out_cols=self.out_cols, save=True)

        eda()

        return eda

    def update_info(self):
        from .backend import lightgbm, tcn, catboost, xgboost
        self.info['version_info'] = get_version_info(
            tf=tf,
            keras=keras,
            torch=torch,
            np=np,
            pd=pd,
            matplotlib=matplotlib,
            h5py=h5py,
            joblib=joblib,
            lightgbm=lightgbm,
            tcn=tcn,
            catboost=catboost,
            xgboost=xgboost
        )
        return

    def print_info(self):
        class_type = ''
        if self.is_binary:
            class_type = "binary"
        elif self.is_multiclass:
            class_type = "multi-class"
        elif self.is_multilabel:
            class_type = "multi-label"

        if isinstance(self.config['model'], dict):
            if 'layers' in self.config['model']:
                model_name = self.__class__.__name__
            else:
                model_name = list(self.config['model'].keys())[0]
        else:
            if isinstance(self.config['model'], str):
                model_name = self.config['model']
            else:
                model_name = self.config['model'].__class__.__name__

        if self.verbosity > 0:
            print('building {} model for {} {} problem using {}'.format(self.category,
                                                                        class_type,
                                                                        self.mode,
                                                                        model_name))
        return

    def get_optimizer(self):
        opt_args = self.get_opt_args()
        optimizer = OPTIMIZERS[self.config['optimizer']](**opt_args)

        return optimizer

    def optimize_hyperparameters(
            self,
            algorithm: str = "bayes",
            num_iterations: int = 14,
            process_results: bool = True,
            update_config: bool = True,
            data: Union[tuple, list] = None,
            **kwargs
    ):
        """
        optimizes the hyperparameters of the built model

        The parameaters that needs to be optimized, must be given as space.

        Arguments:
            algorithm:
                the algorithm to use for optimization
            num_iterations:
                number of iterations for optimization.
            process_results:
                whether to perform postprocessing of optimization results or not
            update_config:
                whether to update the config of model or not.
            data:
                Only required if `data` argument to Model class was not provided
                i.e. user has its own custom data to train on. It must be a tuple
                of x,y pairs.
        Returns:
            an instance of [HyperOpt][ai4water.hyperopt.HyperOpt] which is used for optimization

        Examples:
            >>> from ai4water import Model
            >>> from ai4water.datasets import arg_beach
            >>> from ai4water.hyperopt import Integer, Categorical, Real
            >>> model_config = {"XGBoostRegressor": {"n_estimators": Integer(low=10, high=20),
            >>>                 "max_depth": Categorical([10, 20, 30]),
            >>>                 "learning_rate": Real(0.00001, 0.1)}}
            >>> model = Model(model=model_config, data=arg_beach())
            >>> optimizer = model.optimize_hyperparameters()

            Same can be done if a model is defined using neural networks
            >>> model_conf = {"layers":
            ...     {"LSTM":  {"config": {"units": Integer(32, 64), "activation": "relu"}},
            ...      "Dense1": {"units": 1,
            ...            "activation": Categorical(["relu", "tanh"], name="dense1_act")}}}
            >>> model = Model(model=model_config, data=arg_beach())
            >>> optimizer = model.optimize_hyperparameters()
        """
        from ._optimize import OptimizeHyperparameters  # optimize_hyperparameters

        _optimizer = OptimizeHyperparameters(
            self,
            list(self.opt_paras.values()),
            algorithm=algorithm,
            num_iterations=num_iterations,
            process_results=process_results,
            data=data,
            **kwargs
        ).fit()

        algo_type = list(self.config['model'].keys())[0]

        if update_config:
            new_model_config = update_model_config(self._original_model_config['model'],
                                                   _optimizer.best_paras())
            self.config['model'][algo_type] = new_model_config

            new_other_config = update_model_config(self._original_model_config['other'],
                                                   _optimizer.best_paras())
            self.config.update(new_other_config)

        return _optimizer

    def optimize_transformations(
            self,
            transformations: Union[list, str] = None,
            algorithm: str = "bayes",
            num_iterations: int = 12,
            include: Union[str, list, dict] = None,
            exclude: Union[str, list] = None,
            append: dict = None,
            update_config: bool = True
    ):
        """optimizes the transformations for the input/output features

        The 'val_score' parameter given as input to the Model is used as objective
        function for optimization problem.

        Arguments:
            transformations:
                the transformations to consider. By default, following
                transformations are considered

                - minmax
                - zscore also known as standard scaler
                - power
                - quantile
                - robust
                - log
                - log2
                - log10

            algorithm: str
                The algorithm to use for optimizing transformations
            num_iterations: int
                The number of iterations for optimizatino algorithm.
            include: the names of features to include
            exclude: the name/names of features to exclude
            append:
                the features with custom candidate transformations. For example
                if we want to try only `minmax` and `zscore` on feature `tide_cm`, then
                it can be done as following
                ```python
                >>> append={"tide_cm": ["minmax", "zscore"]}
                ```
            update_config: whether to update the config of model or not.

        Returns:
            an instance of [HyperOpt][ai4water.hyperopt.HyperOpt] class which is used for optimization

        Example:
            >>> from ai4water.datasets import arg_beach
            >>> from ai4water import Model
            >>> model = Model(model="xgboostregressor", data=arg_beach())
            >>> optimizer = model.optimize_transformations(exclude="tide_cm")
            >>> print(optimizer.best_paras())  # find the best/optimized transformations
            >>> model.fit()
            >>> model.predict()
        """
        from ._optimize import OptimizeTransformations  # optimize_transformations

        categories = ["minmax", "zscore", "log", "robust", "quantile", "log2", "log10", "power", "none"]

        if transformations is not None:
            assert isinstance(transformations, list)
            assert all([t in categories for t in transformations]), f"transformations must be one of {categories}"
            categories = transformations

        optimizer = OptimizeTransformations(
            self,
            algorithm=algorithm,
            num_iterations=num_iterations,
            include=include,
            exclude=exclude,
            append=append,
            categories=categories
        ).fit()

        transformations = []
        for feature, method in optimizer.best_paras().items():
            if method == "none":
                pass
            else:
                t = {'method': method, 'features': [feature]}

                if method.startswith("log"):
                    t["replace_nans"] = True
                    t["replace_zeros"] = True

                transformations.append(t)

        if update_config:
            self.config['transformation'] = transformations
            self.dh.config['transformation'] = transformations

        return optimizer

    def permutation_importance(
            self,
            data="test",
            x=None,
            y=None,
            scoring: Union[str, Callable] = "r2",
            n_repeats: int = 5,
            noise: Union[str, np.ndarray] = None,
            use_noise_only: bool = False,
            weights=None,
            plot_type: str = None
    ):
        """Calculates the permutation importance on the given data

        Arguments:
            data:
                one of `training`, `test` or `validation`. By default test data is
                used based upon recommendations of [Christoph Molnar's book](https://christophm.github.io/interpretable-ml-book/feature-importance.html#feature-importance-data)
            x:
                inputs for the model. alternative to data
            y:
                target/observation data for the model. alternative to data
            scoring:
                the scoring to use to calculate importance
            n_repeats:
                number of times the permutation for each feature is performed.
            noise:
                the noise to add when a feature is permutated. It can be a 1D
                array of length equal to len(data) or string defining the
                distribution
            use_noise_only:
                If True, then the feature being perturbed is replaced by the noise
                instead of adding the noise into the feature. This argument is only
                valid if `noise` is not None.
            weights:
            plot_type:
                if not None, it must be either `heatmap` or `boxplot`

        Examples:
            >>> from ai4water import Model
            >>> from ai4water.datasets import arg_beach
            >>> model = Model(model="XGBRegressor", data=arg_beach())
            >>> perm_imp = model.permutation_importance("validation", plot_type="boxplot")
        """
        assert data in ("training", "validation", "test")

        if x is None:
            data = getattr(self, f"{data}_data")()
            x, y = data

        from .postprocessing.explain import PermutationImportance
        pm = PermutationImportance(
            self.predict, x, y, scoring,
            n_repeats, noise, use_noise_only,
            path=os.path.join(self.path, "explain"),
            features=self.dh.input_features,
            weights=weights,
            seed=self.config['seed']
        )

        if plot_type is None:
            return pm.importances
        else:
            assert plot_type in ("boxplot", "heatmap")
            return getattr(pm, f"plot_as_{plot_type}")()


def get_values(outputs):

    if isinstance(outputs, dict) and len(outputs) == 1:
        outputs = list(outputs.values())[0]

    return outputs
