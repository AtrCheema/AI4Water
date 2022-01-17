import math
import os
import json
import time
import random
import warnings
from pickle import PicklingError
from typing import Union, Callable
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
from ai4water.utils.visualizations import ProcessResults
from ai4water.utils.utils import make_model
from ai4water.preprocessing.transformations import Transformations
from ai4water.utils.utils import maybe_three_outputs, get_version_info
from ai4water.models.tensorflow.custom_training import train_step, test_step
from ai4water.utils.utils import maybe_create_path, dict_to_file, dateandtime_now
from ai4water.utils.utils import find_best_weight, reset_seed, update_model_config
from ai4water.preprocessing.datahandler import DataHandler, SiteDistributedDataHandler
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


class BaseModel(NN):

    """ Model class that implements logic of AI4Water. """

    def __init__(self,
                 model: Union[dict, str] = None,
                 x_transformation: Union[str, dict, list] = None,
                 y_transformation:Union[str, dict, list] = None,
                 lr: float = 0.001,
                 optimizer='Adam',
                 loss: Union[str, Callable] = 'mse',
                 quantiles=None,
                 epochs: int = 14,
                 min_val_loss: float = 0.0001,
                 patience: int = 100,
                 save_model: bool = True,
                 metrics: Union[str, list] = None,
                 val_metric: str = None,
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
                the model from one of following libraries

                    - scikit-learn
                    - xgboost
                    - catboost
                    - lightgbm
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
            x_transformation:
                type of transformation to be applied on x data.
                The transformation can be any transformation name from
                ai4water.utils.transformations.py. The user can specify more than
                one transformation. Moreover, the user can also determine which
                transformation to be applied on which input feature. Default is 'minmax'.
                To apply a single transformation on all the data
                ```python
                transformation = 'minmax'
                ```
                To apply different transformations on different input and output features
                ```python
                transformation = [{'method': 'minmax', 'features': ['input1', 'input2']},
                                {'method': 'zscore', 'features': ['input3', 'input4']}
                                ]
                ```
                Here `input1`, `input2`, `input3` and `input4` are the columns in the
                `data`. For more info see [Transformations][ai4water.preprocessing.Transformations]
                and [Transformation][ai4water.preprocessing.Transformation] classes.
            y_transformation:
                type of transformation to be applied on y/label data.
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
                experiment comparison. If not defined then [r2_score][ai4water.postprocessing.SeqMetrics.RegressionMetrics.r2_score]
                will be used for regression and [accuracy][r2_score][ai4water.postprocessing.SeqMetrics.ClassificationMetrics.accuracy]
                will be used for classification.
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

        Note
        -----
            The transformations applied on `x` and `y` data using `x_transformation`
            and `y_transformations` are part of **model**. See [this](https://stats.stackexchange.com/q/555839/314919)
        Example:
            >>>from ai4water import Model
            >>>from ai4water.datasets import busan_beach
            >>>df = busan_beach()
            >>>model_ = Model(input_features=df.columns.tolist()[0:-1],
            ...              batch_size=16,
            ...              output_features=df.columns.tolist()[-1:],
            ...              model={'layers': {'LSTM': 64, 'Dense': 1}},
            ...)
            >>>history = model_.fit(data=df)
            >>>y, obs = model_.predict()
        """
        if self._go_up:
            maker = make_model(
                model=model,
                x_transformation=x_transformation,
                y_transformation=y_transformation,
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

            reset_seed(maker.config['seed'], os=os, random=random, tf=tf, torch=torch)
            if tf is not None:
                # graph should be cleared everytime we build new `Model` otherwise, if two `Models` are prepared in same
                # file, they may share same graph.
                tf.keras.backend.clear_session()

            self.data_config = maker.data_config

            self.opt_paras = maker.opt_paras
            self._original_model_config = maker.orig_model

            NN.__init__(self, config=maker.config)

            self.path = maybe_create_path(path=path, prefix=prefix)
            self.config['path'] = self.path
            self.verbosity = verbosity
            self.category = self.config['category']

            self.info = {}

    @property
    def is_custom_model(self):
        return self.config['is_custom_model_']

    @property
    def model_name(self)->str:
        if self.config.get('model_name_', None):
            return self.config['model_name_']

        model_def = self.config['model']
        if isinstance(model_def, str):
            return model_def
        elif isinstance(model_def, dict):
            return list(model_def.keys())[0]

    @property
    def mode(self)->str:
        from .experiments.utils import regression_models, classification_models
        if self.config.get('mode', None):
            return self.config['mode']
        elif self.is_custom_model:
            if self.config['loss'] in ['sparse_categorical_crossentropy',
                                       'categorical_crossentropy',
                                       'binary_crossentropy']:
                _mode = "classification"
            else:
                _mode = None
        elif self.model_name is not None:
            if self.model_name in classification_models():
                _mode = "classification"
            elif self.model_name in regression_models():
                _mode = "regression"
            elif 'class' in self.model_name.lower():
                _mode = "classification"
            elif "regr" in self.model_name.lower():
                _mode = "regression"
            elif self.config['loss'] in ['sparse_categorical_crossentropy',
                                         'categorical_crossentropy',
                                         'binary_crossentropy']:
                _mode = "classification"
            elif self.model_name == "layers":
                # todo
                _mode = "regression"
            else:
                raise NotImplementedError
        elif self.config['loss'] in ['sparse_categorical_crossentropy',
                                     'categorical_crossentropy',
                                     'binary_crossentropy']:
            _mode = "classification"
        elif self.model_name == "layers":
            # todo
            _mode = "regression"
        else:  # when model_name is None, mode should also be None.
            _mode = None
        # so that next time don't have to go through all these ifelse statements
        self.config['mode'] = _mode
        self.data_config['mode'] = _mode
        return _mode

    @property
    def _estimator_type(self):
        if self.mode == "regression":
            return "regressor"
        return "classifier"

    @property
    def input_features(self):
        if hasattr(self, 'dh_'):
            return self.dh_.input_features
        return self.config['input_features']

    @property
    def num_ins(self):  # raises error if input_features are not defined
        return len(self.input_features)

    @property
    def output_features(self):
        if hasattr(self, 'dh_'):
            return self.dh_.output_features
        return self.config['output_features']

    @property
    def num_outs(self):
        return len(self.output_features)

    @property
    def forecast_len(self):
        if hasattr(self, 'dh_'):
            return self.dh_.forecast_len
        return self.config['forecast_len']

    @property
    def val_metric(self):
        if self.mode=='regression':
            return self.config['val_metric'] or 'r2_score'
        return self.config['val_metric'] or 'accuracy'

    @property
    def forecast_step(self):
        if hasattr(self, 'dh_'):
            return self.dh_.forecast_step
        return self.config['forecast_step']

    @property
    def is_binary(self):
        if hasattr(self, 'dh_'):
            return self.dh_.is_binary
        raise NotImplementedError

    @property
    def is_multiclass(self)->bool:
        if self.mode == "classification":
            if hasattr(self, 'dh_'):
                return self.dh_.is_multiclass
            if len(self.classes_)>2:
                return True
        return False

    @property
    def classes_(self):
        if hasattr(self, 'dh_'):
            return self.dh_.classes
        elif self.category == "ML" and self.mode == "classification":
            return self._model.classes_
        raise NotImplementedError

    @property
    def num_classes(self):
        if hasattr(self, 'dh_'):
            return self.dh_.num_classes
        raise NotImplementedError

    @property
    def is_multilabel(self):
        if hasattr(self, 'dh_'):
            if self.dh_.data is None:
                return None
            return self.dh_.is_multilabel
        raise NotImplementedError

    def _get_dummy_input_shape(self):
        raise NotImplementedError

    def build(self, *args, **kwargs):
        raise NotImplementedError

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

    def loss_name(self):
        raise NotImplementedError

    @property
    def teacher_forcing(self):  # returns None if undefined
        if hasattr(self, 'dh_'):
            return self.dh_.teacher_forcing
        return self.config['teacher_forcing']

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

        if isinstance(callbacks, dict):
            for val in callbacks.values():
                _callbacks.append(val)
        else:
            # if any callback provided by user is similar to what we already prepared, take the one
            # provided by the user
            assert isinstance(callbacks, list)
            cbs_provided = [cb.__class__.__name__ for cb in callbacks]
            for cb in _callbacks:
                if not cb.__class__.__name__ in cbs_provided:
                    callbacks.append(cb)
            _callbacks = callbacks

        return _callbacks

    def get_val_data(self, val_data):
        """Finds out if there is val_data or not"""
        if isinstance(val_data, tuple):
            x,y = val_data
            if x is None and y is None:
                return None
            elif hasattr(x, '__len__') and len(x)==0:
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

    def _call_fit_fn(self, x, **kwargs):
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

    def _fit(self, inputs, outputs, validation_data, validation_steps=None, callbacks=None, **kwargs):

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

        # natively prepared arguments
        _kwargs = {
            'x':inputs,
            'y':None if inputs.__class__.__name__ in ['TorchDataset', 'BatchDataset'] else outputs,
            'epochs':self.config['epochs'],
            'batch_size':batch_siz,
            'validation_data':validation_data,
            'callbacks':callbacks,
            'shuffle':self.config['shuffle'],
            'steps_per_epoch':self.config['steps_per_epoch'],
            'verbose':max(self.verbosity, 0),
            'nans_in_y_exist':nans_in_y_exist,
            'validation_steps':validation_steps,
        }

        # arguments explicitly provided by user during .fit will take priority
        for k,v in kwargs.items():
            if k in _kwargs:
                if k in self.config:  # also update config
                    self.config[k] = v
                _kwargs.pop(k)

        self._call_fit_fn(
            **_kwargs,
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
                           tags=['ai4water', self.api, self.category, self.mode],
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

        self.residual_threshold_not_set_ = False
        if estimator == "RANSACRegressor" and 'residual_threshold' not in kwargs:
            self.residual_threshold_not_set_ = True

        if estimator in ["LGBMRegressor", 'LGBMClassifier']:
            if 'random_state' not in kwargs:
                kwargs['random_state'] = self.config['seed']

        if self.is_custom_model:
            if hasattr(estimator, '__call__'):  # initiate the custom model
                model = estimator(**kwargs)
            else:
                if len(kwargs)>0:
                    raise ValueError("""Initiating args not allowed because you 
                                        provided initiated class in dictionary""")
                model = estimator  # custom model is already instantiated

        # initiate the estimator/model class
        elif estimator in ml_models:
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
            x=None,
            y=None,
            data: Union[np.ndarray, pd.DataFrame, "DataHandler", str] = 'training',
            callbacks: Union[list, dict] = None,
            **kwargs
    ):
        """
        Trains the model with data which is taken from data accoring to `data` arguments.

        Arguments:
            x:
                The input data consisting of input features
            y:
                Correct labels/observations/true data corresponding to 'x'.
            data :
                Raw data fromw which `x`,`y` pairs are prepared. This will be
                passed to [DataHandler][ai4water.preprocessing.DataHandler].
                It can also be an instance if [DataHandler][ai4water.preprocessing.DataHandler] or
                [SiteDistributedDataHandler][ai4water.preprocessing.SiteDistributedDataHandler].
                It can also be name of dataset from [ai4water.datasets][ai4water.datasets.all_datasets]
            callbacks:
                Any callback compatible with keras. If you want to log the output
                to tensorboard, then just use `callbacks={'tensorboard':{}}` or
                to provide additional arguments
                ```python
                >>> callbacks={'tensorboard': {'histogram_freq': 1}}
                ```
            kwargs :
                Any keyword argument for the `fit` method of the underlying algorithm.
                if 'x' is present in kwargs, that will take precedent over `data`.
        Returns:
            A keras history object in case of deep learning model with tensorflow
            as backend or anything returned by `fit` method of underlying model.
        """

        if x is not None:
            assert y is not None

        return self.call_fit(x=x, y=y, data=data, callbacks=callbacks, **kwargs)

    def call_fit(self,
                 x=None,
                 y=None,
                 data='training',
                 callbacks=None,
                 **kwargs):

        visualizer = ProcessResults(path=self.path)
        self.is_training = True

        source = 'training'
        if isinstance(data, str) and data in ['validation', 'test']:
            source = data

        inputs, outputs, _, _, user_defined_x = self._fetch_data(source, x, y, data)

        # apply preprocessing/feature engineering if required.
        inputs = self._transform_x(inputs, 'x_transformer_')
        outputs = self._transform_y(outputs, 'y_transformer_')

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
            if 'validation_data' in kwargs:
                # validation data is provided by user but since transformations are part of
                # model so we must transform it
                val_x, val_y = kwargs['validation_data']
            elif user_defined_x:
                # user defined x,y but did not give validation data using
                # validation_data keyword argument
                val_x, val_y = None, None
            else:
                # either get validation data from DataHandler or maybe the user
                # has overwritten validation_data method, in both cases transformatins
                # are applied inside validation data.
                val_x, val_y = self.validation_data()
            val_x = self._transform_x(val_x, 'val_x_transformer_')
            val_y = self._transform_y(val_y, 'val_y_transformer_')
            kwargs['validation_data'] = val_x, val_y

            history = self._fit(inputs, outputs, callbacks=callbacks, **kwargs)

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

        if len(outputs) == outputs.size:
            outputs = outputs.reshape(-1, )

        self._maybe_change_residual_threshold(outputs)

        history = self._model.fit(inputs, outputs)

        if self._model.__class__.__name__.startswith("XGB") and inputs.__class__.__name__ == "ndarray":
            # by default feature_names of booster as set to f0, f1,...
            self._model.get_booster().feature_names = self.input_features

        self._save_ml_model()

        return history

    def _save_ml_model(self):
        """Saves the non-NN/ML models in the disk."""
        fname = os.path.join(self.w_path, self.model_name)

        if "tpot" not in self.model_name:
            try:
                joblib.dump(self._model, fname)
            except PicklingError:
                print(f"could not pickle {self.model_name} model")

        return

    def cross_val_score(
            self,
            x=None,
            y=None,
            data: Union[pd.DataFrame, np.ndarray, str] = None,
            scoring: str = None,
            refit: bool = False,
    ) -> float:
        """computes cross validation score

        Arguments:
            x:
                input data
            y:
                output corresponding to `x`.
            data:
                raw unprepared data which will be given to [DataHandler][ai4water.preprocessing.DataHandler]
                to prepare x,y from it.
            scoring:
                performance metric to use for cross validation.
                If None, it will be taken from config['val_metric']
            refit:
                If True, the model will be trained on the whole training+validation
                data after calculating cross validation score.
        Returns:
            cross validation score

        Note
        ----
            Currently not working for deep learning models.

        """
        from ai4water.postprocessing.SeqMetrics import RegressionMetrics, ClassificationMetrics

        if self.mode == "classification":
            Metrics = ClassificationMetrics
        else:
            Metrics = RegressionMetrics

        if scoring is None:
            scoring = self.val_metric

        scores = []

        if self.config['cross_validator'] is None:
            raise ValueError("Provide the `cross_validator` argument to the `Model` class upon initiation")

        cross_validator = list(self.config['cross_validator'].keys())[0]
        cross_validator_args = self.config['cross_validator'][cross_validator]

        if data is None:  # prepared data is given
            from .utils.utils import TrainTestSplit
            splitter = TrainTestSplit(x, y, test_fraction=self.config['test_fraction'])
            splits = splitter.KFold_splits(**cross_validator_args)

        else: # we need to prepare data first as x,y

            if callable(cross_validator):
                splits = cross_validator(**cross_validator_args)
            else:
                dh = DataHandler(data=data, **self.data_config)
                setattr(self, 'dh_', dh)
                splits = getattr(self.dh_, f'{cross_validator}_splits')(**cross_validator_args)

        for fold, ((train_x, train_y), (test_x, test_y)) in enumerate(splits):

            verbosity = self.verbosity
            self.verbosity = 0
            # make a new classifier/regressor at every fold
            self.build(self._get_dummy_input_shape())

            self.verbosity = verbosity

            self._maybe_change_residual_threshold(train_y)

            self.fit(x=train_x, y=train_y.reshape(-1, ))

            pred = self.predict(x=test_x, process_results=False)

            metrics = Metrics(test_y.reshape(-1, 1), pred)
            val_score = getattr(metrics, scoring)()

            scores.append(val_score)

            if self.verbosity > 0:
                print(f'fold: {fold} val_score: {val_score}')

        # save all the scores as json in model path`
        cv_name = str(cross_validator)
        fname = os.path.join(self.path, f'{cv_name}_{scoring}.json')
        with open(fname, 'w') as fp:
            json.dump(scores, fp)

        # set it as class attribute so that it can be used
        setattr(self, f'cross_val_{scoring}', scores)

        if refit:
            self.fit_on_all_training_data(data=data)

        # Even if we do not run .fit(), then we should still have model saved in
        # the disk so that it can be used.
        self._save_ml_model()

        cv_score = np.nanmean(scores).item()

        if not math.isfinite(cv_score):
            cv_score = 1.0

        return cv_score

    def fit_on_all_training_data(self, data=None):
        """Fits the model on training+validation data"""
        x_train, y_train = self.training_data(data=data)
        x_val, y_val = self.validation_data()
        if not isinstance(x_train, np.ndarray):
            raise NotImplementedError

        x, y = x_train, y_train
        # if not validation data is available then use only training data
        if x_val is not None:
            if hasattr(x_val, '__len__') and len(x_val)>0:
                x = np.concatenate([x_train, x_val])
                y = np.concatenate([y_train, y_val])

        return self.fit(x=x, y=y)


    def _maybe_change_residual_threshold(self, outputs) -> None:
        # https://stackoverflow.com/a/64396757/5982232
        if self.residual_threshold_not_set_:
            old_value = self._model.residual_threshold or mad(outputs.reshape(-1, ).tolist())
            if np.isnan(old_value) or old_value < 0.001:
                self._model.set_params(residual_threshold=0.001)
                if self.verbosity > 0:
                    print(f"""changing residual_threshold from {old_value} to
                           {self._model.residual_threshold}""")
        return

    def score(self, x=None, y=None, data='test', **kwargs):
        """since preprocesisng is part of Model, so the trained model with
        sklearn as backend must also be able to apply preprocessing on inputs
        before calculating score from sklearn. Currently it just calls the
        `score` function of sklearn by first transforming x and y."""
        if self.category == "ML" and hasattr(self, '_model'):
            x,y, _, _, _ = self._fetch_data(data, x=x,y=y, data=data)
            x = self._transform_x(x, 'x_score_')
            x = self._transform_x(x, 'y_score_')
            return self._model.score(x, y, **kwargs)
        raise NotImplementedError(f"can not calculate score")

    def predict_proba(self, x=None,  data='test', **kwargs):
        """since preprocesisng is part of Model, so the trained model with
        sklearn/xgboost/catboost/lgbm as backend must also be able to apply
        preprocessing on inputs before calling predict_proba from underlying library.
        Currently it just calls the `predict_proba` function of underlying library
        by first transforming x
        """
        if self.category == "ML" and hasattr(self, '_model'):
            x, _, _, _, _ = self._fetch_data(data, x=x, data=data)
            x = self._transform_x(x, 'x_proba_')
            return self._model.predict_proba(x,  **kwargs)
        raise NotImplementedError(f"can not calculate proba")

    def predict_log_proba(self, x=None,  data='test', **kwargs):
        """since preprocesisng is part of Model, so the trained model with
        sklearn/xgboost/catboost/lgbm as backend must also be able to apply
        preprocessing on inputs before calling predict_log_proba from underlying library.
        Currently it just calls the `log_proba` function of underlying library
        by first transforming x
        """
        if self.category == "ML" and hasattr(self, '_model'):
            x, _, _, _, _ = self._fetch_data(data, x=x, data=data)
            x = self._transform_x(x, 'x_log_proba_')
            return self._model.predict_log_proba(x, **kwargs)
        raise NotImplementedError(f"can not calculate log_proba")

    def evaluate(
            self,
            x=None,
            y=None,
            data='test',
            metrics=None,
            **kwargs
    ):
        """
        Evalutes the performance of the model on a given data.
        calls the `evaluate` method of underlying `model`. If the `evaluate`
        method is not available in underlying `model`, then `predict` is called.

        Arguments:
            x:
                inputs
            y:
                outputs/true data corresponding to `x`
            data:
                Raw unprepared data which will be fed to [DataHandler][ai4water.preprocessing.DataHandler]
                to prepare x and y.
                It is is a string then it will identify, which data type to use,
                valid values are `training`, `test` and `validation`. If `x` and
                `y` are given, this argument will have no meaning.
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
            >>> from ai4water.datasets import busan_beach
            >>> model = Model(model={"layers": {"Dense": 1}})
            >>> model.fit(data=busan_beach())

            for evaluation on test data
            >>> model.evaluate()

            for evaluation on training data
            >>> model.evaluate(data='training')

            evaluate on any metric from [Metrics][ai4water.postprocessing.SeqMetrics.RegressionMetrics]
            module
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
        return self.call_evaluate(x=x, y=y, data=data, **kwargs)

    def call_evaluate(self, x=None,y=None, data='test', metrics=None, **kwargs):

        source = 'test'
        if isinstance(data, str) and data in ['training', 'validation', 'test']:
            source = data

        x, y, _, _, _ = self._fetch_data(source, x, y, data)

        # dont' make call to underlying evaluate function rather manually
        # evaluate the given metrics
        if metrics is not None:
            return self._manual_eval(x, y, metrics)

        if hasattr(self._model, 'evaluate'):
            # todo, should we transform?
            return self._model.evaluate(x, y, **kwargs)

        return self.evaluate_fn(x, y, **kwargs)

    def evalute_ml_models(self, x, y, metrics=None):
        if metrics is None:
            metrics = self.val_metric
        return self._manual_eval(x, y, metrics)

    def _manual_eval(self, x, y, metrics):
        """manual evaluation"""
        p = self.predict(x=x, return_true=False, process_results=False)

        if self.mode == "regression":
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
            x=None,
            y=None,
            data: Union[str, pd.DataFrame, np.ndarray, DataHandler] = 'test',
            process_results: bool = True,
            metrics: str = "minimal",
            return_true: bool = False,
            **kwargs
    ):
        """
        Makes prediction from the trained model.

        Arguments:
            x:
                The data on which to make prediction. if given, it will override
                `data`
            y:
                Used for pos-processing etc. if given it will overrite `data`
            data:
                It can either be a string indicating which data to use. In this
                case, possible values are
                    - `training`
                    - `test`
                    - `validation`.
                By default, `test` data is used for predictions.
                It can also be unprepared/raw data which will be given to
                [DataHandler][ai4water.preprocessing.DataHandler]
                to prepare x,y values.

            process_results: bool
                post processing of results
            metrics: str
                only valid if process_results is True. The metrics to calculate.
                Valid values are 'minimal', 'all', 'hydro_metrics'
            return_true: bool
                whether to return the true values along with predicted values
                or not. Default is False, so that this method behaves sklearn type.
            kwargs : any keyword argument for `predict` method.
        Returns:
            An numpy array of predicted values.
            If return_true is True then a tuple of arrays. The first is true
            and the second is predicted. If `x` is given but `y` is not given,
            then, first array which is returned is None.
        """

        assert metrics in ("minimal", "all", "hydro_metrics")

        return self.call_predict(x=x,
                                 y=y,
                                 data=data,
                                 process_results=process_results,
                                 metrics=metrics,
                                 return_true=return_true,
                                 **kwargs)

    def call_predict(self,
                     x=None,
                     y=None,
                     data='test',
                     process_results=True,
                     metrics="minimal",
                     return_true: bool = False,
                     **kwargs):

        source = 'test'
        if isinstance(data, str) and data in ['training', 'validation', 'test']:
            source = data

        inputs, true_outputs, prefix, transformation_key, user_defined_data = self._fetch_data(
            source=source,
            x=x,
            y=y,
            data=data
        )

        inputs = self._transform_x(inputs, 'x_transformer_')

        y_transformer = None
        if true_outputs is not None:
            if self.config['y_transformation']:
                y_transformer = Transformations(self.output_features, self.config['y_transformation'])
                true_outputs = y_transformer.fit_transform(true_outputs)

        if self.category == 'DL':
            # some arguments specifically for DL models
            if 'verbose' not in kwargs:
                kwargs['verbose'] = self.verbosity

            if 'batch_size' in kwargs:  # if given by user
                self.config['batch_size'] = kwargs['batch_size']  # update config
            else:  # otherwise use from config
                kwargs['batch_size'] = self.config['batch_size']

            predicted = self.predict_fn(x=inputs,  **kwargs)

        else:
            if self._model.__class__.__name__.startswith("XGB") and inputs.__class__.__name__ == "ndarray":
                # since we have changed feature_names of booster,
                kwargs['validate_features'] = False

            predicted = self.predict_ml_models(inputs, **kwargs)

        if true_outputs is None:  # only x was given, build transformer from the config
            if self.config.get('y_transformer_', None) is None:
                # when predict/evaluate is called without fitting the model first
                # in such case either we must have true_outputs or y_transformation should be None.
                assert self.config['y_transformation'] is None
            else:
                y_transformer = Transformations.from_config(self.config['y_transformer_'])
                predicted = y_transformer.inverse_transform(predicted)
        elif self.config['y_transformation']:
            # both x,and true_y were given
            predicted = self._inverse_transform_y(predicted, y_transformer)
            true_outputs = self._inverse_transform_y(true_outputs, y_transformer)

        if true_outputs is None:
            if return_true:
                return true_outputs, predicted
            return predicted

        dt_index = np.arange(len(true_outputs))  # dummy/default index when data is user defined

        if not user_defined_data:
            true_outputs, dt_index = self.dh_.deindexify(true_outputs, key=transformation_key)

        if isinstance(true_outputs, np.ndarray) and true_outputs.dtype.name == 'object':
            true_outputs = true_outputs.astype(predicted.dtype)

        if true_outputs is None:
            process_results = False

        # initialize post-processes
        pp = ProcessResults(
            path=self.path,
            forecast_len=self.forecast_len,
            output_features=self.output_features,
            is_multiclass=self.is_multiclass,
            verbosity=self.verbosity,
            config=self.config
        )

        if self.quantiles is None:

            # it true_outputs and predicted are dictionary of len(1) then just get the values
            true_outputs = get_values(true_outputs)
            predicted = get_values(predicted)

            if process_results: # if mode has not been inferred yet, try to infer it
                if self.mode is None:  # because metrics cannot be calculated with wrong mode
                    if len(self.classes_) == 0:
                        self.config['mode'] = "regression"

                if self.mode == 'regression':
                    pp.process_regres_results(true_outputs,
                                                predicted,
                                                metrics=metrics,
                                                prefix=prefix + '_',
                                                index=dt_index,
                                                user_defined_data=user_defined_data)
                else:
                    pp.process_class_results(true_outputs,
                                               predicted,
                                               metrics=metrics,
                                               prefix=prefix,
                                               index=dt_index,
                                               user_defined_data=user_defined_data)

                    pp.confusion_matrx(self, x=inputs, y=true_outputs)
                    # if model does not have predict_proba method, we can't plot following
                    if hasattr(self._model, 'predict_proba'):
                        pp.precision_recall_curve(self, x=inputs, y=true_outputs)
                        pp.roc_curve(self, x=inputs, y=true_outputs)
        else:
            assert self.num_outs == 1

            if true_outputs.ndim == 2:  # todo, this should be avoided
                true_outputs = np.expand_dims(true_outputs, axis=-1)

            pp.quantiles = self.quantiles

            pp.plot_quantiles1(true_outputs, predicted)
            pp.plot_quantiles2(true_outputs, predicted)
            pp.plot_all_qs(true_outputs, predicted)

        if return_true:
            return true_outputs, predicted
        return predicted

    def predict_ml_models(self, inputs, **kwargs):
        """So that it can be overwritten easily for ML models."""
        return self.predict_fn(inputs, **kwargs)

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
            >>> from ai4water import Model
            >>> from ai4water.datasets import busan_beach
            >>> model = Model(model=...)
            >>> model.fit(data=busan_beach())
            >>> model.interpret()
        ```
        """
        # importing ealier will try to import np types as well again
        from ai4water.postprocessing import Interpret

        return Interpret(self)

    def explain(self, *args, **kwargs):
        """Calls the [explain_model][ai4water.postprocessing.explain.explain_model] function
         to explain the model.
         """
        from ai4water.postprocessing.explain import explain_model
        return explain_model(self, *args, **kwargs)

    def _save_indices(self):
        # saves the training and test indices to a json file
        indices = {}
        if hasattr(self, 'dh_'):
            for idx in ['train_indices', 'test_indices']:
                if hasattr(self.dh_, idx):
                    idx_values = getattr(self.dh_, idx)
                    if idx_values is not None and not isinstance(idx_values, str):
                        idx_values = np.array(idx_values, dtype=int).tolist()
                else:
                    idx_values = None

                indices[idx] = idx_values
        dict_to_file(indices=indices, path=self.path)
        return

    def save_config(self, history: dict = None):
        """saves the current state of model in a json file.
        By current state, we mean, train and test indices (if available),
        hyperparameters of related to model and data and current performance
        statistics. All the data is stored in model.path.
        """
        self._save_indices()

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
        config['config']['val_metric'] = self.val_metric  # it is calculated during run time
        config['method'] = self.method

        if 'path' in config['config']:  # we don't want our saved config to have 'path' key in it
            config['config'].pop('path')

        if self.category == "DL":
            config['loss'] = self.loss_name()

        if self.category == "ML" and self.is_custom_model:
            config['config']['model'] = self.model_name

        dict_to_file(config=config, path=self.path)
        return config

    @classmethod
    def from_config(
            cls,
            config: dict,
            make_new_path: bool = False,
            **kwargs
    ):
        """Loads the model from config dictionary i.e. model.config

        Arguments:
            config: dict
                dictionary containing model's parameters i.e. model.config
            make_new_path:
                whether to make new path or not?
            kwargs:
                any additional keyword arguments to Model class.
        Returns:
            an instalnce of Model
        """
        config, path = cls._get_config_and_path(cls, config=config, make_new_path=make_new_path)

        return cls(**config,
                   path=path,
                   **kwargs)

    @classmethod
    def from_config_file(
            cls,
            config_path: str,
            make_new_path: bool = False,
            **kwargs) -> "BaseModel":
        """
        Loads the model from a config file.

        Arguments:
            config_path:
                complete path of config file
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
                path = os.path.dirname(config_path)
        else:
            assert isinstance(config, dict), f"config must be dictionary but it is of type {config.__class__.__name__}"
            path = config['path']

        # todo
        # shouldn't we remove 'path' from Model's init? we just need prefix
        # path is needed in clsas methods only?
        if 'path' in config:
            config.pop('path')

        cls.from_check_point = True

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
            if not os.path.isfile(weight_file):
                raise ValueError(f'weight_file must be complete path of weight file but it is {weight_file}')
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

    def eda(self, data, freq: str = None):
        """Performs comprehensive Exploratory Data Analysis.

        Arguments:
            data:
            freq:
                if specified, small chunks of data will be plotted instead of
                whole data at once. The data will NOT be resampled. This is valid
                only `plot_data` and `box_plot`. Possible values are `yearly`,
                    weekly`, and  `monthly`.
        Returns:
            an instance of [EDA][ai4water.eda.EDA] class
        """
        # importing EDA earlier will import numpy etc as well
        from ai4water.eda import EDA

        # todo, Uniform Manifold Approximation and Projection (UMAP) of input data

        # todo, radial heatmap to show temporal trends http://holoviews.org/reference/elements/bokeh/RadialHeatMap.html
        eda = EDA(data=data,
                  path=self.path,
                  in_cols=self.input_features,
                  out_cols=self.output_features,
                  save=True)

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
            data: Union[tuple, list, pd.DataFrame, np.ndarray],
            algorithm: str = "bayes",
            num_iterations: int = 14,
            process_results: bool = True,
            update_config: bool = True,
            **kwargs
    ):
        """
        optimizes the hyperparameters of the built model

        The parameaters that needs to be optimized, must be given as space.

        Arguments:
            data:
                It can be one of following

                    - tuple of x,y pairs
                    - a list of
            algorithm:
                the algorithm to use for optimization
            num_iterations:
                number of iterations for optimization.
            process_results:
                whether to perform postprocessing of optimization results or not
            update_config:
                whether to update the config of model or not.
        Returns:
            an instance of [HyperOpt][ai4water.hyperopt.HyperOpt] which is used for optimization

        Examples:
            >>> from ai4water import Model
            >>> from ai4water.datasets import busan_beach
            >>> from ai4water.hyperopt import Integer, Categorical, Real
            >>> model_config = {"XGBoostRegressor": {"n_estimators": Integer(low=10, high=20),
            >>>                 "max_depth": Categorical([10, 20, 30]),
            >>>                 "learning_rate": Real(0.00001, 0.1)}}
            >>> model = Model(model=model_config)
            >>> optimizer = model.optimize_hyperparameters(data=busan_beach())

            Same can be done if a model is defined using neural networks
            >>> model_conf = {"layers": {
            ...     "Input": {"input_shape": (15, 13)},
            ...     "LSTM":  {"config": {"units": Integer(32, 64), "activation": "relu"}},
            ...      "Dense1": {"units": 1,
            ...            "activation": Categorical(["relu", "tanh"], name="dense1_act")}}}
            >>> model = Model(model=model_config)
            >>> optimizer = model.optimize_hyperparameters(data=busan_beach())
        """
        from ._optimize import OptimizeHyperparameters  # optimize_hyperparameters

        if isinstance(data, list) or isinstance(data, tuple):
            pass
        else:
            setattr(self, 'dh_', DataHandler(data, **self.data_config))

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
            data:Union[np.ndarray, pd.DataFrame],
            transformations: Union[list, str] = None,
            include: Union[str, list, dict] = None,
            exclude: Union[str, list] = None,
            append: dict = None,
            y_transformations: Union[list, dict] = None,
            algorithm: str = "bayes",
            num_iterations: int = 12,
            update_config: bool = True
    ):
        """optimizes the transformations for the input/output features

        The 'val_score' parameter given as input to the Model is used as objective
        function for optimization problem.

        Arguments:
            data:

            transformations:
                the transformations to consider. By default, following
                transformations are considered for input features

                - `minmax`  rescale from 0 to 1
                - `center`    center the data by subtracting mean from it
                - `scale`     scale the data by dividing it with its standard deviation
                - `zscore`    first performs centering and then scaling
                - `box-cox`
                - `yeo-johnson`
                - `quantile`
                - `robust`
                - `log`
                - `log2`
                - `log10`
                - `sqrt`    square root

            include: the names of input features to include
            exclude: the name/names of input features to exclude
            append:
                the input features with custom candidate transformations. For example
                if we want to try only `minmax` and `zscore` on feature `tide_cm`, then
                it can be done as following
                ```python
                >>> append={"tide_cm": ["minmax", "zscore"]}
                ```
            y_transformations:
                It can either be a list of transformations to be considered for
                output features for example
                ```
                >>> y_transformations = ['log', 'log10', 'log2', 'sqrt']
                ```
                would mean that consider `log`, `log10`, `log2` and `sqrt` are
                to be considered for output transformations during optimization.
                It can also be a dictionary whose keys are names of output features
                and whose values are lists of transformations to be considered for output
                features. For example
                ```
                >>> y_transformations = {'output1': ['log2', 'log10'], 'output2': ['log', 'sqrt']}
                ```
                Default is None, which means do not optimize transformation for output
                features.
            algorithm: str
                The algorithm to use for optimizing transformations
            num_iterations: int
                The number of iterations for optimizatino algorithm.
            update_config: whether to update the config of model or not.

        Returns:
            an instance of [HyperOpt][ai4water.hyperopt.HyperOpt] class which is used for optimization

        Example:
            >>> from ai4water.datasets import busan_beach
            >>> from ai4water import Model
            >>> model = Model(model="xgboostregressor")
            >>> optimizer_ = model.optimize_transformations(data=busan_beach(), exclude="tide_cm")
            >>> print(optimizer_.best_paras())  # find the best/optimized transformations
            >>> model.fit(data=busan_beach())
            >>> model.predict()
        """
        from ._optimize import OptimizeTransformations  # optimize_transformations
        from .preprocessing.transformations.utils import InvalidTransformation

        setattr(self, 'dh_', DataHandler(data=data, **self.data_config))

        allowed_transforamtions = ["minmax", "center", "scale", "zscore", "box-cox", "yeo-johnson",
                      "quantile", "robust", "log", "log2", "log10", "sqrt", "none",
                      ]

        append = append or {}

        categories = allowed_transforamtions
        if transformations is not None:
            assert isinstance(transformations, list)
            for t in transformations:
                if t not in allowed_transforamtions:
                    raise InvalidTransformation(t, allowed_transforamtions)
            categories = transformations

        if y_transformations:

            if isinstance(y_transformations, list):
                for t in y_transformations:
                    if t not in allowed_transforamtions:
                        raise InvalidTransformation(t, allowed_transforamtions)

                for out in self.output_features:
                    append[out] = y_transformations

            else:
                assert isinstance(y_transformations, dict)
                for out_feature, out_transformations in y_transformations.items():

                    assert out_feature in self.output_features
                    assert isinstance(out_transformations, list)
                    for t in out_transformations:
                        if t not in allowed_transforamtions:
                            raise InvalidTransformation(t, allowed_transforamtions)
                    append[out_feature] = out_transformations

        optimizer = OptimizeTransformations(
            self,
            algorithm=algorithm,
            num_iterations=num_iterations,
            include=include,
            exclude=exclude,
            append=append,
            categories=categories,
            data=data,
        ).fit()

        x_transformations = []
        y_transformations = []

        for feature, method in optimizer.best_paras().items():
            if method == "none":
                pass
            else:
                t = {'method': method, 'features': [feature]}

                if method.startswith("log"):
                    t["treat_negatives"] = True
                    t["replace_zeros"] = True
                elif method == "box-cox":
                    t["replace_zeros"] = True
                    t["treat_negatives"] = True
                elif method == "sqrt":
                    t["treat_negatives"] = True
                    t["replace_zeros"] = True

                if feature in self.input_features:
                    x_transformations.append(t)
                else:
                    y_transformations.append(t)

        if update_config:
            self.config['x_transformation'] = x_transformations
            self.config['y_transformation'] = y_transformations or None

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
                used based upon recommendations of
                [Christoph Molnar's book](https://christophm.github.io/interpretable-ml-book/feature-importance.html#feature-importance-data)
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
            >>> from ai4water.datasets import busan_beach
            >>> model = Model(model="XGBRegressor")
            >>> model.fit(data=busan_beach())
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
            feature_names=self.input_features,
            weights=weights,
            seed=self.config['seed']
        )

        if plot_type is None:
            return pm.importances
        else:
            assert plot_type in ("boxplot", "heatmap")
            return getattr(pm, f"plot_as_{plot_type}")()

    def _transform_x(self, x, name):
        """transforms x and puts the transformer in config witht he key name"""
        return self._transform(x, name, self.config['x_transformation'], self.input_features)

    def _transform_y(self, y, name):
        """transforms y and puts the transformer in config witht he key name"""
        return self._transform(y, name, self.config['y_transformation'], self.output_features)

    def _transform(self, data, name, transformation, feature_names):
        """transforms the `data` using `transformation` and puts it in config
        with config `name`."""
        if data is not None and transformation:
            transformer = Transformations(feature_names, transformation)
            data = transformer.fit_transform(data)
            self.config[name] = transformer.config()
        return data

    def _inverse_transform_y(self, y, transformer)->np.ndarray:
        """inverse transforms y where y is supposed to be true or predicted output
        from model."""
        if isinstance(y, np.ndarray):
            # it is ndarray, either num_outs>1 or quantiles>1 or forecast_len>1 some combination of them
            if y.size > len(y):
                if y.ndim == 2:
                    for out in range(y.shape[1]):
                        y[:, out] = transformer.inverse_transform(y[:, out]).reshape(-1, )
                else:
                    # (exs, outs, quantiles) or (exs, outs, forecast_len) or (exs, forecast_len, quantiles)
                    for out in range(y.shape[1]):
                        for q in range(y.shape[2]):
                            y[:, out, q] = transformer.inverse_transform(y[:, out, q]).reshape(-1, )
            else:  # 1d array
                y = transformer.inverse_transform(y)
        else:
            raise ValueError(f"can't inverse transform y of type {type(y)}")

        return y

    def training_data(self, x=None, y=None, data='training', key=None)->tuple:
        #x,y are not used but only given to be used if user overwrites this method
        return self.__fetch('training', data,key)

    def validation_data(self, x=None, y=None, data='validation', key=None, **kwargs)->tuple:
        """This method should return x,y pairs of validation data"""
        return self.__fetch('validation', data,key)

    def test_data(self, x=None, y=None, data='test', key=None)->tuple:
        """This method should return x,y pairs of validation data"""
        return self.__fetch('test', data,key)

    def __fetch(self, source, data=None, key=None):
        """if data is string, then it must either be `trianing`, `validation` or
        `test` or name of a valid dataset. Otherwise data is supposed to be raw
        data which will be given to DataHandler
        """
        if isinstance(data, str):
            if data in ['training', 'test', 'validation']:
                if hasattr(self, 'dh_'):
                    data = getattr(self.dh_, f'{data}_data')(key=key)
                else:
                    raise AttributeError(f"Unable to get {source} data")
            else:
                # e.g. 'CAMELS_AUS'
                dh = DataHandler(data=data, **self.data_config)
                setattr(self, 'dh_', dh)
                data = getattr(dh, f'{source}_data')(key=key)
        else:
            dh = DataHandler(data=data, **self.data_config)
            setattr(self, 'dh_', dh)
            data = getattr(dh, f'{source}_data')(key=key)

        x, y = maybe_three_outputs(data, self.teacher_forcing)

        return x, y

    def _fetch_data(self, source:str, x=None, y=None, data=None):
        """The main idea is that the user should be able to fully customize
        training/test data by overwriting training_data and test_data methods.
        However, if x is given or data is DataHandler then the training_data/test_data
        methods of this(Model) class will not be called."""
        user_defined_x = True
        prefix = f'{source}_{dateandtime_now()}'
        key = None

        if x is None:
            user_defined_x = False
            key=f"5_{source}"

            # the user has provided DataHandler from which training/test data needs to be extracted
            if isinstance(data, DataHandler) or isinstance(data, SiteDistributedDataHandler):
                setattr(self, 'dh_', data)
                data = getattr(data, f'{source}_data')(key=key)

            else:
                data = getattr(self, f'{source}_data')(x=x, y=y, data=data, key=key)

            # data may be tuple/list of three arrays
            x,y = maybe_three_outputs(data, self.teacher_forcing)

        return x,y, prefix, key, user_defined_x


def get_values(outputs):

    if isinstance(outputs, dict) and len(outputs) == 1:
        outputs = list(outputs.values())[0]

    return outputs
