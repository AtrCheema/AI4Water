
import math
import json
import time
import warnings
from types import MethodType
from pickle import PicklingError
from typing import Union, Callable, List, Tuple

from SeqMetrics import RegressionMetrics, ClassificationMetrics

from .nn_tools import NN

from .utils.utils import mad
from .utils.utils import make_model
from .utils.utils import AttribtueSetter
from .utils.utils import get_values
from .utils.utils import DataNotFound
from .utils.utils import maybe_create_path, dict_to_file, dateandtime_now
from .utils.utils import find_best_weight, reset_seed, update_model_config, METRIC_TYPES
from .utils.utils import maybe_three_outputs, get_version_info

from .utils import LossCurve
from .postprocessing import ProcessPredictions
from .postprocessing import feature_interaction
from .postprocessing import prediction_distribution_plot

from .preprocessing import DataSet
from .preprocessing.dataset._main import _DataSet
from .preprocessing.transformations import Transformations

from .models._tensorflow.custom_training import train_step, test_step

import ai4water.backend as K
from .backend import sklearn_models
from ai4water.backend import np, pd, plt, os, random
from .backend import tf, keras, torch, catboost_models, xgboost_models, lightgbm_models

if K.BACKEND == 'tensorflow' and tf is not None:
    from ai4water.tf_attributes import LOSSES, OPTIMIZERS

elif K.BACKEND == 'pytorch' and torch is not None:
    from ai4water.models._torch import LOSSES, OPTIMIZERS


class BaseModel(NN):

    """ Model class that implements logic of AI4Water. """

    def __init__(
            self,
            model: Union[dict, str, Callable] = None,
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
            monitor: Union[str, list] = None,
            val_metric: str = None,
            cross_validator: dict = None,
            wandb_config: dict = None,
            seed: int = 313,
            prefix: str = None,
            path: str = None,
            verbosity: int = 1,
            accept_additional_args: bool = False,
            **kwargs
    ):
        """
        The Model class can take a large number of possible arguments depending
        upon the machine learning model/algorithm used. Not all the arguments
        are applicable in each case. The user must define only the relevant/applicable
        parameters and leave the others as it is.

        Parameters
        ----------
            model :
                This determines the algorithm of ML model or architecture of DL
                model. It can be a dictionary defining machine learning model.
                If you are building
                a non-neural network model then this dictionary must consist of
                name of name of model as key and the keyword arguments to that
                model as dictionary. For example to build a decision forest based model

                >>> {'DecisionTreeRegressor': {"max_depth": 3,
                ...                                    "criterion": "mae"}}

                The key 'DecisionTreeRegressor' should exactly match the name of
                the model from one of following libraries

                - `sklearn`_
                - `xgboost`_
                - `catboost`_
                - `lightgbm`_

                The value {"max_depth": 3, "criterion": "mae"} is another dictionary
                which can be any keyword argument which the `model` (DecisionTreeRegressor
                in this case) accepts. The user must refer to the documentation
                of the underlying library (scikit-learn for DecisionTreeRegressor)
                to find out complete keyword arguments applicable for a particular model.
                See `examples <https://ai4water.readthedocs.io/en/latest/auto_examples/dec_model_def_ml.html>`_
                to learn how to build machine learning models
                If You are building a Deep Learning model using tensorflow, then the key
                must be 'layers' and the value must itself be a dictionary defining layers
                of neural networks. For example we can build an MLP as following

                >>> {'layers': {
                ...             "Dense_0": {'units': 64, 'activation': 'relu'},
                ...              "Flatten": {},
                ...              "Dense_3": {'units': 1}
                >>>             }}

                The MLP in this case consists of dense, and flatten layers. The user
                can define any keyword arguments which is accepted by that layer in
                TensorFlow. For example the `Dense` layer in TensorFlow can accept
                `units` and `activation` keyword argument among others. For details
                on how to buld neural networks using such layered API
                `see examples <https://ai4water.readthedocs.io/en/dev/declarative_def_tf.html>`_
            x_transformation :
                type of transformation to be applied on x/input data.
                The transformation can be any transformation name from
                :py:class:`ai4water.preprocessing.transformations.Transformation` .
                The user can specify more than
                one transformation. Moreover, the user can also determine which
                transformation to be applied on which input feature. Default is 'minmax'.
                To apply a single transformation on all the data

                >>> 'minmax'

                To apply different transformations on different input and output features

                >>> [{'method': 'minmax', 'features': ['input1', 'input2']},
                ...                {'method': 'zscore', 'features': ['input3', 'input4']}
                ...                 ]

                Here `input1`, `input2`, `input3` and `input4` are the columns in the
                `data`. For more info see :py:class:`ai4water.preprocessing.Transformations`
                and :py:class:`ai4water.preprocessing.Transformation` classes.
            y_transformation :
                type of transformation to be applied on y/label/output data.
            lr :, default 0.001.
                learning rate,
            optimizer : str/keras.optimizers like
                the optimizer to be used for neural network training. Default is 'Adam'
            loss : str/callable  Default is `mse`.
                the cost/loss function to be used for training neural networks.
            quantiles : list Default is None
                quantiles to be used when the problem is quantile regression.
            epochs : int  (default=14)
                number of epochs to be used.
            min_val_loss : float, (default = 0.0001).
                minimum value of validatin loss/error to be used for early stopping.
            patience : int, (default=100)
                number of epochs to wait before early stopping. Set this value to None
                if you don't want to use EarlyStopping. For gradient boosting algorithms
                it is equal to ``early_stopping_rounds``
            save_model : bool (default=True)
                whether to save the model or not. For neural networks, the model will
                be saved only an improvement in training/validation loss is observed.
                Otherwise model is not saved.
            monitor : str/list
                metrics to be monitored. e.g. ['nse', 'pbias']
            val_metric : str
                performance metric to be used for validation/cross_validation.
                This metric will be used for hyper-parameter optimizationa and
                experiment comparison. If not defined then
                r2_score_ will be used for regression and accuracy_ will be used
                for classification.
            cross_validator : dict
                selects the type of cross validation to be applied. It can be any
                cross validator from sklear.model_selection. Default is None, which
                means validation will be done using `validation_data`. To use
                kfold cross validation,

                >>> {'KFold': {'n_splits': 5}}

            batches : str
                either `2d` or 3d`.
            wandb_config : dict (default=None)
                Only valid if wandb package is installed.  Default value is None,
                which means, wandb will not be utilized. For simplest case, pass
                a dictionary with at least two keys namely `project` and `entity`.
                Otherwise use a dictionary of all the
                arugments for wandb.init, wandb.log and WandbCallback. For
                `training_data` and `validation_data` in `WandbCallback`, pass
                `True` instead of providing a tuple as shown below

                >>> {'entity': 'entity_name', 'project': 'project_name',
                ...                 'training_data':True, 'validation_data': True}

            seed : int, (default=313)
                random seed for reproducibility. This can be set to None. The seed
                is set to `os`, `tf`, `torch` and `random` modules simultaneously.
                Please note that this seed is not set for numpy because that
                will result in constant sampling during hyperparameter optimization.
                If you want to seed everything, then use following function
                >>> model.seed_everything()
            prefix : str
                prefix to be used for the folder in which the results are saved.
                default is None, which means within
                ./results/model_path
            path : str/path like
                if not given, new model_path path will not be created.
            verbosity : int default is 1
                determines the amount of information being printed. 0 means no
                print information. Can be between 0 and 3. Setting this value to 0
                will also result in not showing some plots such as loss curve or
                regression plot. These plots will only be saved in self.path.
                Setting this value to -1 will result in not saving any file in the disk.
                This value is also passed to XGBoost, LGBM and CatBoost libraries.
            accept_additional_args : bool  Default is False
                If you want to pass any additional argument, then this argument
                must be set to True, otherwise an error will be raise.
            **kwargs:
                keyword arguments for :py:meth:`ai4water.preprocessing.DataSet.__init__`

        Note
        -----
            The transformations applied on `x` and `y` data using `x_transformation`
            and `y_transformation` are part of **model**. See `transformation`_

        Examples
        -------
            >>> from ai4water import Model
            >>> from ai4water.datasets import busan_beach
            >>> from ai4water.models import MLP
            >>> df = busan_beach()
            >>> ann = Model(input_features=df.columns.tolist()[0:-1],
            ...              batch_size=16,
            ...              output_features=df.columns.tolist()[-1:],
            ...              model=MLP(64),
            ... )
            >>> history = ann.fit(data=df)
            >>> y = ann.predict()

        .. _sklearn:
            https://scikit-learn.org/stable/modules/classes.html

        .. _xgboost:
            https://xgboost.readthedocs.io/en/stable/python/index.html

        .. _catboost:
            https://catboost.ai/en/docs/concepts/python-quickstart

        .. _lightgbm:
            https://lightgbm.readthedocs.io/en/latest/

        .. _transformation:
            https://stats.stackexchange.com/q/555839/314919

        .. _RegressionMetrics:
            https://seqmetrics.readthedocs.io/en/latest/rgr.html#regressionmetrics

        .. _r2_score:
            https://seqmetrics.readthedocs.io/en/latest/rgr.html#SeqMetrics.RegressionMetrics.r2_score

        .. _accuracy:
            https://seqmetrics.readthedocs.io/en/latest/cls.html#SeqMetrics.ClassificationMetrics.accuracy
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
                monitor=monitor,
                val_metric=val_metric,
                cross_validator=cross_validator,
                accept_additional_args=accept_additional_args,
                seed=seed,
                wandb_config=wandb_config,
                **kwargs
            )

            reset_seed(maker.config['seed'], os=os, random=random, tf=tf, torch=torch)
            if tf is not None:
                # graph should be cleared everytime we build new `Model` otherwise,
                # if two `Models` are prepared in same
                # file, they may share same graph.
                tf.keras.backend.clear_session()

            self.data_config = maker.data_config

            self.opt_paras = maker.opt_paras
            self._original_model_config = maker.orig_model

            NN.__init__(self, config=maker.config)

            self.path = None
            if verbosity >= 0:
                self.path = maybe_create_path(path=path, prefix=prefix)

            self.config['path'] = self.path
            self.verbosity = verbosity
            self.category = self.config['category']

            self.mode = self.config.get('mode', None)

            if self.config['wandb_config']:
                self.use_wb = True
            else:
                self.use_wb = False

            self.info = {}

    @property
    def use_wb(self):
        return self._use_wb

    @use_wb.setter
    def use_wb(self, x):
        self._use_wb = x

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
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, _mode=None):
        from .experiments.utils import regression_models, classification_models

        if _mode:
            pass
        elif self.config['mode']:
            _mode =  self.config['mode']
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
            elif self.config['loss'] in ['mse']:
                _mode = "regression"
            elif self.model_name == "layers":
                # todo
                _mode = "regression"
            else:
                raise NotImplementedError(f" Can't determine mode for {self.model_name}")

        elif self.config['loss'] in ['sparse_categorical_crossentropy',
                                     'categorical_crossentropy',
                                     'binary_crossentropy']:
            _mode = "classification"
        elif self.config['loss'] in ['mse']:
                _mode = "regression"
        elif self.model_name == "layers":
            # todo
            _mode = "regression"
        else:  # when model_name is None, mode should also be None.
            _mode = None
        # so that next time don't have to go through all these ifelse statements
        self.config['mode'] = _mode
        self.data_config['mode'] = _mode
        self._mode = _mode

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
            if isinstance(self.dh_, DataSet):
                return self.dh_.ts_args['forecast_len']
            else:
                return {k: v['forecast_len'] for k, v in self.dh_.ts_args.items()}
        return self.config['ts_args']['forecast_len']

    @property
    def val_metric(self):
        if self.mode=='regression':
            return self.config['val_metric'] or 'r2_score'
        return self.config['val_metric'] or 'accuracy'

    @property
    def forecast_step(self):
        if hasattr(self, 'dh_'):
            return self.dh_.ts_args['forecast_step']
        return self.config['ts_args']['forecast_step']

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

    @property
    def output_shape_(self)->tuple:
        # for Keras, this calls output_shape attribute of Model class
        # for ML models, this method should calculate output shape
        raise NotImplementedError

    def reset_global_seed(self, seed:int = None)->None:
        """resets seeds of numpy, os, random, tensorflow, torch.
        If any of these module is not available, the seed for that module
        is not set. Also puts the seed in config."""
        if seed is None:
            seed = seed or self.config['seed'] or 313

        self.config['seed'] = seed
        self.data_config['seed'] = seed
        reset_seed(seed=seed, os=os, np=np, tf=tf, torch=torch, random=random)
        return

    def seed_everything(self, seed = None)->None:
        """resets seeds of numpy, os, random, tensorflow, torch.
        If any of these module is not available, the seed for that module
        is not set."""
        warnings.warn("Use `reset_global_seed` instead", DeprecationWarning)
        if seed is None:
            seed = seed or self.config['seed'] or 313
        reset_seed(seed=seed, os=os, np=np, tf=tf, torch=torch, random=random)
        return

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

    def get_callbacks(self, val_data=None, callbacks=None):

        if self.config['backend'] == 'pytorch':
            return self.cbs_for_pytorch(val_data, callbacks)
        else:
            return self.cbs_for_tf(val_data, callbacks)

    def cbs_for_pytorch(self, *args, **kwargs):
        """Callbacks for pytorch training."""
        return []

    def cbs_for_tf(self, val_data=None, callbacks=None):

        if callbacks is None:
            callbacks = {}
        # container to hold all callbacks
        _callbacks = list()

        _monitor = 'val_loss' if val_data is not None else 'loss'
        fname = "{val_loss:.5f}.hdf5" if val_data is not None else "{loss:.5f}.hdf5"

        if self.config['save_model'] and self.verbosity>=0:
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

    def get_val_data(self, validation_data=None):
        """Finds out if there is validation_data"""
        user_defined = True
        if validation_data is None:
            # when validation data is not given in kwargs and validation_data method is overwritten
            try:
                validation_data = self.validation_data()
                user_defined = False
            # when x,y is user defined then validation_data() can give this error
            except DataNotFound:
                validation_data = None

        if isinstance(validation_data, tuple):
            x, y = validation_data

            if x is None and y is None:
                return None
            elif hasattr(x, '__len__') and len(x)==0:
                return None
            else:  # x,y is numpy array
                if not user_defined and self.is_binary_:
                    if y.shape[1] > self.output_shape_[1]:
                        y = np.argmax(y, 1).reshape(-1,1)

                x = self._transform_x(x)
                y = self._transform_y(y)
                validation_data = x,y

        elif validation_data is not None:
            if self.config['backend'] == "tensorflow":
                if isinstance(validation_data, tf.data.Dataset):
                    pass
            elif validation_data.__class__.__name__ in ['TorchDataset', 'BatchDataset']:
                pass
            else:
                raise ValueError(f'Unrecognizable validattion data {validation_data.__class__.__name__}')
            return validation_data

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

    def _fit(self,
             inputs,
             outputs,
             validation_data=None,
             validation_steps=None,
             callbacks=None,
             **kwargs):

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

        outputs = get_values(outputs)

        if isinstance(validation_data, tuple):
            # when val_outs is just a dictionary with 1 key/value pair,
            # we just extract values and consider it validation_data
            val_outs = validation_data[-1]
            val_outs = get_values(val_outs)
            validation_data = (validation_data[0], val_outs)

        callbacks = self.get_callbacks(validation_data, callbacks=callbacks)

        st = time.time()

        # when data is given as generator (tf.data or torchDataset) then
        # we don't set batch size and don't given y argument to fit
        batch_size = self.config['batch_size']
        y = outputs
        if K.BACKEND == "tensorflow":
            if isinstance(inputs, tf.data.Dataset):
                batch_size = None
                y = None
        elif inputs.__class__.__name__ in ["TorchDataset"]:
            batch_size = None
            y = None

        # natively prepared arguments
        _kwargs = {
            'x':inputs,
            'y':y,
            'epochs':self.config['epochs'],
            'batch_size':batch_size,
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

        out = self._call_fit_fn(
            **_kwargs,
            **kwargs,
        )

        self.info['training_time_in_minutes'] = round(float(time.time() - st) / 60.0, 2)

        return out

    def get_wandb_cb(self, callbacks, *args, **kwargs) -> dict:
        """Makes WandbCallback and add it in callback"""
        if self.use_wb:
            from ._wb import WB
            self.wb_ = WB(self, self.config['wandb_config'])
            return self.wb_.callbacks_tf(callbacks, *args, **kwargs)
        return callbacks

    def wb_finish(self):
        """does some stuff related to wandb at the end of training."""
        self.wb_.finish()

        return

    def post_fit(self):
        """Does some stuff after Keras model.fit has been called"""

        if self.category == "DL":
            if K.BACKEND == 'pytorch':
                history = getattr(self, "torch_learner").history

            elif hasattr(self, 'history'):
                history = self.history
            else:
                history = self._model.history

            if self.verbosity >= 0:
                visualizer = LossCurve(path=self.path,
                                       show=bool(self.verbosity),
                                       save=bool(self.verbosity))

                # save all the losses or performance metrics
                df = pd.DataFrame.from_dict(history.history)
                df.to_csv(os.path.join(self.path, "losses.csv"))
                visualizer.plot_loss(history.history)

                # for -ve verbosity, weights are not saved! # todo should raise warning
                self.load_best_weights()
        else:
            class History: history = None
            history = History()

        if self.verbosity >= 0:
            self.save_config(history.history)
            dict_to_file(os.path.join(self.path, 'info.json'), others=self.info)

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
            elif self.config['train_fraction'] < 1.0:
                kwargs.update({'validation_fraction': 1.0 - self.config['train_fraction']})

        # some algorithms allow detailed output during training, this is allowed when self.verbosity is > 1
        if estimator in ['OneClassSVM']:
            kwargs.update({'verbose': True if self.verbosity > 1 else False})

        if estimator in ["CatBoostRegressor", "CatBoostClassifier"]:
            # https://stackoverflow.com/a/52921608/5982232
            if not any([arg in kwargs for arg in ['verbose', 'silent', 'logging_level']]):
                if self.verbosity <= 0:
                    kwargs['logging_level'] = 'Silent'
                elif self.verbosity == 1:
                    kwargs['logging_level'] = 'Verbose'
                else:
                    kwargs['logging_level'] = 'Info'
            if 'random_seed' not in kwargs:
                kwargs['random_seed'] = self.config['seed']

        if estimator in ["XGBRegressor", "XGBClassifier"]:
            if 'random_state' not in kwargs:
                kwargs['random_state'] = self.config['seed']
            kwargs['verbosity'] = max(self.verbosity, 0)  # xgboost does not accept -1

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
            if kwargs.get('boosting_type', None) == "rf" and 'bagging_freq' not in kwargs:
                # https://github.com/microsoft/LightGBM/issues/1333
                # todo, user must be notified
                kwargs['bagging_freq'] = 1
                kwargs['bagging_fraction'] = 0.5
            kwargs['verbosity'] = self.verbosity  # https://lightgbm.readthedocs.io/en/latest/Parameters.html#verbosity

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
            if estimator in ['TweedieRegressor', 'PoissonRegressor', 'GammaRegressor']:
                sk_maj_ver = int(sklearn.__version__.split('.')[0])
                sk_min_ver = int(sklearn.__version__.split('.')[1])

                if sk_maj_ver < 1 and sk_min_ver < 23:
                    raise ValueError(
                        f"{estimator} is available with sklearn version >= 0.23 but you have {version_info['sklearn']}")
            raise ValueError(f"model '{estimator}' not found. {version_info}")

        self._model = model

        return

    def fit(
            self,
            x=None,
            y=None,
            data: Union[np.ndarray, pd.DataFrame, "DataSet", str] = 'training',
            callbacks: Union[list, dict] = None,
            **kwargs
    ):
        """
        Trains the model with data. The data is either ``x`` or it is taken from
        ``data`` by feeding it to DataSet.

        Arguments:
            x:
                The input data consisting of input features. It can also be
                tf.Dataset or TorchDataset.
            y:
                Correct labels/observations/true data corresponding to 'x'.
            data :
                Raw data fromw which ``x``,``y`` pairs are prepared. This will be
                passed to :py:class:`ai4water.preprocessing.DataSet`.
                It can also be an instance if :py:class:`ai4water.preprocessing.DataSet` or
                :py:class:`ai4water.preprocessing.DataSetPipeline`.
                It can also be name of dataset from :py:attr:`ai4water.datasets.all_datasets`
            callbacks:
                Any callback compatible with keras. If you want to log the output
                to tensorboard, then just use `callbacks={'tensorboard':{}}` or
                to provide additional arguments

                >>> {'tensorboard': {'histogram_freq': 1}}

            kwargs :
                Any keyword argument for the `fit` method of the underlying library.
                if 'x' is present in kwargs, that will take precedent over `data`.
        Returns:
            A keras history object in case of deep learning model with tensorflow
            as backend or anything returned by `fit` method of underlying model.

        Examples
        --------
            >>> from ai4water import Model
            >>> from ai4water.datasets import busan_beach
            >>> model = Model(model="XGBRegressor")
            >>> model.fit(data=busan_beach())

            using your own data for training

            >>> import numpy
            >>> new_inputs = numpy.random.random((100, 10))
            >>> new_outputs = numpy.random.random(100)
            >>> model.fit(x=new_inputs, y=new_outputs)
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

        self.is_training = True

        source = 'training'
        if isinstance(data, str) and data in ['validation', 'test']:
            source = data

        inputs, outputs, _, _, user_defined_x = self._fetch_data(source, x, y, data)

        if 'Dataset' in inputs.__class__.__name__:
            pass
        else:
            AttribtueSetter(self, outputs)

        num_examples = _find_num_examples(inputs)
        if num_examples:  # for tf.data, we can't find num_examples
            self._maybe_reduce_nquantiles(num_examples)

        # apply preprocessing/feature engineering if required.
        inputs = self._fit_transform_x(inputs)
        outputs = self._fit_transform_y(outputs)

        outputs = self._verify_output_shape(outputs)

        self.info['training_start'] = dateandtime_now()

        validation_data = self.get_val_data(kwargs.pop('validation_data', None))

        callbacks = self.get_wandb_cb(
            callbacks,
            train_data=(inputs, outputs),
            validation_data=validation_data,
        )

        if self.category == "DL":

            history = self._fit(
                inputs,
                outputs,
                callbacks=callbacks,
                validation_data=validation_data,
                **kwargs)
        else:
            history = self.fit_ml_models(inputs, outputs, **kwargs)

        self.info['training_end'] = dateandtime_now()

        self.post_fit()

        self.is_training = False
        return history

    def _verify_output_shape(self, outputs):
        """verifies that shape of target/true/labels is correct"""
        if isinstance(outputs, np.ndarray) and self.category == "DL":

            if isinstance(self.ai4w_outputs, list):
                assert len(self.ai4w_outputs) == 1
                model_output_shape = tuple(self.ai4w_outputs[0].shape.as_list()[1:])

                if getattr(self, 'quantiles', None) is not None:
                    assert model_output_shape[0] == len(self.quantiles) * self.num_outs

                elif self.mode == 'classification':
                    num_classes = getattr(self, "num_clases_")
                    activation = self.layers[-1].get_config()['activation']
                    if getattr(self, "is_binary_"):
                        if activation == "softmax":
                            assert model_output_shape[0] == num_classes, f"""inferred number of classes are 
                                        {num_classes} while model's output has {model_output_shape[0]} nodes """
                        else:
                            if outputs.shape[1] > model_output_shape[0]:
                                outputs = np.argmax(outputs, 1).reshape(-1, 1)

                    assert model_output_shape[0] == outputs.shape[1]

                elif len(model_output_shape) != len(outputs.shape):
                    assert model_output_shape == outputs.shape[1:], f"""
        ShapeMismatchError: Shape of model's output is {model_output_shape}
        while the prepared targets have shape {outputs.shape[1:]}."""
        return outputs

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

    def ml_callbacks(self, validation_data=None, **kwargs):
        """any callback created here needs to be put into kwargs which
        will then go to .fit as keyword arguments"""
        return kwargs

    def fit_ml_models(self, inputs, outputs, validation_data=None, **kwargs):
        # following arguments are strictly about nn so we don't need to save them in config file
        # so that it does not confuse the reader.
        for arg in ["composite", "optimizer", "lr", "epochs", "learning_rate"]:
            self.config.pop(arg, None)

        if isinstance(outputs, np.ndarray) and len(outputs) == outputs.size:
            outputs = outputs.reshape(-1, )

        self._maybe_change_residual_threshold(outputs)

        kwargs = self.ml_callbacks(validation_data=validation_data, **kwargs)

        out = self._model.fit(inputs, outputs, **kwargs)

        if self._model.__class__.__name__.startswith("XGB") and inputs.__class__.__name__ == "ndarray":
            # by default feature_names of booster as set to f0, f1,...
            self._model.get_booster().feature_names = self.input_features

        if self.verbosity >= 0:
            self._save_ml_model()

        return out

    def _save_ml_model(self):
        """Saves the non-NN/ML models in the disk."""
        import joblib  # some modules don't have joblibe in their requires

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
            cv:str = None,
            cv_kws:dict = None,
            scoring: Union [str, List[str], Callable] = None,
            refit: bool = False,
            process_results:bool = False
    ) -> list:
        """
        computes cross validation score

        Parameters
        ----------
            x :
                input data
            y :
                output corresponding to ``x``.
            data :
                raw unprepared data which will be given to :py:class:`ai4water.preprocessing.DataSet`
                to prepare x,y from it.
            cv : str
                It can be any valid cross validation name from sklearn library such as
                    - KFold
                    - GroupKFold
                if None, ``cross_validator`` argument will be used to define it
            cv_kws :
                keyword arguments that will be given to initialize cv. For example if ``cv`` is ``KFold``
                then cv_kws can be
                    >>> {'n_splits': 5}
            scoring : (default=None)
                performance metric to use for cross validation.
                If None, it will be taken from config['val_metric']
                If callable then it must be a function which can take true and predicted
                arrays and return a float.
            refit : bool, optional (default=False)
                If True, the model will be trained on the whole training+validation
                data after calculating cross validation score.
            process_results : bool, optional
                whether to process results at each cv iteration or not

        Returns
        -------
        list
            cross validation score for each of metric in scoring

        Example
        -------
        >>> from ai4water.datasets import busan_beach
        >>> from ai4water import Model
        >>> model = Model(model="RandomForestRegressor",
        >>>               cross_validator={"KFold": {"n_splits": 5}})
        >>> model.cross_val_score(data=busan_beach())

        We can also have our own performance metric as scoring

        >>> from ai4water.datasets import MtropicsLaos
        >>> df = MtropicsLaos().make_classification(lookback_steps=1)
        >>> def f1_score_(t,p)->float:
        >>>    return ClassificationMetrics(t, p).f1_score(average="macro")
        >>> model = Model(model="RandomForestClassifier",
        ... cross_validator={"KFold": {"n_splits": 5}},)
        >>> model.cross_val_score(data=df, scoring=f1_score_)

        Note
        ----
            Currently not working for deep learning models.

        """

        if self.mode == "classification":
            Metrics = ClassificationMetrics
        else:
            Metrics = RegressionMetrics

        if scoring is None:
            scoring = self.val_metric

        if not isinstance(scoring, list):
            scoring = [scoring]

        scores = []

        if self.config['cross_validator'] is None and cv is None:
            raise ValueError("Provide the `cross_validator` argument to the `Model` class upon initiation")

        cross_validator = cv or list(self.config['cross_validator'].keys())[0]
        cross_validator_args = cv_kws or self.config['cross_validator'][cross_validator]

        if data is None:  # prepared data is given
            from .utils.utils import TrainTestSplit
            splitter = TrainTestSplit(test_fraction=1.0 - self.config['train_fraction'])
            splits = getattr(splitter, cross_validator)(x, y, **cross_validator_args)

        else: # we need to prepare data first as x,y

            if callable(cross_validator):
                splits = cross_validator(**cross_validator_args)
            else:
                ds = DataSet(data=data, **self.data_config)
                splits = getattr(ds, f'{cross_validator}_splits')(**cross_validator_args)

        for fold, ((train_x, train_y), (test_x, test_y)) in enumerate(splits):

            verbosity = self.verbosity
            self.verbosity = min(-1, verbosity)
            # make a new classifier/regressor at every fold
            self.build(self._get_dummy_input_shape())

            self.verbosity = verbosity

            if self.category == "ML":
                self._maybe_change_residual_threshold(train_y)
                self.fit(x=train_x, y=train_y.reshape(-1, ))
            else:
                self.fit(x=train_x, y=train_y)
            # since we have access to true y, it is better to provide it
            # it will be used for processing of results
            pred = self.predict(x=test_x, y=test_y, process_results=process_results)

            metrics = Metrics(test_y.reshape(-1, 1), pred)

            val_scores = []
            for score in scoring:
                if callable(score):
                    val_scores.append(score(test_y.reshape(-1, 1), pred))
                else:
                    val_scores.append(getattr(metrics, score)())

            scores.append(val_scores)

            if self.verbosity > 0:
                print(f'fold: {fold} val_score: {val_scores}')

        if self.verbosity >= 0:
            # save all the scores as json in model path`
            cv_name = str(cross_validator)
            fname = os.path.join(self.path, f'{cv_name}_scores.json')
            with open(fname, 'w') as fp:
                json.dump(scores, fp, indent=True)

        ## set it as class attribute so that it can be used
        setattr(self, f'cross_val_scores', scores)

        if refit:
            self.fit_on_all_training_data(data=data)

        if self.verbosity >= 0:
            # Even if we do not run .fit(), then we should still have model saved in
            # the disk so that it can be used.
            self._save_ml_model()

        scores = np.array(scores)
        cv_scores_ = np.nanmean(scores, axis=0)

        max_val = max(cv_scores_)
        avg_val = np.nanmean(cv_scores_).item()
        if np.isinf(cv_scores_).any():
            # if there is inf, we should not fill it with very large value (999999999)
            # but it should be max(so far experienced) value
            cv_scores_no_inf = cv_scores_.copy()
            cv_scores_no_inf[np.isinf(cv_scores_no_inf)] = np.nan
            cv_scores_no_inf[np.isnan(cv_scores_no_inf)] = np.nanmax(cv_scores_no_inf)
            max_val = max(cv_scores_no_inf)

        # check for both infinity and nans separately
        # because they come due to different reasons
        cv_scores = []
        for cv_score, metric_name in zip(cv_scores_, scoring):
            #  math.isinf(np.nan) will be false therefore
            # first check if cv_score is nan or not, if true, fill it with avg_val
            if math.isnan(cv_score):
                cv_score = fill_val(metric_name, default_min=avg_val)
            # then check if cv_score is infinity because
            elif math.isinf(cv_score):
                cv_score = fill_val(metric_name, default_min=max_val)
            cv_scores.append(cv_score)

        return cv_scores

    def fit_on_all_training_data(self, x=None, y=None, data=None, **kwargs):
        """
        This function trains the model on training + validation data.

        Parameters
        ----------
            x :
                x data which is supposed to be consisting of training and validation.
                If not given, then ``data`` must be given.
            y :
                label/target data corresponding to x data.
            data :
                raw data which will be passed to `py:meth:ai4water.preprocessing.DataSet`
                 to get training and validation x,y pairs.
                The x data from training and validation is concatenated.
                Similarly, y data from training and validation is concatenated
            **kwargs
                any keyword arguments for ``fit`` method.

        """
        if data is None:
            assert x is not None, f"""
            if data is not given, both x, y pairs must be given.
            The provided x,y pairs are of type {type(x)}, {type(y)} respectively."""
            return self.fit(x=x, y=y, **kwargs)

        x_train, y_train = self.training_data(data=data)
        x_val, y_val = self.validation_data()

        if isinstance(x_train, list):
            x = []
            for val in range(len(x_train)):
                if x_val is not None:
                    _val = np.concatenate([x_train[val], x_val[val]])
                    x.append(_val)
                else:
                    _val = x_train[val]

            y = y_train
            if hasattr(y_val, '__len__') and len(y_val) > 0:
                y = np.concatenate([y_train, y_val])

        elif isinstance(x_train, np.ndarray):
            x, y = x_train, y_train
            # if not validation data is available then use only training data
            if x_val is not None:
                if hasattr(x_val, '__len__') and len(x_val)>0:
                    x = np.concatenate([x_train, x_val])
                    y = np.concatenate([y_train, y_val])
        else:
            raise NotImplementedError

        if 'Dataset' in x.__class__.__name__:
            pass
        else:
            AttribtueSetter(self, y)

        if getattr(self, "is_binary_"):
            if len(y) != y.size: # when sigmoid is used for binary
                # convert the output to 1d
                y = np.argmax(y, 1).reshape(-1, 1)

        return self.fit(x=x, y=y, **kwargs)

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
        """since preprocessing is part of Model, so the trained model with
        sklearn as backend must also be able to apply preprocessing on inputs
        before calculating score from sklearn. Currently it just calls the
        `score` function of sklearn by first transforming x and y."""
        if self.category == "ML" and hasattr(self, '_model'):
            x,y, _, _, _ = self._fetch_data(data, x=x,y=y, data=data)
            x = self._transform_x(x)
            y = self._transform_y(y)
            return self._model.score(x, y, **kwargs)
        raise NotImplementedError(f"can not calculate score")

    def predict_proba(self, x=None,  data='test', **kwargs):
        """since preprocessing is part of Model, so the trained model with
        sklearn/xgboost/catboost/lgbm as backend must also be able to apply
        preprocessing on inputs before calling predict_proba from underlying library.
        Currently it just calls the `predict_proba` function of underlying library
        by first transforming x
        """
        if self.category == "ML" and hasattr(self, '_model'):
            x, _, _, _, _ = self._fetch_data(data, x=x, data=data)
            x = self._transform_x(x)

            if self.model_name == "XGBClassifier":
                # https://datascience.stackexchange.com/q/63872
                try:
                    out = self._model.predict_proba(x, **kwargs)
                except ValueError as e:
                    if isinstance(x, np.ndarray) and self.input_features is not None:
                        out = self._model.predict_proba(
                            pd.DataFrame(x, columns=self.input_features))
                    else:
                        raise e
                return out

            return self._model.predict_proba(x,  **kwargs)
        raise NotImplementedError(f"can not calculate proba")

    def predict_log_proba(self, x=None,  data='test', **kwargs):
        """since preprocessing is part of Model, so the trained model with
        sklearn/xgboost/catboost/lgbm as backend must also be able to apply
        preprocessing on inputs before calling predict_log_proba from underlying library.
        Currently it just calls the `log_proba` function of underlying library
        by first transforming x
        """
        if self.category == "ML" and hasattr(self, '_model'):
            x, _, _, _, _ = self._fetch_data(data, x=x, data=data)
            x = self._transform_x(x)
            return self._model.predict_log_proba(x, **kwargs)
        raise NotImplementedError(f"can not calculate log_proba")

    def evaluate(
            self,
            x=None,
            y=None,
            data=None,
            metrics=None,
            **kwargs
    ):
        """
        Evaluates the performance of the model on a given data.
        calls the ``evaluate`` method of underlying `model`. If the `evaluate`
        method is not available in underlying `model`, then `predict` is called.

        Arguments:
            x:
                inputs
            y:
                outputs/true data corresponding to `x`
            data:
                Raw unprepared data which will be fed to :py:class:`ai4water.preprocessing.DataSet`
                to prepare x and y. If ``x`` and ``y`` are given, this argument will have no meaning.
            metrics:
                the metrics to evaluate. It can a string indicating the metric to
                evaluate. It can also be a list of metrics to evaluate. Any metric
                name from RegressionMetrics_ or ClassificationMetrics_ can be given.
                It can also be name of group of metrics to evaluate.
                Following groups are available

                    - ``minimal``
                    - ``all``
                    - ``hydro_metrics``

                If this argument is given, the `evaluate` function of the underlying class
                is not called. Rather the model is evaluated manually for given metrics.
                Otherwise, if this argument is not given, then evaluate method of underlying
                model is called, if available.
            kwargs:
                any keyword argument for the `evaluate` method of the underlying
                model.
        Returns:
            If `metrics` is not given then this method returns whatever is returned
            by `evaluate` method of underlying model. Otherwise the model is evaluated
            for given metric or group of metrics and the result is returned

        Examples
        --------
            >>> import numpy
            >>> from ai4water import Model
            >>> from ai4water.models import MLP
            >>> from ai4water.datasets import busan_beach
            >>> df = busan_beach()
            >>> model = Model(model=MLP(),
            ...               input_features=data.columns.tolist()[0:-1],
            ...               output_features=data.columns.tolist()[-1:])
            >>> model.fit(data=df)

            for evaluation on test data

            >>> model.evaluate(data=df)
            ...

            evaluate on any metric from SeqMetrics_ library

            >>> model.evaluate(data=df, metrics='pbias')
            ...
            ... # to evaluate on custom data, the user can provide its own x and y
            >>> new_inputs = numpy.random.random((10, 13))
            >>> new_outputs = numpy.random.random((10, 1, 1))
            >>> model.evaluate(new_inputs, new_outputs)

            backward compatability
            Since the ai4water's Model is supposed to behave same as Keras' Model
            the following expressions are equally valid.

            >>> model.evaluate(x, y=y)
            >>> model.evaluate(x=x, y=y)

        .. _SeqMetrics:
            https://seqmetrics.readthedocs.io/en/latest/index.html

        .. _RegressionMetrics:
            https://seqmetrics.readthedocs.io/en/latest/rgr.html#regressionmetrics

        .. _ClassificationMetrics:
            https://seqmetrics.readthedocs.io/en/latest/cls.html#classificationmetrics
        """
        return self.call_evaluate(x=x, y=y, data=data, metrics=metrics, **kwargs)

    def evaluate_on_training_data(self, data, metrics=None, **kwargs):
        """evaluates the model on training data.

        Parameters
        ----------
            data:
                Raw unprepared data which will be fed to :py:class:`ai4water.preprocessing.DataSet`
                to prepare x and y. If ``x`` and ``y`` are given, this argument will have no meaning.
            metrics:
                the metrics to evaluate. It can a string indicating the metric to
                evaluate. It can also be a list of metrics to evaluate. Any metric
                name from RegressionMetrics_ or ClassificationMetrics_ can be given.
                It can also be name of group of metrics to evaluate.
                Following groups are available

                    - ``minimal``
                    - ``all``
                    - ``hydro_metrics``

                If this argument is given, the `evaluate` function of the underlying class
                is not called. Rather the model is evaluated manually for given metrics.
                Otherwise, if this argument is not given, then evaluate method of underlying
                model is called, if available.
            kwargs:
                any keyword argument for the `evaluate` method of the underlying
                model.

        Returns
        -------
            If `metrics` is not given then this method returns whatever is returned
            by `evaluate` method of underlying model. Otherwise the model is evaluated
            for given metric or group of metrics and the result is returned as float
            or dictionary

        Examples
        --------
            >>> from ai4water import Model
            >>> from ai4water.models import MLP
            >>> from ai4water.datasets import busan_beach
            >>> df = busan_beach()
            >>> model = Model(model=MLP(),
            ...               input_features=data.columns.tolist()[0:-1],
            ...               output_features=data.columns.tolist()[-1:])
            >>> model.fit(data=df)
            ... # for evaluation on training data
            >>> model.evaluate_on_training_data(data=df)
            >>> model.evaluate(data=df, metrics='pbias')
        """
        x, y = self.training_data(data=data)
        return self.call_evaluate(x=x, y=y, metrics=metrics, **kwargs)

    def evaluate_on_validation_data(self, data, metrics=None, **kwargs):
        """evaluates the model on validation data.

        Parameters
        ----------
            data:
                Raw unprepared data which will be fed to :py:class:`ai4water.preprocessing.DataSet`
                to prepare x and y. If ``x`` and ``y`` are given, this argument will have no meaning.
            metrics:
                the metrics to evaluate. It can a string indicating the metric to
                evaluate. It can also be a list of metrics to evaluate. Any metric
                name from RegressionMetrics_ or ClassificationMetrics_ can be given.
                It can also be name of group of metrics to evaluate.
                Following groups are available

                    - ``minimal``
                    - ``all``
                    - ``hydro_metrics``

                If this argument is given, the `evaluate` function of the underlying class
                is not called. Rather the model is evaluated manually for given metrics.
                Otherwise, if this argument is not given, then evaluate method of underlying
                model is called, if available.
            kwargs:
                any keyword argument for the `evaluate` method of the underlying
                model.

        Returns
        -------
            If `metrics` is not given then this method returns whatever is returned
            by `evaluate` method of underlying model. Otherwise the model is evaluated
            for given metric or group of metrics and the result is returned as float
            or dictionary

        Examples
        --------
            >>> from ai4water import Model
            >>> from ai4water.models import MLP
            >>> from ai4water.datasets import busan_beach
            >>> df = busan_beach()
            >>> model = Model(model=MLP(),
            ...               input_features=data.columns.tolist()[0:-1],
            ...               output_features=data.columns.tolist()[-1:])
            >>> model.fit(data=df)
            ... # for evaluation on validation data
            >>> model.evaluate_on_validation_data(data=df)
            >>> model.evaluate_on_validation_data(data=df, metrics='pbias')
        """

        x, y = self.validation_data(data=data)

        if not _find_num_examples(x):
            raise DataNotFound("Validation")

        return self.call_evaluate(x=x, y=y, metrics=metrics, **kwargs)

    def evaluate_on_test_data(self, data, metrics=None, **kwargs):
        """evaluates the model on test data.

        Parameters
        ----------
            data:
                Raw unprepared data which will be fed to :py:class:`ai4water.preprocessing.DataSet`
                to prepare x and y. If ``x`` and ``y`` are given, this argument will have no meaning.
            metrics:
                the metrics to evaluate. It can a string indicating the metric to
                evaluate. It can also be a list of metrics to evaluate. Any metric
                name from RegressionMetrics_ or ClassificationMetrics_ can be given.
                It can also be name of group of metrics to evaluate.
                Following groups are available

                    - ``minimal``
                    - ``all``
                    - ``hydro_metrics``

                If this argument is given, the `evaluate` function of the underlying class
                is not called. Rather the model is evaluated manually for given metrics.
                Otherwise, if this argument is not given, then evaluate method of underlying
                model is called, if available.
            kwargs:
                any keyword argument for the `evaluate` method of the underlying
                model.

        Returns
        -------
            If `metrics` is not given then this method returns whatever is returned
            by `evaluate` method of underlying model. Otherwise the model is evaluated
            for given metric or group of metrics and the result is returned as float
            or dictionary

        Examples
        --------
            >>> from ai4water import Model
            >>> from ai4water.models import MLP
            >>> from ai4water.datasets import busan_beach
            >>> df = busan_beach()
            >>> model = Model(model=MLP(),
            ...               input_features=data.columns.tolist()[0:-1],
            ...               output_features=data.columns.tolist()[-1:])
            >>> model.fit(data=df)
            ... # for evaluation on test data
            >>> model.evaluate_on_test_data(data=df)
            >>> model.evaluate_on_test_data(data=df, metrics='pbias')
        """
        x, y = self.test_data(data=data)
        if not _find_num_examples(x):
            raise DataNotFound("Test")
        return self.call_evaluate(x=x, y=y, metrics=metrics, **kwargs)

    def evaluate_on_all_data(self, data, metrics=None, **kwargs):
        """evaluates the model on all i.e. training+validation+test data.

        Examples
        --------
            >>> from ai4water import Model
            >>> from ai4water.models import MLP
            >>> from ai4water.datasets import busan_beach
            >>> df = busan_beach()
            >>> model = Model(model=MLP(),
            ...               input_features=data.columns.tolist()[0:-1],
            ...               output_features=data.columns.tolist()[-1:])
            >>> model.fit(data=df)
            ... # for evaluation on all data
            >>> print(model.evaluate_on_all_data(data=df)))
            >>> print(model.evaluate_on_all_data(data=df, metrics='pbias'))
        """
        x, y = self.all_data(data=data)
        return self.call_evaluate(x=x, y=y, metrics=metrics, **kwargs)

    def call_evaluate(self, x=None,y=None, data=None, metrics=None, **kwargs):

        if x is None and data is None:
            data = "test"

        source = 'test'
        if isinstance(data, str) and data in ['training', 'validation', 'test']:
            source = data
            warnings.warn(f"""
            argument {data} is deprecated and will be removed in future. Please 
            use 'evaluate_on_{data}_data' method instead.""")

        x, y, _, _, user_defined = self._fetch_data(source, x, y, data)

        if user_defined:
            pass
        elif len(x) == 0 and source == "test":
            warnings.warn("No test data found. using validation data instead",
                          UserWarning)
            data = "validation"
            source = data
            x, y, _, _, _ = self._fetch_data(source=source, data=data)

            if len(x) == 0:
                warnings.warn("No test and validation data found. using training data instead",
                              UserWarning)
                data = "training"
                source = data
                x, y, _, _, _ = self._fetch_data(source=source, x=x, y=y, data=data)

        if not getattr(self, 'is_fitted', True):
            AttribtueSetter(self, y)

        # dont' make call to underlying evaluate function rather manually
        # evaluate the given metrics
        if metrics is not None:
            return self._manual_eval(x, y, metrics, **kwargs)

        # after this we call evaluate function of underlying model
        # therefore we must transform inputs and outptus
        x = self._transform_x(x)
        y = self._transform_y(y)

        if hasattr(self._model, 'evaluate'):
            return self._model.evaluate(x, y, **kwargs)

        return self.evaluate_fn(x, y, **kwargs)

    def evalute_ml_models(self, x, y, metrics=None):
        if metrics is None:
            metrics = self.val_metric
        return self._manual_eval(x, y, metrics)

    def _manual_eval(self, x, y, metrics, **kwargs):
        """manual evaluation"""
        t, p = self.predict(x=x, y=y, return_true=True, process_results=False, **kwargs)

        if self.mode == "regression":

            errs = RegressionMetrics(t, p)
        else:
            errs = ClassificationMetrics(t, p, multiclass=getattr(self, "is_multiclass_"))

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
            results = metrics(x, t)
        else:
            raise ValueError(f"unknown metrics type {metrics}")

        return results

    def predict(
            self,
            x=None,
            y=None,
            data: Union[str, pd.DataFrame, np.ndarray, DataSet] = 'test',
            process_results: bool = True,
            metrics: str = "minimal",
            return_true: bool = False,
            plots:Union[str, list] = None,
            **kwargs
    ):
        """
        Makes prediction from the trained model.

        Arguments:
            x:
                The data on which to make prediction. if given, it will override
                `data`. It can also be tf.Dataset or TorchDataset
            y:
                Used for pos-processing etc. if given it will overrite `data`
            data:
                It can also be unprepared/raw data which will be given to
                :py:class:`ai4water.preprocessing.DataSet`
                to prepare x,y values.

            process_results: bool
                post processing of results
            metrics: str
                only valid if process_results is True. The metrics to calculate.
                Valid values are ``minimal``, ``all``, ``hydro_metrics``
            return_true: bool
                whether to return the true values along with predicted values
                or not. Default is False, so that this method behaves sklearn type.
            plots : optional (default=None)
                The kind of of plots to draw. Only valid if post_process is True
            kwargs : any keyword argument for ``predict`` method.

        Returns:
            An numpy array of predicted values.
            If return_true is True then a tuple of arrays. The first is true
            and the second is predicted. If ``x`` is given but ``y`` is not given,
            then, first array which is returned is None.

        Examples
        --------
        >>> from ai4water import Model
        >>> from ai4water.datasets import busan_beach
        >>> model = Model(model="RandomForestRegressor")
        >>> model.fit(data=busan_beach())
        >>> pred = model.predict(data=busan_beach())

        get true values

        >>> true, pred = model.predict(data=busan_beach(), return_true=True)

        postprocessing of results

        >>> pred = model.predict(data=busan_beach(), process_results=True)

        calculate all metrics during postprocessing

        >>> pred = model.predict(data=busan_beach(), process_results=True, metrics="all")

        using your own data

        >>> import numpy
        >>> new_input = numpy.random.random((10, 13))
        >>> pred = model.predict(x = new_input)
        """

        assert metrics in ("minimal", "all", "hydro_metrics")

        return self.call_predict(
            x=x,
            y=y,
            data=data,
            process_results=process_results,
            metrics=metrics,
            return_true=return_true,
            plots=plots,
            **kwargs)

    def predict_on_training_data(
            self,
            data,
            process_results=True,
            return_true=False,
            metrics="minimal",
            plots: Union[str, list] = None,
            **kwargs
    ):
        """makes prediction on training data.

        Parameters
        ----------
            data :
                raw, unprepared data from which training data (x,y paris) will be generated.
            process_results : bool, optional
                whether to post-process the results or not
            return_true : bool, optional
                If true, the returned value will be tuple, first is true and second is predicted array
            metrics : str, optional
                the metrics to calculate during post-processing
            plots : optional (default=None)
                The kind of of plots to draw. Only valid if post_process is True
                Following plots are avialble.

                    ``residual``
                    ``regression``
                    ``prediction``
                    ``errors``
                    ``fdc``
                    ``murphy``
                    ``edf``
            **kwargs :
                any keyword argument for .predict method.
        """

        x, y = self.training_data(data=data)

        return self.call_predict(
            x=x,
            y=y,
            process_results=process_results,
            return_true=return_true,
            metrics=metrics,
            plots=plots,
            prefix="training",
            **kwargs
        )

    def predict_on_validation_data(
            self,
            data,
            process_results=True,
            return_true=False,
            metrics="minimal",
            plots: Union[str, list] = None,
            **kwargs
    ):
        """makes prediction on validation data.

        Parameters
        ----------
            data :
                raw, unprepared data from which validation data (x,y paris) will be generated.
            process_results : bool, optional
                whether to post-process the results or not
            return_true : bool, optional
                If true, the returned value will be tuple, first is true and second is predicted array
            metrics : str, optional
                the metrics to calculate during post-processing
            plots : optional (default=None)
                The kind of of plots to draw. Only valid if post_process is True
                Following plots are avialble.

                    ``residual``
                    ``regression``
                    ``prediction``
                    ``errors``
                    ``fdc``
                    ``murphy``
                    ``edf``
            **kwargs :
                any keyword argument for .predict method.
        """

        x, y = self.validation_data(data=data)

        if not _find_num_examples(x):
            raise DataNotFound("Validation")

        return self.call_predict(
            x=x,
            y=y,
            process_results=process_results,
            return_true=return_true,
            metrics=metrics,
            plots=plots,
            prefix="validation",
            **kwargs
        )

    def predict_on_test_data(
            self,
            data,
            process_results=True,
            return_true=False,
            metrics="minimal",
            plots: Union[str, list] = None,
            **kwargs
    ):
        """makes prediction on test data.

        Parameters
        ----------
            data :
                raw, unprepared data from which test data (x,y paris) will be generated.
            process_results : bool, optional
                whether to post-process the results or not
            return_true : bool, optional
                If true, the returned value will be tuple, first is true and second is predicted array
            metrics : str, optional
                the metrics to calculate during post-processing
            plots : optional (default=None)
                The kind of of plots to draw. Only valid if post_process is True
                Following plots are avialble.

                    ``residual``
                    ``regression``
                    ``prediction``
                    ``errors``
                    ``fdc``
                    ``murphy``
                    ``edf``
            **kwargs :
                any keyword argument for .predict method.
        """

        if isinstance(data, _DataSet):
            x, y = data.test_data()
        else:
            x, y = self.test_data(data=data)

        if not _find_num_examples(x):
            raise DataNotFound(f"test")

        return self.call_predict(
            x=x,
            y=y,
            process_results=process_results,
            return_true=return_true,
            metrics=metrics,
            plots=plots,
            prefix="test",
            **kwargs
        )

    def predict_on_all_data(
            self,
            data,
            process_results=True,
            return_true=False,
            metrics="minimal",
            plots: Union[str, list] = None,
            **kwargs
    ):
        """
        It makes prediction on training+validation+test data.

        Parameters
        ----------
            data :
                raw, unprepared data from which x,y paris will be generated.
            process_results : bool, optional
                whether to post-process the results or not
            return_true : bool, optional
                If true, the returned value will be tuple, first is true and second is predicted array
            metrics : str, optional
                the metrics to calculate during post-processing
            plots : optional (default=None)
                The kind of of plots to draw. Only valid if post_process is True
                Following plots are avialble.

                    ``residual``
                    ``regression``
                    ``prediction``
                    ``errors``
                    ``fdc``
                    ``murphy``
                    ``edf``

            **kwargs :
                any keyword argument for .predict method.
        """

        x, y = self.all_data(data=data)

        return self.call_predict(
            x=x,
            y=y,
            process_results=process_results,
            return_true=return_true,
            metrics=metrics,
            plots=plots,
            prefix="all",
            **kwargs
        )

    def call_predict(
            self,
            x=None,
            y=None,
            data=None,
            process_results=True,
            metrics="minimal",
            return_true: bool = False,
            plots:Union[list, str]=None,
            prefix:str=None,
            log_on_wb:bool = False,
            **kwargs
    ):

        source = 'test'
        if x is None and data is None:
            data = "test"

        if isinstance(x, type(None)) and isinstance(data, str) in ['training', 'validation', 'test']:
            warnings.warn(f"""
            argument {data} is deprecated and will be removed in future. Please 
            use 'predict_on_{data}_data' method instead.""")
            source = data

        inputs, true_outputs, _prefix, transformation_key, user_defined_data = self._fetch_data(
            source=source,
            x=x,
            y=y,
            data=data
        )

        if user_defined_data:
            pass
        elif len(inputs) == 0 or (isinstance(inputs, list) and len(inputs[0]) == 0) and source == "test":
            warnings.warn("No test data found. using validation data instead",
                          UserWarning)
            data = "validation"
            source = data
            inputs, true_outputs, _prefix, transformation_key, user_defined_data = self._fetch_data(
                source=source,
                x=x,
                y=y,
                data=data)
            # if we still have no data, then we use training data instead
            if len(inputs)==0 or (isinstance(inputs, list) and len(inputs[0]) == 0):
                warnings.warn("""
                No test and validation data found. using training data instead""",
                              UserWarning)
                data = "training"
                source = data
                inputs, true_outputs, _prefix, transformation_key, user_defined_data = self._fetch_data(
                    source=source,
                    x=x,
                    y=y,
                    data=data)

        if not getattr(self, 'is_fitted', True):
            # prediction without fitting
            AttribtueSetter(self, y)

        prefix = prefix or _prefix

        inputs = self._transform_x(inputs)

        if true_outputs is not None:
            true_outputs = self._transform_y(true_outputs)

        if self.category == 'DL':
            # some arguments specifically for DL models
            if 'verbose' not in kwargs:
                kwargs['verbose'] = self.verbosity

            if 'batch_size' in kwargs:  # if given by user
                ... #self.config['batch_size'] = kwargs['batch_size']  # update config
            elif K.BACKEND == "tensorflow":
                if isinstance(inputs, tf.data.Dataset):
                    ...
            else:  # otherwise use from config
                kwargs['batch_size'] = self.config['batch_size']

            predicted = self.predict_fn(x=inputs,  **kwargs)

        else:
            if self._model.__class__.__name__.startswith("XGB") and isinstance(inputs, np.ndarray):
                # since we have changed feature_names of booster,
                kwargs['validate_features'] = False

            predicted = self.predict_ml_models(inputs, **kwargs)

        true_outputs, predicted = self._inverse_transform_y(
            true_outputs,
            predicted)

        if true_outputs is None:
            if return_true:
                return true_outputs, predicted
            return predicted

        if isinstance(true_outputs, dict):
            dt_index = np.arange(set([len(v) for v in true_outputs.values()]).pop())
        else:
            dt_index = np.arange(len(true_outputs))  # dummy/default index when data is user defined

        if not user_defined_data:
            dt_index = self.dh_.indexes[transformation_key]
            #true_outputs, dt_index = self.dh_.deindexify(true_outputs, key=transformation_key)

        if isinstance(true_outputs, np.ndarray) and true_outputs.dtype.name == 'object':
            true_outputs = true_outputs.astype(predicted.dtype)

        if true_outputs is None or self.verbosity<0:
            process_results = False

        if log_on_wb:
            self.wb_.log_predict(
                true_outputs, predicted,
                mode=self.mode,
                prefix=prefix or source,
                                 )

        if process_results:
            # initialize post-processes
            pp = ProcessPredictions(
                mode=self.mode,
                path=self.path,
                forecast_len=self.forecast_len,
                output_features=self.output_features,
                plots=plots,
                show=bool(self.verbosity),
            )
            pp(true_outputs, predicted, metrics, prefix, dt_index,  inputs, model=self)

        if return_true:
            return true_outputs, predicted
        return predicted

    def predict_ml_models(self, inputs, **kwargs):
        """So that it can be overwritten easily for ML models."""
        return self.predict_fn(inputs, **kwargs)

    def plot_model(self, nn_model, show=False, figsize=None, **kwargs) -> None:

        if int(tf.__version__.split('.')[1]) > 14 and 'dpi' not in kwargs:
            kwargs['dpi'] = 300

        if 'to_file' not in kwargs:
            kwargs['to_file'] = os.path.join(self.path, "model.png")

        try:
            keras.utils.plot_model(
                nn_model,
                show_shapes=True,
                **kwargs)
            drawn = True
        except (AssertionError, ImportError) as e:
            print(f"dot plot of model could not be plotted due to {e}")
            drawn = False

        if drawn and show:
            import matplotlib.image as mpimg
            from easy_mpl import imshow
            img = mpimg.imread(os.path.join(self.path, "model.png"))
            kwargs = {}
            if figsize:
                kwargs['figsize'] = figsize
            ax,_ = imshow(img, show=False, xticklabels=[], yticklabels=[], **kwargs)
            ax.axis('off')
            plt.tight_layout()
            plt.show()
        return

    def get_opt_args(self) -> dict:
        """get input arguments for an optimizer.

        It is being explicitly defined here so that it can be overwritten
        in sub-classes
        """
        kwargs = {'lr': self.config['lr']}

        if self.config['backend'] == 'tensorflow':
            if int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) >= 250:
                kwargs['learning_rate'] = kwargs.pop('lr')
            if tf.__version__ >= "2.10":
                kwargs.pop('lr', None)
                kwargs['learning_rate'] = self.config['lr']

        if self.config['backend'] == 'pytorch':
            # parameters from pytorch model
            kwargs.update({'params': getattr(self, "parameters")()})
        return kwargs

    def get_metrics(self) -> list:
        """Returns the performance metrics to be monitored."""
        _metrics = self.config['monitor']

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
            data = None,
            data_type: str = 'training',
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
            data :
                the data to use when making calls to model for activation calculation
                or for gradient calculation.
            data_type : str
                It can either ``training``, ``validation`` or ``test`` or ``all``.
            x:
                input, alternative to data. If given it will override `data` argument.
            y:
                target/observed/label, alternative to data. If given it will
                override `data` argument.
            examples_to_view :
                the examples to view.
            show:
                whether to show the plot or not!

        Returns:
            An isntance of Visualize :py:class:`ai4water.postprocessing.visualize.Visualize` class.
        """
        from ai4water.postprocessing.visualize import Visualize

        visualizer = Visualize(model=self, show=show)

        visualizer(layer_name,
                   data=data,
                   data_type=data_type,
                   x=x,
                   y=y,
                   examples_to_use=examples_to_view
                   )

        return visualizer

    def interpret(
            self,
            **kwargs
    ):
        """
        Interprets the underlying model. Call it after training.

        Returns:
            An instance of :py:class:`ai4water.postprocessing.interpret.Interpret` class

        Example:
            >>> from ai4water import Model
            >>> from ai4water.datasets import busan_beach
            >>> model = Model(model=...)
            >>> model.fit(data=busan_beach())
            >>> model.interpret()

        """
        # importing ealier will try to import np types as well again
        from ai4water.postprocessing import Interpret

        return Interpret(self)

    def explain(self, *args, **kwargs):
        """Calls the :py:func:ai4water.postprocessing.explain.explain_model` function
         to explain the model.
         Example
         -------
        >>> from ai4water import Model
        >>> from ai4water.datasets import busan_beach
        >>> data = busan_beach()
        >>> model = Model(model="RandomForestRegressor")
        >>> model.fit(data=data)
        >>> model.explain(total_data=data, examples_to_explain=2)
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
        # it is calculated during run time
        config['config']['val_metric'] = self.val_metric
        config['method'] = self.method

        # we don't want our saved config to have 'path' key in it
        if 'path' in config['config']:
            config['config'].pop('path')

        if self.category == "DL":
            config['loss'] = self.loss_name()

        if self.category == "ML" and self.is_custom_model:
            config['config']['model'] = self.model_name

        # following parameters are not set during build and are not in "config" which
        # builds the model because they are set during run time from data
        # putting them in metaconfig is necessary because when we build model
        # from config, and want to make some plots such as roc curve,
        # it will need classes_ attribute.
        for attr in ['classes_', 'num_classes_', 'is_binary_',
                     'is_multiclass_', 'is_multilabel_']:
            config[attr] = getattr(self, attr, None)

        dict_to_file(config=config, path=self.path)
        return config

    @classmethod
    def from_config(
            cls,
            config: dict,
            make_new_path: bool = False,
            **kwargs
    )->"BaseModel":
        """Loads the model from config dictionary i.e. model.config

        Arguments
        ---------
            config: dict
                dictionary containing model's parameters i.e. model.config
            make_new_path : bool, optional
                whether to make new path or not?
            **kwargs:
                any additional keyword arguments to Model class.

        Returns
        -------
            an instalnce of :py:class:`ai4water.Model`

        Example
        -------
            >>> from ai4water import Model
            >>> import numpy
            >>> from ai4water.datasets import busan_beach
            >>> data = busan_beach()
            >>> old_model = Model(model="XGBRegressor")
            >>> old_model.fit(data=data)
            ... # now construct a new model instance from config dictionary
            >>> model = Model.from_config(old_model.config)
            >>> model.update_weights()
            >>> x = numpy.random.random((100, 14))
            >>> prediction = model.predict(x=x)
        """
        return cls._get_config_and_path(
            cls,
            config=config,
            make_new_path=make_new_path,
            **kwargs)

    @classmethod
    def from_config_file(
            cls,
            config_path: str,
            make_new_path: bool = False,
            **kwargs) -> "BaseModel":
        """
        Loads the model from a config file.

        Arguments
        ----------
            config_path :
                complete path of config file
            make_new_path : bool, optional
                If true, then it means we want to use the config
                file, only to build the model and a new path will be made. We
                would not normally update the weights in such a case.
            **kwargs :
                any additional keyword arguments for the :py:class:`ai4water.Model`

        Return
        ------
            an instance of :py:class:`ai4water.Model` class

        Example
        -------
            >>> from ai4water import Model
            >>> config_file_path = "../path/to/config.json"
            >>> model_ = Model.from_config_file(config_file_path)
            >>> x = np.random.random((100, 14))
            >>> prediction = model_.predict(x=x)
        """

        # when an instance of Model is created, config is written, which
        # will overwrite these attributes, so we need to keep track of them
        # so that after building the model, we can set these attributes to Model
        attrs = {}
        with open(config_path, 'r') as fp:
            config = json.load(fp)
        for attr in ['classes_', 'num_classes_', 'is_binary_', 'is_multiclass_', 'is_multilabel_']:
            if attr in config:
                if attr == 'classes_':
                    # because classes_ attribute should be np array instead of list
                    attrs[attr] = np.array(config[attr])
                else:
                    attrs[attr] = config[attr]

        model = cls._get_config_and_path(
            cls,
            config_path=config_path,
            make_new_path=make_new_path,
            **kwargs
        )

        for attr in ['classes_', 'num_classes_', 'is_binary_', 'is_multiclass_', 'is_multilabel_']:
            if attr in attrs:
                setattr(model, attr, attrs[attr])
         # now we need to save the config again
        model.save_config()

        return model

    @staticmethod
    def _get_config_and_path(
            cls,
            config_path: str = None,
            config=None,
            make_new_path=False,
            **kwargs
    )->"BaseModel":
        """Sets some attributes of the cls so that it can be built from config.

        Also fetches config and path which are used to initiate cls."""
        if config is not None and config_path is not None:
            raise ValueError

        if config is None:
            assert config_path is not None
            with open(config_path, 'r') as fp:
                meta_config = json.load(fp)
                config = meta_config['config']
                path = os.path.dirname(config_path)
        else:
            assert isinstance(config, dict), f"""
                config must be dictionary but it is of type {config.__class__.__name__}"""
            path = config['path']

        # todo
        # shouldn't we remove 'path' from Model's init? we just need prefix
        # path is needed in clsas methods only?
        if 'path' in config:
            config.pop('path')

        if make_new_path:
            allow_weight_loading = False
            path = None
        else:
            allow_weight_loading = True

        model = cls(**config, path=path, **kwargs)

        model.allow_weight_loading = allow_weight_loading
        model.from_check_point = True

        return model

    def update_weights(self, weight_file: str = None):
        """
        Updates the weights of the underlying model.

        Parameters
        ----------
            weight_file : str, optional
                complete path of weight file. If not given, the
                weights are updated from model.w_path directory. For neural
                network based models, the best weights are updated if more
                than one weight file is present in model.w_path.

        Returns
        -------
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
            import joblib # some modules don't have joblib in their requires
            self._model = joblib.load(weight_file_path)
        else:
            # loads the weights of keras model from weight file `w_file`.
            if self.api == 'functional' and self.config['backend'] == 'tensorflow':
                self._model.load_weights(weight_file_path)
            elif self.config['backend'] == 'pytorch':
                getattr(self, "load_state_dict")(torch.load(weight_file_path))
            else:
                getattr(self, "load_weights")(weight_file_path)

        if self.verbosity > 0:
            print("{} Successfully loaded weights from {} file {}".format('*' * 10, weight_file, '*' * 10))
        return

    def eda(self, data, *args, **kwargs):
        """Performs comprehensive Exploratory Data Analysis.

        Parameters
        ----------
            data :

        Returns
        -------
            an instance of EDA :py:class:`ai4water.eda.EDA` class
        """
        # importing EDA earlier will import numpy etc as well
        from ai4water.eda import EDA

        eda = EDA(data=data,
                  path=self.path,
                  in_cols=self.input_features,
                  out_cols=self.output_features,
                  save=True,
                  *args, **kwargs)

        eda()

        return eda

    def update_info(self):

        self.info["version_info"] = get_version_info()
        return

    def print_info(self):

        if self.verbosity > 0:
            class_type = ''
            print(f"""
            building {self.category} model for {class_type} 
            {self.mode} problem using {self.model_name}""")
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
            refit: bool = True,
            **kwargs
    ):
        """
        optimizes the hyperparameters of the built model

        The parameaters that needs to be optimized, must be given as space.

        Arguments:
            data :
                It can be one of following

                    - raw unprepared data in the form of a numpy array or pandas dataframe
                    - a tuple of x,y pairs
                If it is unprepared data, it is passed to :py:class:`ai4water.preprocessing.DataSet`.
                which prepares x,y pairs from it. The ``DataSet`` class also
                splits the data into training, validation and tests sets. If it
                is a tuple of x,y pairs, it is split into training and validation.
                In both cases, the loss on validation set is used as objective function.
                The loss calculated using ``val_metric``.
            algorithm: str, optional (default="bayes")
                the algorithm to use for optimization
            num_iterations: int, optional (default=14)
                number of iterations for optimization.
            process_results: bool, optional (default=True)
                whether to perform postprocessing of optimization results or not
            refit: bool, optional (default=True)
                whether to retrain the model using both training and validation data

        Returns:
            an instance of :py:class:`ai4water.hyperopt.HyperOpt` which is used for optimization

        Examples:
            >>> from ai4water import Model
            >>> from ai4water.datasets import busan_beach
            >>> from ai4water.hyperopt import Integer, Categorical, Real
            >>> model_config = {"XGBRegressor": {"n_estimators": Integer(low=10, high=20),
            >>>                 "max_depth": Categorical([10, 20, 30]),
            >>>                 "learning_rate": Real(0.00001, 0.1)}}
            >>> model = Model(model=model_config)
            >>> optimizer = model.optimize_hyperparameters(data=busan_beach())

            Same can be done if a model is defined using neural networks

            >>> lookback = 14
            >>> model_config = {"layers": {
            ...     "Input": {"input_shape": (lookback, 13)},
            ...     "LSTM":  {"config": {"units": Integer(32, 64), "activation": "relu"}},
            ...      "Dense": {"units": 1,
            ...            "activation": Categorical(["relu", "tanh"], name="dense1_act")}}}
            >>> model = Model(model=model_config, ts_args={"lookback": lookback})
            >>> optimizer = model.optimize_hyperparameters(data=busan_beach(),
            ...          refit=False)
        """
        from ._optimize import OptimizeHyperparameters  # optimize_hyperparameters

        if isinstance(data, list) or isinstance(data, tuple):
            pass
        else:
            setattr(self, 'dh_', DataSet(data, **self.data_config))

        if self.opt_paras:
            space = list(self.opt_paras.values())

        elif self.category == "ML":

            from .experiments.utils import regression_models, classification_models
            from .experiments.utils import regression_space, classification_space

            if self.model_name in regression_models():
                space = regression_space(len(data), 5)[self.model_name]['param_space']

            elif self.model_name in classification_models():
                space = classification_space(len(data), 5)[self.model_name]['param_space']

            else:
                raise ValueError
        else:
            raise ValueError

        _optimizer = OptimizeHyperparameters(
            self,
            space = space,
            algorithm=algorithm,
            num_iterations=num_iterations,
            process_results=process_results,
            **kwargs
        ).fit(data=data)

        algo_type = list(self.config['model'].keys())[0]

        if self.opt_paras:
            new_model_config = update_model_config(self._original_model_config['model'],
                                               _optimizer.best_paras())
        else:
            new_model_config = _optimizer.best_paras()
        self.config['model'][algo_type] = new_model_config

        # find parameters which were in model_config but not optimized
        # and update the config!
        new_other_config = update_model_config(self._original_model_config['other'],
                                               _optimizer.best_paras())
        self.config.update(new_other_config)

        # if ts_args are optimized, update them as well
        for k, v in _optimizer.best_paras().items():
            if k in self.config['ts_args']:
                self.config['ts_args'][k] = v

        if refit:
            # for ml models, we must build them again
            # TODO, why not for DL models
            if self.category == "ML":
                self.build_ml_model()
                if isinstance(data, (list, tuple)):
                    assert len(data)==2
                    x, y = data
                    self.fit_on_all_training_data(x=x, y=y)
                else:
                    self.fit_on_all_training_data(data=data)
            else:
                raise NotImplementedError(f"""
            Setting `refit` is not supported for neural network
            based models. please call optimize_hyperparmeters(..., refit=False)
""")

        return _optimizer

    def optimize_transformations(
            self,
            data:Union[np.ndarray, pd.DataFrame, Tuple[np.ndarray, np.ndarray]],
            transformations: Union[list, str] = None,
            include: Union[str, list, dict] = None,
            exclude: Union[str, list] = None,
            append: dict = None,
            y_transformations: Union[list, dict] = None,
            algorithm: str = "bayes",
            num_iterations: int = 12,
            process_results: bool = True,
            update_config: bool = True
    ):
        """
        optimizes the transformations for the input/output features

        The 'val_score' parameter given as input to the Model is used as objective
        function for optimization problem.

        Arguments:
            data :
                It can be one of following

                    - raw unprepared data in the form of a numpy array or pandas dataframe
                    - a tuple of x,y pairs
                If it is unprepared data, it is passed to :py:class:`ai4water.preprocessing.DataSet`.
                which prepares x,y pairs from it. The ``DataSet`` class also
                splits the data into training, validation and tests sets. If it
                is a tuple of x,y pairs, it is split into training and validation.
                In both cases, the loss on validation set is used as objective function.
                The loss calculated using ``val_metric``.
            transformations :
                the transformations to consider for input features. By default,
                following transformations are considered for input features

                - ``minmax``  rescale from 0 to 1
                - ``center``    center the data by subtracting mean from it
                - ``scale``     scale the data by dividing it with its standard deviation
                - ``zscore``    first performs centering and then scaling
                - ``box-cox``
                - ``yeo-johnson``
                - ``quantile``
                - ``robust``
                - ``log``
                - ``log2``
                - ``log10``
                - ``sqrt``    square root

            include : list, dict, str, optional
                the name/names of input features to include. If you don't want
                to include any feature. Set this to an empty list
            exclude :
                the name/names of input features to exclude
            append :
                the input features with custom candidate transformations. For example
                if we want to try only `minmax` and `zscore` on feature `tide_cm`, then
                it can be done as following

                >>> {"tide_cm": ["minmax", "zscore"]}

            y_transformations : list/str
                It can either be a list of transformations to be considered for
                output features for example

                >>> ['log', 'log10', 'log2', 'sqrt']

                would mean that consider `log`, `log10`, `log2` and `sqrt` are
                to be considered for output transformations during optimization.
                It can also be a dictionary whose keys are names of output features
                and whose values are lists of transformations to be considered for output
                features. For example

                >>> {'output1': ['log2', 'log10'], 'output2': ['log', 'sqrt']}

                Default is None, which means do not optimize transformation for output
                features.
            algorithm : str
                The algorithm to use for optimizing transformations
            num_iterations : int
                The number of iterations for optimizatino algorithm.
            process_results : bool
                whether to perform postprocessing of optimization results or not
            update_config: whether to update the config of model or not.

        Returns:
            an instance of HyperOpt :py:class:`ai4water.hyperopt.HyperOpt` class
            which is used for optimization

        Example:
            >>> from ai4water.datasets import busan_beach
            >>> from ai4water import Model
            >>> model = Model(model="XGBRegressor")
            >>> optimizer_ = model.optimize_transformations(data=busan_beach(), exclude="tide_cm")
            >>> print(optimizer_.best_paras())  # find the best/optimized transformations
            >>> model.fit(data=busan_beach())
            >>> model.predict()
        """
        from ._optimize import OptimizeTransformations  # optimize_transformations
        from .preprocessing.transformations.utils import InvalidTransformation

        # if isinstance(data, list) or isinstance(data, tuple):
        #     pass
        # else:
        #     setattr(self, 'dh_', DataSet(data=data, **self.data_config))

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
            process_results=process_results,
        ).fit(data=data)

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
            data = None,
            data_type: str = "test",
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

        Parameters
        ----------
            data :
                Raw unprepared data from which x,y paris of training and test
                data are prepared.
            data_type : str
                one of `training`, `test` or `validation`. By default test data is
                used based upon recommendations of Christoph Molnar's book_. Only
                valid if ``data`` argument is given.
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
                if not None, it must be either ``heatmap`` or ``boxplot`` or ``bar_chart``

        Returns
        -------
        an instance of :py:class:`ai4water.postprocessing.PermutationImprotance`

        Examples
        --------
            >>> from ai4water import Model
            >>> from ai4water.datasets import busan_beach
            >>> model = Model(model="XGBRegressor")
            >>> model.fit(data=busan_beach())
            >>> perm_imp = model.permutation_importance(data=busan_beach(),
            ...  data_type="validation", plot_type="boxplot")
            >>> perm_imp.importances

        .. _book:
            https://christophm.github.io/interpretable-ml-book/feature-importance.html#feature-importance-data
        """
        assert data_type in ("training", "validation", "test")

        if x is None:
            data = getattr(self, f"{data_type}_data")(data=data)
            x, y = data

        from .postprocessing.explain import PermutationImportance

        save = False
        path = None
        if self.verbosity >= 0:
            path = os.path.join(self.path, "explain")

        pm = PermutationImportance(
            model=self.predict,
            inputs=x,
            target=y,
            scoring=scoring,
            n_repeats=n_repeats,
            noise=noise,
            use_noise_only=use_noise_only,
            path=path,
            feature_names=self.input_features,
            weights=weights,
            seed=self.config['seed'],
            save=save
        )

        if plot_type is not None:
            assert plot_type in ("boxplot", "heatmap", "bar_chart")
            if plot_type == "heatmap":
                pm.plot_as_heatmap()
            else:
                pm.plot_1d_pimp(plot_type=plot_type)
        return pm

    def sensitivity_analysis(
            self,
            data=None,
            bounds=None,
            sampler="morris",
            analyzer:Union[str, list]="sobol",
            sampler_kwds: dict = None,
            analyzer_kwds: dict = None,
            save_plots: bool = True,
            names: List[str] = None
    )->dict:
        """performs sensitivity analysis of the model w.r.t input features in data.

        The model and its hyperprameters remain fixed while the input data is changed.

        Parameters
        ----------
        data :
            data which will be used to get the bounds/limits of input features. If given,
            it must be 2d numpy array. It should be remembered that the given data
            is not used during sensitivity analysis. But new synthetic data is prepared
            on which sensitivity analysis is performed.
        bounds : list,
            alternative to data
        sampler : str, optional
            any sampler_ from SALib library. For example ``morris``, ``fast_sampler``,
            ``ff``, ``finite_diff``, ``latin``, ``saltelli``, ``sobol_sequence``
        analyzer : str, optional
            any analyzer_ from SALib lirary. For example ``sobol``, ``dgsm``, ``fast``
            ``ff``, ``hdmr``, ``morris``, ``pawn``, ``rbd_fast``. You can also choose
            more than one analyzer. This is useful when you want to compare results
            of more than one analyzers. It should be noted that having more than
            one analyzers does not increases computation time except for ``hdmr``
            and ``delta`` analyzers. The ``hdmr`` and ``delta`` analyzers ane computation
            heavy. For example
            >>> analyzer = ["morris", "sobol", "rbd_fast"]
        sampler_kwds : dict
            keyword arguments for sampler
        analyzer_kwds : dict
            keyword arguments for analyzer
        save_plots : bool, optional
        names : list, optional
            names of input features. If not given, names of input features will be used.

        Returns
        -------
        dict :
            a dictionary whose keys are names of analyzers and values and sensitivity
            results for that analyzer.

        Examples
        --------
        >>> from ai4water import Model
        >>> from ai4water.datasets import busan_beach
        >>> df = busan_beach()
        >>> input_features=df.columns.tolist()[0:-1]
        >>> output_features = df.columns.tolist()[-1:]
        ... # build the model
        >>> model=Model(model="RandomForestRegressor",
        >>>     input_features=input_features,
        >>>     output_features=output_features)
        ... # train the model
        >>> model.fit(data=df)
        ... # perform sensitivity analysis
        >>> si = model.sensitivity_analysis(data=df[input_features].values,
        >>>                    sampler="morris", analyzer=["morris", "sobol"],
        >>>                        sampler_kwds={'N': 100})

        .. _sampler:
            https://salib.readthedocs.io/en/latest/api/SALib.sample.html

        .. _analyzer:
            https://salib.readthedocs.io/en/latest/api/SALib.analyze.html

        """
        try:
            import SALib
        except (ImportError, ModuleNotFoundError):
            warnings.warn("""
            You must have SALib library installed in order to perform sensitivity analysis.
            Please install it using 'pip install SALib' and make sure that it is importable
            """)
            return {}

        from ai4water.postprocessing._sa import sensitivity_analysis, sensitivity_plots, make_predict_func

        if data is not None:
            if not isinstance(data, np.ndarray):
                assert isinstance(data, pd.DataFrame)
                data = data.values
            x = data

            # calculate bounds
            assert isinstance(x, np.ndarray)
            bounds = []
            for feat in range(x.shape[1]):
                bound = [np.min(x[:, feat]), np.max(x[:, feat])]
                bounds.append(bound)
        else:
            assert bounds is not None
            assert isinstance(bounds, list)
            assert all([isinstance(bound, list) for bound in bounds])

        analyzer_kwds = analyzer_kwds or {}

        if self.lookback >1:
            if self.category == "DL":
                func = make_predict_func(self, verbose=0)
            else:
                func = make_predict_func(self)
        else:
            func = self.predict

        if names is None:
            names = self.input_features

        results = sensitivity_analysis(
            sampler,
            analyzer,
            func,
            bounds=bounds,
            sampler_kwds = sampler_kwds,
            analyzer_kwds = analyzer_kwds,
            names=names
        )

        if save_plots:
            for _analyzer, result in results.items():
                res_df = result.to_df()
                if isinstance(res_df, list):
                    for idx, res in enumerate(res_df):
                        fname = os.path.join(self.path, f"{_analyzer}_{idx}_results.csv")
                        res.to_csv(fname)
                else:
                    res_df.to_csv(os.path.join(self.path, f"{_analyzer}_results.csv"))

                sensitivity_plots(_analyzer, result, self.path)

        return results

    def shap_values(
            self,
            data,
            layer=None
    )->np.ndarray:
        """
        returns shap values

        Parameters
        ----------
            data :
                raw unprepared data from which training and test data are extracted.
            layer :

        Returns
        -------

        Examples
        --------
            >>> from ai4water import Model
            >>> from ai4water.datasets import busan_beach
            >>> df = busan_beach()
            >>> model = Model(model="RandomForestRegressor")
            >>> model.fit(data=df)
            >>> model.shap_values(data=df)

        """

        from .postprocessing.explain import explain_model_with_shap

        explainer = explain_model_with_shap(
            self,
            total_data=data,
            layer=layer,
        )

        return explainer.shap_values

    def explain_example(
            self,
            data,
            example_num:int,
            method="shap"
    ):
        """explains a single exmaple either using shap or lime

        Parameters
        ----------
            data :
                the data to use
            example_num :
                the example/sample number/index to explain
            method :
                either ``shap`` or ``lime``

        Examples
        --------
            >>> from ai4water import Model
            >>> from ai4water.datasets import busan_beach
            >>> df = busan_beach()
            >>> model = Model(model="RandomForestRegressor")
            >>> model.fit(data=df)
            >>> model.explain_example(data=df, example_num=2)

        """

        assert method in ("shap", "lime")

        if method == "shap":
            from .postprocessing.explain import explain_model_with_shap

            explainer = explain_model_with_shap(
                self,
                total_data=data,
                examples_to_explain=example_num)
        else:
            from .postprocessing.explain import explain_model_with_lime

            explainer = explain_model_with_lime(
                self,
                total_data=data,
                examples_to_explain=example_num)

        return explainer

    def partial_dependence_plot(
            self,
            x=None,
            data=None,
            data_type="all",
            feature_name=None,
            num_points:int=100,
            show:bool = True,
    ):
        """Shows partial depedence plot for a feature.

        Parameters
        ----------
            x :
                the input data to use. If not given, then ``data`` must be given.
            data :
                raw unprepared data from which x,y paris are to be made. If
                given, ``x`` must not be given.
            data_type : str
                the kind of the data to be used. It is only valid when
                ``data`` is given.
            feature_name : str/list
                name/names of features. If only one feature is given, 1 dimensional
                partial dependence plot is plotted. You can also provide a list of
                two feature names, in which case 2d interaction plot will be plotted.
            num_points : int
                number of points. It is used to define grid.
            show : bool
                whether to show the plot or not!

        Returns
        -------
            an instance of :py:class:`ai4water.postprocessing.PartialDependencePlot`

        Examples
        --------
            >>> from ai4water import Model
            >>> from ai4water.datasets import busan_beach
            >>> df = busan_beach()
            >>> model = Model(model="RandomForestRegressor")
            >>> model.fit(data=df)
            >>> model.partial_dependence_plot(x=df.iloc[:, 0:-1], feature_name="tide_cm")
            ...
            >>> model.partial_dependence_plot(data=df, feature_name="tide_cm")

        """
        if x is None:
            assert data is not None, f"either x or data must be given"
            x, _ = getattr(self, f"{data_type}_data")(data=data)

        from .postprocessing.explain import PartialDependencePlot

        pdp = PartialDependencePlot(
            self.predict,
            data=x,
            feature_names=self.input_features,
            num_points=num_points,
            show=show
        )

        if isinstance(feature_name, str):
            pdp.plot_1d(feature=feature_name)
        else:
            assert isinstance(feature_name, list)
            assert len(feature_name) == 2
            pdp.plot_interaction(features=feature_name)

        return pdp

    def prediction_analysis(
            self,
            features:Union[list, str],
            x:Union[np.ndarray, pd.DataFrame]=None,
            y:np.ndarray=None,
            data = None,
            data_type:str = "all",
            feature_names:Union[list, str]=None,
            num_grid_points:Union[int, Tuple[int, int]] = None,
            grid_types = "percentile",
            percentile_ranges = None,
            grid_ranges = None,
            custom_grid:list = None,
            show_percentile: bool = False,
            show_outliers: bool = False,
            end_point:bool = True,
            which_classes=None,
            ncols=2,
            figsize: tuple = None,
            kind:str = None,
            annotate:bool = True,
            annotate_kws:dict = None,
            cmap="YlGn",
            border=False,
            show:bool=True,
            save_metadata:bool=True
    )->plt.Axes:
        """shows prediction distribution with respect to two input features.

        Parameters
        ----------
            x :
                input data to the model.
            y :
                true data corresponding to ``x``.
            data :
                raw unprepared data from which x,y pairs for training,validation and test
                are generated. It must only be given if ``x`` is not given.
            data_type : str, optional (default="test")
                The kind of data to be used. It is only valid if ``data`` argument is used.
                It should be one of ``training``, ``validation``, ``test`` or ``all``.
            features: str/list
                name or names of features to investigate. If one feauture is given then
                distribution of prediction within buckets defined by the grid is plotted
                plotted as bar/box/violin . If two features are given then a heatmap
                is plotted.
            feature_names: list
                feature names
            num_grid_points: list, optional, default=None
                number of grid points for each feature
            grid_types: list, optional, default=None
                type of grid points for each feature
            percentile_ranges: list of tuple, optional, default=None
                percentile range to investigate for each feature
            grid_ranges: list of tuple, optional, default=None
                value range to investigate for each feature
            custom_grid: list of (Series, 1d-array, list), optional, default=None
                customized list of grid points for each feature
            show_percentile: bool, optional, default=False
                whether to display the percentile buckets for both feature
            show_outliers: bool, optional, default=False
                whether to display the out of range buckets for both features
            end_point: bool, optional
                If True, stop is the last grid point, default=True
                Otherwise, it is not included
            which_classes: list, optional, default=None
                which classes to plot, only use when it is a multi-class problem
            figsize: tuple or None, optional, default=None
                size of the figure, (width, height)
            ncols: integer, optional, default=2
                number subplot columns, used when it is multi-class problem
            kind : str (default=None)
                The kind of plot to draw. This is only valid if only one feature
                is provided for ``features`` argument.
            annotate: bool, default=False
                whether to annotate the points
            annotate_kws : dict, optional
                a dictionary of keyword arguments with following keys
                    counts : bool, default=False
                        whether to annotate counts or not.
                    colors : tuple
                        pair of colors
                    color_threshold : float
                        threshold value for annotation
                    fmt : str
                        format string for annotation.
                    fontsize : int, optinoal (default=7)
                        fontsize for annotation
            cmap :
                The colormap to use for heatmap.
            border :
                whether to draw teh border around heatmap or not
            show : bool, optional (default=True)
                whether to show the  plot or not
            save_metadata : bool, optional, default=True
                whether to save the information as csv or not

        Returns
        -------
        tuple
            a pandas dataframe and matplotlib Axes

        Examples
        --------
        >>> from ai4water.datasets import busan_beach
        >>> from ai4water import Model
        ...
        >>> model = Model(model="XGBRegressor")
        >>> model.fit(data=busan_beach())
        >>> model.prediction_analysis(features="tide_cm",
        ... data=busan_beach(), show_percentile=True)
        ... # for multiple features
        >>> model.prediction_analysis(
        ...     ['tide_cm', 'sal_psu'],
        ...     data=busan_beach(),
        ...     annotate_kws={"counts":True,
        ...     "colors":("black", "black"),
        ...     "fontsize":10},
        ...     custom_grid=[[-41.4, -20.0, 0.0, 20.0, 42.0],
        ...                       [33.45, 33.7, 33.9, 34.05, 34.4]],
        ... )

        """
        if x is None:
            assert data is not None
            x, _ = getattr(self, f"{data_type}_data")(data=data)

            y = self.predict(x)

        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x, columns=self.input_features)

        if feature_names is None:
            if isinstance(features, str):
                feature_names = [features]
            else:
                assert isinstance(features, list)
                feature_names = ["Feature"]

        if not isinstance(features, list):
            features = [features]

        _annotate_kws = {
            'counts': True,
            'colors':("black", "white"),
            'color_threshold': None,
            'fmt': None,
            'fontsize': 8
        }

        if annotate_kws is None:
            annotate_kws = dict()

        _annotate_kws.update(annotate_kws)

        annotate_kws = {}
        for k,v in _annotate_kws.items():
            k = "annotate_" + k
            annotate_kws[k] = v

        if len(features) == 1:
            ax, summary_df = prediction_distribution_plot(
                self.mode,
                inputs=x,
                prediction=y,
                feature=features[0],
                feature_name=feature_names[0],
                n_classes=getattr(self, "num_classes_"),
                num_grid_points=num_grid_points or 10,
                grid_type=grid_types,
                percentile_range=percentile_ranges,
                grid_range=grid_ranges,
                cust_grid_points=custom_grid,
                show_percentile=show_percentile,
                show_outliers=show_outliers,
                end_point=end_point,
                kind=kind,
                figsize=figsize,
                ncols=ncols,
                show=False,
            )

        else:
            ax, summary_df = feature_interaction(
                self.predict,
                x,
                features=features,
                feature_names=feature_names,
                num_grid_points=num_grid_points,
                grid_types=grid_types,
                percentile_ranges=percentile_ranges,
                grid_ranges=grid_ranges,
                cust_grid_points=custom_grid,
                show_percentile=show_percentile,
                show_outliers=show_outliers,
                end_point=end_point,
                which_classes=which_classes,
                ncols=ncols,
                figsize=figsize,
                annotate=annotate,
                cmap=cmap,
                border=border,
                **annotate_kws,
            )

        fname = f"prediction_analysis_{features[0] if len(features)==1 else features[0]+features[1]}"
        if save_metadata:
            summary_df.to_csv(os.path.join(self.path, f"{fname}.csv"))

        if show:
            plt.show()

        return ax

    def _transform(self, data, name_in_config:str):
        """transforms the data using the transformer which has already been fit"""

        if name_in_config not in self.config:
            transformation_type = name_in_config[0]
            raise NotImplementedError(f"""You have not trained the model using .fit.
            Making predictions from model or evaluating model without training
            is not allowed when applying transformations. Because the transformation
            parameters are calculated using training data. Either train the model
            first by using.fit() method or remove {transformation_type}_transformation
            arguments.""")

        transformer = Transformations.from_config(self.config[name_in_config])

        # because in _fit_transform(), DataFrame is also converted so we do it here
        # as well
        if isinstance(data, pd.DataFrame):
            data = data.values
        return transformer.transform(data=data)

    def _transform_x(self, x):
        """transforms the x data using the transformer which has already been fit"""
        if self.config['x_transformation']:
            return self._transform(x, 'x_transformer_')
        return x

    def _transform_y(self, y):
        """transforms the y according the transformer which has already been fit."""
        if self.config['y_transformation']:
            return self._transform(y, 'y_transformer_')
        return y

    def _fit_transform_x(self, x):
        """fits and transforms x and puts the transformer in config witht he key
        'x_transformer_'"""
        return self._fit_transform(
            x,
            'x_transformer_',
            self.config['x_transformation'],
            self.input_features)

    def _fit_transform_y(self, y):
        """fits and transforms y and puts the transformer in config witht he key
        'y_transformer_'"""
        return self._fit_transform(
            y,
            'y_transformer_',
            self.config['y_transformation'],
            self.output_features)

    def _fit_transform(self, data, key, transformation, feature_names):
        """fits and transforms the `data` using `transformation` and puts it in config
        with config `name`."""
        if data is not None and transformation:
            transformer = Transformations(feature_names, transformation)
            if isinstance(data, pd.DataFrame):
                data = data.values
            data = transformer.fit_transform(data)
            self.config[key] = transformer.config()
        return data

    def _inverse_transform_y(self, true_outputs, predicted):
        """inverse transformation of y/labels for both true and predicted"""

        if self.config['y_transformation']:

            y_transformer = Transformations.from_config(self.config['y_transformer_'])

            if self.config['y_transformation']:  # only if we apply transformation on y
                # both x,and true_y were given
                true_outputs = self.__inverse_transform_y(true_outputs, y_transformer)
                # because observed y may have -ves or zeros which would have been
                # removed during fit and are put back into it during inverse transform, so
                # in such case predicted y should not be treated by zero indices or negative indices
                # of true y. In other words, parameters of true y should not have impact on inverse
                # transformation of predicted y.
                predicted = self.__inverse_transform_y(predicted, y_transformer, postprocess=False)

        return true_outputs, predicted

    def __inverse_transform_y(self,
                              y,
                              transformer,
                              method="inverse_transform",
                              postprocess=True
                              )->np.ndarray:
        """inverse transforms y where y is supposed to be true or predicted output
        from model."""
        # todo, if train_y had zeros or negatives then this will be wrong
        if isinstance(y, np.ndarray):
            # it is ndarray, either num_outs>1 or quantiles>1 or forecast_len>1 some combination of them
            if y.size > len(y):
                if y.ndim == 2:
                    y = getattr(transformer, method)(y, postprocess=postprocess)
                else:
                    # (exs, outs, quantiles) or (exs, outs, forecast_len) or (exs, forecast_len, quantiles)
                    for out in range(y.shape[1]):
                        for q in range(y.shape[2]):
                            y[:, out, q] = getattr(transformer, method)(y[:, out, q],
                                                                        postprocess=postprocess).reshape(-1, )
            else:  # 1d array
                y = getattr(transformer, method)(y, postprocess=postprocess)
        # y can be None for example when we call model.predict(x=x),
        # in this case we don't know what is y
        elif y is not None:
            raise ValueError(f"can't inverse transform y of type {type(y)}")

        return y

    def training_data(self, x=None, y=None, data='training', key='train')->tuple:
        """
        returns the x,y pairs for training. x,y are not used but
        only given to be used if user overwrites this method for further processing
        of x, y as shown below.

        >>> from ai4water import Model
        >>> class MyModel(Model):
        >>>     def training_data(self, *args, **kwargs) ->tuple:
        >>>         train_x, train_y = super().training_data(*args, **kwargs)
        ...         # further process x, y
        >>>         return train_x, train_y
        """
        return self.__fetch('training', data,key)

    def validation_data(self, x=None, y=None, data='validation', key="val")->tuple:
        """
        returns the x,y pairs for validation. x,y are not used but
        only given to be used if user overwrites this method for further processing
        of x, y as shown below.

        >>> from ai4water import Model
        >>> class MyModel(Model):
        >>>     def validation_data(self, *args, **kwargs) ->tuple:
        >>>         train_x, train_y = super().training_data(*args, **kwargs)
        ...         # further process x, y
        >>>         return train_x, train_y
        """

        return self.__fetch('validation', data, key)

    def test_data(self, x=None, y=None, data='test', key="test")->tuple:
        """
        returns the x,y pairs for test. x,y are not used but
        only given to be used if user overwrites this method for further processing
        of x, y as shown below.

        >>> from ai4water import Model
        >>> class MyModel(Model):
        >>>     def ttest_data(self, *args, **kwargs) ->tuple:
        >>>         train_x, train_y = super().training_data(*args, **kwargs)
        ...         # further process x, y
        >>>         return train_x, train_y
        """
        return self.__fetch('test', data, key)

    def all_data(self, x=None, y=None, data=None)->tuple:
        """it returns all data i.e. training+validation+test
        after extracting them ``data``.

        Examples
        --------
        >>> from ai4water import Model
        >>> from ai4water.datasets import busan_beach
        >>> model = Model(model="XGBRegressor")
        >>> x_train, y_train = model.training_data(data=data)
        >>> print(x_train.shape, y_train.shape)
        >>> x_valid, y_valid = model.validation_data(data=data)
        >>> print(x_valid.shape, y_valid.shape)
        ... # all_data will contain both training and validation data
        >>> all_x, all_y = model.all_data(data=data)
        >>> print(all_x.shape, all_y.shape
        """

        train_x, train_y = self.training_data(x=x, y=y, data=data)
        val_x, val_y = self.validation_data(x=x, y=y, data=data)
        test_x, test_y = self.test_data(x=x, y=y, data=data)

        x = []
        y = []

        if isinstance(train_x, list):
            for val in range(len(train_x)):

                # if val data is not available
                if hasattr(val_x[val], '__len__') and len(val_x[val])==0:
                    x_val = np.concatenate([train_x[val], test_x[val]])

                # if test data is not available
                elif hasattr(test_x[val], '__len__') and len(test_x[val])==0:
                    x_val = np.concatenate([train_x[val], val_x[val]])
                # supposing all three data are available
                else:
                    x_val = np.concatenate([train_x[val], val_x[val], test_x[val]])
                x.append(x_val)
        else:
            for _x in [train_x, val_x, test_x]:
                if _x is not None and (hasattr(_x, '__len__') and len(_x)>0):
                    x.append(_x)
            x = np.concatenate(x)

        for _y in [train_y, val_y, test_y]:
            if _y is not None and (hasattr(_y, '__len__') and len(_y)>0):
                y.append(_y)

        y = np.concatenate(y)
        return x, y

    def __fetch(self, source, data=None, key=None):
        """if data is string, then it must either be `trianing`, `validation` or
        `test` or name of a valid dataset. Otherwise data is supposed to be raw
        data which will be given to DataSet
        """
        if isinstance(data, str):
            if data in ['training', 'test', 'validation']:
                if hasattr(self, 'dh_'):
                    data = getattr(self.dh_, f'{data}_data')(key=key)
                else:
                    raise DataNotFound(source)
            else:
                # e.g. 'CAMELS_AUS'
                dh = DataSet(data=data, **self.data_config)
                setattr(self, 'dh_', dh)
                data = getattr(dh, f'{source}_data')(key=key)
        else:
            dh = DataSet(data=data, **self.data_config)
            setattr(self, 'dh_', dh)
            data = getattr(dh, f'{source}_data')(key=key)

        x, y = maybe_three_outputs(data, self.teacher_forcing)

        return x, y

    def _fetch_data(self, source:str, x=None, y=None, data=None):
        """The main idea is that the user should be able to fully customize
        training/test data by overwriting training_data and test_data methods.
        However, if x is given or data is DataSet then the training_data/test_data
        methods of this(Model) class will not be called."""
        user_defined_x = True
        prefix = f'{source}_{dateandtime_now()}'
        key = None

        if x is None:
            user_defined_x = False
            key=f"5_{source}"

            if isinstance(data, _DataSet):
                # the user has provided DataSet from which training/test data
                # needs to be extracted
                setattr(self, 'dh_', data)
                data = getattr(data, f'{source}_data')(key=key)

            else:
                data = getattr(self, f'{source}_data')(x=x, y=y, data=data, key=key)

            # data may be tuple/list of three arrays
            x, y = maybe_three_outputs(data, self.teacher_forcing)

        return x, y, prefix, key, user_defined_x

    def _maybe_reduce_nquantiles(self, num_exs:int)->None:

        self.config['x_transformation'] = _reduce_nquantiles_in_config(self.config['x_transformation'], num_exs)
        self.config['y_transformation'] = _reduce_nquantiles_in_config(self.config['y_transformation'], num_exs)

        return

def _reduce_nquantiles_in_config(config:Union[str, list, dict], num_exs:int):

    if isinstance(config, str) and config in ['quantile', 'quantile_normal']:
        config = {'method': 'quantile', 'n_quantiles': num_exs}

    elif isinstance(config, dict):
        if 'method' not in config:
            # for multiinput cases when x_transformation is defined as
            # {'inp_1d': 'minmax', 'inp_2d': None}
            pass # todo
        elif config['method'] in ['quantile', 'quantile_normal']:
            config['n_quantiles'] = min(config.get('n_quantiles', num_exs), num_exs)

    elif isinstance(config, list):

        for idx, transformer in enumerate(config):

            if isinstance(transformer, str) and transformer in ['quantile', 'quantile_normal']:
                config[idx] = {'method': 'quantile', 'n_quantiles': num_exs}

            elif isinstance(transformer, dict) and transformer['method'] in ['quantile', 'quantile_normal']:
                transformer['n_quantiles'] = min(transformer.get('n_quantiles', num_exs), num_exs)
                config[idx] = transformer

    return config


def fill_val(metric_name, default="min", default_min=99999999):
    if METRIC_TYPES.get(metric_name, default) == "min":
        return default_min
    return 0.0


def _find_num_examples(inputs)->Union[int, None]:
    """find number of examples in inputs"""
    if isinstance(inputs, (pd.DataFrame, np.ndarray)):
        return len(inputs)
    if isinstance(inputs, list):
        return len(inputs[0])
    elif hasattr(inputs, '__len__'):
        return len(inputs)

    return None
