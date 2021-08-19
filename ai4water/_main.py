import os
import json
import time
import pprint
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
import matplotlib.pyplot as plt

from ai4water.nn_tools import NN
from ai4water._data import DataHandler
from ai4water.backend import tpot_models
from ai4water.backend import sklearn_models
from ai4water.utils.plotting_tools import Plots
from ai4water.utils.utils import ts_features, make_model
from ai4water.utils.utils import find_best_weight, reset_seed
from ai4water.models.custom_training import train_step, test_step
from ai4water.utils.visualizations import Visualizations, Interpret
from ai4water.utils.SeqMetrics import RegressionMetrics, ClassificationMetrics
from ai4water.utils.utils import maybe_create_path, save_config_file, dateandtime_now
from .backend import tf, keras, torch, VERSION_INFO, catboost_models, xgboost_models, lightgbm_models
from .backend import BACKEND

LOSSES = {}

if BACKEND == 'tensorflow' and tf is not None:

    import ai4water.keract_mod as keract
    from ai4water.tf_attributes import LOSSES, OPTIMIZERS

elif BACKEND == 'pytorch' and torch is not None:
    from ai4water.pytorch_attributes import LOSSES, OPTIMIZERS



class BaseModel(NN, Plots):
    """
    Model class that implements logic of AI4Water.
    """

    def __init__(self,
                 model: dict = None,
                 data = None,
                 lr: float = 0.001,
                 optimizer = 'adam',
                 loss:Union[str, Callable] = 'mse',
                 quantiles = None,
                 epochs:int = 14,
                 min_val_loss:float = 0.0001,
                 patience:int = 100,
                 save_model:bool = True,
                 metrics:Union[str, list] = None,
                 cross_validator:dict=None,
                 seed:int = 313,
                 prefix: str = None,
                 path: str = None,
                 verbosity: int = 1,
                 accept_additional_args:bool = False,
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
                on how to buld neural networks using such layered API [see](https://ai4water.readthedocs.io/en/latest/build_dl_models/)
            lr  float:, default 0.001.
                learning rate,
            optimizer str/keras.optimizers like:
                the optimizer to be used for neural network training. Default is 'adam'
            loss str/callable:  Default is `mse`.
                the cost/loss function to be used for training neural networks.
            quantiles list: Default is None
                quantiles to be used when the problem is quantile regression.
            epochs int:  Default is 14
                number of epochs to be used.
            min_val_loss float:  Default is 0.0001.
                minimum value of validatin loss/error to be used for early stopping.
            patience int:
                number of epochs to wait before early stopping.
            save_model bool:,
                whether to save the model or not. For neural networks, the model will
                be saved only an improvement in training/validation loss is observed.
                Otherwise model is not saved.
            subsequences int: Default is 3.
                The number of sub-sequences. Relevent for building CNN-LSTM based models.
            metrics str/list:
                metrics to be monitored. e.g. ['nse', 'pbias']
            cross_validator :
                selects the type of cross validation to be applied. It can be any
                cross validator from sklear.model_selection. Default is None, which
                means validation will be done using `validation_data`. To use
                kfold cross validation,
                ```python
                cross_validator = {'kfold': {'n_splits': 5}}
                ```
            batches str:
                either `2d` or 3d`.
            seed int:
                random seed for reproducibility
            prefix str:
                prefix to be used for the folder in which the results are saved.
                default is None, which means within
                ./results/model_path
            path str/path like:
                if not given, new model_path path will not be created.
            verbosity int: default is 1.
                determines the amount of information being printed. 0 means no
                print information. Can be between 0 and 3.
            accept_additional_args bool:  Default is False
                If you want to pass any additional argument, then this argument
                must be set to True, otherwise an error will be raise.
            kwargs : keyword arguments for `DataHandler` class

        Example
        ---------
        ```python
        >>>from ai4water import Model
         >>>from ai4water.utils.datasets import arg_beach
        >>>df = arg_beach()
        >>>model = Model(data=df,
        ...              batch_size=16,
        ...           model={'layers': {'LSTM': 64}},
        ...)
        >>>history = model.fit()
        >>>y, obs = model.predict()
        ```
        """
        if self._go_up:
            maker = make_model(data,
                               model=model,
                               prefix=prefix,
                               path=path,
                               verbosity=verbosity,
                               lr=lr,
                               optimizer=optimizer,
                               loss = loss,
                               quantiles = quantiles,
                               epochs = epochs,
                               min_val_loss=min_val_loss,
                               patience = patience,
                               save_model = save_model,
                               metrics = metrics or ['nse'],
                               cross_validator = cross_validator,
                               accept_additional_args = accept_additional_args,
                               seed = seed,
                               **kwargs)

            # data_config, model_config = config['data_config'], config['model_config']
            reset_seed(maker.config['seed'], os, random, np, tf, torch)
            if tf is not None:
                # graph should be cleared everytime we build new `Model` otherwise, if two `Models` are prepared in same
                # file, they may share same graph.
                tf.keras.backend.clear_session()

            self.dh = DataHandler(data=data, **maker.data_config)

            NN.__init__(self, config=maker.config)

            self.path = maybe_create_path(path=path, prefix=prefix)
            self.verbosity = verbosity
            self.category = self.config['category']
            self.problem = self.config['problem']
            self.info = {}

            Plots.__init__(self, self.path, self.problem, self.category,
                           config=maker.config)

    def __getattr__(self, item):
        # instead of doing model.dh.training
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
        """alias for keras.MOdel.outputs"""
        if hasattr(self, 'outputs'):
            return self.outputs
        elif hasattr(self._model, 'outputs'):
            return self._model.outputs
        else:
            return None

    def trainable_parameters(self)->int:
        """Calculates trainable parameters in the model
        https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
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

        if self.config['backend']=='pytorch':
            return LOSSES[self.config['loss'].upper()]()

        return LOSSES[self.config['loss'].upper()]

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
        """callbacks for pytorch training."""
        return []

    def cbs_for_tf(self, val_data, callbacks=None):

        if callbacks is None:
            callbacks = {}

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

        _callbacks.append(keras.callbacks.EarlyStopping(
            monitor=_monitor, min_delta=self.config['min_val_loss'],
            patience=self.config['patience'], verbose=0, mode='auto'
        ))

        if 'tensorboard' in callbacks:
            tb_kwargs = callbacks['tensorboard']
            if 'log_dir' not in tb_kwargs: tb_kwargs['log_dir'] = self.path
            _callbacks.append(keras.callbacks.TensorBoard(**tb_kwargs))
            callbacks.pop('tensorboard')

        for key, val in callbacks.items():
            _callbacks.append(val)

        return _callbacks

    def get_val_data(self, val_data):
        """Finds out if there is val_data or not"""
        validation_data = None

        if val_data is not None:
            if isinstance(val_data, tuple):
                x = val_data[0]
                if x is not None:
                    if isinstance(x, np.ndarray):
                        if x.size>0:
                            validation_data = val_data
                    elif isinstance(x, dict):  # x may be a dictionary
                        for k,v in x.items():
                            if v.size>0:
                                validation_data = val_data
                                break
                    elif isinstance(x, list):
                        for v in x:
                            if v.size>0:
                                validation_data = val_data
                                break
                    else:
                        raise ValueError(f'Unrecognizable validattion data {val_data.__class__.__name__}')

        return validation_data

    def DO_fit(self, x, **kwargs):
        """If nans are present in y, then tf.keras.model.fit is called as it is otherwise it is called with custom
        train_step and test_step which avoids calculating loss at points containing nans."""
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
            for out_name, out_array in outputs.items():
                if np.isnan(out_array).sum() > 0:
                    nans_in_y_exist = True

        validation_data = self.get_val_data(validation_data)
        callbacks = self.get_callbacks(validation_data, callbacks=callbacks)

        st = time.time()

        outputs = get_values(outputs)

        if validation_data is not None:
            val_outs = validation_data[-1]
            val_outs = get_values(val_outs)
            validation_data = (validation_data[0], val_outs)


        self.DO_fit(x=inputs,
                    y=None if inputs.__class__.__name__ in ['TorchDataset', 'BatchDataset'] else outputs,
                    epochs=self.config['epochs'],
                    batch_size=None if inputs.__class__.__name__ in ['TorchDataset', 'BatchDataset'] else self.config['batch_size'],
                    validation_data=validation_data,
                    callbacks=callbacks,
                    shuffle=self.config['shuffle'],
                    steps_per_epoch=self.config['steps_per_epoch'],
                    verbose=self.verbosity,
                    nans_in_y_exist=nans_in_y_exist,
                    validation_steps=validation_steps,
                    **kwargs,
                    )

        self.info['training_time_in_minutes'] = round(float(time.time() - st) / 60.0, 2)

        return self.post_kfit()

    def post_kfit(self):
        """Does some stuff after Keras model.fit has been called"""
        if BACKEND == 'pytorch':
            history = self.torch_learner.history

        elif hasattr(self, 'history'):
            history = self.history
        else:
            history = self._model.history

        self.save_config(history.history)

        # save all the losses or performance metrics
        df = pd.DataFrame.from_dict(history.history)
        df.to_csv(os.path.join(self.path, "losses.csv"))

        return history

    def maybe_not_3d_data(self, true, predicted):

        if true.ndim < 3:
            assert self.forecast_len == 1, f'{self.forecast_len}'
            axis = 2 if true.ndim == 2 else (1, 2)
            true = np.expand_dims(true, axis=axis)

        if predicted.ndim < 3:
            assert self.forecast_len == 1
            axis = 2 if predicted.ndim == 2 else (1, 2)
            predicted = np.expand_dims(predicted, axis=axis)

        return true, predicted

    def process_class_results(self,
                              true: np.ndarray,
                              predicted: np.ndarray,
                              prefix=None,
                              index=None
                              ):
        """post-processes classification results."""

        if self.is_multiclass:
            pred_labels = [f"pred_{i}" for i in range(predicted.shape[1])]
            true_labels = [f"true_{i}" for i in range(true.shape[1])]
            fname = os.path.join(self.path, f"{prefix}_prediction.csv")
            pd.DataFrame(np.concatenate([true, predicted], axis=1), columns=true_labels + pred_labels, index=index).to_csv(fname)
            metrics = ClassificationMetrics(true, predicted, categorical=True)

            save_config_file(self.path,
                             errors=metrics.calculate_all(),
                             name=f"{prefix}_{dateandtime_now()}.json"
                             )
        else:
            if predicted.ndim==1:
                predicted = predicted.reshape(-1,1)
            for idx, _class in enumerate(self.out_cols):
                _true = true[:, idx]
                _pred = predicted[:, idx]

                fpath = os.path.join(self.path, _class)
                if not os.path.exists(fpath):
                    os.makedirs(fpath)

                metrics = ClassificationMetrics(_true, _pred, categorical=False)
                save_config_file(fpath,
                                 errors=metrics.calculate_all(),
                                 name=f"{prefix}_{_class}_{dateandtime_now()}.json"
                                 )

                fname = os.path.join(fpath, f"{prefix}_{_class}.csv")
                array = np.concatenate([_true.reshape(-1, 1), _pred.reshape(-1, 1)], axis=1)
                pd.DataFrame(array,
                             columns=['true', 'predicted'],
                             index=index).to_csv(fname)


        return

    def process_regres_results(
            self,
            true: np.ndarray,
            predicted: np.ndarray,
            prefix=None,
            index=None,
            remove_nans=True
    ):
        """
        predicted, true are arrays of shape (examples, outs, forecast_len)
        """
        visualizer = Visualizations(path=self.path)

        # for cases if they are 2D/1D, add the third dimension.
        true, predicted = self.maybe_not_3d_data(true, predicted)

        forecast_len = self.forecast_len
        if isinstance(forecast_len, dict):
            forecast_len = np.unique(list(forecast_len.values())).item()


        out_cols = list(self.out_cols.values())[0] if isinstance(self.out_cols, dict) else self.out_cols
        for idx, out in enumerate(out_cols):

            horizon_errors = {metric_name:[] for metric_name in ['nse', 'rmse']}
            for h in range(forecast_len):

                errs = dict()

                fpath = os.path.join(self.path, out)
                if not os.path.exists(fpath):
                    os.makedirs(fpath)

                t = pd.DataFrame(true[:, idx, h], index=index, columns=['true_' + out])
                p = pd.DataFrame(predicted[:, idx, h], index=index, columns=['pred_' + out])
                df = pd.concat([t, p], axis=1)
                df = df.sort_index()
                fname = prefix + out + '_' + str(h) + ".csv"
                df.to_csv(os.path.join(fpath, fname), index_label='index')

                visualizer.plot_results(t, p, name=prefix + out + '_' + str(h), where=out)

                if remove_nans:
                    nan_idx = np.isnan(t)
                    t = t.values[~nan_idx]
                    p = p.values[~nan_idx]

                errors = RegressionMetrics(t, p)
                errs[out + '_errors_' + str(h)] = errors.calculate_all()
                errs[out + 'true_stats_' + str(h)] = ts_features(t)
                errs[out + 'predicted_stats_' + str(h)] = ts_features(p)

                save_config_file(fpath, errors=errs, name=prefix)

                [horizon_errors[p].append(getattr(errors, p)()) for p in horizon_errors.keys()]

            if forecast_len>1:
                visualizer.horizon_plots(horizon_errors, f'{prefix}_{out}_horizons.png')
        return

    def build_ml_model(self):
        """ builds models that follow sklearn api such as xgboost, catboost, lightgbm and obviously sklearn."""

        ml_models = {**sklearn_models, **xgboost_models, **catboost_models, **lightgbm_models, **tpot_models}
        _model = list(self.config['model'].keys())[0]
        regr_name = _model.upper()

        kwargs = list(self.config['model'].values())[0]

        if regr_name in ['HISTGRADIENTBOOSTINGREGRESSOR', 'SGDREGRESSOR', 'MLPREGRESSOR']:
            if self.config['val_fraction'] > 0.0:
                kwargs.update({'validation_fraction': self.config['val_fraction']})
            elif self.config['test_fraction'] > 0.0:
                kwargs.update({'validation_fraction': self.config['test_fraction']})

        # some algorithms allow detailed output during training, this is allowed when self.verbosity is > 1
        if regr_name in ['ONECLASSSVM']:
            kwargs.update({'verbose': True if self.verbosity > 1 else False})

        if regr_name in ['TPOTREGRESSOR', 'TPOTCLASSIFIER']:
            if 'verbosity' not in kwargs:
                kwargs.update({'verbosity': self.verbosity})

        if regr_name in ml_models:
            model = ml_models[regr_name](**kwargs)
        else:
            if regr_name in ['TWEEDIEREGRESSOR', 'POISSONREGRESSOR', 'LGBMREGRESSOR', 'LGBMCLASSIFIER',
                             'GAMMAREGRESSOR']:
                if int(VERSION_INFO['sklearn'].split('.')[1]) < 23:
                    raise ValueError(
                        f"{regr_name} is available with sklearn version >= 0.23 but you have {VERSION_INFO['sklearn']}")
            raise ValueError(f"model {regr_name} not found. {VERSION_INFO}")

        self._model = model

        return

    def fit(self,
            data:str = 'training',
            callbacks:dict=None,
            **kwargs
            ):
        """
        Trains the model with data which is taken from data accoring to `data` arguments.

        Arguments:
            data : data to use for model training. Default is 'training`.
            callbacks : Any callback compatible with keras. If you want to log the output
                to tensorboard, then just use `callbacks={'tensorboard':{}}` or
                to provide additional arguments
                ```python
                callbacks={'tensorboard': {'histogram_freq': 1}}
                ```
            kwargs : Any keyword argument for the `fit` method of the underlying algorithm.
                if 'x' is present in kwargs, that will take precedent over `data`.
        Returns:
            A keras history object in case of deep learning model with tensorflow
            as backend or anything returned by `fit` method of underlying model.
        """

        assert data in ['training', 'test', 'validation']

        return self.call_fit(data=data, callbacks=callbacks, **kwargs)

    def call_fit(self,
                 data='training',
                 callbacks=None,
                 **kwargs):

        visualizer = Visualizations(path=self.path)
        self.is_training = True

        if 'x' not in kwargs:
            train_data = getattr(self, f'{data}_data')()
            inputs, outputs = maybe_three_outputs(train_data, self.dh.teacher_forcing)
        else:
            outputs = None
            if 'y' in kwargs:
                outputs = kwargs['y']
                kwargs.pop('y')
            inputs = kwargs.pop('x')

        if isinstance(outputs, np.ndarray) and self.category.upper() == "DL":
            if isinstance(self.ai4w_outputs, list):
                assert len(self.ai4w_outputs) == 1
                model_output_shape = tuple(self.ai4w_outputs[0].shape.as_list()[1:])

                if getattr(self, 'quantiles', None) is not None:
                    assert model_output_shape[0] == len(self.quantiles) * self.num_outs

                # todo, it is assumed that there is softmax as the last layer
                elif self.problem == 'classification':
                    # todo, don't know why it is working
                    assert model_output_shape[0] == self.num_classes, f"""inferred number of classes are 
                            {self.num_classes} while model's output has {model_output_shape[0]} nodes """
                    assert model_output_shape[0] == outputs.shape[1]
                else:
                    assert model_output_shape == outputs.shape[1:], f"""
    ShapeMismatchError: Shape of model's output is {model_output_shape}
    while the targets in prepared have shape {outputs.shape[1:]}."""

        self.info['training_start'] = dateandtime_now()

        if self.category.upper() == "DL":
            val_data = self.validation_data()
            val_x, val_y = maybe_three_outputs(val_data, self.dh.teacher_forcing)
            val_data = (val_x, val_y)
            history = self._FIT(inputs, outputs, val_data, callbacks=callbacks, **kwargs)

            visualizer.plot_loss(history.history)

            self.load_best_weights()
        else:
            history = self.fit_ml_models(inputs, outputs)

        self.info['training_end'] = dateandtime_now()
        self.save_config()
        save_config_file(os.path.join(self.path, 'info.json'), others=self.info)

        self.is_training = False
        return history

    def load_best_weights(self)->None:
        if self.config['backend'] != 'pytorch':
            # load the best weights so that the best weights can be used during model.predict calls
            best_weights = find_best_weight(os.path.join(self.path, 'weights'))
            if best_weights is None:
                warnings.warn("best weights could not be found and are not loaded", UserWarning)
            else:
                self.allow_weight_loading = True
                self.update_weights(best_weights)
        return

    def fit_ml_models(self, inputs, outputs):

        if self.dh.is_multiclass:
            outputs = outputs
        else:
            outputs = outputs.reshape(-1, )

        history = self._model.fit(inputs, outputs)

        model_name = list(self.config['model'].keys())[0]
        fname = os.path.join(self.w_path, self.category + '_' + self.problem + '_' + model_name)

        if "TPOT" not in model_name.upper():
            joblib.dump(self._model, fname)

        if model_name.lower().startswith("xgb"):
            self._model.save_model(fname + ".json")
        return history

    def cross_val_score(self, scoring='mse'):

        scores = []

        cross_validator = list(self.config['cross_validator'].keys())[0]
        cross_validator_args = self.config['cross_validator'][cross_validator]

        if callable(cross_validator):
            splits = cross_validator(**cross_validator_args)
        else:
            splits = getattr(self.dh, f'{cross_validator}_splits')(**cross_validator_args)

        for fold, ((train_x, train_y), (test_x, test_y)) in enumerate(splits):

            self._model.fit(train_x, y=train_y.reshape(-1, self.num_outs))

            pred = self._model.predict(test_x)

            metrics = RegressionMetrics(test_y.reshape(-1, self.num_outs), pred)
            val_score = getattr(metrics, scoring)()

            scores.append(val_score)

            if self.verbosity>0:
                print(f'fold: {fold} val_loss: {val_score}')

        return np.mean(scores)

    def evaluate(self, data='training', **kwargs):
        """
        Evalutes the performance of the model on a given data.
        calls the `evaluate` method of underlying `model`. If the `evaluate`
        method is not available in underlying `model`, then `predict` is called.
        Arguments:
            data : which data type to use, valid values are `training`, `test`
                and `validation`. You can also provide your own x,y values as keyword
                arguments. In such a case, this argument will have no meaning.
            kwargs : any keyword argument for the `evaluate` method of the underlying
                model.
        Returns:
            whatever is returned by `evaluate` method of underlying model.
        """
        return self.call_evaluate(data, **kwargs)

    def call_evaluate(self, data=None, **kwargs):

        if data:
            assert data in ['training', 'test', 'validation']

            # get the relevant data
            data = getattr(self, f'{data}_data')()
            data = maybe_three_outputs(data, self.dh.teacher_forcing)

        if 'x' in kwargs:  # expecting it to be called by keras' fit loop
            assert data is None

            if self.category == 'ML':
                if hasattr(self._model, 'evaluate'):
                    return self._model.evaluate(kwargs['x'])
                else:
                    return self._model.predict(kwargs['x'])

            return self.evaluate_fn(**kwargs)

        # this will mostly be the validation data.
        elif data is not None:
            # if data.__class__.__name__ in ["Dataset"]:
            #     if 'x' not in kwargs:
            #         #if self.api == 'functional':
            #         eval_output = self.evaluate_fn(self.val_dataset, **kwargs)
            #
            #     else:  # give priority to xy
            #         eval_output = self._evaluate_with_xy(**kwargs)

            #else:
            eval_output = self._evaluate_with_xy(data, **kwargs)

        else:
            raise ValueError

        acc, loss = None, None

        if self.category == "DL":
            if BACKEND == 'tensorflow':
                loss, acc = eval_output
            else:
                loss = eval_output

        eval_report = f"{'*' * 30}\n{dateandtime_now()}\n Accuracy: {acc}\n Loss: {loss}\n"

        fname = os.path.join(self.path, 'eval_report.txt')
        with open(fname, 'a+') as fp:
            fp.write(eval_report)

        return eval_output

    def _evaluate_with_xy(self, data, **kwargs):
        x, y = data

        # the user provided x,y and batch_size values should have priority
        if 'x' in kwargs:
            x = kwargs.pop('x')
        if 'y' in kwargs:
            y = kwargs.pop('y')
        if 'batch_size' in kwargs:
            batch_size = kwargs.pop('batch_size')
        else:
            batch_size = self.config['batch_size']

        y = get_values(y)

        return self.evaluate_fn(
            x=x,
            y=y,
            batch_size=batch_size,
            verbose=self.verbosity,
            **kwargs
        )

    def predict(self,
                data: str='test',
                x=None,
                y=None,
                prefix: str = None,
                process_results:bool = True,
                **kwargs
                )->Tuple[np.ndarray, np.ndarray]:
        """
        Makes prediction from the trained model.
        Arguments:
            data : which data to use. Possible values are `training`, `test` or `validation`.
                By default, `test` data is used for predictions.
            x : if given, it will override `data`
            y : Used for pos-processing etc. if given it will overrite `data`
            process_results : post processing of results
            prefix : prefix used with names of saved results
            kwargs : any keyword argument for `fit` method.
        Returns:
            a tuple of arrays. The first is true and the second is predicted.
            If `x` is given but `y` is not given, then, first array which is
            returned is None.
        """
        assert data in ['training', 'test', 'validation']

        return self.call_predict(data=data, x=x, y=y, process_results=process_results, **kwargs)

    def call_predict(self,
                     data='test',
                     x = None,
                     y = None,
                     process_results=True,
                     **kwargs):

        transformation_key = '5'

        if x is None:
            prefix = data
            data = getattr(self, f'{data}_data')(key=transformation_key)
            inputs, true_outputs = maybe_three_outputs(data, self.dh.teacher_forcing)
        else:
            prefix = 'x'
            inputs = x
            true_outputs = y

        if self.category == 'DL':
            predicted = self.predict_fn(x= inputs,
                                        batch_size = self.config['batch_size'],
                                        verbose= self.verbosity,
                                        **kwargs)
        else:
            predicted = self.predict_fn(inputs, **kwargs)

        if y is None:
            return y, predicted

        true_outputs, predicted = self.inverse_transform(true_outputs, predicted, transformation_key)

        true_outputs, dt_index = self.dh.deindexify(true_outputs, key=transformation_key)

        if isinstance(true_outputs, np.ndarray) and true_outputs.dtype.name == 'object':
            true_outputs = true_outputs.astype(predicted.dtype)

        if self.quantiles is None:

            # it true_outputs and predicted are dictionary of len(1) then just get the values
            true_outputs = get_values(true_outputs)
            predicted = get_values(predicted)

            if process_results:
                if self.problem == 'regression':
                    self.process_regres_results(true_outputs, predicted, prefix=prefix + '_', index=dt_index)
                else:
                    self.process_class_results(true_outputs, predicted, prefix=prefix, index=dt_index)

        else:
            assert self.num_outs == 1
            self.plot_quantiles1(true_outputs, predicted)
            self.plot_quantiles2(true_outputs, predicted)
            self.plot_all_qs(true_outputs, predicted)

        return true_outputs, predicted

    def inverse_transform(self,
                          true:Union[np.ndarray, dict],
                          predicted:Union[np.ndarray, dict],
                          key:str
                          )->Tuple[np.ndarray, np.ndarray]:

        if self.dh.source_is_dict or self.dh.source_is_list:
            true = self.dh.inverse_transform(true, key=key)
            if isinstance(predicted, np.ndarray):
                assert len(true) == 1
                predicted = {list(true.keys())[0]: predicted}
            predicted = self.dh.inverse_transform(predicted, key=key)

        else:
            true_shape, pred_shape = true.shape, predicted.shape
            if isinstance(true, np.ndarray) and self.forecast_len==1 and isinstance(self.num_outs, int):
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

    def plot_model(self, nn_model)->None:
        kwargs = {}
        if int(tf.__version__.split('.')[1]) > 14:
            kwargs['dpi'] = 300

        try:
            keras.utils.plot_model(nn_model, to_file=os.path.join(self.path, "model.png"), show_shapes=True, **kwargs)
        except (AssertionError, ImportError) as e:
            print("dot plot of model could not be plotted")
        return

    def get_opt_args(self)->dict:
        """ get input arguments for an optimizer. It is being explicitly defined here so that it can be overwritten
        in sub-classes"""
        kwargs = {'lr': self.config['lr']}
        if self.config['backend'] == 'pytorch':
            kwargs.update({'params': self.parameters()})  # parameters from pytorch model
        return kwargs

    def get_metrics(self) -> list:
        """ returns the performance metrics specified"""
        _metrics = self.config['metrics']

        metrics = None
        if _metrics is not None:
            if not isinstance(_metrics, list):
                assert isinstance(_metrics, str)
                _metrics = [_metrics]

            from ai4water.utils.tf_losses import nse, kge, pbias, tf_r2

            METRICS = {'NSE': nse,
                       'KGE': kge,
                       "R2": tf_r2,
                       'PBIAS': pbias}

            metrics = []
            for m in _metrics:
                if m.upper() in METRICS.keys():
                    metrics.append(METRICS[m.upper()])
                else:
                    metrics.append(m)
        return metrics

    def get_2d_batches(self, data, ins, outs):
        if not isinstance(data, np.ndarray):
            if isinstance(data, pd.DataFrame):
                data = data.values
            else:
                raise TypeError(f"unknown data type {data.__class__.__name__} for data ")

        # for case when there is not lookback, i.e first layer is dense layer and takes 2D input
        input_x, input_y, label_y = data[:, 0:ins], data[:, -outs:], data[:, -outs:]

        assert self.lookback == 1, """lookback should be one for MLP/Dense layer based model, but it is {}
            """.format(self.lookback)
        return self.check_nans(data, input_x, input_y, np.expand_dims(label_y, axis=2), outs, self.lookback,
                               self.config['allow_nan_labels'])

    def activations(self,
                    layer_names:Union[list, str]=None,
                    x=None,
                    data:str='training')->dict:
        """gets the activations of any layer of the Keras Model.
        Activation here means output of a layer of a deep learning model.
        Arguments:
            layer_names : name of list of names of layers whose activations are
                to be returned.
            x : If provided, it will override `data`.
            data : data to use to get activations. Only relevent if `x` is not
                provided. By default training data is used. Possible values are
                `training`, `test` or `validation`.
        Returns:
            a dictionary whose keys are names of layers and values are weights
            of those layers as numpy arrays
        """

        # if layer names are not specified, this will get get activations of allparameters
        if x is None:
            data = getattr(self, f'{data}_data')()
            x, y = maybe_three_outputs(data, self.dh.teacher_forcing)

        activations = keract.get_activations(self.dl_model, x, layer_names=layer_names, auto_compile=True)

        return activations

    def display_activations(self, layer_name: str = None, x=None, data:str='training', **kwargs):
        # not working currently because it requres the shape of activations to be (1, output_h, output_w, num_filters)
        activations = self.activations(x=x, data=data, layer_names=layer_name)

        assert isinstance(activations, dict)

        if layer_name is None:
            activations = activations
        else:
            activations = activations[layer_name]

        keract.display_activations(activations=activations, **kwargs)

    def gradients_of_weights(self, x=None, y=None, data:str='training') -> dict:

        if x is None:
            data = getattr(self, f'{data}_data')()
            x, y = maybe_three_outputs(data)

        return keract.get_gradients_of_trainable_weights(self.dl_model, x, y)

    def gradients_of_activations(self, x=None, y=None, data:str='training', layer_name=None) -> dict:
        """
        either x,y or data is required
        x = input data. Will overwrite `data`
        y = corresponding label of x. Will overwrite `data`.
        data : one of `training`, `test` or `validation`
        """

        if x is None:
            data = getattr(self, f'{data}_data')()
            x, y = maybe_three_outputs(data)

        return keract.get_gradients_of_activations(self.dl_model, x, y, layer_names=layer_name)

    def trainable_weights_(self):
        """ returns all trainable weights as arrays in a dictionary"""
        weights = {}
        for weight in self.dl_model.trainable_weights:
            if tf.executing_eagerly():
                weights[weight.name] = weight.numpy()
            else:
                weights[weight.name] = keras.backend.eval(weight)
        return weights

    def find_num_lstms(self) -> list:
        """Finds names of lstm layers in model"""

        lstm_names = []
        for lyr, config in self.config['model']['layers'].items():
            if "LSTM" in lyr.upper():
                config = config.get('config', config)
                prosp_name = config.get('name', lyr)

                lstm_names.append(prosp_name)

        return lstm_names

    def get_rnn_weights(self, weights: dict) -> dict:
        """Finds RNN related weights and combine kernel, recurrent curnel and bias
        of each layer into a list."""
        lstm_weights = {}
        if self.config['model'] is not None and 'layers' in self.config['model']:
            if "LSTM" in self.config['model']['layers']:
                lstms = self.find_num_lstms()
                for lstm in lstms:
                    lstm_w = []
                    for w in ["kernel", "recurrent_kernel", "bias"]:
                        w_name = lstm + "/lstm_cell/" + w
                        for k, v in weights.items():
                            if w_name in k:
                                lstm_w.append(v)

                    lstm_weights[lstm] = lstm_w

        return lstm_weights

    def plot_weights(self, save=True):
        weights = self.trainable_weights_()

        if self.verbosity > 0:
            print("Plotting trainable weights of layers of the model.")

        rnn_weights = self.get_rnn_weights(weights)
        for k, w in rnn_weights.items():
            self.rnn_histogram(w, name=k + "_weight_histogram", save=save)

        for _name, weight in weights.items():
            title = _name + " Weights"
            fname = _name + '_weights'

            rnn_args = None
            if "LSTM" in title.upper():
                rnn_args = {'n_gates': 4,
                            'gate_names_str': "(input, forget, cell, output)"}

            if np.ndim(weight) == 2 and weight.shape[1] > 1:

                self._imshow(weight, title, save, fname, rnn_args=rnn_args, where='weights')

            elif len(weight) > 1 and np.ndim(weight) < 3:
                self.plot1d(weight, title, save, fname, rnn_args=rnn_args, where='weights')

            elif "conv" in _name.lower() and np.ndim(weight) == 3:
                _name = _name.replace("/", "_")
                _name = _name.replace(":", "_")
                self.features_2d(data=weight, save=save, name=_name, where='weights',
                                 slices=64, slice_dim=2, tight=True, borderwidth=1,
                                 norm=(-.1, .1))
            else:
                print("ignoring weight for {} because it has shape {}".format(_name, weight.shape))

    def plot_layer_outputs(self, save: bool = True, lstm_activations=False, x=None, data:str='training'):
        """Plots outputs of intermediate layers except input and output.
        If called without any arguments then it will plot outputs of all layers.
        By default do not plot LSTM activations."""
        activations = self.activations(x=x, data=data)

        if self.verbosity > 0:
            print("Plotting activations of layers")

        for lyr_name, activation in activations.items():
            # activation may be tuple e.g if input layer receives more than 1 input
            if isinstance(activation, np.ndarray):
                self._plot_layer_outputs(activation, lyr_name, save, lstm_activations=lstm_activations)

            elif isinstance(activation, tuple):
                for act in activation:
                    self._plot_layer_outputs(act, lyr_name, save)
        return

    def _plot_layer_outputs(self, activation, lyr_name, save, lstm_activations=False):

        if "LSTM" in lyr_name.upper() and np.ndim(activation) in (2, 3) and lstm_activations:

            self.features_2d(activation, save=save, name=lyr_name + "_outputs", title="Outputs", norm=(-1, 1))

        elif np.ndim(activation) == 2 and activation.shape[1] > 1:
            self._imshow(activation, lyr_name + " Outputs", save, lyr_name)
        elif np.ndim(activation) == 3:
            self._imshow_3d(activation, lyr_name, save=save)
        elif np.ndim(activation) == 2:  # this is now 1d
            # shape= (?, 1)
            self.plot1d(activation, label=lyr_name + ' Outputs', save=save,
                        fname=lyr_name + '_outputs')
        else:
            print("ignoring activations for {} because it has shape {}, {}".format(lyr_name, activation.shape,
                                                                                   np.ndim(activation)))
        return

    def plot_weight_grads(self, save: bool = True, **kwargs):
        """ plots gradient of all trainable weights"""

        gradients = self.gradients_of_weights(**kwargs)

        if self.verbosity > 0:
            print("Plotting gradients of trainable weights")

        rnn_weights = self.get_rnn_weights(gradients)
        for k, w in rnn_weights.items():
            self.rnn_histogram(w, name=k + "_weight_grads_histogram", save=save)

        for lyr_name, gradient in gradients.items():

            title = lyr_name + "Weight Gradients"
            fname = lyr_name + '_weight_grads'
            rnn_args = None

            if "LSTM" in title.upper():
                rnn_args = {'n_gates': 4,
                            'gate_names_str': "(input, forget, cell, output)"}

                if np.ndim(gradient) == 3:
                    self.rnn_histogram(gradient, name=fname, title=title, save=save)

            if np.ndim(gradient) == 2 and gradient.shape[1] > 1:
                self._imshow(gradient, title, save, fname, rnn_args=rnn_args)

            elif len(gradient) and np.ndim(gradient) < 3:
                self.plot1d(gradient, title, save, fname, rnn_args=rnn_args)
            else:
                print("ignoring weight gradients for {} because it has shape {} {}".format(lyr_name, gradient.shape,
                                                                                           np.ndim(gradient)))
            return

    def plot_act_grads(self, save: bool = True, **kwargs):
        """ plots activations of intermediate layers except input and output"""
        gradients = self.gradients_of_activations(**kwargs)
        return self._plot_act_grads(gradients, save=save)

    def _plot_act_grads(self, gradients, save=True):
        if self.verbosity > 0:
            print("Plotting gradients of activations of layersr")

        for lyr_name, gradient in gradients.items():
            fname = lyr_name + "_output_grads"
            title = lyr_name + " Output Gradients"
            if "LSTM" in lyr_name.upper() and np.ndim(gradient) in (2, 3):

                self.features_2d(gradient, name=fname, title=title, save=save, n_rows=8, norm=(-1e-4, 1e-4))
                self.features_1d(gradient[0], show_borders=False, name=fname, title=title, save=save, n_rows=8)

                if gradient.ndim == 2:
                    self.features_0d(gradient, name=fname, title=title, save=save)

            elif np.ndim(gradient) == 2:
                if gradient.shape[1] > 1:
                    # (?, ?)
                    self._imshow(gradient, title, save, fname)
                elif gradient.shape[1] == 1:
                    # (? , 1)
                    self.plot1d(np.squeeze(gradient), title, save, fname)

            elif np.ndim(gradient) == 3 and gradient.shape[1] == 1:
                if gradient.shape[2] == 1:
                    # (?, 1, 1)
                    self.plot1d(np.squeeze(gradient), title, save, fname)
                else:
                    # (?, 1, ?)
                    self._imshow(np.squeeze(gradient), title, save, fname)
            elif np.ndim(gradient) == 3:
                if gradient.shape[2] == 1:
                    # (?, ?, 1)
                    self._imshow(np.squeeze(gradient), title, save, fname)
                elif gradient.shape[2] > 1:
                    # (?, ?, ?)
                    self._imshow_3d(gradient, lyr_name, save)
            else:
                print("ignoring activation gradients for {} because it has shape {} {}".format(lyr_name, gradient.shape,
                                                                                               np.ndim(gradient)))

    def view_model(self, **kwargs):
        """shows all activations, weights and gradients of the model.

        Arguments:
            kwargs : keyword arguments for specifying the data. These arguments
                must be same which are used by `fit` method for specifying data.
        """

        if self.category.upper() == "DL":
            self.plot_act_grads(**kwargs)
            self.plot_weight_grads(**kwargs)
            self.plot_layer_outputs(**kwargs)
            self.plot_weights()

        return

    def interpret(self, save=True, **kwargs):
        """
        Interprets the underlying model. Call it after training.

        Example
        -------
        ```python
        model.fit()
        model.interpret()
        ```
        """
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        if 'layers' not in self.config['model']:

            self.plot_treeviz_leaves()

            if self.problem.lower().startswith("cl"):
                self.plot_treeviz_leaves()
                self.decision_tree(which="sklearn", **kwargs)

                data = self.test_data()
                x, y = maybe_three_outputs(data)
                self.confusion_matrx(x=x, y=y)
                self.precision_recall_curve(x=x, y=y)
                self.roc_curve(x=x, y=y)

            if list(self.config['model'].keys())[0].lower().startswith("xgb"):
                self.decision_tree(which="xgboost", **kwargs)

        interpreter = Interpret(self)

        if self.category == 'ML':
            interpreter.plot_feature_importance(save=save)

        return

    def plot_act_along_lookback(self, activations, sample=0):

        assert isinstance(activations, np.ndarray)

        activation = activations[sample, :, :]
        act_t = activation.transpose()

        fig, axis = plt.subplots()

        for idx, _name in enumerate(self.in_cols):
            axis.plot(act_t[idx, :], label=_name)

        axis.set_xlabel('Lookback')
        axis.set_ylabel('Input attention weight')
        axis.legend(loc="best")

        plt.show()
        return

    def inputs_for_attention(self, inputs):
        """ it is being provided separately so that it can be overwritten for cases when attention mechanism depends
        on more than one inputs or the model applying attention has more than one inputs. """
        if isinstance(inputs, list):
            inputs = inputs[0]

        inputs = inputs[:, -1, :]  # why 0, why not -1

        assert inputs.shape[1] == self.num_ins

        return inputs

    def prepare_batches(self, df: pd.DataFrame, ins, outs):

        assert self.num_outs == 1
        target = self.config['output_features'][0]

        x = np.zeros((len(df), self.lookback, df.shape[1] - 1))
        prev_y = np.zeros((len(df), self.lookback, 1))

        for i, name in enumerate(list(df.columns[:-1])):
            for j in range(self.lookback):
                x[:, j, i] = df[name].shift(self.lookback - j - 1).fillna(method="bfill")

        for j in range(self.lookback):
            prev_y[:, j, 0] = df[target].shift(self.lookback - j - 1).fillna(method="bfill")

        fl = self.config['forecast_len']
        _y = np.zeros((df.shape[0], fl))
        for i in range(df.shape[0] - fl):
            _y[i - 1, :] = df[target].values[i:i + fl]

        input_x = x[self.lookback:-fl, :]
        prev_y = prev_y[self.lookback:-fl, :]
        y = _y[self.lookback:-fl, :].reshape(-1, outs, self.forecast_len)

        return self.check_nans(df, input_x, prev_y, y, outs, self.lookback, self.config['allow_nan_labels'])

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
        save_config_file(indices=indices, path=self.path)
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

        config['config'] = self.config
        config['method'] = self.method
        config['category'] = self.category
        config['problem'] = self.problem
        config['quantiles'] = self.quantiles

        if self.category == "DL":
            config['loss'] = self.loss_name()

        save_config_file(config=config, path=self.path)
        return config

    @classmethod
    def from_config(cls,
                    config_path: str,
                    data,
                    make_new_path: bool = False,
                    **kwargs)->"BaseModel":
        """
        Loads the model from a config file.

        Arguments:
            config_path : complete path of config file
            data : data for Model
            make_new_path : If true, then it means we want to use the config
                file, only to build the model and a new path will be made. We
                would not normally update the weights in such a case.
            kwargs : any additional keyword arguments for the `Model`
        return:
            a `Model` instance
        """
        config, path = cls._get_config_and_path(cls, config_path, make_new_path)

        return cls(**config['config'],
                   data=data,
                   path=path,
                   **kwargs)

    @staticmethod
    def _get_config_and_path(cls, config_path, make_new_path):
        """Sets some attributes of the cls so that it can be built from config.
        Also fetches config and path which are used to initiate cls."""
        with open(config_path, 'r') as fp:
            config = json.load(fp)

        if 'path' in config['config']: config['config'].pop('path')

        idx_file = os.path.join(os.path.dirname(config_path), 'indices.json')
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
            path = os.path.dirname(config_path)

        return config, path

    def update_weights(self, weight_file: str):
        """
        Updates the weights of the underlying model.
        Arguments:
            weight_file str: name of file which contains parameters of model.
        """
        weight_file_path = os.path.join(self.w_path, weight_file)
        if not self.allow_weight_loading:
            raise ValueError(f"Weights loading not allowed because allow_weight_loading is {self.allow_weight_loading}"
                             f"and model path is {self.path}")

        if self.category == "ML":
            if list(self.config['model'].keys())[0].lower().startswith("xgb"):
                self._model.load_model(weight_file_path)
            else:
                # for sklearn based models
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

    def eda(self, freq=None, cols=None):
        """Performs comprehensive Exploratory Data Analysis.
        freq: str, if specified, small chunks of data will be plotted instead of whole data at once. The data will NOT
        be resampled. This is valid only `plot_data` and `box_plot`. Possible values are `yearly`, weekly`, and
        `monthly`."""
        if self.data is None:
            print("data is None so eda can not be performed.")
            return
        # todo, radial heatmap to show temporal trends http://holoviews.org/reference/elements/bokeh/RadialHeatMap.html
        visualizer = Visualizations(data=self.data, path=self.path, in_cols=self.in_cols, out_cols=self.out_cols)

        # plot number if missing vals
        visualizer.plot_missing(cols=cols)

        # show data as heatmapt
        visualizer.heatmap(cols=cols)

        # line plots of input/output data
        visualizer.plot_data(cols=cols, freq=freq, subplots=True, figsize=(12, 14), sharex=True)

        # plot feature-feature correlation as heatmap
        visualizer.feature_feature_corr(cols=cols)

        # print stats about input/output data
        self.stats()

        # box-whisker plot
        self.box_plot(freq=freq)

        # principle components
        visualizer.plot_pcs()

        # scatter plots of input/output data
        visualizer.grouped_scatter(cols=cols)

        # distributions as histograms
        visualizer.plot_histograms(cols=cols)

        return

    def stats(self, precision=3, inputs=True, outputs=True, fpath=None, out_fmt="csv"):
        """Finds the stats of inputs and outputs and puts them in a json file.
        inputs: bool
        fpath: str, path like
        out_fmt: str, in which format to save. csv or json"""
        cols = []
        fname = "data_description_"
        if inputs:
            cols += self.in_cols
            fname += "inputs_"
        if outputs:
            cols += self.out_cols
            fname += "outputs_"

        fname += str(dateandtime_now())

        def save_stats(_description, _fpath):

            if out_fmt == "csv":
                pd.DataFrame.from_dict(_description).to_csv(_fpath + ".csv")
            else:
                save_config_file(others=_description, path=_fpath + ".json")

        description = {}
        if isinstance(self.data, pd.DataFrame):
            description = {}
            for col in cols:
                if col in self.data:
                    description[col] = ts_features(self.data[col], precision=precision, name=col)

            fpath = os.path.join(self.data_path, fname) if fpath is None else fpath
            save_stats(description, fpath)

        elif isinstance(self.data, list):
            description = {}

            for idx, data in enumerate(self.data):
                _description = {}

                if isinstance(data, pd.DataFrame):

                    for col in cols:
                        if col in data:
                            _description[col] = ts_features(data[col], precision=precision, name=col)

                description['data' + str(idx)] = _description
                _fpath = os.path.join(self.data_path, fname + f'_{idx}') if fpath is None else fpath
                save_stats(_description, _fpath)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                _description = {}
                if isinstance(data, pd.DataFrame):
                    for col in data.columns:
                        _description[col] = ts_features(data[col], precision=precision, name=col)

                description[f'data_{data_name}'] = _description
                _fpath = os.path.join(self.data_path, fname + f'_{data_name}') if fpath is None else fpath
                save_stats(_description, _fpath)
        else:
            print(f"description can not be found for data type of {self.data.__class__.__name__}")

        return description

    def update_info(self):

        VERSION_INFO.update({'numpy': str(np.__version__),
                             'pandas': str(pd.__version__),
                             'matplotlib': str(matplotlib.__version__),
                             'h5py': h5py.__version__ if h5py is not None else None,
                            'joblib': joblib.__version__})
        self.info['version_info'] = VERSION_INFO
        return

    def print_info(self):
        class_type = None
        if self.is_binary:
            class_type = "binary"
        elif self.is_multiclass:
            class_type = "multi-class"
        elif self.is_multilabel:
            class_type = "multi-label"

        if self.verbosity > 0:
            print('building {} model for {} {} problem'.format(self.category, class_type, self.problem))
        return

    def get_optimizer(self):
        opt_args = self.get_opt_args()
        optimizer = OPTIMIZERS[self.config['optimizer'].upper()](**opt_args)

        return optimizer


def print_something(something, prefix=''):
    """prints shape of some python object"""
    if isinstance(something, np.ndarray):
        print(f"{prefix} shape: ", something.shape)
    elif isinstance(something, list):
        print(f"{prefix} shape: ", [thing.shape for thing in something if isinstance(thing, np.ndarray)])
    elif isinstance(something, dict):
        print(f"{prefix} shape: ")
        pprint.pprint({k: v.shape for k, v in something.items()}, width=40)


def maybe_three_outputs(data, teacher_forcing=False):
    """num_outputs: how many outputs from data we want"""
    if teacher_forcing:
        num_outputs = 3
    else:
        num_outputs = 2

    if num_outputs == 2:
        if len(data) == 2:
            return data[0], data[1]
        elif len(data) == 3:
            return data[0], data[2]
    else:
        return [data[0], data[1]], data[2]


def get_values(outputs):

    if isinstance(outputs, dict) and len(outputs) == 1:
        outputs = list(outputs.values())[0]

    return outputs