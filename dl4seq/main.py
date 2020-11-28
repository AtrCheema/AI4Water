import numpy as np
import pandas as pd
from TSErrors import FindErrors
import matplotlib.pyplot as plt
import matplotlib # for version info
import os
from sklearn.model_selection import train_test_split
import json
import h5py
import random
import math

from dl4seq.nn_tools import NN
from dl4seq.backend import tf, keras, tcn, VERSION_INFO
from dl4seq.utils.utils import plot_results, plot_loss, maybe_create_path, save_config_file, get_index, get_sklearn_models
from dl4seq.utils.plotting_tools import Plots
from dl4seq.utils.scalers import Scalers

def reset_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


if tf is not None:
    import dl4seq.keract_mod as keract
    from dl4seq.tf_attributes import LOSSES, OPTIMIZERS

class Model(NN, Plots):
    """

    data_config: dict
    nn_config: dict
    data: pd.Dataframe or any other conforming type
    prefix: str, prefix to be used for the folder in which the results are saved
    path: str/path like, if given, new path will not be created
    verbosity: int, determines the amount of information being printed
    category: str, one of "DL" or "ML", signifying, whether a machine learning model or deep learning model
    problem: str, tell whether it is is classification problem or regression problem.

    """
    def __init__(self, data_config: dict,
                 nn_config: dict,
                 data=None,
                 intervals=None,
                 prefix: str = None,
                 path: str = None,
                 verbosity=1,
                 category="DL",
                 problem="reg"):

        reset_seed(data_config['seed'])
        if tf is not None:
            # graph should be cleared everytime we build new `Model` otherwise, if two `Models` are prepared in same
            # file, they may share same graph.
            tf.keras.backend.clear_session()

        super(Model, self).__init__(data_config, nn_config)

        self.intervals = intervals
        self.data = data
        self.in_cols = self.data_config['inputs']
        self.out_cols = self.data_config['outputs']
        self.loss = LOSSES[self.nn_config['loss'].upper()]
        self.KModel = keras.models.Model if keras is not None else None
        self.path, self.act_path, self.w_path = maybe_create_path(path=path, prefix=prefix)
        self.verbosity = verbosity
        self.category = category
        self.problem = problem

    @property
    def forecast_step(self):
        # should only be changed using self.data_config
        return self.data_config['forecast_step']

    @property
    def forecast_len(self):
        # should only be changed using self.data_config
        return self.data_config['forecast_length']

    @property
    def data(self):
        # so that .data attribute can be customized
        return self._data

    @data.setter
    def data(self, x):
        if isinstance(x, pd.DataFrame):
            _data = x[self.data_config['inputs'] + self.data_config['outputs']]
        else:
            _data = x
        self._data = _data

    @property
    def intervals(self):
        # so that itnervals can be set from outside the Model class. A use case can be when we want to implement
        # intervals during train time but not during test/predict time.
        return self._intervals

    @intervals.setter
    def intervals(self, x: list):
        self._intervals = x

    @property
    def in_cols(self):
        return self._in_cols

    @in_cols.setter
    def in_cols(self, x: list):
        self._in_cols = x

    @property
    def out_cols(self):
        return self._out_cols

    @out_cols.setter
    def out_cols(self, x: list):
        self._out_cols = x

    @property
    def ins(self):
        return len(self.in_cols)

    @property
    def outs(self):
        return len(self.out_cols)

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, x):
        self._loss = x

    @property
    def KModel(self):
        """In case when we want to customize the model such as for
        implementing custom `train_step`, we can provide the customized model as input the this Model class"""
        return self._k_model

    @KModel.setter
    def KModel(self, x):
        self._k_model = x

    @property
    def layer_names(self):
        _all_layers = []
        for layer in self._model.layers:
            _all_layers.append(layer.name)
        return _all_layers

    @property
    def layers_in_shapes(self) -> dict:
        """ returns the shapes of inputs to all layers"""
        shapes = {}

        for lyr in self._model.layers:
            shapes[lyr.name] = lyr.input_shape

        return shapes

    @property
    def layers_out_shapes(self) -> dict:
        """ returns shapes of outputs from all layers in model as dictionary"""
        shapes = {}

        for lyr in self._model.layers:
            shapes[lyr.name] = lyr.output_shape

        return shapes

    @property
    def num_input_layers(self) -> int:
        if self.category.upper() != "DL":
            return np.inf
        else:
            return len(self._model.inputs)

    def feature_imporance(self):
        if self.category.upper() == "ML":
            return self._model.feature_importances_

    def fetch_data(self, data: pd.DataFrame, st: int = 0, en=None,
                   shuffle: bool = False,  # TODO, is this arg requried?
                   write_data=False,
                   noise: int = 0,
                   indices: list = None,
                   scaler_key: str = '0',
                   use_datetime_index=False):
        """
        :param data:
        :param st:
        :param en:
        :param shuffle:
        :param write_data: writes the created batches/generated data in h5 file. The filename will be data_scaler_key.h5
        :param noise:
        :param indices:
        :param scaler_key: in case we are calling fetch_data multiple times, each data will be scaled with a unique
                   MinMaxScaler object and can be saved with a unique key in memory.
        :param use_datetime_index: if True, first value in returned `x` will be datetime index. This can be used when
                                fetching data during predict but must be separated before feeding in NN for prediction.
        :return:
        """

        if indices is not None:
            assert isinstance(indices, list), "indices must be list"
            if en is not None or st != 0:
                raise ValueError
        if en is None:
            en = data.shape[0]

        df = self.indexify_data(data, use_datetime_index)

        # # add random noise in the data
        df = self.add_noise(df, noise)


        if self.data_config['normalize']:  # TODO when train_dataand test_data are externally set, normalization can't be done.
            df, _ = self.normalize(df, scaler_key)

        if self.intervals is None:

            df = df[st:en]
            df.columns = self.in_cols + self.out_cols

            x, y, label = self.get_batches(df,
                                           len(self.in_cols),
                                           len(self.out_cols)
                                           )
            if indices is not None:
                # because the x,y,z have some initial values removed
                indices = np.subtract(np.array(indices), self.lookback - 1).tolist()

                # if indices are given then this should be done after `get_batches` method
                x = x[indices]
                y = y[indices]
                label = label[indices]
        else:
            xs, ys, labels = [], [], []
            for _st, _en in self.intervals:
                df1 = df[_st:_en]

                df.columns = self.in_cols + self.out_cols
                df1.columns = self.in_cols + self.out_cols

                if df1.shape[0] > 0:
                    x, y, label = self.get_batches(df1,
                                                   len(self.in_cols),
                                                   len(self.out_cols))
                    xs.append(x)
                    ys.append(y)
                    labels.append(label)

            if indices is None:
                x = np.vstack(xs)[st:en]
                y = np.vstack(ys)[st:en]
                label = np.vstack(labels)[st:en]
            else:
                x = np.vstack(xs)[indices]
                y = np.vstack(ys)[indices]
                label = np.vstack(labels)[indices]

        if shuffle:
            x, y, label = unison_shuffled_copies(x, y, label)

        x = self.conform_shape(x, datetime_index=use_datetime_index)

        if use_datetime_index:
            self.in_cols.remove("dt_index")
        else:
            x = x.astype(np.float32)

        if write_data:
            self.write_cache('data_' + scaler_key, x, y, label)

        return x, y, label

    def indexify_data(self, data, use_datetime_index):

        if use_datetime_index:
            assert isinstance(data.index, pd.DatetimeIndex), """\nInput dataframe must have index of type
             pd.DateTimeIndex. A dummy datetime index can be inserted by using following command:
            `data.index = pd.date_range("20110101", periods=len(data), freq='H')`
            If the data file already contains `datetime` column, then use following command
            `data.index = pd.to_datetime(data['datetime'])`
            or use set `use_datetime_index` in `predict` method to False.
            """
            dt_index = list(map(int, np.array(data.index.strftime('%Y%m%d%H%M'))))  # datetime index
            # pandas will add the 'datetime' column as first column. This columns will only be used to keep
            # track of indices of train and test data.
            data.insert(0, 'dt_index', dt_index)
            self.in_cols = ['dt_index'] + self.in_cols

        return data

    def add_noise(self, df, noise):

        if noise > 0:
            x = pd.DataFrame(np.random.randint(0, 1, (len(df), noise)))
            df = pd.concat([df, x], axis=1)
            prev_inputs = self.in_cols
            self.in_cols = prev_inputs + list(x.columns)
            ys = []
            for y in self.out_cols:
                ys.append(df.pop(y))
            df[self.out_cols] = ys

        return df

    def normalize(self, df, key):
        """ should return the transformed dataframe and the key with which scaler is put in memory. """
        scaler = None
        transformation = self.data_config['normalize']
        if transformation is not None:

            if isinstance(transformation, dict):
                df, scaler = Scalers(data=df, **transformation)('normalize')
            else:
                assert isinstance(transformation, str)
                df, scaler = Scalers(data=df, method=transformation)('normalize')
            self.scalers[key] = scaler

        return df, scaler

    def check_batches(self, x, prev_y, y: np.ndarray):
        """
        Will have no effect if drop_remainder is False or batches_per_epoch are not specified.
        :param x: a list consisting of one or more 1D arrays
        :param prev_y:
        :param y: nd array
        :return:
        """
        steps_per_epoch = self.data_config['batches_per_epoch']
        if steps_per_epoch is None:

            examples = x[0].shape[0] if isinstance(x, list) else x.shape[0]

            if self.data_config['drop_remainder']:
                iterations = math.floor(examples/self.data_config['batch_size'])
                return self._check_batches(x, prev_y, y, iterations)

            return x, prev_y, y
        else:

             return self._check_batches(x, prev_y, y, steps_per_epoch)

    def _check_batches(self, x, prev_y, y, iterations):

        if self.verbosity > 0:
            print(f"Number of total batches are {iterations}")

        x_is_list = True
        if not isinstance(x, list):
            x = [x]
            x_is_list = False

        assert isinstance(y, np.ndarray)

        if prev_y is None:
            prev_y = y.copy()

        batch_size = self.data_config['batch_size']
        _x = [[None for _ in range(iterations)] for _ in range(len(x))]
        _prev_y = [None for _ in range(iterations)]
        _y = [None for _ in range(iterations)]

        st, en = 0, batch_size
        for batch in range(iterations):
            for ex in range(len(x)):

                _x[ex][batch] = x[ex][st:en, :]


            _prev_y[batch] = prev_y[st:en, :]
            _y[batch] = y[st:en, :]

            st += batch_size
            en += batch_size

        x = [np.vstack(_x[i]) for i in range(len(_x))]
        prev_y = np.vstack(_prev_y)
        y = np.vstack(_y)

        if not x_is_list:
            x = x[0]

        return x, prev_y, y

    def conform_shape(self, x, datetime_index):
        """
    makes sure that the shape of x corresponds to first layer of NN. This comes handy if the first
    layer is CNN in CNNLSTM case or first layer is Conv2DLSTM or LSTM or a dense layer. No change in shape
    will be performed when more than one `Input` layers exist.
        """
        input_layers = self.num_input_layers

        if input_layers > 1:
            return x
        elif input_layers == 1:

            first_layer_shape = self.first_layer_shape()
            if datetime_index:
                first_layer_shape[-1] = first_layer_shape[-1] + 1
            return x.reshape(tuple(first_layer_shape))

        else:
            raise ValueError(" {} Input layers found".format(input_layers))

    def first_layer_shape(self):
        """ instead of tuple, returning a list so that it can be moified if needed"""
        if self.num_input_layers > 1:
            return None
        shape = []
        for idx, d in enumerate(self._model.layers[0].input.shape):
            if int(tf.__version__[0]) == 1:
                if isinstance(d, tf.Dimension):  # for tf 1.x
                    d = d.value

            if idx == 0:  # the first dimension must remain undefined so that the user may define batch_size
                d = -1
            shape.append(d)
        return shape

    def get_callbacks(self, **callbacks):

        _callbacks = list()

        _monitor = 'val_loss' if self.data_config['val_fraction'] > 0.0 else 'loss'
        fname = "{val_loss:.4f}.hdf5" if self.data_config['val_fraction'] > 0.0 else "{loss:.5f}.hdf5"
        if self.nn_config['save_model']:
            _callbacks.append(keras.callbacks.ModelCheckpoint(
                filepath=self.w_path + "\\weights_{epoch:03d}_" + fname,
                save_weights_only=True,
                monitor=_monitor,
                mode='min',
                save_best_only=True))

        _callbacks.append(keras.callbacks.EarlyStopping(
            monitor=_monitor, min_delta=self.nn_config['min_val_loss'],
            patience=self.nn_config['patience'], verbose=0, mode='auto'
        ))

        if 'tensorboard' in callbacks:
            _callbacks.append(keras.callbacks.TensorBoard(log_dir=self.path, histogram_freq=1))
            callbacks.pop('tensorboard')

        for key, val in callbacks.items():
            _callbacks.append(val)

        return _callbacks

    def fit(self, inputs, outputs, validation_data, **callbacks):

        callbacks = self.get_callbacks(**callbacks)

        self._model.fit(inputs,
                         outputs,
                         epochs=self.nn_config['epochs'],
                         batch_size=self.data_config['batch_size'],
                         validation_split=self.data_config['val_fraction'],
                         validation_data=validation_data,
                         callbacks=callbacks,
                         shuffle=self.nn_config['shuffle'],
                         steps_per_epoch=self.data_config['steps_per_epoch'],
                         verbose=self.verbosity
                         )
        history = self._model.history

        self.save_config(history.history)

        # save all the losses or performance metrics
        df = pd.DataFrame.from_dict(history.history)
        df.to_csv(os.path.join(self.path, "losses.csv"))

        return history

    def get_indices(self, indices=None):
        # returns only train indices if `indices` is None
        if isinstance(indices, str) and indices.upper() == 'RANDOM':
            if self.data_config['ignore_nans']:
                tot_obs = self.data.shape[0]
            else:
                if self.outs == 1:
                    # TODO it is being supposed that intervals correspond with Nans. i.e. all values outside intervals
                    # are NaNs. However this can be wrong when the user just wants to skip some chunks of data even
                    # though they are not NaNs. In this case, there will be no NaNs and tot_obs will be more than
                    # required. In such case we have to use `self.vals_in_intervals` to calculate tot_obs. But that
                    # creates problems when larger intervals are provided. such as [NaN, NaN, 1, 2, 3, NaN] we provide
                    # (0, 5) instead of (2, 4). Is it correct/useful to provide (0, 5)?
                    more = len(self.intervals) * self.lookback if self.intervals is not None else self.lookback
                    tot_obs = self.data.shape[0] - int(self.data[self.out_cols].isna().sum()) - more
                else:
                    # data contains nans and target series are > 1, we want to make sure that they have same nan counts
                    tot_obs = self.data.shape[0] - int(self.data[self.out_cols[0]].isna().sum())
                    nans = self.data[self.out_cols].isna().sum()
                    assert np.all(nans.values == int(nans.sum() / self.outs)), f"toal nan values in data are {nans}."

            idx = np.arange(tot_obs - self.lookback)
            train_indices, test_idx = train_test_split(idx, test_size=self.data_config['test_fraction'],
                                                       random_state=self.data_config['seed'])
            setattr(self, 'test_indices', list(test_idx))
            indices = list(train_indices)

        setattr(self, 'train_indices', indices)

        return indices

    def vals_in_intervals(self):
        """
        Supposing that when intervals are present than nans are present only outsite the intervals and No Nans are
        present within intervals.
        Not implementing it for the time being when self.outs>1.
        """
        if self.intervals is not None:
            interval_length = 0
            for interval in self.intervals:
                interval_length += interval[1] - interval[0]
        else:
            interval_length = self.data.shape[0]

        return interval_length

    def train_data(self, **kwargs):
        """ prepare data on which to train the NN"""
        x, y, label = self.fetch_data(self.data, **kwargs)

        x, y, label = self.check_batches(x, y, label)

        if self.verbosity > 0:
            print('input_X shape:', x.shape)
            print('prev_Y shape:', y.shape)
            print('label shape:', label.shape)

        return [x], label

    def maybe_not_3d_data(self, true, predicted):
        if true.ndim ==2:
            assert self.forecast_len == 1
            axis = 2 if true.ndim == 2 else (1, 2)
            true = np.expand_dims(true, axis=axis)
        if predicted.ndim < 3:
            assert self.forecast_len == 1
            axis = 2 if predicted.ndim == 2 else (1,2)
            predicted = np.expand_dims(predicted, axis=axis)
        return true, predicted

    def process_results(self, true: np.ndarray, predicted: np.ndarray, prefix=None, index=None, **plot_args):
        """
        predicted, true are arrays of shape (examples, outs, forecast_len)
        """
        # for cases if they are 2D/1D, add the third dimension.
        true, predicted = self.maybe_not_3d_data(true, predicted)

        errs = dict()

        for idx, out in enumerate(self.out_cols):
            for h in range(self.forecast_len):

                t = pd.DataFrame(true[:, idx, h], index=index, columns=[out])
                p = pd.DataFrame(predicted[:, idx, h], index=index, columns=[out])
                df = pd.concat([t, p], axis=1)
                fname = prefix + '_' + out + '_' + str(h) +  ".csv"
                df.to_csv(os.path.join(self.path, fname), index_label='time')

                errors = FindErrors(t, p, warn="ignore")
                errs[out + '_errors_' + str(h)] = errors.calculate_all()
                errs[out + '_stats_' + str(h)] = errors.stats()

                plot_results(t, p, name=os.path.join(self.path, prefix + out + '_' + str(h)),
                             **plot_args)

        save_config_file(self.path, errors=errs, name=prefix)

        return

    def build(self):

        if self.verbosity > 0:
            print('building {} layer based model for {} problem'.format(self.category, self.problem))

        if self.category.upper() == "DL":
            inputs, predictions = self.add_layers(self.nn_config['layers'])

            self._model = self.compile(inputs, predictions)

        else:
            self.build_ml_model()

        if self.verbosity > 0:
            if 'tcn' in self.nn_config['layers']:
                tcn.tcn_full_summary(self._model, expand_residual_blocks=True)

        return

    def build_ml_model(self):
        """ currently only sklearn based  ML models. """
        
        regr_name = self.nn_config['ml_model'].upper()
        sklearn_models = get_sklearn_models()
        regr = sklearn_models[regr_name](**self.nn_config['sklearn_model_args'])
        self._model = regr

        return

    def val_data(self, **kwargs):
        """ This method can be overwritten in child classes. """
        val_data = None
        if self.data_config['val_data'] is not None:
            if isinstance(self.data_config['val_data'], str):
                val_data = NotImplementedError("Define the method `val_data` in your model class.")
            else:
                # `val_data` might have been provided (then use it as it is) or it is None
                val_data = self.data_config['val_data']

        return val_data

    def train(self, st=0, en=None, indices=None, **callbacks):

        indices = self.get_indices(indices)

        inputs, outputs = self.train_data(st=st, en=en, indices=indices)

        if self.category.upper() == "DL":
            history = self.fit(inputs, outputs, self.val_data(), **callbacks)

            plot_loss(history.history, name=os.path.join(self.path, "loss_curve"))

        else:
            history = self._model.fit(*inputs, outputs.reshape(-1, ))

        return history

    def test_data(self, **kwargs):
        """ just providing it so that it can be overwritten in sub-classes."""
        return self.train_data(**kwargs)

    def prediction_step(self, inputs):
        if self.category.upper() == "DL":
            predicted = self._model.predict(x=inputs,
                                             batch_size=self.data_config['batch_size'],
                                             verbose=self.verbosity)
        else:
            predicted = self._model.predict(*inputs)

        return predicted

    def predict(self, st=0, en=None, indices=None, scaler_key: str = '5', pref: str = 'test',
                use_datetime_index=False, pp=True, **plot_args):
        """
        scaler_key: if None, the data will not be indexed along date_time index.
        pp: post processing
        """

        if indices is not None:
            setattr(self, 'predict_indices', indices)

        inputs, true_outputs = self.test_data(st=st, en=en, indices=indices, scaler_key=scaler_key,
                                              use_datetime_index=use_datetime_index)

        first_input, inputs, dt_index = self.deindexify_input_data(inputs, use_datetime_index=use_datetime_index)

        predicted = self.prediction_step(inputs)

        if self.problem.upper().startswith("CLASS"):
            self.plot_roc_curve(inputs, true_outputs)

        predicted, true_outputs = self.denormalize_data(first_input, predicted, true_outputs, scaler_key)

        if self.quantiles is None:

            if pp:
                self.process_results(true_outputs, predicted, prefix=pref + '_', index=dt_index, **plot_args)

        else:
            assert self.outs == 1
            self.plot_quantiles1(true_outputs, predicted)
            self.plot_quantiles2(true_outputs, predicted)
            self.plot_all_qs(true_outputs, predicted)

        return true_outputs, predicted

    def denormalize_data(self, inputs: np.ndarray, predicted: np.ndarray, true: np.ndarray, scaler_key:str):
        """
        predicted, true are arrays of shape (examples, outs, forecast_len)
        """
        # for cases if they are 2D, add the third dimension.
        true, predicted = self.maybe_not_3d_data(true, predicted)

        if self.data_config['normalize']:
            if np.ndim(inputs) == 4:
                inputs = inputs[:, -1, 0, :]
            elif np.ndim(inputs) == 5:
                inputs = inputs[:, -1, 0, -1, :]
            elif np.ndim(inputs) == 3:
                inputs = inputs[:, -1, :]
            elif np.ndim(inputs) == 2:
                pass
            else:
                raise ValueError(f"Input data has dimension {np.ndim(inputs)}.")

            true_denorm = np.full(true.shape, np.nan)
            pred_denorm = np.full(predicted.shape, np.nan)

            for h in range(self.forecast_len):
                t = true[:, :, h]
                p = predicted[:, :, h]
                in_obs = np.hstack([inputs, t])
                in_pred = np.hstack([inputs, p])
                scaler = self.scalers[scaler_key]['scaler']
                in_obs_den = scaler.inverse_transform(in_obs)
                in_pred_den = scaler.inverse_transform(in_pred)
                true_denorm[:, :, h] = in_obs_den[:, -self.outs:]
                pred_denorm[:, :, h] = in_pred_den[:, -self.outs:]

            predicted = pred_denorm
            true = true_denorm

        return predicted, true


    def deindexify_input_data(self, inputs:list, sort:bool = False, use_datetime_index:bool = False):

        first_input = inputs[0]
        dt_index = np.arange(len(first_input))

        if use_datetime_index:
            if np.ndim(first_input) == 2:
                dt_index = get_index(np.array(first_input[:, 0], dtype=np.int64))
            elif np.ndim(first_input) == 3:
                dt_index = get_index(np.array(first_input[:, -1, 0], dtype=np.int64))
            elif np.ndim(first_input) == 4:
                dt_index = get_index(np.array(first_input[:, -1, -1, 0], dtype=np.int64))

            # remove the first of first inputs which is datetime index
            first_input = first_input[..., 1:].astype(np.float32)

            if sort:
                first_input =first_input[np.argsort(dt_index.to_pydatetime())]

        inputs[0] = first_input

        return first_input, inputs, dt_index

    def compile(self, model_inputs, outputs):

        k_model = self.KModel(inputs=model_inputs, outputs=outputs)

        opt_args = self.get_opt_args()
        optimizer = OPTIMIZERS[self.nn_config['optimizer'].upper()](**opt_args)

        k_model.compile(loss=self.loss, optimizer=optimizer, metrics=self.get_metrics())

        if self.verbosity > 0:
            k_model.summary()

        try:
            keras.utils.plot_model(k_model, to_file=os.path.join(self.path, "model.png"), show_shapes=True, dpi=300)
        except (AssertionError, ImportError) as e:
            print("dot plot of model could not be plotted")
        return k_model

    def get_opt_args(self):
        """ get input arguments for an optimizer. It is being explicitly defined here so that it can be overwritten
        in sub-classes"""
        return {'lr': self.nn_config['lr']}

    def get_metrics(self) -> list:
        """ returns the performance metrics specified"""
        _metrics = self.data_config['metrics']

        metrics = None
        if _metrics is not None:
            if not isinstance(_metrics, list):
                assert isinstance(_metrics, str)
                _metrics = [_metrics]

            from dl4seq.utils.tf_losses import nse, kge, pbias

            METRICS = {'NSE': nse,
                       'KGE': kge,
                       'PBIAS': pbias}

            metrics = []
            for m in _metrics:
                if m.upper() in METRICS.keys():
                    metrics.append(METRICS[m.upper()])
                else:
                    metrics.append(m)

        return metrics

    def get_2d_batches(self, df, ins, outs):
        # for case when there is not lookback, i.e first layer is dense layer and takes 2D input
        input_x, input_y, label_y = df.iloc[:, 0:ins].values, df.iloc[:, -outs:].values, df.iloc[:, -outs:].values

        assert self.lookback == 1, """lookback should be one for MLP/Dense layer based model, but it is {}
        """.format(self.lookback)
        return self.check_nans(df, input_x, input_y, np.expand_dims(label_y, axis=2), outs)

    def get_3d_batches(self, df, ins, outs, lookback, in_step, forecast_step, forecast_len):
        # Provide lookback/history/seq length in input data
        input_x = []
        input_y = []
        label_y = []

        row_length = len(df)
        column_length = df.columns.size

        for i in range(row_length - lookback * in_step + 1 - forecast_step - forecast_len + 1):
            stx, enx = i, i + lookback * in_step
            x_data = df.iloc[stx:enx:in_step, 0:column_length - outs]

            st, en = i, i + (lookback - 1) * in_step
            y_data = df.iloc[st:en:in_step, column_length - outs:]

            sty = enx + forecast_step - in_step
            eny = sty + forecast_len
            label_data = df.iloc[sty:eny, column_length - outs:]

            input_x.append(np.array(x_data))
            input_y.append(np.array(y_data))
            label_y.append(np.array(label_data))

        input_x = np.array(input_x, dtype=np.float64).reshape(-1, lookback, ins)
        input_y = np.array(input_y, dtype=np.float32).reshape(-1, lookback - 1, outs)
        # transpose because we want labels to be of shape (examples, outs, forecast_length)
        label_y = np.array([np.array(i, dtype=np.float32).T for i in label_y], dtype=np.float32)

        return self.check_nans(df, input_x, input_y, label_y, outs)

    def get_batches(self, df, ins, outs):

        if self.num_input_layers > 1:
            # if the model takes more than 1 input, we must define what kind of inputs must be made because
            # using this method means we are feeding same shaped inputs so we can easily say, 2d or 3d. If the
            # the model takes more than 1 input and they are of different shapes, theen the user has to (is this
            # common sense?) overwrite `train_paras` and or `test_paras` methods.
            if self.data_config['batches'].upper() == "2D":
                return self.get_2d_batches(df, ins, outs)
            else:
                return self.get_3d_batches(df, ins, outs, self.lookback, self.data_config['input_step'],
                                           self.forecast_step, self.forecast_len)
        else:
            if len(self.first_layer_shape()) == 2:
                return self.get_2d_batches(df, ins, outs)

            else:
                return self.get_3d_batches(df, ins, outs, self.lookback, self.data_config['input_step'],
                                           self.forecast_step, self.forecast_len)

    def check_nans(self, df, input_x, input_y, label_y, outs):
        """ checks whether anns are present or not and checks shapes of arrays being prepared.
        """
        nans = df[self.out_cols].isna().sum()
        if int(nans.sum()) > 0:
            if self.data_config['ignore_nans']:
                print("\n{} Ignoring NANs in predictions {}\n".format(10 * '*', 10 * '*'))
            else:
                if self.method == 'dual_attention':
                    raise ValueError
                if outs > 1:
                    assert np.all(nans.values == int(nans.sum() / outs)), """output columns contains nan values at
                     different indices"""

                if self.verbosity > 0:
                    print('\n{} Removing Samples with nan labels  {}\n'.format(10 * '*', 10 * '*'))
                # y = df[df.columns[-1]]
                nan_idx = np.isnan(label_y) if self.outs == 1 else np.isnan(label_y[:, 0])  # y.isna()
                # nan_idx_t = nan_idx[self.lookback - 1:]
                # non_nan_idx = ~nan_idx_t.values
                non_nan_idx = ~nan_idx.reshape(-1, )
                label_y = label_y[non_nan_idx]
                input_x = input_x[non_nan_idx]
                input_y = input_y[non_nan_idx]

                assert np.isnan(label_y).sum() < 1, "label still contains {} nans".format(np.isnan(label_y).sum())

        assert input_x.shape[0] == input_y.shape[0] == label_y.shape[0], "shapes are not same"

        assert np.isnan(input_x).sum() == 0, "input still contains {} nans".format(np.isnan(input_x).sum())

        return input_x, input_y, label_y

    def activations(self, layer_names=None, **kwargs):
        # if layer names are not specified, this will get get activations of allparameters
        inputs, _ = self.test_data(**kwargs)

        # samples/examples in inputs may not be ordered/sorted so we should order them
        # remvoe the first column from x data
        _, inputs, _ = self.deindexify_input_data(inputs, sort=True, use_datetime_index=kwargs.get('use_datetime_index', False))

        activations = keract.get_activations(self._model, inputs, layer_names=layer_names, auto_compile=True)
        return activations, inputs

    def display_activations(self, layer_name: str = None, st=0, en=None, indices=None, **kwargs):
        # not working currently because it requres the shape of activations to be (1, output_h, output_w, num_filters)
        activations, _ = self.activations(st=st, en=en, indices=indices, layer_names=layer_name)
        if layer_name is None:
            activations = activations
        else:
            activations = activations[layer_name]

        keract.display_activations(activations=activations, **kwargs)

    def gradients_of_weights(self, **kwargs) -> dict:

        x, y = self.test_data(**kwargs)

        _, x, _ = self.deindexify_input_data(x, sort=True, use_datetime_index=kwargs.get('use_datetime_index', False))

        return keract.get_gradients_of_trainable_weights(self._model, x, y)

    def gradients_of_activations(self, st=0, en=None, indices=None, layer_name=None, **kwargs) -> dict:

        x, y = self.test_data(st=st, en=en, indices=indices)

        _, x, _ = self.deindexify_input_data(x, sort=True, use_datetime_index=kwargs.get('use_datetime_index', False))

        return keract.get_gradients_of_activations(self._model, x, y, layer_names=layer_name)

    def trainable_weights(self):
        """ returns all trainable weights as arrays in a dictionary"""
        weights = {}
        for weight in self._model.trainable_weights:
            if tf.executing_eagerly():
                weights[weight.name] = weight.numpy()
            else:
                weights[weight.name] = keras.backend.eval(weight)
        return weights

    def plot_weights(self, save=True):
        weights = self.trainable_weights()

        if self.verbosity > 0:
            print("Plotting trainable weights of layers of the model.")

        for _name, weight in weights.items():
            title = _name + " Weights"
            fname = _name + '_weights'

            rnn_args = None
            if "LSTM" in title.upper():
                rnn_args = {'n_gates': 4,
                            'gate_names_str': "(input, forget, cell, output)"}

            if np.ndim(weight) == 2 and weight.shape[1] > 1:

                self._imshow(weight, title, save, fname, rnn_args=rnn_args)

            elif len(weight) > 1 and np.ndim(weight) < 3:
                self.plot1d(weight, title, save, fname, rnn_args=rnn_args)
            else:
                print("ignoring weight for {} because it has shape {}".format(_name, weight.shape))

    def plot_activations(self, save: bool = True, **kwargs):
        """ plots activations of intermediate layers except input and output.
        If called without any arguments then it will plot activations of all layers."""
        activations, _ = self.activations(**kwargs)

        if self.verbosity > 0:
            print("Plotting activations of layers")

        for lyr_name, activation in activations.items():
            # activation may be tuple e.g if input layer receives more than 1 input
            if isinstance(activation, np.ndarray):
                self._plot_activation(activation, lyr_name, save)

            elif isinstance(activation, tuple):
                for act in activation:
                    self._plot_activation(act, lyr_name, save)
        return

    def _plot_activation(self, activation, lyr_name, save):
        if "LSTM" in lyr_name.upper() and np.ndim(activation) in (2, 3):
            self.features_2d(activation, save=save, lyr_name=lyr_name, norm=(-1, 1))
        elif np.ndim(activation) == 2 and activation.shape[1] > 1:
            self._imshow(activation, lyr_name + " Activations", save, lyr_name)
        elif np.ndim(activation) == 3:
            self._imshow_3d(activation, lyr_name, save=save)
        elif np.ndim(activation) == 2:  # this is now 1d
            # shape= (?, 1)
            self.plot1d(activation, label=lyr_name + ' Activations', save=save,
                        fname=lyr_name + '_activations')
        else:
            print("ignoring activations for {} because it has shape {}, {}".format(lyr_name, activation.shape,
                                                                                   np.ndim(activation)))
        return

    def plot_weight_grads(self, save: bool = True, **kwargs):
        """ plots gradient of all trainable weights"""

        gradients = self.gradients_of_weights(**kwargs)

        if self.verbosity > 0:
            print("Plotting gradients of trainable weights")

        for lyr_name, gradient in gradients.items():

            title = lyr_name + "Weight Gradients"
            fname = lyr_name + '_weight_grads'
            rnn_args = None

            if "LSTM" in title.upper():
                rnn_args = {'n_gates': 4,
                            'gate_names_str': "(input, forget, cell, output)"}

            if np.ndim(gradient) == 2 and gradient.shape[1] > 1:
                self._imshow(gradient, title, save, fname, rnn_args=rnn_args)

            elif len(gradient) and np.ndim(gradient) < 3:
                self.plot1d(gradient, title, save, fname, rnn_args=rnn_args)
            else:
                print("ignoring weight gradients for {} because it has shape {} {}".format(lyr_name, gradient.shape,
                                                                                           np.ndim(gradient)))

    def plot_act_grads(self, save: bool = True, **kwargs):
        """ plots activations of intermediate layers except input and output"""
        gradients = self.gradients_of_activations(**kwargs)

        if self.verbosity > 0:
            print("Plotting gradients of activations of layersr")

        for lyr_name, gradient in gradients.items():
            fname = lyr_name + "_activation gradients"
            title = lyr_name + " Activation Gradients"
            if "LSTM" in lyr_name.upper() and np.ndim(gradient) in (2, 3):
                self.features_2d(gradient, lyr_name=lyr_name, save=save)

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
        """ shows all activations, weights and gradients of the keras model."""
        self.plot_act_grads(**kwargs)
        self.plot_weight_grads(**kwargs)
        self.plot_activations(**kwargs)
        self.plot_weights()
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

        assert inputs.shape[1] == self.ins

        return inputs

    def plot_act_along_inputs(self, layer_name: str, name: str = None, vmin=0, vmax=0.8, **kwargs):

        assert isinstance(layer_name, str), "layer_name must be a string, not of {} type".format(type(layer_name))

        predictions, observations = self.predict(pp=False, **kwargs)

        activation, data = self.activations(layer_names=layer_name, **kwargs)

        activation = activation[layer_name]
        data = self.inputs_for_attention(data)

        assert data.shape[1] == self.ins

        plt.close('all')

        for out in range(self.outs):
            pred = predictions[out]
            obs = observations[out]
            out_name = self.out_cols[out]

            for idx in range(self.ins):

                fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='all')
                fig.set_figheight(10)

                ax1.plot(data[:, idx], label=self.in_cols[idx])
                ax1.legend()
                ax1.set_title('activations w.r.t ' + self.in_cols[idx])
                ax1.set_ylabel(self.in_cols[idx])

                ax2.plot(pred.values, label='Prediction')
                ax2.plot(obs.values, '.', label='Observed')
                ax2.legend()

                im = ax3.imshow(activation[:, :, idx].transpose(), aspect='auto', vmin=vmin, vmax=vmax)
                ax3.set_ylabel('lookback')
                ax3.set_xlabel('samples')
                fig.colorbar(im, orientation='horizontal', pad=0.2)
                plt.subplots_adjust(wspace=0.005, hspace=0.005)
                if name is not None:
                    _name = out_name + '_' + name
                    plt.savefig(os.path.join(self.act_path, _name) + self.in_cols[idx], dpi=400, bbox_inches='tight')
                else:
                    plt.show()
                plt.close('all')
            return

    def prepare_batches(self, df: pd.DataFrame, ins, outs):

        assert self.outs == 1
        target = self.data_config['outputs'][0]

        x = np.zeros((len(df), self.lookback, df.shape[1] - 1))
        prev_y = np.zeros((len(df), self.lookback, 1))

        for i, name in enumerate(list(df.columns[:-1])):
            for j in range(self.lookback):
                x[:, j, i] = df[name].shift(self.lookback - j - 1).fillna(method="bfill")

        for j in range(self.lookback):
            prev_y[:, j, 0] = df[target].shift(self.lookback - j - 1).fillna(method="bfill")

        fl = self.data_config['forecast_length']
        _y = np.zeros((df.shape[0], fl))
        for i in range(df.shape[0] - fl):
            _y[i - 1, :] = df[target].values[i:i + fl]

        input_x = x[self.lookback:-fl, :]
        prev_y = prev_y[self.lookback:-fl, :]
        y = _y[self.lookback:-fl, :].reshape(-1, outs, self.forecast_len)

        return self.check_nans(df, input_x, prev_y, y, outs)

    def save_config(self, history: dict):

        test_indices = np.array(self.test_indices, dtype=int).tolist() if self.test_indices is not None else None
        train_indices = np.array(self.train_indices, dtype=int).tolist() if self.train_indices is not None else None

        save_config_file(indices={'test_indices': test_indices,
                                  'train_indices': train_indices}, path=self.path)

        config = dict()
        config['min_val_loss'] = int(np.min(history['val_loss'])) if 'val_loss' in history else None
        config['min_loss'] = int(np.min(history['loss'])) if 'val_loss' in history else None
        config['nn_config'] = self.nn_config
        config['data_config'] = self.data_config
        config['intervals'] = self.intervals
        config['method'] = self.method
        config['quantiles'] = self.quantiles
        config['loss'] = self._model.loss.__name__ if self._model is not None else None
        config['params'] = int(self._model.count_params()) if self._model is not None and self.category == "DL" else None,

        VERSION_INFO.update({'numpy_version': str(np.__version__),
                             'pandas_version': str(pd.__version__),
                             'matplotlib_version': str(matplotlib.__version__)})
        config['version_info'] = VERSION_INFO

        save_config_file(config=config, path=self.path)
        return config

    @classmethod
    def from_config(cls, config_path: str, data, use_pretrained_model=True):
        with open(config_path, 'r') as fp:
            config = json.load(fp)

        idx_file = os.path.join(os.path.dirname(config_path), 'indices.json')
        with open(idx_file, 'r') as fp:
            indices = json.load(fp)

        data_config = config['data_config']
        nn_config = config['nn_config']
        if 'intervals' in config:
            intervals = config['intervals']
        else:
            intervals = None

        cls.from_check_point = True

        # These paras neet to be set here because they are not withing init method
        cls.test_indices = indices["test_indices"]
        cls.train_indices = indices["train_indices"]

        if use_pretrained_model:
            path = os.path.dirname(config_path)
        else:
            path = None
        return cls(data_config=data_config,
                   nn_config=nn_config,
                   data=data,
                   intervals=intervals,
                   path=path)

    def load_weights(self, w_file: str):
        # loads the weights of keras model from weight file `w_file`.
        cpath = os.path.join(self.w_path, w_file)
        self._model.load_weights(cpath)
        print("{} Successfully loaded weights {}".format('*' * 10, '*' * 10))
        return

    def write_cache(self, _fname, input_x, input_y, label_y):
        fname = os.path.join(self.path, _fname)
        h5 = h5py.File(fname, 'w')
        h5.create_dataset('input_X', data=input_x)
        h5.create_dataset('input_Y', data=input_y)
        h5.create_dataset('label_Y', data=label_y)
        h5.close()
        return


def unison_shuffled_copies(a, b, c):
    """makes sure that all the arrays are permuted similarly"""
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]
