import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib # for version info
import os
from types import MethodType

from sklearn.model_selection import train_test_split
import json
import joblib
import h5py
import random
import math
import warnings
import time

from dl4seq.nn_tools import NN
from dl4seq.backend import tf, keras, tcn, torch, VERSION_INFO, catboost_models, xgboost_models, lightgbm_models
from dl4seq.backend import tpot_models
from dl4seq.backend import imputations, sklearn_models
from dl4seq.utils.utils import maybe_create_path, save_config_file, get_index, dateandtime_now
from dl4seq.utils.utils import train_val_split, split_by_indices, stats, make_model, make_3d_batches
from dl4seq.utils.plotting_tools import Plots
from dl4seq.utils.transformations import Transformations
from dl4seq.utils.imputation import Imputation
from dl4seq.models.custom_training import train_step, test_step
from dl4seq.utils.TSErrors import FindErrors

def reset_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if tf is not None:
        if int(tf.__version__.split('.')[0]) == 1:
            tf.compat.v1.random.set_random_seed(seed)
        elif int(tf.__version__.split('.')[0]) > 1:
            tf.random.set_seed(seed)


LOSSES = {}
if tf is not None:
    import dl4seq.keract_mod as keract
    from dl4seq.tf_attributes import LOSSES, OPTIMIZERS
elif torch is not None:  # TODO, what if both tf and torch are installed and we want to run pytorch-based model?
    from dl4seq.torch_attributes import LOSSES as pt_losses
    LOSSES.update(pt_losses)

class Model(NN, Plots):
    """
    Model class that implements logic of dl4seq.

    data: pd.Dataframe or any other conforming type
    prefix: str, prefix to be used for the folder in which the results are saved
    path: str/path like, if given, new path will not be created
    verbosity: int, determines the amount of information being printed
    kwargs: any argument for model building/pre-processing etc.
            for details see make_model in utils.utils.py

    """
    def __init__(self,
                 *,
                 data=None,
                 prefix: str = None,
                 path: str = None,
                 verbosity=1,
                 **kwargs):

        config = make_model(**kwargs)

        #data_config, model_config = config['data_config'], config['model_config']
        reset_seed(config.config['seed'])
        if tf is not None:
            # graph should be cleared everytime we build new `Model` otherwise, if two `Models` are prepared in same
            # file, they may share same graph.
            tf.keras.backend.clear_session()

        #super(Model, self).__init__(**config)
        NN.__init__(self, config=config.config)

        self.intervals = config.config['intervals']
        self.data = data
        self.in_cols = self.config['inputs']
        self.out_cols = self.config['outputs']
        self.loss = LOSSES[self.config['loss'].upper()]
        self.KModel = keras.models.Model if keras is not None else None
        self.path = maybe_create_path(path=path, prefix=prefix)
        self.verbosity = verbosity
        self.category = self.config['category']
        self.problem = self.config['problem']
        self.info = {}

        Plots.__init__(self, self.path, self.problem, self.category, self._model,
                       config=config.config)

        self.build() # will initialize ML models or build NNs

    @property
    def act_path(self):
        return os.path.join(self.path, 'activations')

    @property
    def w_path(self):
        return os.path.join(self.path, 'weights')

    @property
    def data_path(self):
        return os.path.join(self.path, 'data')

    @property
    def forecast_step(self):
        # should only be changed using self.data_config
        return self.config['forecast_step']

    @property
    def forecast_len(self):
        # should only be changed using self.data_config
        return self.config['forecast_length']

    @property
    def data(self):
        # so that .data attribute can be customized
        return self._data

    @data.setter
    def data(self, x):
        if isinstance(x, pd.DataFrame):
            _data = x[self.config['inputs'] + self.config['outputs']]
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
    def in_cols(self, x):
        self._in_cols = x

    @property
    def out_cols(self):
        return self._out_cols

    @out_cols.setter
    def out_cols(self, x):
        self._out_cols = x

    @property
    def ins(self):

        if isinstance(self.in_cols, dict):
            return {k:len(inp) for k,inp in self.in_cols.items()}
        else:
            return len(self.in_cols)

    @property
    def outs(self):

        if isinstance(self.out_cols, dict):
            return {k:len(inp) for k,inp in self.out_cols.items()}
        else:
            return len(self.out_cols)

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, x):
        self._loss = x

    @property
    def quantiles(self):
        return self.config['quantiles']

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

    @property
    def input_layer_names(self) -> list:

        return [lyr.name.split(':')[0] for lyr in self._model.inputs]

    def fetch_data(self,
                   data: pd.DataFrame,
                   inps,
                   outs,
                   transformation=None,
                   st: int = 0,
                   en=None,
                   shuffle: bool = False,  # TODO, is this arg requried?
                   write_data=False,
                   noise: int = 0,
                   indices: list = None,
                   scaler_key: str = '0',
                   use_datetime_index=False):
        """
        :param data:
        :param inps,
        :param outs
        :param transformation:
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
        data = data.copy()
        if st is not None:
            assert isinstance(st, int), "starting point must be integer."
        if indices is not None:
            assert isinstance(np.array(indices), np.ndarray), "indices must be array like"
            if en is not None or st != 0:
                raise ValueError
        if en is None:
            en = data.shape[0]

        # # add random noise in the data
        df = self.add_noise(data, noise)

        if transformation:  # TODO when train_dataand test_data are externally set, normalization can't be done.
            df, _ = self.normalize(df, scaler_key, transformation)

        # indexification should happen after transformation, because datetime column should not be transformed.
        df = self.indexify_data(df, use_datetime_index)

        if self.intervals is None:

            df = df[st:en]
            if isinstance(df, pd.DataFrame):
                df = df.values

            x, y, label = self.get_batches(df,
                                           len(inps),
                                           len(outs)
                                           )
            if indices is not None and not isinstance(indices, str):

                if getattr(self, 'nans_removed_4m_st', 0)==0:
                    # Either NaNs were not present in outputs or present but not at the start
                    if not hasattr(self, 'nans_removed_4m_st'):
                        # NaNs were not present in outputs
                        additional = self.config['forecast_length']+1 if self.config['forecast_length']>1 else 0
                        to_subtract = self.lookback + additional - 1
                    else:
                    # Either not present at the start or ?
                    # TODO
                    # verify this situation
                        to_subtract = 0
                        # so that when get_batches is called next time, with new data, this should be False by default
                        self.nans_removed_4m_st = 0
                else:
                    # nans were present in output columns which were removed
                    if getattr(self, 'nans_removed_4m_st', 0) > 0:
                        additional = self.config['forecast_length']+1 if self.config['forecast_length'] > 1 else 0
                        self.offset = abs(self.lookback + additional - self.nans_removed_4m_st-1)
                        warnings.warn(f"""lookback is {self.lookback}, due to which first {self.nans_removed_4m_st} nan
                                      containing values were skipped from start. This may lead to some wrong examples
                                      at the start or an offset of {self.offset} in indices.""",
                                      UserWarning)
                        to_subtract = self.offset
                        self.nans_removed_4m_st = 0
                    # we don't want to subtract anything from indices, possibly because the number of nans removed from
                    # start are > lookback.
                    elif self.nans_removed_4m_st == -9999:
                        to_subtract = 0
                    else:
                        # because the x,y,z have some initial values removed
                        to_subtract = self.lookback - 1
                indices = np.subtract(np.array(indices), to_subtract).tolist()

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
                    x, y, label = self.get_batches(df1.values,
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
            if 'dt_index' in df:  # TODO, is it necessary?
                df.pop('dt_index') # because self.data belongs to class, this should remain intact.

        #x = x.astype(np.float32)

        if write_data:
            self.write_cache('data_' + scaler_key, x, y, label)

        return x, y, label

    def indexify_data(self, data, use_datetime_index:bool):

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

    def normalize(self, df, key, transformation):
        """ should return the transformed dataframe and the key with which scaler is put in memory. """
        scaler = None

        if transformation is not None:

            if isinstance(transformation, dict):
                df, scaler = Transformations(data=df, **transformation)('transformation', return_key=True)
                self.scalers[key] = scaler

            # we want to apply multiple transformations
            elif isinstance(transformation, list):
                for trans in transformation:
                    if trans['method'] is not None:
                        df, scaler = Transformations(data=df, **trans)('transformation', return_key=True)
                        self.scalers[key+str(trans['method'])] = scaler
            else:
                assert isinstance(transformation, str)
                df, scaler = Transformations(data=df, method=transformation)('transformation', return_key=True)
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
        steps_per_epoch = self.config['batches_per_epoch']
        if steps_per_epoch is None:

            if isinstance(x, list):
                examples = x[0].shape[0]
            elif isinstance(x, dict):
                examples = list(x.values())[0].shape[0]
            else:
                examples =  x.shape[0]

            if self.config['drop_remainder']:
                iterations = math.floor(examples/self.config['batch_size'])
                return self._check_batches(x, prev_y, y, iterations)

            return x, prev_y, y
        else:

             return self._check_batches(x, prev_y, y, steps_per_epoch)

    def _check_batches(self, x, prev_y, y, iterations):
        if isinstance(x, dict):
            raise NotImplementedError

        if self.verbosity > 0:
            print(f"Number of total batches are {iterations}")

        x_is_list = True
        if not isinstance(x, list):
            x = [x]
            x_is_list = False

        assert isinstance(y, np.ndarray)

        if prev_y is None:
            prev_y = y.copy()

        batch_size = self.config['batch_size']
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
            shapes = {}
            for lyr in self._model.inputs:
                shapes[lyr.name] = lyr.shape
            return shapes
        shape = []
        for idx, d in enumerate(self._model.layers[0].input.shape):
            if int(tf.__version__[0]) == 1:
                if isinstance(d, tf.Dimension):  # for tf 1.x
                    d = d.value

            if idx == 0:  # the first dimension must remain undefined so that the user may define batch_size
                d = -1
            shape.append(d)
        return shape

    def get_callbacks(self, val_data_present, **callbacks):

        _callbacks = list()

        _monitor = 'val_loss' if val_data_present else 'loss'
        fname = "{val_loss:.5f}.hdf5" if val_data_present else "{loss:.5f}.hdf5"

        if int(''.join(tf.__version__.split('.')[0:2])) <= 115:
            for lyr_name in self.layer_names:
                if 'HA_weighted_input' in lyr_name or 'SeqWeightedAttention_weights' in lyr_name:
                    self.config['save_model'] = False

                    warnings.warn("Can not save Heirarchical model with tf<= 1.15")

        if self.config['save_model']:
            _callbacks.append(keras.callbacks.ModelCheckpoint(
                filepath=self.w_path + "\\weights_{epoch:03d}_" + fname,
                save_weights_only=True,
                monitor=_monitor,
                mode='min',
                save_best_only=True))

        _callbacks.append(keras.callbacks.EarlyStopping(
            monitor=_monitor, min_delta=self.config['min_val_loss'],
            patience=self.config['patience'], verbose=0, mode='auto'
        ))

        if 'tensorboard' in callbacks:
            _callbacks.append(keras.callbacks.TensorBoard(log_dir=self.path, histogram_freq=1))
            callbacks.pop('tensorboard')

        for key, val in callbacks.items():
            _callbacks.append(val)

        return _callbacks

    def use_val_data(self, val_split, val_data):
        """Finds out if there is val_data or not"""
        val_data_present=False
        if val_split>0.0:
            val_data_present=True

        # it is possible that val_split=0.0 but we do have val_data.
        if val_data is not None:
            val_data_present=True

        return val_data_present

    def do_fit(self, *args, **kwargs):
        """If nans are present in y, then tf.keras.model.fit is called as it is otherwise it is called with custom
        train_step and test_step which avoids calculating loss at points containing nans."""
        if kwargs.pop('nans_in_y_exist'):

            self._model.train_step = MethodType(train_step, self._model)
            self._model.test_step = MethodType(test_step, self._model)

        return self._model.fit(*args, **kwargs)

    def _fit(self, inputs, outputs, validation_data, **callbacks):

        nans_in_y_exist = False
        if isinstance(outputs, np.ndarray):
            if np.isnan(outputs).sum()>0:
                nans_in_y_exist=True
        elif isinstance(outputs, list):
            for out_array in outputs:
                if np.isnan(out_array).sum()>0:
                    nans_in_y_exist = True
        elif isinstance(outputs, dict):
            for out_name, out_array in outputs.items():
                if np.isnan(out_array).sum()>0:
                    nans_in_y_exist = True

        inputs, outputs, validation_data = self.to_tf_data(inputs, outputs, validation_data)

        validation_split = 0.0 if isinstance(inputs, tf.data.Dataset) else self.config['val_fraction']

        callbacks = self.get_callbacks(self.use_val_data(validation_split, validation_data), **callbacks)

        st = time.time()

        self.do_fit(inputs,
                    y=None if isinstance(inputs, tf.data.Dataset) else outputs,
                    epochs=self.config['epochs'],
                    batch_size=None if isinstance(inputs, tf.data.Dataset) else self.config['batch_size'],
                    validation_split=validation_split,
                    validation_data=validation_data,
                    callbacks=callbacks,
                    shuffle=self.config['shuffle'],
                    steps_per_epoch=self.config['steps_per_epoch'],
                    verbose=self.verbosity,
                    nans_in_y_exist=nans_in_y_exist
                         )

        self.info['training_time_in_minutes'] = float(time.time() - st)/60.0

        return self.post_kfit()

    def to_tf_data(self, inputs, outputs, val_data):
        """This method has only effect when we want to use same data for validation and test. This will work in following
           scenarios:
            if val_data is string,
            if val_data is is defined by user and is in (x,y) form.
        In both above scenarios, val_dataset will be returned which will be instance of tf.data.Dataset. (since the name
        to_tf_data) The test and validation data is set as Model's attribute in such cases.
        Following attributes will be generated
          - val_dataset
          - train_dataset
          """

        if isinstance(val_data, str):
            # we need to get validation data from training data depending upon value of 'val_fraction' and convert it
            # to tf.data
            val_frac = self.config['test_fraction']  # assuming that val_fraction has been set to 0.0
            if not hasattr(self, 'test_indices'):
                self.test_indices = None
            if self.test_indices is not None:
                x_train, y_train, x_val, y_val = split_by_indices(inputs, outputs,
                                                                  self.test_indices)
            elif self.train_indices == 'random': # val_data needs to be defined randomly
                # split the data into train/val randomly
                self.train_indices, self.test_indices = train_test_split(np.arange(outputs.shape[0]), test_size=self.config['test_fraction'])
                x_train, y_train = split_by_indices(inputs, outputs, self.train_indices)
                x_val, y_val = split_by_indices(inputs, outputs, self.test_indices)
            else:
                # split the data into train/val by chunk not randomly
                x_train, y_train, x_val, y_val = train_val_split(inputs, outputs, val_frac)

        elif hasattr(val_data, '__len__'):
            # val_data has already been in the form of x,y paris
            x_train = inputs
            y_train = outputs

            assert len(val_data) <= 3
            x_val, y_val = val_data

        else:
            return inputs, outputs, val_data

        if self.verbosity > 0.0:
            self.info['train_examples'] = len(y_train)
            self.info['val_examples'] = len(y_val)
            print(f"Train on {len(y_train)} and validation on {len(y_val)} examples")

        if self.num_input_layers == 1:
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train[0], y_train))
        elif self.num_input_layers > 1 and isinstance(x_train, list):
            assert len(self._model.outputs) == 1
            def train_generator():
                for idx, l in enumerate(y_train):
                    x = [x_train[i][idx] for i in range(len(x_train))]
                    yield {inp_name:inp for inp_name, inp in zip(self.input_layer_names, x)}, l

            inp_lyr_shapes = [self.layers_out_shapes[inp_name][0][1:] for inp_name in self.input_layer_names]
            # assuming that there is only one output
            out_lyr_shape = tuple(self._model.outputs[0].shape.as_list()[1:])
            train_dataset = tf.data.Dataset.from_generator(
                train_generator,
                output_shapes=({inp_name: inp_shp for inp_name, inp_shp in zip(self.input_layer_names, inp_lyr_shapes)},
                               out_lyr_shape),
                output_types=({inp_name: tf.float32 for inp_name in self.input_layer_names}, tf.float32))
        else:
            raise NotImplementedError

        if self.config['shuffle']:
            train_dataset = train_dataset.shuffle(self.config['buffer_size'])
        train_dataset = train_dataset.batch(self.config['batch_size'],
                                             drop_remainder=self.config['drop_remainder'])
        if x_val is not None:
            if self.num_input_layers == 1:
                if isinstance(x_val, list):
                    x_val = x_val[0]
                assert isinstance(x_val, np.ndarray)
                val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            else:
                def val_generator():
                    for idx, l in enumerate(y_val):
                        x = [x_val[i][idx] for i in range(len(x_val))]
                        yield {inp_name:inp for inp_name, inp in zip(self.input_layer_names, x)}, l

                inp_lyr_shapes = [self.layers_out_shapes[inp_name][0][1:] for inp_name in self.input_layer_names]
                # assuming that there is only one output
                out_lyr_shape = tuple(self._model.outputs[0].shape.as_list()[1:])
                val_dataset = tf.data.Dataset.from_generator(
                    val_generator,
                    output_shapes=({inp_name: inp_shp for inp_name, inp_shp in zip(self.input_layer_names, inp_lyr_shapes)},
                               out_lyr_shape),
                    output_types=({inp_name: tf.float32 for inp_name in self.input_layer_names}, tf.float32))

            val_dataset = val_dataset.batch(self.config['batch_size'],
                                                drop_remainder=self.config['drop_remainder'])
        else:
            val_dataset = val_data

        self.val_dataset = val_dataset
        self.train_dataset = train_dataset

        return train_dataset, outputs,  val_dataset

    def post_kfit(self):
        """Does some stuff after Keras model.fit has been called"""
        history = self._model.history

        self.save_config(history.history)

        # save all the losses or performance metrics
        df = pd.DataFrame.from_dict(history.history)
        df.to_csv(os.path.join(self.path, "losses.csv"))

        return history

    def get_indices(self, indices=None):
        if isinstance(self.data, dict):
            key = list(self.out_cols.keys())[0]
            data = self.data[key]
            out_cols =self.out_cols[key]
        else:
            data = self.data
            out_cols = self.out_cols
        # returns only train indices if `indices` is None
        if isinstance(indices, str) and indices.upper() == 'RANDOM':
            if self.config['allow_nan_labels'] == 2:
                tot_obs = self.data.shape[0]
            elif self.config['allow_nan_labels'] == 1:
                label_y = self.data[self.out_cols].values
                idx = ~np.array([all([np.isnan(x) for x in label_y[i]]) for i in range(len(label_y))])
                tot_obs = np.sum(idx)
            else:
                if self.outs == 1:
                    # TODO it is being supposed that intervals correspond with Nans. i.e. all values outside intervals
                    # are NaNs. However this can be wrong when the user just wants to skip some chunks of data even
                    # though they are not NaNs. In this case, there will be no NaNs and tot_obs will be more than
                    # required. In such case we have to use `self.vals_in_intervals` to calculate tot_obs. But that
                    # creates problems when larger intervals are provided. such as [NaN, NaN, 1, 2, 3, NaN] we provide
                    # (0, 5) instead of (2, 4). Is it correct/useful to provide (0, 5)?
                    more = len(self.intervals) * self.lookback if self.intervals is not None else 0 #self.lookback
                    tot_obs = data.shape[0] - int(data[out_cols].isna().sum()) - more
                else:
                    # data contains nans and target series are > 1, we want to make sure that they have same nan counts
                    tot_obs = data.shape[0] - int(data[out_cols[0]].isna().sum())
                    nans = data[out_cols].isna().sum()
                    assert np.all(nans.values == int(nans.sum() / len(out_cols))), f"toal nan values in data are {nans}."

            idx = np.arange(tot_obs)
            train_indices, test_idx = train_test_split(idx, test_size=self.config['test_fraction'],
                                                       random_state=self.config['seed'])
            setattr(self, 'test_indices', list(test_idx))
            indices = list(train_indices)

        if self.is_training:
            setattr(self, 'train_indices', indices)
            if self.config['val_data'] == 'same':
                if isinstance(indices, list):
                    # indices have been provided externally so use them instead.
                    return indices
                else:
                    # indices have not been created here and `indices` is None which we don't want to feed to fetch_data
                    return None

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

    def train_data(self, data=None, **kwargs):
        """ prepare data on which to train the NN."""

        # For cases we don't want to use any of data-preprocessing utils, the train_data can be called directly with 'data'
        # keyword argumentnt, and that will be returned.
        if data is not None:
            return data

        transformation = self.config['transformation']
        if isinstance(self.data, dict):
            x, prev_y, label = {}, {}, {}
            for k, data in self.data.items():
                _x, _y, _label = self.fetch_data(data, self.in_cols[k],  self.out_cols.get(k, []),
                                                 transformation=transformation[k] if transformation else transformation,
                                                 **kwargs)
                x[k] = _x
                prev_y[k] = _y
                if _label.sum() != 0.0:
                    label[k] = _label
        else:
            x, prev_y, label = self.fetch_data(self.data, self.in_cols, self.out_cols,
                                          transformation=self.config['transformation'], **kwargs)

        x, prev_y, label = self.check_batches(x, prev_y, label)

        if isinstance(label, dict):
            assert len(self._model.outputs) == len(label)
            if len(label) ==1:
                label = list(label.values())[0]

        if self.category.upper() == "ML" and self.outs == 1:
            label = label.reshape(-1,)

        if self.verbosity > 0:
            print_something(x, "input_x")
            print_something(prev_y, "input_y")
            print_something(label, "label")

        if isinstance(x, np.ndarray):
            return [x], prev_y, label

        elif isinstance(x, list):
            return x, prev_y, label

        elif isinstance(x, dict):
            return x, prev_y, label

        else:
            raise ValueError

    def maybe_not_3d_data(self, true, predicted):
        if true.ndim < 3:
            assert self.forecast_len == 1
            axis = 2 if true.ndim == 2 else (1, 2)
            true = np.expand_dims(true, axis=axis)
        if predicted.ndim < 3:
            assert self.forecast_len == 1
            axis = 2 if predicted.ndim == 2 else (1,2)
            predicted = np.expand_dims(predicted, axis=axis)
        return true, predicted

    def process_results(self,
                        true: np.ndarray,
                        predicted: np.ndarray,
                        prefix=None,
                        index=None,
                        remove_nans=True,
                        **plot_args):
        """
        predicted, true are arrays of shape (examples, outs, forecast_len)
        """
        # for cases if they are 2D/1D, add the third dimension.
        true, predicted = self.maybe_not_3d_data(true, predicted)

        out_cols = list(self.out_cols.values())[0] if isinstance(self.out_cols, dict) else self.out_cols
        for idx, out in enumerate(out_cols):
            for h in range(self.forecast_len):

                errs = dict()

                fpath = os.path.join(self.path, out)
                if not os.path.exists(fpath):
                    os.makedirs(fpath)

                t = pd.DataFrame(true[:, idx, h], index=index, columns=['true_' + out])
                p = pd.DataFrame(predicted[:, idx, h], index=index, columns=['pred_' + out])
                df = pd.concat([t, p], axis=1)
                df = df.sort_index()
                fname = prefix + out + '_' + str(h) +  ".csv"
                df.to_csv(os.path.join(fpath, fname), index_label='time')

                self.plot_results(t, p, name=prefix + out + '_' + str(h), where=fpath)

                if remove_nans:
                    nan_idx = np.isnan(t)
                    t = t.values[~nan_idx]
                    p = p.values[~nan_idx]

                errors = FindErrors(t, p)
                errs[out + '_errors_' + str(h)] = errors.calculate_all()
                errs[out + 'true_stats_' + str(h)] = stats(t)
                errs[out + 'predicted_stats_' + str(h)] = stats(p)

                save_config_file(fpath, errors=errs, name=prefix)
        return

    def build(self):

        if self.verbosity > 0:
            print('building {} based model for {} problem'.format(self.category, self.problem))

        if self.category.upper() == "DL":
            inputs, predictions = self.add_layers(self.config['model']['layers'])

            self._model = self.compile(inputs, predictions)

            self.info['model_parameters'] = int(self._model.count_params()) if self._model is not None else None

            if self.verbosity > 0:
                if 'tcn' in self.config['model']['layers']:
                    tcn.tcn_full_summary(self._model, expand_residual_blocks=True)
        else:
            self.build_ml_model()

        # fit main fail so better to save config before as well. This will be overwritten once the fit is complete
        self.save_config()

        VERSION_INFO.update({'numpy_version': str(np.__version__),
                             'pandas_version': str(pd.__version__),
                             'matplotlib_version': str(matplotlib.__version__)})
        self.info['version_info'] = VERSION_INFO

        return

    def build_ml_model(self):
        """ builds models that follow sklearn api such as xgboost, catboost, lightgbm and obviously sklearn."""

        ml_models = {**sklearn_models, **xgboost_models, **catboost_models, **lightgbm_models, **tpot_models}
        _model = list(self.config['model'].keys())[0]
        regr_name = _model.upper()

        kwargs = list(self.config['model'].values())[0]

        if regr_name in ['HISTGRADIENTBOOSTINGREGRESSOR', 'SGDREGRESSOR', 'MLPREGRESSOR']:
            kwargs.update({'validation_fraction': self.config['val_fraction']})

        # some algorithms allow detailed output during training, this is allowed when self.verbosity is > 1
        if regr_name in ['ONECLASSSVM']:
            kwargs.update({'verbose': True if self.verbosity>1 else False})

        if regr_name in ['TPOTREGRESSOR', 'TPOTCLASSIFIER']:
            if 'verbosity' not in kwargs:
                kwargs.update({'verbosity': self.verbosity})

        if regr_name in ml_models:
            model = ml_models[regr_name](**kwargs)
        else:
            if regr_name in ['TWEEDIEREGRESSOR', 'POISSONREGRESSOR', 'LGBMREGRESSOR', 'LGBMCLASSIFIER', 'GAMMAREGRESSOR']:
                if int(VERSION_INFO['sklearn'].split('.')[1]) < 23:
                    raise ValueError(f"{regr_name} is available with sklearn version >= 0.23 but you have {VERSION_INFO['sklearn']}")
            raise ValueError(f"model {regr_name} not found. {VERSION_INFO}")

        self._model = model

        return

    def val_data(self, **kwargs):
        """ This method can be overwritten in child classes. """
        val_data = None
        if self.config['val_data'] is not None:
            if isinstance(self.config['val_data'], str):

                assert self.config['val_data'].lower() == "same"

                # It is good that the user knows  explicitly that either of val_fraction or test_fraction is used so
                # one of them must be set to 0.0
                if self.config['val_fraction'] > 0.0:
                    warnings.warn(f"Setting val_fraction from {self.config['val_fraction']} to 0.0")
                    self.config['val_fraction'] = 0.0

                assert self.config['test_fraction'] > 0.0, f"test_fraction should be > 0.0. It is {self.config['test_fraction']}"

                if hasattr(self, 'test_indices'):
                    x, prev_y, label = self.fetch_data(self.data,  self.in_cols, self.out_cols,
                                          transformation=self.config['transformation'],  indices=self.test_indices)

                    if self.category.upper() == "ML" and self.outs == 1:
                        label = label.reshape(-1, )
                    return x, label
                return self.config['val_data']
            else:
                # `val_data` might have been provided (then use it as it is) or it is None
                val_data = self.config['val_data']

        return val_data

    def fit(self, st=0, en=None, indices=None, data=None, **callbacks):
        """data: if not None, it will directlry passed to fit."""
        self.is_training = True
        indices = self.get_indices(indices)

        inputs, _, outputs = self.train_data(st=st, en=en, indices=indices, data=data)

        if isinstance(outputs, np.ndarray) and self.category.upper() == "DL":
            if isinstance(self._model.outputs, list):
                assert len(self._model.outputs) == 1
                model_output_shape = tuple(self._model.outputs[0].shape.as_list()[1:])
                assert model_output_shape == outputs.shape[1:], f"""
ShapeMismatchError: Shape of model's output is {model_output_shape}
while the targets in prepared have shape {outputs.shape[1:]}."""

        self.info['training_start'] = dateandtime_now()

        if self.category.upper() == "DL":
            history = self._fit(inputs, outputs, self.val_data(), **callbacks)

            self.plot_loss(history.history)

        else:
            history = self._model.fit(*inputs, outputs.reshape(-1, ))
            model_name = list(self.config['model'].keys())[0]
            fname = os.path.join(self.w_path, self.category + '_' + self.problem + '_' + model_name)

            if "TPOT" not in model_name.upper():
                joblib.dump(self._model, fname)

            if model_name.lower().startswith("xgb"):

                self._model.save_model(fname + ".json")

        self.info['training_end'] = dateandtime_now()
        self.save_config()
        save_config_file(os.path.join(self.path, 'info.json'), others=self.info)

        self.is_training = False
        return history

    def test_data(self, scaler_key='5', **kwargs):
        """ just providing it so that it can be overwritten in sub-classes."""
        if 'data' in kwargs:
            if kwargs['data'] is not None:
                return kwargs['data']

        return self.train_data(scaler_key=scaler_key, **kwargs)

    def prediction_step(self, inputs):
        if self.category.upper() == "DL":
            predicted = self._model.predict(x=inputs,
                                             batch_size=self.config['batch_size'],
                                             verbose=self.verbosity)
        else:
            predicted = self._model.predict(*inputs)

        return predicted

    def predict(self, st=0, en=None, indices=None, data=None, scaler_key: str = None, pref: str = 'test',
                use_datetime_index=False, pp=True, **plot_args):
        """
        scaler_key: if None, the data will not be indexed along date_time index.
        pp: post processing
        data: if not None, this will diretly passed to predict. If data_config['transformation'] is True, do provide
            the scaler_key that was used when the data was transformed. By default that is '0'. If the data was not
            transformed with Model, then make sure that data_config['transformation'] is None.
        """

        if indices is not None:
            if not hasattr(self, 'predict_indices'):
                setattr(self, 'predict_indices', indices)

        if data is not None:
            if scaler_key is None:
                if self.config['transformation'] is not None:
                    raise ValueError(f"""The transformation argument in config_file was set to {self.config['transformation']}
                                    but a scaler_key given. Provide the either 'scaler_key' argument while calling
                                    'predict' or set the transformation while initializing model to None."""
                                     )
        if scaler_key is None:
            scaler_key = '5'

        inputs, _, true_outputs = self.test_data(st=st, en=en, indices=indices, data=data,
                                              scaler_key=scaler_key,
                                              use_datetime_index=use_datetime_index)

        first_input, inputs, dt_index = self.deindexify_input_data(inputs, use_datetime_index=use_datetime_index)

        predicted = self.prediction_step(inputs)

        if self.problem.upper().startswith("CLASS"):
            self.roc_curve(inputs, true_outputs)

        if self.config['transformation']:
            transformation = self.config['transformation']
            if isinstance(self.out_cols, dict):
                assert len(self.out_cols) == 1
                for k, out_cols in self.out_cols.items():
                    predicted, true_outputs = self.denormalize_data(inputs[k], predicted, true_outputs,
                                                                    self.in_cols[k], out_cols,
                                                                    scaler_key, transformation[k])
            else:
                predicted, true_outputs = self.denormalize_data(first_input, predicted, true_outputs,
                                                                self.in_cols, self.out_cols,
                                                                scaler_key, self.config['transformation'])

        if self.quantiles is None:

            if pp:
                self.process_results(true_outputs, predicted, prefix=pref + '_', index=dt_index, **plot_args)

        else:
            assert self.outs == 1
            self.plot_quantiles1(true_outputs, predicted)
            self.plot_quantiles2(true_outputs, predicted)
            self.plot_all_qs(true_outputs, predicted)

        return true_outputs, predicted

    def impute(self, method, imputer_args=None, inputs=False, outputs=False, cols=None):
        """impute the missing data. One of either inputs, outputs or cols can be used."""
        if cols is not None:
            if not isinstance(cols, list):
                assert isinstance(cols, str)
                cols = [cols]
            assert not inputs and not outputs
        else:
            if inputs:
                cols = self.in_cols
            elif outputs:
                cols = self.out_cols
        # if self.data is dataframe, default
        if isinstance(self.data, pd.DataFrame):
            initial_nans = sum(self.data[cols].isna().sum())
            self.data[cols] = Imputation(self.data[cols], method=method, imputer_args=imputer_args)()
            if self.verbosity>0:
                print(f"Number of nans changed from {initial_nans} to {self.data[cols].isna().sum()}")
        # may be self.data is list of dataframes
        elif isinstance(self.data, list):
            data_holder = []
            for data in self.data:
                if isinstance(data, pd.DataFrame):
                    initial_nans = data[cols].isna().sum()
                    data[cols] = Imputation(data[cols], method=method, imputer_args=imputer_args)()
                    if self.verbosity > 0:
                        print(f"Number of nans changed from {initial_nans} to {data[cols].isna().sum()}")
                data_holder.append(data)
            self.data = data_holder

        elif isinstance(self.data, dict):
            for data_name, data in self.data.values():
                if isinstance(data, pd.DataFrame):
                    initial_nans = data[cols].isna().sum()
                    data[cols] = Imputation(data[cols], method=method, imputer_args=imputer_args)()
                    if self.verbosity > 0:
                        print(f"Number of nans changed from {initial_nans} to {data[cols].isna().sum()}")
                    self.data[data_name] = data
        return

    def denormalize_data(self,
                         inputs: np.ndarray,
                         predicted: np.ndarray,
                         true: np.ndarray,
                         in_cols, out_cols,
                         scaler_key: str, transformation=None):
        """
        predicted, true are arrays of shape (examples, outs, forecast_len)
        """
        # for cases if they are 2D, add the third dimension.
        true, predicted = self.maybe_not_3d_data(true, predicted)

        if transformation:
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

                in_obs = pd.DataFrame(in_obs, columns=in_cols + out_cols)
                in_pred = pd.DataFrame(in_pred, columns=in_cols + out_cols)
                if isinstance(transformation, list):  # for cases when we used multiple transformatinos
                    for trans in transformation:
                        if trans['method'] is not None:
                            scaler = self.scalers[scaler_key + trans['method']]['scaler']
                            in_obs = Transformations(data=in_obs, **trans)(what='inverse', scaler=scaler)
                            in_pred = Transformations(data=in_pred, **trans)(what='inverse', scaler=scaler)
                elif isinstance(transformation, dict):
                    scaler = self.scalers[scaler_key]['scaler']
                    in_obs = Transformations(data=in_obs, **transformation)(what='inverse', scaler=scaler)
                    in_pred = Transformations(data=in_pred, **transformation)(what='inverse', scaler=scaler)
                else:
                    assert isinstance(transformation, str)
                    scaler = self.scalers[scaler_key]['scaler']
                    in_obs = Transformations(data=in_obs, method=transformation)(what='inverse', scaler=scaler)
                    in_pred = Transformations(data=in_pred, method=transformation)(what='inverse', scaler=scaler)

                in_obs_den = in_obs.values
                in_pred_den = in_pred.values
                true_denorm[:, :, h] = in_obs_den[:, -len(out_cols):]
                pred_denorm[:, :, h] = in_pred_den[:, -len(out_cols):]

            predicted = pred_denorm
            true = true_denorm

        return predicted, true

    def deindexify_input_data(self, inputs:list, sort:bool = False, use_datetime_index:bool = False):
        """Removes the index columns from inputs. If inputs is a list then it removes index column from each
        of the input in inputs. Index column is usually datetime column. It is added to keep track of indices of
        input data.
        """

        # `first_input` is only used to extract datetime_index.
        if isinstance(inputs, list):
            first_input = inputs[0]
        elif isinstance(inputs, dict):
            first_input = list(inputs.values())[0]
        else:
            raise NotImplementedError

        dt_index = np.arange(len(first_input))
        new_inputs = inputs

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

            if isinstance(inputs, list):
                new_inputs = []
                for _input in inputs:
                    if sort:
                        _input = _input[np.argsort(dt_index.to_pydatetime())]
                    new_inputs.append(_input[..., 1:])
            elif isinstance(inputs, dict):
                new_inputs = {}
                for inp_name, _inp in inputs.values():
                    if sort:
                        _inp = _inp[np.argsort(dt_index.to_pydatetime())]
                    new_inputs[inp_name] = _inp[..., 1:]
            else:
                raise NotImplementedError

        return first_input, new_inputs, dt_index

    def compile(self, model_inputs, outputs):

        k_model = self.KModel(inputs=model_inputs, outputs=outputs)

        opt_args = self.get_opt_args()
        optimizer = OPTIMIZERS[self.config['optimizer'].upper()](**opt_args)

        k_model.compile(loss=self.loss, optimizer=optimizer, metrics=self.get_metrics())

        if self.verbosity > 0:
            k_model.summary()

        kwargs = {}
        if int(tf.__version__.split('.')[1])  > 14:
            kwargs['dpi'] = 300

        try:
            keras.utils.plot_model(k_model, to_file=os.path.join(self.path, "model.png"), show_shapes=True, **kwargs)
        except (AssertionError, ImportError) as e:
            print("dot plot of model could not be plotted")
        return k_model

    def get_opt_args(self):
        """ get input arguments for an optimizer. It is being explicitly defined here so that it can be overwritten
        in sub-classes"""
        return {'lr': self.config['lr']}

    def get_metrics(self) -> list:
        """ returns the performance metrics specified"""
        _metrics = self.config['metrics']

        metrics = None
        if _metrics is not None:
            if not isinstance(_metrics, list):
                assert isinstance(_metrics, str)
                _metrics = [_metrics]

            from dl4seq.utils.tf_losses import nse, kge, pbias, tf_r2

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
                raise TypeError(f"unknown data type for data {type(data)}")

        # for case when there is not lookback, i.e first layer is dense layer and takes 2D input
        input_x, input_y, label_y = data[:, 0:ins], data[:, -outs:], data[:, -outs:]

        assert self.lookback == 1, """lookback should be one for MLP/Dense layer based model, but it is {}
        """.format(self.lookback)
        return self.check_nans(data, input_x, input_y, np.expand_dims(label_y, axis=2), outs, self.lookback,
                               self.config['allow_nan_labels'])

    def imputation(self, df, ins:int, outs):

        if self.config['input_nans'] is not None:

            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df, columns=self.in_cols + self.out_cols)

            if isinstance(self.config['input_nans'], list):
                # separate imputation method to be applied on each column/feature of df
                assert len(self.config['input_nans']) == ins

            elif isinstance(self.config['input_nans'], dict):
                kwargs = list(self.config['input_nans'].values())[0]

                if len(self.config['input_nans']) > 1:

                    for key, val in self.config['input_nans'].items():

                        how = list(val.keys())[0]
                        kwargs = list(val.values())[0]
                        df[key] = impute_df(pd.DataFrame(df[key]), how, **kwargs)
                else:
                    df[self.in_cols] = impute_df(df[self.in_cols], list(self.config['input_nans'].keys())[0], **kwargs)

            else:
                raise ValueError(f"Unknown value '{self.config['input_nans']} to deal with with nans in inputs")

        if isinstance(df, pd.DataFrame):
            df = df.values

        return df

    def get_batches(self, df, ins, outs):

        df = self.imputation(df, ins, outs)

        if self.num_input_layers > 1:
            # if the model takes more than 1 input, we must define what kind of inputs must be made because
            # using this method means we are feeding same shaped inputs so we can easily say, 2d or 3d. If the
            # the model takes more than 1 input and they are of different shapes, theen the user has to (is this
            # common sense?) overwrite `train_paras` and or `test_paras` methods.
            if self.config['batches'].upper() == "2D":
                return self.get_2d_batches(df, ins, outs)
            else:
                return self.check_nans(df, *make_3d_batches(df,
                                                            num_outputs=outs,
                                                            lookback_steps=self.lookback,
                                                            input_steps=self.config['input_step'],
                                                            forecast_step=self.forecast_step,
                                                            forecast_len=self.forecast_len),
                                       outs, self.lookback,
                                       self.config['allow_nan_labels'])
        else:
            if len(self.first_layer_shape()) == 2:
                return self.get_2d_batches(df, ins, outs)

            else:
                return self.check_nans(df, *make_3d_batches(df,
                                                            num_outputs=outs,
                                                            lookback_steps=self.lookback,
                                                            input_steps=self.config['input_step'],
                                                            forecast_step=self.forecast_step,
                                                            forecast_len=self.forecast_len),
                                       outs, self.lookback,
                                       self.config['allow_nan_labels'])

    def check_nans(self, data, input_x, input_y, label_y, outs, lookback, allow_nan_labels, allow_input_nans=False):
        """Checks whether anns are present or not and checks shapes of arrays being prepared.
        """
        # TODO, nans in inputs should be ignored at all cost because this causes error in results, when we set allow_nan_labels
        # to True, then this should apply only to target/labels, and examples with nans in inputs should still be ignored.
        if isinstance(data, pd.DataFrame):
            nans = data[self.out_cols].isna()
            data = data.values
        else:
            nans = np.isnan(data[:, -outs:]) # df[self.out_cols].isna().sum()
        if int(nans.sum()) > 0:
            if allow_nan_labels == 2:
                print("\n{} Ignoring NANs in predictions {}\n".format(10 * '*', 10 * '*'))
            elif allow_nan_labels == 1:
                print("\n{} Ignoring examples whose all labels are NaNs {}\n".format(10 * '*', 10 * '*'))
                idx = ~np.array([all([np.isnan(x) for x in label_y[i]]) for i in range(len(label_y))])
                input_x = input_x[idx]
                input_y = input_y[idx]
                label_y = label_y[idx]
                if int(np.isnan(data[:, -outs:][0:lookback]).sum() / outs) >= lookback:
                    self.nans_removed_4m_st = -9999
            else:
                if self.method == 'dual_attention':
                    raise ValueError
                if outs > 1:
                    assert np.all(nans == int(nans.sum() / outs)), """output columns contains nan values at
                     different indices"""

                if self.verbosity > 0:
                    print('\n{} Removing Samples with nan labels  {}\n'.format(10 * '*', 10 * '*'))
                if outs == 1:
                    # find out how many nans were present from start of data until lookback, these nans will be removed
                    self.nans_removed_4m_st = np.isnan(data[:, -outs:][0:lookback]).sum()
                nan_idx = np.isnan(label_y) if outs == 1 else np.isnan(label_y[:, 0])  # y.isna()
                # nan_idx_t = nan_idx[self.lookback - 1:]
                # non_nan_idx = ~nan_idx_t.values
                non_nan_idx = ~nan_idx.reshape(-1, )
                label_y = label_y[non_nan_idx]
                input_x = input_x[non_nan_idx]
                input_y = input_y[non_nan_idx]

                assert np.isnan(label_y).sum() < 1, "label still contains {} nans".format(np.isnan(label_y).sum())

        assert input_x.shape[0] == input_y.shape[0] == label_y.shape[0], "shapes are not same"

        if not allow_input_nans:
            assert np.isnan(input_x).sum() == 0, "input still contains {} nans".format(np.isnan(input_x).sum())

        return input_x, input_y, label_y

    def activations(self, layer_names=None, **kwargs):
        # if layer names are not specified, this will get get activations of allparameters
        inputs, _, _ = self.test_data(**kwargs)

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

        x, _, y = self.test_data(**kwargs)

        _, x, _ = self.deindexify_input_data(x, sort=True, use_datetime_index=kwargs.get('use_datetime_index', False))

        return keract.get_gradients_of_trainable_weights(self._model, x, y)

    def gradients_of_activations(self, st=0, en=None, indices=None, data=None, layer_name=None, **kwargs) -> dict:

        x, _, y = self.test_data(st=st, en=en, indices=indices, data=data)

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

    def find_num_lstms(self)->list:
        """Finds names of lstm layers in model"""

        lstm_names = []
        for lyr, config in self.config['layers'].items():
            if "LSTM" in lyr.upper():

                prosp_name = config["config"]['name'] if "name" in config['config'] else lyr

                lstm_names.append(prosp_name)

        return lstm_names

    def get_rnn_weights(self, weights:dict)->dict:
        """Finds RNN related weights and combine kernel, recurrent curnel and bias
        of each layer into a list."""
        lstm_weights = {}
        if "LSTM" in self.config['model']['layers']:
            lstms = self.find_num_lstms()
            for lstm in lstms:
                lstm_w = []
                for w in ["kernel", "recurrent_kernel", "bias"]:
                    w_name = lstm + "/lstm_cell/" + w
                    for k,v in weights.items():
                        if w_name in k:
                            lstm_w.append(v)

                lstm_weights[lstm] = lstm_w

        return lstm_weights

    def plot_weights(self, save=True):
        weights = self.trainable_weights()

        if self.verbosity > 0:
            print("Plotting trainable weights of layers of the model.")

        rnn_weights = self.get_rnn_weights(weights)
        for k,w in rnn_weights.items():
            self.rnn_histogram(w, name=k+"_weight_histogram", save=save)

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

    def plot_layer_outputs(self, save: bool = True, **kwargs):
        """Plots outputs of intermediate layers except input and output.
        If called without any arguments then it will plot outputs of all layers."""
        activations, _ = self.activations(**kwargs)

        if self.verbosity > 0:
            print("Plotting activations of layers")

        for lyr_name, activation in activations.items():
            # activation may be tuple e.g if input layer receives more than 1 input
            if isinstance(activation, np.ndarray):
                self._plot_layer_outputs(activation, lyr_name, save)

            elif isinstance(activation, tuple):
                for act in activation:
                    self._plot_layer_outputs(act, lyr_name, save)
        return

    def _plot_layer_outputs(self, activation, lyr_name, save):

        if "LSTM" in lyr_name.upper() and np.ndim(activation) in (2, 3):

            self.features_2d(activation, save=save, name=lyr_name+"_outputs", title="Outputs", norm=(-1, 1))

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
        for k,w in rnn_weights.items():
            self.rnn_histogram(w, name=k+"_weight_grads_histogram", save=save)

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

    def _plot_act_grads(self, gradients,  save=True):
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
        """ shows all activations, weights and gradients of the keras model."""
        if 'layers' not in self.config['model']:

            self.plot_feature_importance()

            self.plot_treeviz_leaves()

            if self.problem.lower().startswith("cl"):
                self.plot_treeviz_leaves()
                self.decision_tree(which="sklearn", **kwargs)

                x, _, y = self.test_data()
                self.confusion_matrx(x=x, y=y)
                self.precision_recall_curve(x=x, y=y)
                self.roc_curve(x=x, y=y)


            if list(self.config['model'].keys())[0].lower().startswith("xgb"):
                self.decision_tree(which="xgboost", **kwargs)

        if self.category.upper() == "DL":
            self.plot_act_grads(**kwargs)
            self.plot_weight_grads(**kwargs)
            self.plot_layer_outputs(**kwargs)
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
        target = self.config['outputs'][0]

        x = np.zeros((len(df), self.lookback, df.shape[1] - 1))
        prev_y = np.zeros((len(df), self.lookback, 1))

        for i, name in enumerate(list(df.columns[:-1])):
            for j in range(self.lookback):
                x[:, j, i] = df[name].shift(self.lookback - j - 1).fillna(method="bfill")

        for j in range(self.lookback):
            prev_y[:, j, 0] = df[target].shift(self.lookback - j - 1).fillna(method="bfill")

        fl = self.config['forecast_length']
        _y = np.zeros((df.shape[0], fl))
        for i in range(df.shape[0] - fl):
            _y[i - 1, :] = df[target].values[i:i + fl]

        input_x = x[self.lookback:-fl, :]
        prev_y = prev_y[self.lookback:-fl, :]
        y = _y[self.lookback:-fl, :].reshape(-1, outs, self.forecast_len)

        return self.check_nans(df, input_x, prev_y, y, outs, self.lookback, self.config['allow_nan_labels'])

    def save_indices(self):
        indices={}
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

    def save_config(self, history: dict=None):

        self.save_indices()

        config = dict()
        if history is not None:
            config['min_val_loss'] = int(np.nanmin(history['val_loss'])) if 'val_loss' in history else None
            config['min_loss'] = int(np.nanmin(history['loss'])) if 'val_loss' in history else None

        config['config'] = self.config
        config['method'] = self.method
        config['category'] = self.category
        config['problem'] = self.problem
        config['quantiles'] = self.quantiles

        if self.category == "DL":
            config['loss'] = self.loss_name()

        save_config_file(config=config, path=self.path)
        return config

    def loss_name(self):
        if isinstance(self._model.loss, str):
            return self._model.loss
        else:
            return self._model.loss.__name__

    @classmethod
    def from_config(cls, config_path: str, data, make_new_path=False, **kwargs):
        """
        :param config_path:
        :param data:
        :param make_new_path: bool, If true, then it means we want to use the config file, only to build the model and a new
                              path will be made. We should not load the weights in such a case.
        :param kwargs:
        :return:
        """
        with open(config_path, 'r') as fp:
            config = json.load(fp)

        idx_file = os.path.join(os.path.dirname(config_path), 'indices.json')
        with open(idx_file, 'r') as fp:
            indices = json.load(fp)

        cls.from_check_point = True

        # These paras neet to be set here because they are not withing init method
        cls.test_indices = indices["test_indices"]
        cls.train_indices = indices["train_indices"]

        if make_new_path:
            cls.allow_weight_loading=False
            path=None
        else:
            cls.allow_weight_loading = True
            path = os.path.dirname(config_path)

        return cls(**config['config'],
                   data=data,
                   path=path,
                   **kwargs)

    def load_weights(self, weight_file: str):
        """
        weight_file: str, name of file which contains parameters of model.
        """
        weight_file = os.path.join(self.w_path, weight_file)
        if not self.allow_weight_loading:
            raise ValueError(f"Weights loading not allowed because allow_weight_loading is {self.allow_weight_loading}"
                             f"and model path is {self.path}")

        if self.category == "ML":
            if list(self.config['model'].keys())[0].lower().startswith("xgb"):
                self._model.load_model(weight_file)
            else:
                # for sklearn based models
                self._model = joblib.load(weight_file)
        else:
            # loads the weights of keras model from weight file `w_file`.
            self._model.load_weights(weight_file)
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

    def eda(self, freq=None, cols=None, **kwargs):
        """Performs comprehensive Exploratory Data Analysis.
        freq: str, if specified, small chunks of data will be plotted instead of whole data at once. The data will NOT
        be resampled. This is valid only `plot_data` and `box_plot`. Possible values are `yearly`, weekly`, and
        `monthly`."""
        # plot number if missing vals
        self.plot_missing(cols=cols)

        # show data as heatmapt
        self.data_heatmap(cols=cols)

        # line plots of input/output data
        self.plot_data(cols=cols, freq=freq, subplots=True, figsize=(12, 14), sharex=True)

        # plot feature-feature correlation as heatmap
        self.plot_feature_feature_corr(cols=cols)

        # print stats about input/output data
        self.stats()

        # box-whisker plot
        self.box_plot(freq=freq)

        # principle components
        self.plot_pcs()

        # scatter plots of input/output data
        self.grouped_scatter(cols=cols)

        # distributions as histograms
        self.plot_histograms(cols=cols)

        return

    def stats(self, precision=3, inputs=True, outputs=True, fpath=None, out_fmt="csv"):
        """Finds the stats of inputs and outputs and puts them in a json file.
        inputs: bool
        fpath: str, path like
        out_fmt: str, in which format to save. csv or json"""
        cols = []
        fname = "description_"
        if inputs:
            cols += self.in_cols
            fname += "inputs_"
        if outputs:
            cols += self.out_cols
            fname += "outputs_"

        fname += str(dateandtime_now())

        def save_stats(_description, fpath):

            if out_fmt == "csv":
                pd.DataFrame.from_dict(_description).to_csv(fpath + ".csv")
            else:
                save_config_file(others=_description, path=fpath + ".json")

        description = {}
        if isinstance(self.data, pd.DataFrame):
            description = {}
            for col in cols:
                if col in self.data:
                    description[col] = stats(self.data[col], precision=precision, name=col)

            fpath = os.path.join(self.data_path, fname) if fpath is None else fpath
            save_stats(description, fpath)

        elif isinstance(self.data, list):
            description = {}

            for idx, data in enumerate(self.data):
                _description = {}

                if isinstance(data, pd.DataFrame):

                    for col in cols:
                        if col in data:
                            _description[col] = stats(data[col], precision=precision, name=col)

                description['data' + str(idx)] = _description
                _fpath = os.path.join(self.data_path, fname+f'_{idx}') if fpath is None else fpath
                save_stats(_description, _fpath)
        else:
            print(f"description can not be found for data type of {type(self.data)}")

        return description



def impute_df(df:pd.DataFrame, how:str, **kwargs):
    """Given the dataframe df, will input missing values by how e.g. by df.fillna or df.interpolate"""
    if  how.lower() not in ['fillna', 'interpolate', 'knnimputer', 'iterativeimputer', 'simpleimputer']:
        raise ValueError(f"Unknown method to fill missing values `{how}`.")

    if how.lower() in ['fillna', 'interpolate']:
        for col in df.columns:
            df[col] = getattr(df[col], how)(**kwargs)
    else:
        imputer = imputations[how.upper()](**kwargs)
        df = imputer.fit_transform(df.values)

    return df


def unison_shuffled_copies(a, b, c):
    """makes sure that all the arrays are permuted similarly"""
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]


def print_something(something, prefix=''):
    """prints shape of some python object"""
    if isinstance(something, np.ndarray):
        print(f"{prefix} shape: ", something.shape)
    elif isinstance(something, list):
        print(f"{prefix} shape: ", [thing.shape for thing in something if isinstance(thing, np.ndarray)])
    elif isinstance(something, dict):
        print(f"{prefix} shape: ", [thing.shape for thing in something.values() if isinstance(thing, np.ndarray)])