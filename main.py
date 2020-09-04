__all__ = ["Model"]

import numpy as np
from weakref import WeakKeyDictionary
import pandas as pd
from TSErrors import FindErrors
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import json

import keract_mod as keract
from models.global_variables import keras, tf, tcn
from utils import plot_results, plot_loss, maybe_create_path, save_config_file
from models.global_variables import LOSSES, ACTIVATION_LAYERS, ACTIVATION_FNS


KModel = keras.models.Model
layers = keras.layers

tf.compat.v1.disable_eager_execution()
# tf.enable_eager_execution()

np.random.seed(313)
if int(tf.__version__[0]) == 1:
    tf.compat.v1.set_random_seed(313)
elif int(tf.__version__[0]) > 1:
    tf.random.set_seed(313)

np.set_printoptions(suppress=True) # to suppress scientific notation while printing arrays

class AttributeNotSetYet:
    def __init__(self, func_name):
        self.data = WeakKeyDictionary()
        self.func_name = func_name

    def __get__(self, instance, owner):
        raise AttributeError("run the function {} first to get {}".format(self.func_name, self.name))

    def __set_name__(self, owner, name):
        self.name = name

class AttributeStore(object):
    """ a class which will just make sure that attributes are set at its childs class level and not here.
    It's purpose is just to avoid cluttering of __init__ method of its child classes. """
    k_model = AttributeNotSetYet("`build_nn` to build neural network")
    method = None
    ins = None
    outs = None
    en_densor_We = None
    en_LSTM_cell = None
    auto_enc_composite = None
    de_LSTM_cell = None
    de_densor_We = None
    test_indices = None
    train_indices = None
    scalers = {}
    cnn_counter = 0
    lstm_counter = 0
    act_counter = 0
    time_dist_counter = 0
    conv2d_lstm_counter = 0
    run_paras = AttributeNotSetYet("You must define the `run_paras` method first")


class Model(AttributeStore):

    def __init__(self, data_config: dict,
                 nn_config: dict,
                 data: pd.DataFrame,
                 intervals=None,
                 path: str=None):
        self.data_config = data_config
        self.nn_config = nn_config
        self.intervals = intervals
        self.data = data[data_config['inputs'] + data_config['outputs']]
        self.in_cols = self.data_config['inputs']
        self.out_cols = self.data_config['outputs']
        self.loss = LOSSES[self.nn_config['loss']]
        self.KModel = KModel
        self.path = maybe_create_path(path=path)

    @property
    def data(self):
        # so that .data attribute can be customized
        return self._data

    @data.setter
    def data(self, x):
        self._data = x

    @property
    def intervals(self):
        # so that itnervals can be set from outside the Model class. A use case can be when we want to implement
        # intervals during train time but not during test/predict time.
        return self._intervals

    @intervals.setter
    def intervals(self, x:list):
        self._intervals = x

    @property
    def lookback(self):
        return self.data_config['lookback']

    @property
    def in_cols(self):
        return self._in_cols

    @in_cols.setter
    def in_cols(self, x:list):
        self._in_cols = x

    @property
    def out_cols(self):
        return self._out_cols

    @out_cols.setter
    def out_cols(self, x:list):
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
        for layer in self.k_model.layers:
            _all_layers.append(layer.name)
        return _all_layers

    def fetch_data(self, data: pd.DataFrame,  st: int, en=None,
                   shuffle: bool = True,
                   cache_data=True,
                   noise: int = 0,
                   indices: list = None,
                   scaler_key:str = '0',
                   return_dt_index=False):
        """

        :param data:
        :param st:
        :param en:
        :param shuffle:
        :param cache_data:
        :param noise:
        :param indices:
        :param scaler_key: in case we are calling fetch_data multiple times, each data will be scaled with a unique
                   MinMaxScaler object and can be saved with a unique key in memory.
        :param return_dt_index: if True, first value in returned `x` will be datetime index. This can be used when
                                fetching data during predict but must be separated before feeding in NN for prediction.
        :return:
        """

        if indices is not None:
            assert isinstance(indices, list), "indices must be list"
            if en is not None or st != 0:
                raise ValueError

        df = data
        dt_index=None
        if return_dt_index:
            assert isinstance(data.index, pd.DatetimeIndex), """\nInput dataframe must have index of type pd.DateTimeIndex.
            A dummy datetime index can be inserted by using following command:
            `data.index = pd.date_range("20110101", periods=len(data), freq='H')`
            If the data file already contains `datetime` column, then use following command
            `data.index = pd.to_datetime(data['datetime'])`
            or use set `use_datetime_index` in `predict` method to False.
            """
            dt_index = list(map(int, np.array(data.index.strftime('%Y%m%d%H%M'))))  # datetime index

        # # add random noise in the data
        if noise > 0:
            x = pd.DataFrame(np.random.randint(0, 1, (len(df), noise)))
            df = pd.concat([df, x], axis=1)
            prev_inputs = self.in_cols
            self.in_cols = prev_inputs + list(x.columns)
            ys = []
            for y in self.out_cols:
                ys.append(df.pop(y))
            df[self.out_cols] = ys

        cols = self.in_cols + self.out_cols
        df = df[cols]

        scaler = MinMaxScaler()
        data = scaler.fit_transform(df)
        df = pd.DataFrame(data)

        if return_dt_index:
            # pandas will add the 'datetime' column as first column. This columns will only be used to keep
            # track of indices of train and test data.
            df.insert(0, 'dt_index', dt_index)
            self.in_cols = ['dt_index'] + self.in_cols

        self.scalers['scaler_' + scaler_key] = scaler

        if en is None:
            en = df.shape[0]

        if self.intervals is None:

            df = df[st:en]
            df.columns = self.in_cols + self.out_cols

            x, y, label = self.get_data(df,
                                        len(self.in_cols),
                                        len(self.out_cols)
                                        )
            if indices is not None:
                # because the x,y,z have some initial values removed
                indices = np.subtract(np.array(indices), self.lookback - 1).tolist()

                # if indices are given then this should be done after `get_data` method
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
                    x, y, label = self.get_data(df1,
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

        # data_path = self.data_config['data_path']
        # cache_path = os.path.join(os.path.dirname(data_path), 'cache')
        #
        # if cache_data and os.path.isdir(cache_path) is False:
        #     os.mkdir(cache_path)
        # fname = os.path.join(cache_path, 'nasdaq_T{}.h5'.format(self.lookback))
        # if os.path.exists(fname) and cache_data:
        #     input_x, input_y, label_y = read_cache(fname)
        #     print('load %s successfully' % fname)
        # else:
        #     input_x, input_y, label_y = get_data(data_path, self.lookback, st, en)
        #     if cache_data:
        #         write_cache(fname, input_x, input_y, label_y)

        if shuffle:
            x, y, label = unison_shuffled_copies(x, y, label)

        print('input_X shape:', x.shape)
        print('input_Y shape:', y.shape)
        print('label_Y shape:', label.shape)

        if return_dt_index:
            self.in_cols.remove("dt_index")
        else:
            x = x.astype(np.float32)

        return x, y, label

    def fit(self, inputs, outputs, **callbacks):

        _callbacks = list()

        _monitor = 'val_loss' if self.data_config['val_fraction'] > 0.0 else 'loss'
        _callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=self.path + "\\weights_{epoch:03d}_{val_loss:.4f}.hdf5",
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
        self.k_model.fit(inputs,
                         outputs,
                         epochs=self.nn_config['epochs'],
                         batch_size=self.data_config['batch_size'],
                         validation_split=self.data_config['val_fraction'],
                         callbacks=_callbacks
                         )
        history = self.k_model.history

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
                    tot_obs = self.data.shape[0] - int(self.data[self.out_cols].isna().sum()) - self.data_config['batch_size']
                else:
                    # data contains nans and target series are > 1, we want to make sure that they have same nan counts
                    tot_obs = self.data.shape[0] - int(self.data[self.out_cols[0]].isna().sum())
                    nans = self.data[self.out_cols].isna().sum()
                    assert np.all(nans.values == int(nans.sum() / self.outs))

            idx = np.arange(tot_obs - self.lookback)
            train_indices, test_idx = train_test_split(idx, test_size=self.data_config['val_fraction'], random_state=313)
            setattr(self, 'test_indices', list(test_idx))
            indices = list(train_indices)

        setattr(self, 'train_indices', indices)

        return indices

    def run_paras(self, **kwargs):

        x, y, label = self.fetch_data(self.data, **kwargs)
        return [x], label

    def process_results(self, true:list, predicted:list, name=None, **plot_args):

        errs = dict()
        for out in range(self.outs):

            t = true[out]
            p = predicted[out]

            if np.isnan(t).sum() > 0:
                mask = np.invert(np.isnan(t.reshape(-1,)))
                t = t[mask]
                p = p[mask]

            errors = FindErrors(t, p)
            errs['out'] = errors.calculate_all()

            plot_results(t, p, name=os.path.join(self.path, name + self.data_config['outputs'][out]),
                         **plot_args)

        save_config_file(self.path, errors=errs)

        return

    def build_nn(self):
        print('building simple lstm model')

        # lstm = self.nn_config['lstm_config']

        inputs, predictions = self.simple_lstm(self.nn_config['lstm_config'], outs=self.outs)

        self.k_model = self.compile(inputs, predictions)

        return

    def train_nn(self, st=0, en=None, indices=None, **callbacks):

        indices = self.get_indices(indices)

        inputs, outputs = self.run_paras(st=st, en=en, indices=indices)

        history = self.fit(inputs, outputs, **callbacks)

        plot_loss(history.history, name=os.path.join(self.path, "loss_curve"))

        return history

    def predict(self, st=0, en=None, indices=None, scaler_key:str='5', pref:str='test',
                use_datetime_index=True, **plot_args):
        """
        scaler_key: if None, the data will not be indexed along date_time index.

        """

        if indices is not None:
            setattr(self, 'predict_indices', indices)

        inputs, true_outputs = self.run_paras(st=st, en=en, indices=indices, scaler_key=scaler_key,
                                              return_dt_index=use_datetime_index)

        first_input = inputs[0]
        dt_index=None  # default case when datetime_index is not present in input data
        if use_datetime_index:
            # remove the first of first inputs which is datetime index
            dt_index = get_index(np.array(first_input[:, -1, 0], dtype=np.int64))
            first_input1 = first_input[:, :, 1:].astype(np.float32)
            inputs[0] = first_input1

        predicted = self.k_model.predict(x=inputs,
                                         batch_size=self.data_config['batch_size'],
                                         verbose=1)

        if self.outs == 1:
            # denormalize the data
            if np.ndim(first_input) > 3:
                first_input = first_input[:, -1, 0, :]
            else:
                first_input = first_input[:, -1, :]
            in_obs = np.hstack([first_input, true_outputs])
            in_pred = np.hstack([first_input, predicted])
            scaler = self.scalers['scaler_'+scaler_key]
            in_obs_den = scaler.inverse_transform(in_obs)
            in_pred_den = scaler.inverse_transform(in_pred)
            true_outputs = in_obs_den[:, -self.outs]
            predicted = in_pred_den[:, -self.outs]

        if not isinstance(true_outputs, list):
            true_outputs = [true_outputs]
        if not isinstance(predicted, list):
            predicted = [predicted]

        # convert each output in ture_outputs and predicted lists as pd.Series with datetime indices sorted
        true_outputs = [pd.Series(t_out, index=dt_index).sort_index() for t_out in true_outputs]
        predicted = [pd.Series(p_out, index=dt_index).sort_index() for p_out in predicted]

        # save the results
        for idx, out in enumerate(self.out_cols):
            p = predicted[idx]
            t = true_outputs[idx]
            df = pd.DataFrame(np.stack([p, t], axis=1), columns=['true_' + str(out), 'pred_' + str(out)],
                              index=dt_index)
            df.to_csv(os.path.join(self.path, pref + '_' + str(out) + ".csv"), index_label='time')

        self.process_results(true_outputs, predicted, pref+'_', **plot_args)

        return predicted, true_outputs

    def add_lstm(self, inputs, config):

        self.lstm_counter += self.lstm_counter

        if 'name' not in config:
            config['name'] = 'lstm_lyr_' + str(self.lstm_counter)

        # check whether to apply any activation layer or not.
        act_layer = check_act_layer(config)

        config, activation = check_act_fn(config)

        lstm_activations = layers.LSTM(**config)(inputs)
        config['activation'] = activation

        # apply activation layer if applicable
        lstm_activations = self.add_act_layer(act_layer, lstm_activations)

        return lstm_activations

    def conv2d_lstm(self, config:dict, inputs):
        """
        Adds a CONVLSTM2D layer. It is possible to add activation layer after conv2d layer.
        """
        self.conv2d_lstm_counter += self.conv2d_lstm_counter
        if 'name' not in config:
            config['name'] = 'conv_lstm_' + str(self.conv2d_lstm_counter)

        # checks whether to apply activation layer at the end or not
        act_layer = check_act_layer(config)

        config, activation = check_act_fn(config)

        conv_lstm_outs = keras.layers.ConvLSTM2D(**config)(inputs)

        # apply activation layer if applicable
        conv_lstm_outs = self.add_act_layer(act_layer, conv_lstm_outs)

        return  keras.layers.Flatten()(conv_lstm_outs)


    def add_1d_cnn(self, inputs, config: dict):
        """
        adds a 1d CNN collowed by an optional activation layer, max pool layer and then flattens the output
        Fore more options in in `config` file see
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
        """
        self.cnn_counter += 1
        if 'name' in config:
            rand_name = config['name']
        else:
            rand = '1dcnn_' + str(self.cnn_counter)
            rand_name = rand
            config['name'] = rand

        # check whether to apply any activation layer or not.
        act_layer = check_act_layer(config)

        config, activation = check_act_fn(config)

        max_pool_size = config.pop('max_pool_size')

        cnn_activations = layers.Conv1D(**config)(inputs)
        config['activation'] = activation

        # apply activation layer if applicable
        cnn_activations = self.add_act_layer(act_layer, cnn_activations)

        max_pool_lyr = layers.MaxPooling1D(pool_size=max_pool_size,
                                           name='max_pool_lyr_'+rand_name)(cnn_activations)

        flat_lyr = layers.Flatten(name='flat_lyr_'+rand_name)(max_pool_lyr)

        return flat_lyr

    def cnn_model(self, cnn_config:dict, outs:int):
        """ basic structure of a simple CNN based model"""

        inputs = layers.Input(shape=(self.lookback, self.ins))

        # check whether to apply any activation layer or not.
        act_layer = check_act_layer(cnn_config)

        cnn_configs = check_cnn_config(cnn_config)

        cnn_outputs = inputs
        cnn_inputs = inputs
        for cnn in cnn_configs:
            cnn_outputs = self.add_1d_cnn(cnn_inputs, cnn)
            cnn_inputs = cnn_outputs

        # apply activation layer if applicable
        cnn_outputs = self.add_act_layer(act_layer, cnn_outputs)

        predictions = layers.Dense(outs)(cnn_outputs)

        return inputs, predictions

    def simple_lstm(self, lstm_config:dict, outs:int):
        """ basic structure of a simple LSTM based model"""

        inputs = layers.Input(shape=(self.lookback, self.ins))

        # check whether to apply any activation layer or not.
        act_layer = check_act_layer(lstm_config)

        lstm_configs = check_lstm_config(lstm_config)

        lstm_activations = inputs
        lstm_inputs = inputs
        for lstm in lstm_configs:
            lstm_activations = self.add_lstm(lstm_inputs, lstm)
            lstm_inputs = lstm_activations

        # apply activation layer if applicable
        lstm_activations = self.add_act_layer(act_layer, lstm_activations)

        predictions = layers.Dense(outs)(lstm_activations)

        return inputs, predictions

    def add_time_dist_cnn(self, cnn_config:dict, inputs):

        # check whether to apply any activation layer or not.
        act_layer = check_act_layer(cnn_config)

        cnn_config, activation = check_act_fn(cnn_config)

        if 'name' not in cnn_config:
            self.cnn_counter += 1
            cnn_config['name'] = 'td_cnn_' + str(self.cnn_counter)

        self.time_dist_counter += 1
        name = 'time_dist_' + str(self.time_dist_counter)

        cnn_lyr = layers.TimeDistributed(layers.Conv1D(**cnn_config),
                                         name=name)(inputs)
        cnn_config['activation'] = activation

        # apply activation layer if applicable
        cnn_lyr = self.add_act_layer(act_layer, cnn_lyr, True)

        return cnn_lyr

    def cnn_lstm(self, cnn_config:dict,  lstm_config:dict,  outs:int):
        """ blue print for developing a CNN-LSTM model.
        :param cnn_config: con contain input arguments for one or more than one cnns, all given as dictionaries
        :param lstm_config: cna contain input arguments for one or more than one LSTM layers, all given as dictionaries
        :param outs:
        It is possible to insert activation layer after each cnn/lstm and/or after whole lstm/cnn block
        """
        timesteps = self.lookback // self.nn_config['subsequences']

        inputs = layers.Input(shape=(None, timesteps, self.ins))

        # check whether to apply any activation layer or not.
        act_layer = check_act_layer(cnn_config)

        cnn_configs = check_cnn_config(cnn_config)

        cnn_outputs = inputs
        cnn_inputs = inputs
        for cnn in cnn_configs:
            cnn_outputs = self.add_time_dist_cnn(cnn, cnn_inputs)
            cnn_inputs = cnn_outputs

        # apply activation layer if applicable
        cnn_outputs = self.add_act_layer(act_layer, cnn_outputs, True)

        max_pool_lyr = layers.TimeDistributed(layers.MaxPooling1D(pool_size=cnn_config['max_pool_size']))(cnn_outputs)
        flat_lyr = layers.TimeDistributed(layers.Flatten())(max_pool_lyr)

        # check whether to apply any activation layer or not.
        act_layer = check_act_layer(lstm_config)

        lstm_configs = check_lstm_config(lstm_config)

        lstm_activations = flat_lyr
        lstm_inputs = flat_lyr
        for lstm in lstm_configs:
            lstm_activations = self.add_lstm(lstm_inputs, lstm)
            lstm_inputs = lstm_activations

        # apply activation layer if applicable
        lstm_activations = self.add_act_layer(act_layer, lstm_activations, True)

        predictions = layers.Dense(outs)(lstm_activations)

        return inputs, predictions

    def dense_layers(self, out_config:dict, dense_inputs):
        """ adds one or more dense layers following their configurations in `out_config`.
        out_config={1:{units=1}}  one dense layer which outputs 1
        out_config={8: {units=8}, 4: {units=2}, 1: {units=1, use_bias=False}}  adds three dense layers.
        For more options of a dense layer see https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        """
        dense_outs = None
        for dense_units, dense_config in out_config.items():
            dense_outs = layers.Dense(**dense_config)(dense_inputs)
            dense_inputs = dense_outs

        return dense_outs

    def lstm_cnn(self, lstm_config:dict, cnn_config:dict, outs:int):
        """ basic structure of a simple LSTM -> CNN based model"""

        inputs = layers.Input(shape=(self.lookback, self.ins))

        lstm_config['return_sequences'] = True   # forcing it to be True

        lstm_activations = self.add_lstm(inputs, lstm_config)

        cnn_outputs = self.add_1d_cnn(lstm_activations, cnn_config)

        predictions = layers.Dense(outs)(cnn_outputs)

        return inputs, predictions

    def lstm_autoencoder(self, enc_config:dict, dec_config:dict, composite:bool, outs:int):
        """
        basic structure of an LSTM autoencoder
        """
        inputs = layers.Input(shape=(self.lookback, self.ins))

        # build encoder
        encoder = self.add_lstm(inputs, enc_config)

        # define predict decoder
        decoder1 = layers.RepeatVector(self.lookback-1)(encoder)
        decoder1 = self.add_lstm(decoder1, dec_config)
        decoder1 = layers.Dense(outs)(decoder1)
        if composite:
            # define reconstruct decoder
            decoder2 = layers.RepeatVector(self.lookback)(encoder)
            decoder2 = layers.LSTM(100, activation='relu', return_sequences=True)(decoder2)
            decoder2 = layers.TimeDistributed(layers.Dense(self.ins))(decoder2)

            outputs = [decoder1, decoder2]
        else:
            outputs = decoder1

        return inputs, outputs

    def tcn_model(self, tcn_options:dict, outs:int):
        """
        basic tcn model
        """
        tcn_options['return_sequences'] = False

        inputs = layers.Input(batch_shape=(None, self.lookback, self.ins))

        tcn_outs = tcn.TCN(**tcn_options)(inputs)  # The TCN layers are here.
        predictions = layers.Dense(outs)(tcn_outs)

        return inputs,predictions

    def conv_lstm_model(self, enc_config:dict, dec_config:dict, outs:int):
        """
         basic structure of a ConvLSTM based encoder decoder model.
        """
        sub_seq = self.nn_config['subsequences']
        sub_seq_lens = int(self.lookback / sub_seq)

        inputs = keras.layers.Input(shape=(sub_seq, 1, sub_seq_lens, self.ins))

        enc_outs = self.conv2d_lstm(enc_config, inputs)

        interm_outs = keras.layers.RepeatVector(self.outs)(enc_outs)

        lstm_outs = self.add_lstm(interm_outs, dec_config)

        predictions = keras.layers.Dense(outs)(lstm_outs)

        return inputs, predictions

    def compile(self, model_inputs, outputs):

        k_model = self.KModel(inputs=model_inputs, outputs=outputs)
        adam = keras.optimizers.Adam(lr=self.nn_config['lr'])
        k_model.compile(loss=self.loss, optimizer=adam, metrics=['mse'])
        k_model.summary()

        try:
            keras.utils.plot_model(k_model, to_file=os.path.join(self.path, "model.png"), show_shapes=True, dpi=300)
        except AssertionError:
            print("dot plot of model could not be plotted")
        return k_model

    def get_data(self, df, ins, outs):
        input_x = []
        input_y = []
        label_y = []

        row_length = len(df)
        column_length = df.columns.size
        for i in range(row_length - self.lookback+1):
            x_data = df.iloc[i:i+self.lookback, 0:column_length-outs]
            y_data = df.iloc[i:i+self.lookback-1, column_length-outs:]
            label_data = df.iloc[i+self.lookback-1, column_length-outs:]
            input_x.append(np.array(x_data))
            input_y.append(np.array(y_data))
            label_y.append(np.array(label_data))
        input_x = np.array(input_x, dtype=np.float64).reshape(-1, self.lookback, ins)
        input_y = np.array(input_y, dtype=np.float32).reshape(-1, self.lookback-1, outs)
        label_y = np.array(label_y, dtype=np.float32).reshape(-1, outs)

        nans = df[self.out_cols].isna().sum()
        if int(nans.sum()) > 0:
            if self.data_config['ignore_nans']:
                print("\n{} Ignoring NANs in predictions {}\n".format(10*'*', 10*'*'))
            else:
                if self.method == 'dual_attention':
                    raise ValueError
                if outs> 1:
                    assert np.all(nans.values == int(nans.sum()/outs)), "output columns contains nan values at different indices"
                print('\n{} Removing Samples with nan labels  {}\n'.format(10*'*', 10*'*'))
                y = df[df.columns[-1]]
                nan_idx = y.isna()
                nan_idx_t = nan_idx[self.lookback - 1:]
                non_nan_idx = ~nan_idx_t.values
                label_y = label_y[non_nan_idx]
                input_x = input_x[non_nan_idx]
                input_y = input_y[non_nan_idx]

                assert np.isnan(label_y).sum() < 1, "label still contains nans"

        assert input_x.shape[0] == input_y.shape[0] == label_y.shape[0], "shapes are not same"

        assert np.isnan(input_x).sum() == 0, "input still contains nans"

        return input_x, input_y, label_y

    def activations(self,layer_names=None, **kwargs):
        # if layer names are not specified, this will get get activations of allparameters
        inputs, outputs = self.run_paras(**kwargs)

        activations = keract.get_activations(self.k_model, inputs, layer_names=layer_names, auto_compile=True)
        return activations, inputs

    def display_activations(self, layer_name: str=None, st=0, en=None, indices=None, **kwargs):
        # not working currently because it requres the shape of activations to be (1, output_h, output_w, num_filters)
        activations, _ = self.activations(st=st, en=en, indices=indices, layer_names=layer_name)
        if layer_name is None:
            activations = activations
        else:
            activations = activations[layer_name]

        keract.display_activations(activations=activations,**kwargs)

    def gradients_of_weights(self, **kwargs) -> dict:

        x, y = self.run_paras(**kwargs)

        return keract.get_gradients_of_trainable_weights(self.k_model, x, y)

    def gradients_of_activations(self, st=0, en=None, indices=None, layer_name=None) -> dict:

        x, y = self.run_paras(st=st, en=en, indices=indices)

        return keract.get_gradients_of_activations(self.k_model, x, y, layer_names=layer_name)

    def trainable_weights(self):
        """ returns all trainable weights as arrays in a dictionary"""
        weights = {}
        for weight in self.k_model.trainable_weights:
            if tf.executing_eagerly():
               weights[weight.name] = weight.numpy()
            else:
                weights[weight.name] = keras.backend.eval(weight)
        return weights

    def plot_weights(self, save=True):
        weights = self.trainable_weights()
        for _name, weight in weights.items():
            title = _name + " Weights"
            fname = _name + '_weights'

            if np.ndim(weight) == 2 and weight.shape[1] > 1:
                self._imshow(weight, title, save, fname)
            elif len(weight) > 1 and np.ndim(weight) < 3:
                self.plot1d(weight, title, save, fname)
            else:
                print("ignoring weight for {} because it has shape {}".format(_name, weight.shape))

    def plot_activations(self, save: bool=False, **kwargs):
        """ plots activations of intermediate layers except input and output.
        If called without any arguments then it will plot activations of all layers."""
        activations, _ = self.activations(**kwargs)

        for lyr_name, activation in activations.items():
            if np.ndim(activation) == 2 and activation.shape[1] > 1:
                self._imshow(activation, lyr_name + " Activations", save, os.path.join(self.path, lyr_name))
            elif np.ndim(activation) == 3:
                self._imshow_3d(activation, lyr_name, save=save)
            elif np.ndim(activation) == 2: # this is now 1d
                # shape= (?, 1)
                self.plot1d(activation, label=lyr_name+' Activations', save=save,
                            fname=os.path.join(self.path, lyr_name+'_activations'))
            else:
                print("ignoring activations for {} because it has shape {}, {}".format(lyr_name, activation.shape,
                                                                                           np.ndim(activation)))

    def plot_weight_grads(self, save: bool=True, **kwargs):
        """ plots gradient of all trainable weights"""

        gradients = self.gradients_of_weights(**kwargs)
        for lyr_name, gradient in gradients.items():

            title = lyr_name+ "Weight Gradients"
            fname = os.path.join(self.path, lyr_name+'_weight_grads')

            if np.ndim(gradient) == 2 and gradient.shape[1] > 1:
                self._imshow(gradient, title, save,  fname)
            elif len(gradient) and np.ndim(gradient) < 3:
                self.plot1d(gradient, title, save, fname)
            else:
                print("ignoring weight gradients for {} because it has shape {} {}".format(lyr_name, gradient.shape,
                                                                                           np.ndim(gradient)))

    def plot_act_grads(self, save: bool=None, **kwargs):
        """ plots activations of intermediate layers except input and output"""
        gradients = self.gradients_of_activations(**kwargs)

        for lyr_name, gradient in gradients.items():
            fname = os.path.join(self.path, lyr_name + "_activation gradients")
            title = lyr_name + " Activation Gradients"
            if np.ndim(gradient) == 2:
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

    def plot_act_along_inputs(self, layer_name:str, name:str=None, **kwargs):

        pred, obs = self.predict(**kwargs)

        activation, data = self.activations(layer_names=layer_name, **kwargs)

        if isinstance(data, list):
            data = data[0]

        activation = activation[layer_name]
        data = data[:, 0, :]

        assert data.shape[1] == self.ins

        plt.close('all')

        for idx in range(self.ins):

            fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='all')
            fig.set_figheight(10)

            ax1.plot(data[:, idx], label=self.in_cols[idx])
            ax1.legend()
            ax1.set_title('activations w.r.t ' + self.in_cols[idx])
            ax1.set_ylabel(self.in_cols[idx])

            ax2.plot(pred[0], label='Prediction')
            ax2.plot(obs[0], label='Observed')
            ax2.legend()

            im = ax3.imshow(activation[:, :, idx].transpose(), aspect='auto', vmin=0, vmax=0.8)
            ax3.set_ylabel('lookback')
            ax3.set_xlabel('samples')
            fig.colorbar(im, orientation='horizontal', pad=0.2)
            plt.subplots_adjust(wspace=0.005, hspace=0.005)
            if name is not None:
                plt.savefig(os.path.join(self.path, name) + str(idx), dpi=400, bbox_inches='tight')
            else:
                plt.show()

        return

    def plot2d_act_for_a_sample(self, activations, sample=0, name:str=None):
        fig, axis = plt.subplots()
        fig.set_figheight(8)
        # for idx, ax in enumerate(axis):
        im = axis.imshow(activations[sample, :, :].transpose(), aspect='auto')
        axis.set_xlabel('lookback')
        axis.set_ylabel('inputs')
        print(self.in_cols)
        axis.set_title('Activations of all inputs at different lookbacks for sample ' + str(sample))
        fig.colorbar(im)
        if name is not None:
            plt.savefig(os.path.join(self.path, name) + '_' + str(sample), dpi=400, bbox_inches='tight')
        else:
            plt.show()
        return

    def plot1d_act_for_a_sample(self, activations, sample=0, name=None):
        fig, axis = plt.subplots()

        for idx in range(self.lookback-1):
            axis.plot(activations[sample, idx, :].transpose(), label='lookback '+str(idx))
        axis.set_xlabel('inputs')
        axis.set_ylabel('activation weight')
        axis.set_title('Activations at different lookbacks for all inputs for sample ' + str(sample))
        if name is not None:
            plt.savefig(os.path.join(self.path, name) + '_' + str(sample), dpi=400, bbox_inches='tight')
        else:
            plt.show()

    def prepare_batches(self, data: pd.DataFrame, target: str):

        assert self.outs == 1

        x = np.zeros((len(data), self.lookback, data.shape[1] - 1))
        y = np.zeros((len(data), self.lookback, 1))

        for i, name in enumerate(list(data.columns[:-1])):
            for j in range(self.lookback):
                x[:, j, i] = data[name].shift(self.lookback - j - 1).fillna(method="bfill")

        for j in range(self.lookback):
            y[:, j, 0] = data[target].shift(self.lookback - j - 1).fillna(method="bfill")

        prediction_horizon = 1
        target = data[target].shift(-prediction_horizon).fillna(method="ffill").values

        x = x[self.lookback:]
        y = y[self.lookback:]
        target = target[self.lookback:]

        return x, y, target

    def _imshow(self, img, label:str='', save=True, fname=None):
        assert np.ndim(img)==2, "can not plot {} with shape {} and ndim {}".format(label, img.shape, np.ndim(img))
        plt.close('all')
        plt.imshow(img, aspect='auto')
        plt.colorbar()
        plt.title(label)
        self.save_or_show(save, fname)

    def _imshow_3d(self, activation, lyr_name, save=True):
        act_2d = []
        for i in range(activation.shape[0]):
            act_2d.append(activation[i, :])
        activation_2d = np.vstack(act_2d)
        self._imshow(activation_2d, lyr_name + " Activations (3d of {})".format(activation.shape),
                     save, os.path.join(self.path, lyr_name))

    def plot1d(self, array, label:str='', save=True, fname=None):
        plt.close('all')
        plt.plot(array, '.')
        plt.title(label)
        self.save_or_show(save, fname)


    def save_or_show(self, save:bool=True, fname=None):
        if save:
            assert isinstance(fname, str)
            if "/" in fname:
                fname = fname.replace("/", "_")
            plt.savefig(os.path.join(self.path, fname))
        else:
            plt.show()


    def save_config(self, history:dict):

        config = dict()
        config['min_val_loss'] = int(np.min(history['val_loss'])) if 'val_loss' in history else None
        config['min_loss'] = int(np.min(history['loss'])) if 'val_loss' in history else None
        config['nn_config'] = self.nn_config
        config['data_config'] = self.data_config
        config['test_indices'] = np.array(self.test_indices, dtype=int).tolist() if self.test_indices is not None else None
        config['train_indices'] = np.array(self.train_indices, dtype=int).tolist() if self.train_indices is not None else None
        config['intervals'] = self.intervals
        config['method'] = self.method

        save_config_file(config=config, path=self.path)
        return config

    @classmethod
    def from_config(cls, config_path:str, data):
        with open(config_path, 'r') as fp:
            config = json.load(fp)

        data_config = config['data_config']
        nn_config = config['nn_config']
        if 'intervals' in config:
            intervals = config['intervals']
        else:
            intervals = None

        cls.from_check_point = True

        # These paras neet to be set here because they are not withing init method
        cls.test_indices = config["test_indices"]
        cls.train_indices = config["train_indices"]

        return cls(data_config=data_config,
                   nn_config=nn_config,
                   data=data,
                   intervals=intervals,
                   path=os.path.dirname(config_path))

    def load_weights(self, w_file:str):
        # loads the weights of keras model from weight file `w_file`.
        cpath = os.path.join(self.path, w_file)
        self.k_model.load_weights(cpath)
        return

    def add_act_layer(self, act_layer, inputs, time_distributed:bool=False):
        # this is used if we want to apply activation as a separate layer

        outputs = inputs
        if act_layer is not None:
            assert isinstance(act_layer, str)
            self.act_counter += 1
            name = 'act_' + str(self.act_counter)
            if time_distributed:
                outputs = layers.TimeDistributed(ACTIVATION_LAYERS[act_layer.upper()](name=name))(inputs)
            else:
                outputs = ACTIVATION_LAYERS[act_layer.upper()](name=name)(inputs)

        return outputs

def unison_shuffled_copies(a, b, c):
    """makes sure that all the arrays are permuted similarly"""
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]

def get_index(idx_array, fmt='%Y%m%d%H%M'):
    """ converts a numpy 1d array into pandas DatetimeIndex type."""

    if not isinstance(idx_array, np.ndarray):
        raise TypeError

    return pd.to_datetime(idx_array.astype(str), format=fmt)

def check_cnn_config(cnn_config:dict)->list:
    """ checks whether the `cnn_config` dictionary contains configuration for one cnn layer or multiple cnn layers
    """

    if 'n_layers' in cnn_config:
        n_cnns = cnn_config['n_layers']
        assert len(cnn_config) > n_cnns + 1, "{} cnn configs provided but cnofig says add {}".format(
            len(cnn_config) - 1, n_cnns)
        cnn_configs = []
        for k, v in cnn_config.items():
            if k not in ['n_layers', 'max_pool_size']:
                assert isinstance(v, dict), "cnn config must be of type `dict` but {} provided".format(v)
                cnn_configs.append(v)
    else:
        cnn_configs = [cnn_config]

    return cnn_configs


def check_lstm_config(lstm_config:dict)->list:

    if 'n_layers' in lstm_config:
        n_lstms = lstm_config['n_layers']
        assert len(lstm_config) > n_lstms, "{} lstm configs provided but cnofig says add {}".format(
            len(lstm_config), n_lstms)
        lstm_configs = []
        for k, v in lstm_config.items():
            if k not in ['n_layers']:
                assert isinstance(v, dict), "cnn config must be of type `dict` but {} provided".format(v)
                lstm_configs.append(v)
    else:
        lstm_configs = [lstm_config]

    return lstm_configs


def check_act_layer(config:dict):
    act_layer = None
    if 'act_layer' in config:
        act_layer = config.pop('act_layer')
    return act_layer


def check_act_fn(config:dict):
    """ it is possible that the config file does not have activation argument or activation is None"""
    activation = None
    if 'activation' in config:
        activation = config['activation']
    if activation is not None:
        assert isinstance(activation, str)
        config['activation'] = ACTIVATION_FNS[activation.upper()]

    return config, activation
