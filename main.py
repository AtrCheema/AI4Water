__all__ = ["Model", "CNNModel", "CNNLSTMModel", "LSTMCNNModel", "DualAttentionModel", "TCNModel",
           "LSTMAutoEncoder", "InputAttentionModel", "OutputAttentionModel", "NBeatsModel"]

import numpy as np

from tensorflow import keras
# import keras

import pandas as pd
from layer_definition import MyTranspose, MyDot
from TSErrors import FindErrors
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from tcn import TCN, tcn_full_summary
from nbeats_keras import NBeatsNet

# from .dual_attention import read_cache, get_data, write_cache
from utils import plot_results, plot_loss
import tf_losses as tf_losses

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

layers = keras.layers
KModel = keras.models.Model
Sequential = keras.models.Sequential

ACTIVATIONS = {'LeakyRelu': lambda name='leaky_relu': layers.LeakyReLU(name=name),
               'PRelu': lambda name='p_relu': layers.PReLU(name=name),  # https://arxiv.org/pdf/1502.01852v1.pdf
               'relu': lambda name='relu': layers.Activation('relu', name=name),
               'tanh': lambda name='tanh': layers.Activation('tanh', name=name),
               'elu': lambda name='elu': layers.ELU(name=name),
               'TheresholdRelu': lambda name='threshold_relu': layers.ThresholdedReLU(name=name),
               # 'SRelu': layers.advanced_activations.SReLU()
               }


LOSSES = {
    'mse': keras.losses.mse,
    'mae': keras.losses.mae,
    'mape': keras.losses.MeanAbsolutePercentageError,
    'male': keras.losses.MeanSquaredLogarithmicError,
    'nse': tf_losses.tf_nse,
}


class AttributeNotSetYet(object):
    k_model = None
    method = None
    ins = None
    outs = None
    en_densor_We = None
    en_LSTM_cell = None
    auto_enc_composite = None
    de_LSTM_cell = None
    de_densor_We = None


class Model(AttributeNotSetYet):

    def __init__(self, data_config: dict, nn_config: dict, data: pd.DataFrame, intervals=None):
        self.data_config = data_config
        self.nn_config = nn_config
        self.intervals = intervals
        self.data = data[data_config['inputs'] + data_config['outputs']]
        self.ins = len(self.data_config['inputs'])
        self.outs = len(self.data_config['outputs'])
        self.loss = LOSSES[self.nn_config['loss']]
        self.KModel = KModel

    @property
    def lookback(self):
        return self.data_config['lookback']

    @property
    def ins(self):
        return self._ins

    @ins.setter
    def ins(self, x):
        self._ins = x

    @property
    def outs(self):
        return self._outs

    @outs.setter
    def outs(self, x):
        self._outs = x

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

    def fetch_data(self, start: int, ende=None,
                   shuffle: bool = True,
                   cache_data=True,
                   noise: int = 0,
                   indices: list = None):

        if indices is not None:
            assert isinstance(indices, list), "indices must be list"
            if ende is not None or start != 0:
                raise ValueError

        df = pd.read_csv(self.data_config['data_path'])

        setattr(self, 'data_shape', df.shape)

        # # add random noise in the data
        if noise > 0:
            x = pd.DataFrame(np.random.randint(0, 1, (len(df), noise)))
            df = pd.concat([df, x], axis=1)
            prev_inputs = self.data_config['inputs']
            self.data_config['inputs'] = prev_inputs + list(x.columns)
            ys = []
            for y in self.data_config['outputs']:
                ys.append(df.pop(y))
            df[self.data_config['outputs']] = ys

        cols = self.data_config['inputs'] + self.data_config['outputs']
        df = df[cols]

        scaler = MinMaxScaler()
        data = scaler.fit_transform(df)
        df = pd.DataFrame(data)

        if ende is None:
            ende = df.shape[0]

        if self.intervals is None:

            df = df[start:ende]

            x, y, label = self.get_data(df,
                                        len(self.data_config['inputs']),
                                        len(self.data_config['outputs'])
                                        )
            if indices is not None:
                # if indices are given then this should be done after `get_data` method
                x = x[indices]
                y = y[indices]
                label = label[indices]
        else:
            xs, ys, labels = [], [], []
            for st, en in self.intervals:
                df1 = df[st:en]
                if df1.shape[0] > 0:
                    x, y, label = self.get_data(df1, len(self.data_config['inputs']), len(self.data_config['outputs']))
                    xs.append(x)
                    ys.append(y)
                    labels.append(label)

            if indices is None:
                x = np.vstack(xs)[start:ende]
                y = np.vstack(ys)[start:ende]
                label = np.vstack(labels)[start:ende]
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

        return x, y, label

    def fit(self, inputs, outputs, **callbacks):

        if 'tensorboard' in callbacks:
            log_dir = "./logs/fit/"
            callbacks = [keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]
        self.k_model.fit(inputs,
                         outputs,
                         epochs=self.nn_config['epochs'],
                         batch_size=self.data_config['batch_size'], validation_split=self.data_config['val_size'],
                         callbacks=callbacks
                         )
        history = self.k_model.history
        return history

    def train_nn(self, st=0, en=None, indices=None, **callbacks):

        setattr(self, 'train_indices', indices)

        train_x, train_y, train_label = self.fetch_data(start=st, ende=en, shuffle=True,
                                                        cache_data=self.data_config['CACHEDATA'],
                                                        indices=indices)

        if self.method == 'CNN_LSTM':
            subseq = self.nn_config['subsequences']
            examples = train_x.shape[0]
            timesteps = self.lookback // subseq
            train_x = train_x.reshape(examples, subseq, timesteps, self.ins)

        history = self.fit(train_x, train_label, **callbacks)

        plot_loss(history)

        return history

    def predict(self, st=0, ende=None, indices=None):

        setattr(self, 'predict_indices', indices)

        test_x, test_y, test_label = self.fetch_data(start=st, ende=ende, shuffle=False,
                                                     cache_data=False,
                                                     indices=indices)

        if self.method == 'CNN_LSTM':
            subseq = self.nn_config['subsequences']
            examples = test_x.shape[0]
            timesteps = self.lookback // subseq
            test_x = test_x.reshape(examples, subseq, timesteps, self.ins)

        predicted = self.k_model.predict(x=test_x,
                                         batch_size=test_x.shape[0],
                                         verbose=1)

        self.process_results(test_label, predicted, str(st) + '_' + str(ende))

        return predicted, test_label

    def process_results(self, true, predicted, name=None):
        errors = FindErrors(true, predicted)
        for er in ['mse', 'rmse', 'r2', 'nse', 'kge', 'rsr', 'percent_bias']:
            print(er, getattr(errors, er)())

        plot_results(true, predicted, name=name)
        return

    def build_nn(self):
        print('building simple lstm model')

        # lstm = self.nn_config['lstm_config']

        inputs = layers.Input(shape=(self.lookback, self.ins))

        lstm_activations = self.add_LSTM(inputs, self.nn_config['lstm_config'])

        predictions = layers.Dense(self.outs)(lstm_activations)

        self.k_model = self.compile(inputs, predictions)

        return

    def add_LSTM(self, inputs, config, seq=False):

        lstm_lyr = layers.LSTM(config['lstm_units'],
                               # input_shape=(self.lookback, self.ins),
                               dropout=config['dropout'],
                               recurrent_dropout=config['rec_dropout'],
                               return_sequences=seq,
                               name='lstm_lyr')(inputs)

        lstm_activations = ACTIVATIONS[config['lstm_act']](name='lstm_act')(lstm_lyr)

        return lstm_activations

    def add_1dCNN(self, inputs, cnn):

        cnn_layer = layers.Conv1D(filters=cnn['filters'],
                                  kernel_size=cnn['kernel_size'],
                                  # activation=cnn['activation'],
                                  name='cnn_lyr')(inputs)

        cnn_activations = ACTIVATIONS[cnn['activation']](name='cnn_act')(cnn_layer)

        max_pool_lyr = layers.MaxPooling1D(pool_size=cnn['max_pool_size'],
                                           name='max_pool_lyr')(cnn_activations)
        flat_lyr = layers.Flatten(name='flat_lyr')(max_pool_lyr)

        return flat_lyr

    def compile(self, model_inputs, outputs):

        k_model = self.KModel(inputs=model_inputs, outputs=outputs)
        adam = keras.optimizers.Adam(lr=self.nn_config['lr'])
        k_model.compile(loss=self.loss, optimizer=adam, metrics=['mse'])
        k_model.summary()

        return k_model

    def _encoder(self, config, lstm2_seq=True):

        self.en_densor_We = layers.Dense(self.lookback, name='enc_We')
        self.en_LSTM_cell = layers.LSTM(config['n_h'], return_state=True, name='encoder_LSTM')

        enc_input = layers.Input(shape=(self.lookback, self.ins), name='enc_input')  # Enter time series data
        # initialize the first cell state
        s0 = layers.Input(shape=(config['n_s'],), name='enc_first_cell_state')
        # initialize the first hidden state
        h0 = layers.Input(shape=(config['n_h'],), name='enc_first_hidden_state')

        enc_attn_out = self.encoder_attention(enc_input, s0, h0)
        print('encoder attention output:', enc_attn_out)
        enc_lstm_in = layers.Reshape((self.lookback, self.ins), name='enc_lstm_input')(enc_attn_out)
        print('input to encoder LSTM:', enc_lstm_in)
        enc_lstm_out = layers.LSTM(config['m'], return_sequences=lstm2_seq,
                                   name='LSTM_after_encoder')(enc_lstm_in)  # h_en_all
        print('Output from LSTM out: ', enc_lstm_out)
        return enc_lstm_out, enc_input,  h0, s0

    def one_encoder_attention_step(self, h_prev, s_prev, x, t):
        """
        :param h_prev: previous hidden state
        :param s_prev: previous cell state
        :param x: (T,n),n is length of input series at time t,T is length of time series
        :param t: time-step
        :return: x_t's attention weights,total n numbers,sum these are 1
        """
        _concat = layers.Concatenate()([h_prev, s_prev])  # (none,1,2m)
        result1 = self.en_densor_We(_concat)   # (none,1,T)
        result1 = layers.RepeatVector(x.shape[2],)(result1)  # (none,n,T)
        x_temp = MyTranspose(axis=(0, 2, 1))(x)  # X_temp(None,n,T)
        result2 = MyDot(self.lookback, name='eq_8_mul_'+str(t))(x_temp)  # (none,n,T) Ue(T,T), Ue * Xk in eq 8 of paper
        result3 = layers.Add()([result1, result2])  # (none,n,T)
        result4 = layers.Activation(activation='tanh')(result3)  # (none,n,T)
        result5 = MyDot(1)(result4)
        result5 = MyTranspose(axis=(0, 2, 1), name='eq_8_' + str(t))(result5)  # etk/ equation 8
        alphas = layers.Activation(activation='softmax', name='eq_9_'+str(t))(result5)  # equation 9

        return alphas

    def encoder_attention(self, _input, _s0, _h0):

        s = _s0
        _h = _h0
        print('encoder cell state:', s)
        # initialize empty list of outputs
        attention_weight_t = None
        for t in range(self.lookback):
            print('encoder input:', _input)
            _context = self.one_encoder_attention_step(_h, s, _input, t)  # (none,1,n)
            print('context:', _context)
            x = layers.Lambda(lambda x: _input[:, t, :])(_input)
            x = layers.Reshape((1, self.ins))(x)
            print('x:', x)
            _h, _, s = self.en_LSTM_cell(x, initial_state=[_h, s])
            if t != 0:
                print('attention_weight_:'+str(t), attention_weight_t)
                # attention_weight_t = layers.Merge(mode='concat', concat_axis=1,
                #                                    name='attn_weight_'+str(t))([attention_weight_t,
                #                                                                _context])
                attention_weight_t = layers.Concatenate(axis=1)([attention_weight_t, _context])
                print('salam')
            else:
                attention_weight_t = _context
            print('Now attention_weight_:' + str(t), attention_weight_t)
            print('encoder hidden state:', _h)
            print('_:', _)
            print('encoder cell state:', s)
            print('time-step', t)
            # break

        # get the driving input series
        enc_output = layers.Multiply(name='enc_output')([attention_weight_t, _input])  # equation 10 in paper
        print('output from encoder attention:', enc_output)
        return enc_output

    def one_decoder_attention_step(self, _h_de_prev, _s_de_prev, _h_en_all, t):
        """
        :param _h_de_prev: previous hidden state
        :param _s_de_prev: previous cell state
        :param _h_en_all: (None,T,m),n is length of input series at time t,T is length of time series
        :param t: int, timestep
        :return: x_t's attention weights,total n numbers,sum these are 1
        """
        print('h_en_all:', _h_en_all)
        # concatenation of the previous hidden state and cell state of the LSTM unit in eq 12
        _concat = layers.Concatenate(name='eq_12_'+str(t))([_h_de_prev, _s_de_prev])  # (None,1,2p)
        result1 = self.de_densor_We(_concat)   # (None,1,m)
        result1 = layers.RepeatVector(self.lookback)(result1)  # (None,T,m)
        result2 = MyDot(self.nn_config['enc_config']['m'])(_h_en_all)
        print('result1:', result1)
        print('result2:', result2)
        result3 = layers.Add()([result1, result2])  # (None,T,m)
        result4 = layers.Activation(activation='tanh')(result3)  # (None,T,m)
        result5 = MyDot(1)(result4)

        beta = layers.Activation(activation='softmax', name='eq_13_'+str(t))(result5)    # equation 13
        _context = layers.Dot(axes=1, name='eq_14_'+str(t))([beta, _h_en_all])  # (1,m)  # equation 14 in paper
        return _context

    def decoder_attention(self, _h_en_all, _y, _s0, _h0):
        s = _s0
        _h = _h0
        print('y_prev into decoder: ', _y)
        for t in range(self.lookback-1):
            y_prev = layers.Lambda(lambda y_prev: _y[:, t, :])(_y)
            y_prev = layers.Reshape((1, 1))(y_prev)   # (None,1,1)
            print('\ny_prev at {}'.format(t), y_prev)
            _context = self.one_decoder_attention_step(_h, s, _h_en_all, t)  # (None,1,20)
            # concatenation of decoder input and computed context vector  # ??
            y_prev = layers.Concatenate(axis=2)([y_prev, _context])   # (None,1,21)
            print('y_prev1 at {}:'.format(t), y_prev)
            y_prev = layers.Dense(1, name='eq_15_'+str(t))(y_prev)       # (None,1,1),                   Eq 15  in paper
            print('y_prev2 at {}:'.format(t), y_prev)
            _h, _, s = self.de_LSTM_cell(y_prev, initial_state=[_h, s])   # eq 16  ??
            print('decoder hidden state:', _h)
            print('_:', _)
            print('decoder cell state:', s)

        _context = self.one_decoder_attention_step(_h, s, _h_en_all, 'final')
        return _h, _context

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
        input_x = np.array(input_x).reshape(-1, self.lookback, ins)
        input_y = np.array(input_y).reshape(-1, self.lookback-1, outs)
        label_y = np.array(label_y).reshape(-1, outs)

        if df.isna().sum().values.sum() > 0:
            if self.method == 'dual_attention':
                raise ValueError
            print('input dataframe contain nans')
            y = df[df.columns[-1]]
            nan_idx = y.isna()
            nan_idx_t = nan_idx[self.lookback - 1:]
            non_nan_idx = ~nan_idx_t.values
            label_y = label_y[non_nan_idx]
            input_x = input_x[non_nan_idx]
            input_y = input_y[non_nan_idx]

            assert np.isnan(label_y).sum() < 1, "label still contains nans"

        assert input_x.shape[0] == input_y.shape[0] == label_y.shape[0], "shapes are not same"

        return input_x, input_y, label_y

    def get_activations(self, st, en, lyr_name='attn_weight_8'):
        """
        lyr_name: attn_weight_8, 'enc_lstm_input' or any other valid layer name
        """

        test_x, test_y, test_label = self.fetch_data(start=st, ende=en, shuffle=False)

        s0_test = h0_test = np.zeros((test_x.shape[0], self.nn_config['enc_config']['n_h']))

        bs = test_x.shape[0]  # self.data_config['batch_size']

        intermediate_model = KModel(
            inputs=self.k_model.inputs,
            outputs=self.k_model.get_layer(name=lyr_name).output)

        if self.method == 'input_attention':

            activations = intermediate_model.predict([test_x, s0_test, h0_test],
                                                     batch_size=bs,
                                                     verbose=1)
        elif self.method == 'dual_attention':
            h_de0_test = s_de0_test = np.zeros((test_x.shape[0], self.nn_config['dec_config']['p']))
            activations = intermediate_model.predict([test_x, test_y, s0_test, h0_test, s_de0_test, h_de0_test],
                                                     batch_size=bs,
                                                     verbose=1)
        elif self.method in ['lstm_cnn', 'simple_lstm', 'cnn_lstm', 'lstm_autoencoder']:

            if self.method == 'cnn_lstm':
                subseq = self.nn_config['subsequences']
                examples = test_x.shape[0]
                timesteps = self.lookback // subseq
                test_x = test_x.reshape(examples, subseq, timesteps, self.ins)

            activations = intermediate_model.predict(test_x,
                                                     batch_size=bs,
                                                     verbose=1)
        else:
            raise ValueError

        return activations, test_x

    def plot_act_along_lookback(self, activations, sample=0):

        activation = activations[sample, :, :]
        act_t = activation.transpose()

        fig, axis = plt.subplots()

        for idx, _name in enumerate(self.data_config['inputs']):
            axis.plot(act_t[idx, :], label=_name)

        axis.set_xlabel('Lookback')
        axis.set_ylabel('Input attention weight')
        axis.legend(loc="best")

        plt.show()
        return

    def plot_act_along_inputs(self, st, en, lyr_name, name=None):

        pred, obs = self.predict(st, en)

        activation, data = self.get_activations(st, en, lyr_name)

        data = data[:, 0, :]

        assert data.shape[1] == self.ins

        plt.close('all')

        for idx in range(self.ins):

            fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='all')
            fig.set_figheight(10)

            ax1.plot(data[:, idx], label=self.data_config['inputs'][idx])
            ax1.legend()
            ax1.set_title('activations w.r.t ' + self.data_config['inputs'][idx])
            ax1.set_ylabel(self.data_config['inputs'][idx])

            ax2.plot(pred, label='Prediction')
            ax2.plot(obs, label='Observed')
            ax2.legend()

            ax3.imshow(activation[:, :, idx].transpose())
            ax3.set_ylabel('lookback')
            ax3.set_xlabel('samples')
            plt.subplots_adjust(wspace=0.005, hspace=0.005)
            if name is not None:
                plt.savefig(name + str(idx), dpi=400, bbox_inches='tight')
            plt.show()

        return

    def plot2d_act_for_a_sample(self, activations, sample=0, name=None):
        fig, axis = plt.subplots()
        fig.set_figheight(8)
        # for idx, ax in enumerate(axis):
        axis.imshow(activations[sample, :, :].transpose())
        axis.set_xlabel('lookback')
        axis.set_ylabel('inputs')
        print(self.data_config['inputs'])
        axis.set_title('Activations of all inputs at different lookbacks for sample ' + str(sample))
        if name is not None:
            plt.savefig(name + '_' + str(sample), dpi=400, bbox_inches='tight')
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
            plt.savefig(name + '_' + str(sample), dpi=400, bbox_inches='tight')
        plt.show()

    def prepare_batches(self, data: pd.DataFrame, target: str):

        scaler = MinMaxScaler()
        cols = data.columns
        data = scaler.fit_transform(data)
        data = pd.DataFrame(data, columns=cols)

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


class CNNModel(Model):

    def __init__(self, **kwargs):

        self.method = 'simple_CNN'

        super(CNNModel, self).__init__(**kwargs)

    def build_nn(self):

        print('building simple cnn model')

        cnn = self.nn_config['cnn_config']

        inputs = layers.Input(shape=(self.lookback, self.ins))

        cnn_outputs = self.add_1dCNN(inputs, cnn)

        predictions = layers.Dense(self.outs)(cnn_outputs)

        self.k_model = self.compile(inputs, predictions)

        return self.k_model


class LSTMCNNModel(Model):
    def __init__(self, **kwargs):

        self.method = 'LSTM_CNN'

        super(LSTMCNNModel, self).__init__(**kwargs)

    def build_nn(self):

        lstm = self.nn_config['lstm_config']
        cnn = self.nn_config['cnn_config']

        inputs = layers.Input(shape=(self.lookback, self.ins))

        lstm_activations = self.add_LSTM(inputs, lstm, seq=True)

        cnn_outputs = self.add_1dCNN(lstm_activations, cnn)

        predictions = layers.Dense(self.outs)(cnn_outputs)

        self.k_model = self.compile(inputs, predictions)

        return

    def train_nn(self, st=0, en=None, indices=None, **callbacks):

        setattr(self, 'train_indices', indices)

        train_x, train_y, train_label = self.fetch_data(start=st, ende=en, shuffle=True,
                                                        cache_data=self.data_config['CACHEDATA'],
                                                        indices=indices)

        if self.auto_enc_composite:
            outputs = [train_x, train_label]
            history = self.fit(train_x, outputs, **callbacks)
        else:
            history = self.fit(train_x, train_label, **callbacks)

        plot_loss(history)

        return history


class CNNLSTMModel(Model):

    def __init__(self, **kwargs):

        self.method = 'CNN_LSTM'

        super(CNNLSTMModel, self).__init__(**kwargs)

    def build_nn(self):
        """
        Here we sub-divide each sample into further subsequences. The CNN model will interpret each sub-sequence
        and the LSTM will piece together the interpretations from the subsequences.
        https://machinelearningmastery.com/how-to-get-started-with-deep-learning-for-time-series-forecasting-7-day-mini-course/
        """
        print("building cnn -> lstm model")

        lstm = self.nn_config['lstm_config']
        cnn = self.nn_config['cnn_config']

        assert self.lookback % self.nn_config['subsequences'] == int(0), "lookback must be multiple of subsequences"

        timesteps = self.lookback // self.nn_config['subsequences']

        inputs = layers.Input(shape=(None, timesteps, self.ins))

        cnn_lyr = layers.TimeDistributed(layers.Conv1D(filters=cnn['filters'],
                                         kernel_size=cnn['kernel_size'],
                                         padding='same'),
                                         )(inputs)
        cnn_activations = layers.TimeDistributed(ACTIVATIONS[cnn['activation']](name='cnn_act'))(cnn_lyr)

        max_pool_lyr = layers.TimeDistributed(layers.MaxPooling1D(pool_size=cnn['max_pool_size']))(cnn_activations)
        flat_lyr = layers.TimeDistributed(layers.Flatten())(max_pool_lyr)

        lstm_lyr = layers.LSTM(lstm['lstm_units'],
                               # input_shape=(self.lookback, self.ins),
                               dropout=lstm['dropout'],
                               recurrent_dropout=lstm['rec_dropout'],
                               name='lstm_lyr')(flat_lyr)

        lstm_activations = ACTIVATIONS[lstm['lstm_act']](name='lstm_act')(lstm_lyr)
        # lstm_activations = self.add_LSTM(flat_lyr, lstm)

        predictions = layers.Dense(self.outs)(lstm_activations)

        self.k_model = self.compile(inputs, predictions)

        return


class DualAttentionModel(Model):

    def __init__(self, **kwargs):

        self.method = 'dual_attention'

        super(DualAttentionModel, self).__init__(**kwargs)

    def build_nn(self):

        dec_config = self.nn_config['dec_config']
        enc_config = self.nn_config['enc_config']

        self.de_LSTM_cell = layers.LSTM(dec_config['p'], return_state=True, name='decoder_LSTM')
        self.de_densor_We = layers.Dense(enc_config['m'])

        h_de0 = layers.Input(shape=(dec_config['n_hde0'],), name='dec_1st_hidden_state')
        s_de0 = layers.Input(shape=(dec_config['n_sde0'],), name='dec_1st_cell_state')
        input_y = layers.Input(shape=(self.lookback - 1, 1), name='input_y')

        enc_lstm_out, enc_input,  h0, s0 = self._encoder(self.nn_config['enc_config'])
        print('encoder output before reshaping: ', enc_lstm_out)
        # originally the last dimentions was -1 but I put it equal to 'm'
        # eq 11 in paper
        enc_out = layers.Reshape((self.lookback, enc_config['m']), name='enc_out_eq_11')(enc_lstm_out)
        print('output from Encoder LSTM:', enc_out)

        h, context = self.decoder_attention(enc_out, input_y, s_de0, h_de0)
        h = layers.Reshape((1, dec_config['p']))(h)
        # concatenation of decoder hidden state and the context vector.
        last_concat = layers.Concatenate(axis=2, name='last_concat')([h, context])  # (None, 1, 50)
        print('last_concat', last_concat)
        sec_dim = enc_config['m'] + dec_config['p']  # original it was not defined but in tf-keras we need to define it
        last_reshape = layers.Reshape((sec_dim,), name='last_reshape')(last_concat)  # (None, 50)
        print('reshape:', last_reshape)
        result = layers.Dense(dec_config['p'], name='eq_22')(last_reshape)  # (None, 30)  # equation 22
        print('result:', result)
        output = layers.Dense(1)(result)

        self.k_model = self.compile(model_inputs=[enc_input, input_y, s0, h0, s_de0, h_de0], outputs=output)

        return

    def train_nn(self, st=0, en=None, indices=None, **callbacks):

        setattr(self, 'train_indices', indices)

        train_x, train_y, train_label = self.fetch_data(start=st, ende=en, shuffle=True,
                                                        cache_data=self.data_config['CACHEDATA'],
                                                        indices=indices)

        s0_train = np.zeros((train_x.shape[0], self.nn_config['enc_config']['n_s']))
        h0_train = np.zeros((train_x.shape[0], self.nn_config['enc_config']['n_h']))

        h_de0_train = s_de0_train = np.zeros((train_x.shape[0], self.nn_config['dec_config']['p']))

        inputs = [train_x, train_y, s0_train, h0_train, s_de0_train, h_de0_train]
        history = self.fit(inputs, train_label, **callbacks)

        plot_loss(history)

        return history

    def predict(self, st=0, ende=None, indices=None):

        setattr(self, 'predict_indices', indices)

        test_x, test_y, test_label = self.fetch_data(start=st, ende=ende, shuffle=False,
                                                     cache_data=False,
                                                     indices=indices)

        h0_test = np.zeros((test_x.shape[0], self.nn_config['enc_config']['n_h']))
        s0_test = np.zeros((test_x.shape[0], self.nn_config['enc_config']['n_s']))
        h_de0_test = s_de0_test = np.zeros((test_x.shape[0], self.nn_config['dec_config']['p']))

        predicted = self.k_model.predict([test_x, test_y, s0_test, h0_test, s_de0_test, h_de0_test],
                                         batch_size=test_x.shape[0],
                                         verbose=1)

        self.process_results(test_label, predicted, str(st) + '_' + str(ende))

        return predicted, test_label


class TCNModel(Model):

    def build_nn(self):
        """ temporal convolution networks
        https://github.com/philipperemy/keras-tcn
        TCN can also be used as a keras layer.
        """

        setattr(self, 'method', 'TCN')

        tcn_options = self.nn_config['tcn_options']
        tcn_options['return_sequences'] = False

        inputs = layers.Input(batch_shape=(None, self.lookback, self.ins))

        tcn_outs = TCN(**tcn_options)(inputs)  # The TCN layers are here.
        predictions = layers.Dense(1)(tcn_outs)

        self.k_model = self.compile(inputs, predictions)
        tcn_full_summary(self.k_model, expand_residual_blocks=True)

        return


class LSTMAutoEncoder(Model):

    def build_nn(self, method='prediction'):
        """

        :param method: str, if 'composite', then autoencoder is trained to reproduce inputs and
                       predictions at the same time.
        :return:
        """

        setattr(self, 'auto_enc_composite', False)
        setattr(self, 'method', 'lstm_autoencoder')

        # define encoder
        inputs = layers.Input(shape=(self.lookback, self.ins))
        encoder = layers.LSTM(100, activation='relu')(inputs)
        # define predict decoder
        decoder1 = layers.RepeatVector(self.lookback-1)(encoder)
        decoder1 = layers.LSTM(100, activation='relu', return_sequences=False)(decoder1)
        decoder1 = layers.Dense(1)(decoder1)
        if method == 'composite':
            setattr(self, 'auto_enc_composite', True)
            # define reconstruct decoder
            decoder2 = layers.RepeatVector(self.lookback)(encoder)
            decoder2 = layers.LSTM(100, activation='relu', return_sequences=True)(decoder2)
            decoder2 = layers.TimeDistributed(layers.Dense(self.ins))(decoder2)

            outputs = [decoder1, decoder2]
        else:
            outputs = decoder1

        self.k_model = self.compile(inputs, outputs)

        return


class InputAttentionModel(Model):

    def build_nn(self):

        setattr(self, 'method', 'input_attention')
        print('building input attention')

        lstm_out, enc_input, h0, s0 = self._encoder(self.nn_config['enc_config'], lstm2_seq=False)

        act_out = layers.LeakyReLU()(lstm_out)
        predictions = layers.Dense(self.outs)(act_out)
        print('predictions: ', predictions)

        self.k_model = self.compile(model_inputs=[enc_input, s0, h0], outputs=predictions)

        return

    def train_nn(self, st=0, en=None, indices=None, tensorboard=None):

        setattr(self, 'train_indices', indices)

        train_x, train_y, train_label = self.fetch_data(start=st, ende=en, shuffle=True,
                                                        cache_data=self.data_config['CACHEDATA'],
                                                        indices=indices)

        s0_train = np.zeros((train_x.shape[0], self.nn_config['enc_config']['n_s']))
        h0_train = np.zeros((train_x.shape[0], self.nn_config['enc_config']['n_h']))

        history = self.fit([train_x, s0_train, h0_train], train_label, callbacks=tensorboard)
        plot_loss(history)

        return history

    def predict(self, st=0, ende=None, indices=None):

        setattr(self, 'predict_indices', indices)

        test_x, test_y, test_label = self.fetch_data(start=st, ende=ende, shuffle=False,
                                                     cache_data=False,
                                                     indices=indices)

        h0_test = np.zeros((test_x.shape[0], self.nn_config['enc_config']['n_h']))
        s0_test = np.zeros((test_x.shape[0], self.nn_config['enc_config']['n_s']))

        predicted = self.k_model.predict([test_x, s0_test, h0_test],
                                         batch_size=test_x.shape[0],
                                         verbose=1)

        self.process_results(test_label, predicted, str(st) + '_' + str(ende))

        return predicted, test_label


class OutputAttentionModel(Model):

    def build_nn(self):

        setattr(self, 'method', 'output_attention')

        inputs = layers.Input(shape=(self.lookback, self.ins))

        enc_config = self.nn_config['enc_config']
        dec_config = self.nn_config['dec_config']

        self.de_densor_We = layers.Dense(enc_config['m'])
        h_de0 = layers.Input(shape=(dec_config['n_hde0'],), name='dec_1st_hidden_state')
        s_de0 = layers.Input(shape=(dec_config['n_sde0'],), name='dec_1st_cell_state')
        prev_output = layers.Input(shape=(self.lookback - 1, 1), name='input_y')

        enc_lstm_out = layers.LSTM(self.nn_config['lstm_config']['lstm_units'], return_sequences=True,
                                   name='starting_LSTM')(inputs)
        print('Output from LSTM: ', enc_lstm_out)
        enc_out = layers.Reshape((self.lookback, -1), name='enc_out_eq_11')(enc_lstm_out)  # eq 11 in paper
        print('output from first LSTM:', enc_out)

        h, context = self.decoder_attention(enc_out, prev_output, s_de0, h_de0)
        h = layers.Reshape((1, dec_config['p']))(h)
        # concatenation of decoder hidden state and the context vector.
        concat = layers.Concatenate(axis=2)([h, context])
        concat = layers.Reshape((-1,))(concat)
        print('decoder concat:', concat)
        result = layers.Dense(dec_config['p'], name='eq_22')(concat)   # equation 22
        print('result:', result)
        predictions = layers.Dense(1)(result)

        k_model = self.compile([inputs, prev_output, s_de0, h_de0], predictions)

        return k_model

    def train_nn(self, st=0, en=None, indices=None, **callbacks):

        train_x, train_y, train_label = self.fetch_data(start=st, ende=en, shuffle=True,
                                                        cache_data=self.data_config['CACHEDATA'],
                                                        indices=indices)

        h_de0_train = s_de0_train = np.zeros((train_x.shape[0], self.nn_config['dec_config']['p']))

        inputs = [train_x, train_y, s_de0_train, h_de0_train]
        history = self.fit(inputs, train_label, **callbacks)

        plot_loss(history)

        return history

    def predict(self, st=0, ende=None, indices=None):
        setattr(self, 'predict_indices', indices)

        test_x, test_y, test_label = self.fetch_data(start=st, ende=ende, shuffle=False,
                                                     cache_data=False,
                                                     indices=indices)

        h_de0_test = s_de0_test = np.zeros((test_x.shape[0], self.nn_config['dec_config']['p']))

        predicted = self.k_model.predict([test_x, test_y, s_de0_test, h_de0_test],
                                         batch_size=test_x.shape[0],
                                         verbose=1)

        self.process_results(test_label, predicted, str(st) + '_' + str(ende))

        return predicted, test_label


class NBeatsModel(Model):
    """
    original paper https://arxiv.org/pdf/1905.10437.pdf which is implemented by https://github.com/philipperemy/n-beats
    must be used with normalization
    """
    def build_nn(self):

        nbeats_options = self.nn_config['nbeats_options']
        nbeats_options['input_dim'] = self.outs
        nbeats_options['exo_dim'] = self.ins
        self.k_model = NBeatsNet(**nbeats_options)

        adam = keras.optimizers.Adam(lr=self.nn_config['lr'])
        self.k_model.compile(loss=self.loss, optimizer=adam, metrics=['mse'])

        return

    def train_nn(self, st=0, en=None, indices=None, **callbacks):
        exo_x, x, label = self.prepare_batches(data=self.data[st:en], target=self.data_config['outputs'][0])

        history = self.fit(inputs=[x, exo_x], outputs=label.reshape(-1, 1, 1), **callbacks)
        plot_loss(history)

        return history

    def predict(self, st=0, ende=None, indices=None):

        exo_x, x, label = self.prepare_batches(data=self.data[st:ende], target=self.data_config['outputs'][0])

        predicted = self.k_model.predict([x, exo_x],
                                         batch_size=exo_x.shape[0],
                                         verbose=1)

        self.process_results(label, predicted.reshape(-1,), str(st) + '_' + str(ende))

        return predicted, label


def unison_shuffled_copies(a, b, c):
    """makes sure that all the arrays are permuted similarly"""
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]
