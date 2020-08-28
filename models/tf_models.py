__all__ = ["CNNModel", "CNNLSTMModel", "LSTMCNNModel", "DualAttentionModel", "TCNModel",
           "LSTMAutoEncoder", "InputAttentionModel", "OutputAttentionModel", "NBeatsModel"]

import numpy as np

from main import Model
from .global_variables import keras, tcn
from models.layer_definition import MyTranspose, MyDot

from .nbeats_keras import NBeatsNet
from utils import plot_loss
layers = keras.layers
KModel = keras.models.Model


class CNNModel(Model):

    def __init__(self, **kwargs):

        self.method = 'simple_CNN'

        super(CNNModel, self).__init__(**kwargs)

    def build_nn(self):

        print('building simple cnn model')

        cnn = self.nn_config['cnn_config']

        inputs, predictions = self.cnn_model(cnn, self.outs)

        self.k_model = self.compile(inputs, predictions)

        return self.k_model


class LSTMCNNModel(Model):
    def __init__(self, **kwargs):

        self.method = 'LSTM_CNN'

        super(LSTMCNNModel, self).__init__(**kwargs)

    def build_nn(self):

        lstm = self.nn_config['lstm_config']
        cnn = self.nn_config['cnn_config']

        inputs, predictions = self.lstm_cnn(lstm, cnn, self.outs)

        self.k_model = self.compile(inputs, predictions)

        return


class CNNLSTMModel(Model):

    def __init__(self, **kwargs):

        self.method = 'CNN_LSTM'

        super(CNNLSTMModel, self).__init__(**kwargs)

    def run_paras(self, **kwargs):

        x, y, labels = self.fetch_data(self.data, **kwargs)

        subseq = self.nn_config['subsequences']
        examples = x.shape[0]
        timesteps = self.lookback // subseq
        x = x.reshape(examples, subseq, timesteps, self.ins)

        return x, labels

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

        inputs, predictions = self.cnn_lstm(cnn=cnn, lstm=lstm, outs=self.outs)

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
        enc_input = keras.layers.Input(shape=(self.lookback, self.ins), name='enc_input')

        enc_lstm_out, h0, s0 = self._encoder(enc_input, self.nn_config['enc_config'])
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

    def _encoder(self,enc_inputs, config, lstm2_seq=True, suf:str = '1'):

        self.en_densor_We = layers.Dense(self.lookback, name='enc_We_'+suf)
        self.en_LSTM_cell = layers.LSTM(config['n_h'], return_state=True, activation=config['enc_lstm1_act'],
                                        name='encoder_LSTM_'+suf)

        # initialize the first cell state
        s0 = layers.Input(shape=(config['n_s'],), name='enc_first_cell_state_'+suf)
        # initialize the first hidden state
        h0 = layers.Input(shape=(config['n_h'],), name='enc_first_hidden_state_'+suf)

        enc_attn_out = self.encoder_attention(enc_inputs, s0, h0, suf)
        print('encoder attention output:', enc_attn_out)
        enc_lstm_in = layers.Reshape((self.lookback, self.ins), name='enc_lstm_input_'+suf)(enc_attn_out)
        print('input to encoder LSTM:', enc_lstm_in)
        enc_lstm_out = layers.LSTM(config['m'], return_sequences=lstm2_seq, activation=config['enc_lstm2_act'],
                                   name='LSTM_after_encoder_'+suf)(enc_lstm_in)  # h_en_all
        print('Output from LSTM out: ', enc_lstm_out)
        return enc_lstm_out, h0, s0

    def one_encoder_attention_step(self, h_prev, s_prev, x, t, suf:str = '1'):
        """
        :param h_prev: previous hidden state
        :param s_prev: previous cell state
        :param x: (T,n),n is length of input series at time t,T is length of time series
        :param t: time-step
        :param suf: str, Suffix to be attached to names
        :return: x_t's attention weights,total n numbers,sum these are 1
        """
        _concat = layers.Concatenate()([h_prev, s_prev])  # (none,1,2m)
        result1 = self.en_densor_We(_concat)   # (none,1,T)
        result1 = layers.RepeatVector(x.shape[2],)(result1)  # (none,n,T)
        x_temp = MyTranspose(axis=(0, 2, 1))(x)  # X_temp(None,n,T)
        result2 = MyDot(self.lookback, name='eq_8_mul_'+str(t)+'_'+suf)(x_temp)  # (none,n,T) Ue(T,T), Ue * Xk in eq 8 of paper
        result3 = layers.Add()([result1, result2])  # (none,n,T)
        result4 = layers.Activation(activation='tanh')(result3)  # (none,n,T)
        result5 = MyDot(1)(result4)
        result5 = MyTranspose(axis=(0, 2, 1), name='eq_8_' + str(t)+'_'+suf)(result5)  # etk/ equation 8
        alphas = layers.Activation(activation='softmax', name='eq_9_'+str(t)+'_'+suf)(result5)  # equation 9

        return alphas

    def encoder_attention(self, _input, _s0, _h0, suf:str = '1'):

        s = _s0
        _h = _h0
        print('encoder cell state:', s)
        # initialize empty list of outputs
        attention_weight_t = None
        for t in range(self.lookback):
            print('encoder input:', _input)
            _context = self.one_encoder_attention_step(_h, s, _input, t, suf=suf)  # (none,1,n)
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
                attention_weight_t = layers.Concatenate(axis=1,
                                                        name='attn_weight_'+str(t)+'_'+suf)([attention_weight_t, _context])
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
        enc_output = layers.Multiply(name='enc_output_'+suf)([attention_weight_t, _input])  # equation 10 in paper
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

    def run_paras(self, **kwargs):

        x, y, labels = self.fetch_data(self.data, **kwargs)

        s0 = np.zeros((x.shape[0], self.nn_config['enc_config']['n_s']))
        h0 = np.zeros((x.shape[0], self.nn_config['enc_config']['n_h']))

        h_de0 = s_de0 = np.zeros((x.shape[0], self.nn_config['dec_config']['p']))

        return [x, s0, h0, h_de0, s_de0], labels


class TCNModel(Model):

    def build_nn(self):
        """ temporal convolution networks
        https://github.com/philipperemy/keras-tcn
        TCN can also be used as a keras layer.
        """

        setattr(self, 'method', 'TCN')

        tcn_options = self.nn_config['tcn_options']

        inputs, predictions = self.tcn_model(tcn_options, self.outs)

        self.k_model = self.compile(inputs, predictions)
        tcn.tcn_full_summary(self.k_model, expand_residual_blocks=True)

        return


class LSTMAutoEncoder(Model):

    def __init__(self, nn_config, **kwargs):
        # because composite attribute is used in this Model
        self.composite = nn_config['autoenc_config']['composite']

        super(LSTMAutoEncoder, self).__init__(nn_config=nn_config, **kwargs)

    def build_nn(self, method='prediction'):
        """

        :param method: str, if 'composite', then autoencoder is trained to reproduce inputs and
                       predictions at the same time.
        :return:
        """

        setattr(self, 'method', 'lstm_autoencoder')

        # define encoder
        config = self.nn_config['autoenc_config']

        enc_config = config['enc_config']
        dec_config = config['dec_config']

        inputs, outputs = self.lstm_autoencoder(enc_config, dec_config, self.composite, self.outs)

        self.k_model = self.compile(inputs, outputs)

        return

    def run_paras(self, **kwargs):

        x, y, labels = self.fetch_data(self.data, **kwargs)

        if self.composite:
            outputs = [x, labels]
        else:
            outputs = labels

        return x, outputs


class InputAttentionModel(DualAttentionModel):

    def build_nn(self):

        setattr(self, 'method', 'input_attention')
        print('building input attention')

        enc_input = keras.layers.Input(shape=(self.lookback, self.ins), name='enc_input1')
        lstm_out, h0, s0 = self._encoder(enc_input, self.nn_config['enc_config'], lstm2_seq=False)

        act_out = layers.LeakyReLU()(lstm_out)
        predictions = layers.Dense(self.outs)(act_out)
        print('predictions: ', predictions)

        self.k_model = self.compile(model_inputs=[enc_input, s0, h0], outputs=predictions)

        return

    def run_paras(self, **kwargs):
        x, y, labels = self.fetch_data(self.data, **kwargs)

        s0 = np.zeros((x.shape[0], self.nn_config['enc_config']['n_s']))
        h0 = np.zeros((x.shape[0], self.nn_config['enc_config']['n_h']))

        return [x, s0, h0], labels


class OutputAttentionModel(DualAttentionModel):

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

        train_x, train_y, train_label = self.fetch_data(self.data, st=st, en=en, shuffle=True,
                                                        cache_data=self.data_config['CACHEDATA'],
                                                        indices=indices)

        h_de0_train = s_de0_train = np.zeros((train_x.shape[0], self.nn_config['dec_config']['p']))

        inputs = [train_x, train_y, s_de0_train, h_de0_train]
        history = self.fit(inputs, train_label, **callbacks)

        plot_loss(history)

        return history

    def predict(self, st=0, ende=None, indices=None):
        setattr(self, 'predict_indices', indices)

        test_x, test_y, test_label = self.fetch_data(self.data, st=st, en=ende, shuffle=False,
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