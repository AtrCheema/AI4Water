__all__ = ["CNNModel", "CNNLSTMModel", "LSTMCNNModel", "DualAttentionModel", "TCNModel",
           "LSTMAutoEncoder", "InputAttentionModel", "OutputAttentionModel", "NBeatsModel"]

import numpy as np

from main import Model
from .global_variables import keras, tcn, ACTIVATIONS

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

        tcn_outs = tcn.TCN(**tcn_options)(inputs)  # The TCN layers are here.
        predictions = layers.Dense(1)(tcn_outs)

        self.k_model = self.compile(inputs, predictions)
        tcn.tcn_full_summary(self.k_model, expand_residual_blocks=True)

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