__all__ = ["DualAttentionModel", "InputAttentionModel", "OutputAttentionModel", "NBeatsModel"]

import numpy as np
import pandas as pd

from ai4water.backend import tf
from ai4water.main import Model
from ai4water.functional import Model as FModel
from ai4water.backend import keras
from ai4water._main import print_something
from ai4water.nn_tools import check_act_fn
from ai4water.layer_definition import MyTranspose, MyDot

layers = keras.layers
KModel = keras.models.Model


class DualAttentionModel(FModel):
    _enc_config = {'n_h': 20,  # length of hidden state m
                   'n_s': 20,  # length of hidden state m
                   'm': 20,  # length of hidden state m
                   'enc_lstm1_act': None,
                   'enc_lstm2_act': None,
                                }
    # arguments for decoder/outputAttention in Dual stage attention
    _dec_config = {
        'p': 30,
        'n_hde0': 30,
        'n_sde0': 30
    }

    def __init__(self, enc_config:dict=None, dec_config:dict=None, **kwargs):

        self.method = 'dual_attention'
        if enc_config is None:
            enc_config = DualAttentionModel._enc_config
        else:
            assert isinstance(enc_config, dict)

        if dec_config is None:
            dec_config = DualAttentionModel._dec_config
        else:
            assert isinstance(dec_config, dict)
        self.enc_config = enc_config
        self.dec_config = dec_config

        super(DualAttentionModel, self).__init__(**kwargs)

        setattr(self, 'category', "DL")

    def build(self, input_shape=None):

        self.config['dec_config'] = self.dec_config
        self.config['enc_config'] = self.enc_config

        self.de_LSTM_cell = layers.LSTM(self.dec_config['p'], return_state=True, name='decoder_LSTM')
        self.de_densor_We = layers.Dense(self.enc_config['m'])

        h_de0 = layers.Input(shape=(self.dec_config['n_hde0'],), name='dec_1st_hidden_state')
        s_de0 = layers.Input(shape=(self.dec_config['n_sde0'],), name='dec_1st_cell_state')

        #h_de0 = tf.zeros((self.config['batch_size'], self.dec_config['n_hde0']), name='dec_1st_hidden_state')
        #s_de0 = tf.zeros((self.config['batch_size'], self.dec_config['n_sde0']), name='dec_1st_cell_state')

        input_y = layers.Input(shape=(self.lookback - 1, self.num_outs), name='input_y')
        enc_input = keras.layers.Input(shape=(self.lookback, self.num_ins), name='enc_input')

        enc_lstm_out, s0, h0 = self._encoder(enc_input, self.config['enc_config'])

        if self.verbosity > 2:
            print('encoder output before reshaping: ', enc_lstm_out)

        # originally the last dimentions was -1 but I put it equal to 'm'
        # eq 11 in paper
        enc_out = layers.Reshape((self.lookback, self.enc_config['m']), name='enc_out_eq_11')(enc_lstm_out)

        if self.verbosity > 1:
            print('output from Encoder LSTM:', enc_out)

        h, context = self.decoder_attention(enc_out, input_y, s_de0, h_de0)
        h = layers.Reshape((self.num_outs, self.dec_config['p']))(h)
        # concatenation of decoder hidden state and the context vector.
        last_concat = layers.Concatenate(axis=2, name='last_concat')([h, context])  # (None, 1, 50)

        if self.verbosity > 2:
            print('last_concat', last_concat)
        sec_dim = self.enc_config['m'] + self.dec_config['p']  # original it was not defined but in tf-keras we need to define it
        last_reshape = layers.Reshape((sec_dim,), name='last_reshape')(last_concat)  # (None, 50)

        if self.verbosity > 2:
            print('reshape:', last_reshape)

        result = layers.Dense(self.dec_config['p'], name='eq_22')(last_reshape)  # (None, 30)  # equation 22

        if self.verbosity > 1:
            print('result:', result)

        output = layers.Dense(self.num_outs)(result)
        output = layers.Reshape(target_shape=(self.num_outs, self.forecast_len))(output)

        if self.config['drop_remainder']:
            self._model = self.compile(model_inputs=[enc_input, input_y, s_de0, h_de0], outputs=output)
        else:
            self._model = self.compile(model_inputs=[enc_input, input_y, s0, h0, s_de0, h_de0], outputs=output)

        return

    def _encoder(self, enc_inputs, config, lstm2_seq=True, suf: str = '1', s0=None, h0=None):

        #if not self.config['drop_remainder']:
        #    self.config['drop_remainder'] = True
        #    print(f"setting `drop_remainder` to {self.config['drop_remainder']}")

        self.en_densor_We = layers.Dense(self.lookback, name='enc_We_'+suf)
        _config, act_str = check_act_fn({'activation': config['enc_lstm1_act']})
        self.en_LSTM_cell = layers.LSTM(config['n_h'], return_state=True, activation=_config['activation'],
                                        name='encoder_LSTM_'+suf)
        config['enc_lstm1_act'] = act_str

        # initialize the first cell state
        if s0 is None:
            if self.config['drop_remainder']:
                s0 =  tf.zeros((self.config['batch_size'], config['n_s']), name=f'enc_first_cell_state_{suf}')
            else:
                s0 = layers.Input(shape=(config['n_s'],), name='enc_first_cell_state_' + suf)
        # initialize the first hidden state
        if h0 is None:
            if self.config['drop_remainder']:
                h0 = tf.zeros((self.config['batch_size'], config['n_h']), name=f'enc_first_hidden_state_{suf}')
            else:
                h0 = layers.Input(shape=(config['n_h'],), name='enc_first_hidden_state_' + suf)

        enc_attn_out = self.encoder_attention(enc_inputs, s0, h0, suf)
        if self.verbosity > 2:
            print('encoder attention output:', enc_attn_out)
        enc_lstm_in = layers.Reshape((self.lookback, self.num_ins), name='enc_lstm_input_'+suf)(enc_attn_out)
        if self.verbosity > 2:
            print('input to encoder LSTM:', enc_lstm_in)

        _config, act_str = check_act_fn({'activation': config['enc_lstm2_act']})
        enc_lstm_out = layers.LSTM(config['m'], return_sequences=lstm2_seq, activation=_config['activation'],
                                   name='LSTM_after_encoder_'+suf)(enc_lstm_in)  # h_en_all
        config['enc_lstm2_act'] = act_str

        if self.verbosity > 2:
            print('Output from LSTM out: ', enc_lstm_out)
        return enc_lstm_out, h0, s0

    def one_encoder_attention_step(self, h_prev, s_prev, x, t, suf: str = '1'):
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

    def encoder_attention(self, _input, _s0, _h0, suf: str = '1'):

        s = _s0
        _h = _h0
        if self.verbosity > 2:
            print('encoder cell state:', s)
        # initialize empty list of outputs
        attention_weight_t = None
        for t in range(self.lookback):
            if self.verbosity > 2:
                print('encoder input:', _input)
            _context = self.one_encoder_attention_step(_h, s, _input, t, suf=suf)  # (none,1,n)
            if self.verbosity > 2:
                print('context:', _context)
            x = layers.Lambda(lambda x: _input[:, t, :])(_input)
            x = layers.Reshape((1, self.num_ins))(x)
            if self.verbosity > 2:
                print('x:', x)
            _h, _, s = self.en_LSTM_cell(x, initial_state=[_h, s])
            if t != 0:
                if self.verbosity > 1:
                    print('attention_weight_:'+str(t), attention_weight_t)
                # attention_weight_t = layers.Merge(mode='concat', concat_axis=1,
                #                                    name='attn_weight_'+str(t))([attention_weight_t,
                #                                                                _context])
                attention_weight_t = layers.Concatenate(axis=1,
                                                        name='attn_weight_'+str(t)+'_'+suf)([attention_weight_t,
                                                                                             _context])
            else:
                attention_weight_t = _context

            if self.verbosity > 2:
                print('Now attention_weight_:' + str(t), attention_weight_t)
                print('encoder hidden state:', _h)
                print('_:', _)
                print('encoder cell state:', s)
                print('time-step', t)
            # break

        # get the driving input series
        enc_output = layers.Multiply(name='enc_output_'+suf)([attention_weight_t, _input])  # equation 10 in paper
        if self.verbosity > 1:
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
        if self.verbosity > 2:
            print('h_en_all:', _h_en_all)
        # concatenation of the previous hidden state and cell state of the LSTM unit in eq 12
        _concat = layers.Concatenate(name='eq_12_'+str(t))([_h_de_prev, _s_de_prev])  # (None,1,2p)
        result1 = self.de_densor_We(_concat)   # (None,1,m)
        result1 = layers.RepeatVector(self.lookback)(result1)  # (None,T,m)
        result2 = MyDot(self.config['enc_config']['m'])(_h_en_all)
        if self.verbosity > 2:
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
        if self.verbosity > 2:
            print('y_prev into decoder: ', _y)
        for t in range(self.lookback-1):
            y_prev = layers.Lambda(lambda y_prev: _y[:, t, :])(_y)
            y_prev = layers.Reshape((1, self.num_outs))(y_prev)   # (None,1,1)
            if self.verbosity > 2:
                print('\ny_prev at {}'.format(t), y_prev)
            _context = self.one_decoder_attention_step(_h, s, _h_en_all, t)  # (None,1,20)
            # concatenation of decoder input and computed context vector  # ??
            y_prev = layers.Concatenate(axis=2)([y_prev, _context])   # (None,1,21)
            if self.verbosity > 2:
                print('y_prev1 at {}:'.format(t), y_prev)
            y_prev = layers.Dense(self.num_outs, name='eq_15_'+str(t))(y_prev)       # (None,1,1),                   Eq 15  in paper
            if self.verbosity > 2:
                print('y_prev2 at {}:'.format(t), y_prev)
            _h, _, s = self.de_LSTM_cell(y_prev, initial_state=[_h, s])   # eq 16  ??
            if self.verbosity > 2:
                print('decoder hidden state:', _h)
                print('_:', _)
                print('decoder cell state:', s)

        _context = self.one_decoder_attention_step(_h, s, _h_en_all, 'final')
        return _h, _context

    def fetch_data(self, source, **kwargs):
        self.dh.teacher_forcing = True
        x, prev_y, labels = getattr(self.dh, f'{source}_data')(**kwargs)
        self.dh.teacher_forcing = False

        n_s_feature_dim = self.config['enc_config']['n_s']
        n_h_feature_dim = self.config['enc_config']['n_h']
        p_feature_dim = self.config['dec_config']['p']

        if kwargs.get('use_datetime_index', False):  # during deindexification, first feature will be removed.
            n_s_feature_dim += 1
            n_h_feature_dim += 1
            p_feature_dim += 1
            idx = np.expand_dims(x[:, 1:, 0], axis=-1)   # extract the index from x
            prev_y = np.concatenate([prev_y, idx], axis=2)  # insert index in prev_y
        s0 = np.zeros((x.shape[0], n_s_feature_dim))
        h0 = np.zeros((x.shape[0], n_h_feature_dim))

        h_de0 = s_de0 = np.zeros((x.shape[0], p_feature_dim))

        x, prev_y, labels = self.check_batches([x, prev_y, h_de0, s_de0], prev_y, labels)
        x, prev_y, h_de0, s_de0 = x

        if self.verbosity > 0:
            print_something([x, prev_y, s0, h0, h_de0, s_de0], "input_x")
            print_something(labels, "target")

        return [x, prev_y, s0, h0, h_de0, s_de0], prev_y, labels

    def training_data(self, **kwargs):
        return self.fetch_data('training', **kwargs)

    def validation_data(self, **kwargs):
        return self.fetch_data('validation', **kwargs)

    def test_data(self,  **kwargs):
        return self.fetch_data('test', **kwargs)


class InputAttentionModel(DualAttentionModel):

    def build(self, input_shape=None):

        self.config['enc_config'] = self.enc_config

        setattr(self, 'method', 'input_attention')
        print('building input attention model')

        enc_input = keras.layers.Input(shape=(self.lookback, self.num_ins), name='enc_input1')
        lstm_out, h0, s0 = self._encoder(enc_input, self.enc_config, lstm2_seq=False)

        act_out = layers.LeakyReLU()(lstm_out)
        predictions = layers.Dense(self.num_outs)(act_out)
        predictions = layers.Reshape(target_shape=(self.num_outs, self.forecast_len))(predictions)
        if self.verbosity > 2:
            print('predictions: ', predictions)

        inputs = [enc_input]
        if not self.config['drop_remainder']:
            inputs = inputs + [s0, h0]
        self._model = self.compile(model_inputs=inputs, outputs=predictions)

        return

    def fetch_data(self, source, **kwargs):
        self.dh.teacher_forcing = True

        x, prev_y, labels = getattr(self.dh, f'{source}_data')(**kwargs)
        self.dh.teacher_forcing = False
        #if self.config['drop_remainder']:

        #x, prev_y, labels = self.fetch_data(self.data, self.in_cols, self.out_cols,
        #                                  transformation=self.config['transformation'], **kwargs)

        n_s_feature_dim = self.config['enc_config']['n_s']
        n_h_feature_dim = self.config['enc_config']['n_h']

        if kwargs.get('use_datetime_index', False):  # during deindexification, first feature will be removed.
            n_s_feature_dim += 1
            n_h_feature_dim += 1
            idx = np.expand_dims(x[:, 1:, 0], axis=-1)   # extract the index from x
            prev_y = np.concatenate([prev_y, idx], axis=2)  # insert index in prev_y

        #x, prev_y, labels = self.check_batches(x, prev_y, labels)

        if not self.config['drop_remainder']:
            s0 = np.zeros((x.shape[0], n_s_feature_dim))
            h0 = np.zeros((x.shape[0], n_h_feature_dim))
            x = [x, s0, h0]

        if self.verbosity > 0:
            print_something(x, "input_x")
            print_something(labels, "target")

        return x, prev_y, labels


class OutputAttentionModel(DualAttentionModel):

    def build(self):
        self.config['dec_config'] = self.dec_config
        self.config['enc_config'] = self.enc_config

        setattr(self, 'method', 'output_attention')

        inputs = layers.Input(shape=(self.lookback, self.num_ins))

        self.de_densor_We = layers.Dense(self.enc_config['m'])
        h_de0 = layers.Input(shape=(self.dec_config['n_hde0'],), name='dec_1st_hidden_state')
        s_de0 = layers.Input(shape=(self.dec_config['n_sde0'],), name='dec_1st_cell_state')
        prev_output = layers.Input(shape=(self.lookback - 1, 1), name='input_y')

        enc_lstm_out = layers.LSTM(self.config['lstm_config']['lstm_units'], return_sequences=True,
                                   name='starting_LSTM')(inputs)
        print('Output from LSTM: ', enc_lstm_out)
        enc_out = layers.Reshape((self.lookback, -1), name='enc_out_eq_11')(enc_lstm_out)  # eq 11 in paper
        print('output from first LSTM:', enc_out)

        h, context = self.decoder_attention(enc_out, prev_output, s_de0, h_de0)
        h = layers.Reshape((1, self.dec_config['p']))(h)
        # concatenation of decoder hidden state and the context vector.
        concat = layers.Concatenate(axis=2)([h, context])
        concat = layers.Reshape((-1,))(concat)
        print('decoder concat:', concat)
        result = layers.Dense(self.dec_config['p'], name='eq_22')(concat)   # equation 22
        print('result:', result)
        predictions = layers.Dense(1)(result)

        self._model = self.compile([inputs, prev_output, s_de0, h_de0], predictions)

        return

    def train(self, st=0, en=None, indices=None, **callbacks):

        train_x, train_y, train_label = self.fetch_data(self.data,
                                                        st=st,
                                                        en=en,
                                                        inps=self.in_cols,
                                                        outs=self.out_cols,
                                                        shuffle=True,
                                                        write_data=self.config['CACHEDATA'],
                                                        indices=indices)

        h_de0_train = s_de0_train = np.zeros((train_x.shape[0], self.config['dec_config']['p']))

        inputs = [train_x, train_y, s_de0_train, h_de0_train]
        history = self.fit(inputs, train_label, **callbacks)

        self.plot_loss(history)

        return history

    def predict(self,
                st=0,
                en=None,
                indices=None,
                pref: str = "test",
                scaler_key: str = '5',
                use_datetime_index=False,
                **plot_args):
        setattr(self, 'predict_indices', indices)

        test_x, test_y, test_label = self.fetch_data(self.data,
                                                     st=st,
                                                     en=en, shuffle=False,
                                                     write_data=False,
                                                     inps=self.in_cols,
                                                     outs=self.out_cols,
                                                     indices=indices)

        h_de0_test = s_de0_test = np.zeros((test_x.shape[0], self.config['dec_config']['p']))

        predicted = self._model.predict([test_x, test_y, s_de0_test, h_de0_test],
                                         batch_size=test_x.shape[0],
                                         verbose=1)

        self.process_results(test_label, predicted, pref, **plot_args)

        return predicted, test_label


class NBeatsModel(Model):
    """
    original paper https://arxiv.org/pdf/1905.10437.pdf which is implemented by https://github.com/philipperemy/n-beats
    must be used with normalization
    """

    def training_data(self, data=None, data_keys=None, **kwargs):

        exo_x, x, label = self.fetch_data(data=self.data,
                                          inps=self.in_cols,
                                          outs=self.out_cols,
                                          transformation=self.config['transformation'],
                                          **kwargs)

        return [x, exo_x[:, :, 0:-1]], exo_x[:, :, 0:-1], label   # TODO  .reshape(-1, 1, 1) ?

    def test_data(self, scaler_key='5', data_keys=None, **kwargs):
        return self.training_data(scaler_key=scaler_key, data_keys=data_keys, **kwargs)

    def get_batches(self, df, ins, outs):
        df = pd.DataFrame(df, columns=self.in_cols + self.out_cols)
        return self.prepare_batches(df, ins, outs)

    def deindexify_input_data(self, inputs:list, sort:bool = False, use_datetime_index:bool = False):

        _, inputs, dt_index = Model.deindexify_input_data(self,
                                                          inputs,
                                                          sort=sort,
                                                          use_datetime_index=use_datetime_index)

        return inputs[0], inputs, dt_index