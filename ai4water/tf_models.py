__all__ = ["DualAttentionModel", "InputAttentionModel", "OutputAttentionModel", "NBeatsModel"]

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ai4water.backend import tf
from ai4water.main import Model
from ai4water.functional import Model as FModel
from ai4water.backend import keras
from ai4water.utils.utils import print_something
from ai4water.nn_tools import check_act_fn
from ai4water.layer_definition import MyTranspose, MyDot
from ai4water.utils.utils import plot_activations_along_inputs

layers = keras.layers
KModel = keras.models.Model


class DualAttentionModel(FModel):
    """
    This is Dual-Attention LSTM model of [Qin et al., 2017](https://arxiv.org/abs/1704.02971).
    The code is adopted from [this](https://github.com/chensvm/A-Dual-Stage-Attention-Based-Recurrent-Neural-Network-for-Time-Series-Prediction) repository

    Arguments:
        enc_config : dict
            dictionary defining configuration of encoder/input attention. It must
            have following three keys

                - n_h: 20
                - n_s: 20
                - m: 20
                - enc_lstm1_act: None
                - enc_lstm2_act: None

        dec_config : dict
            dictionary defining configuration of decoder/output attention. It must have
            following three keys

                - p: 30
                - n_hde0: None
                - n_sde0: None

        teacher_forcing : bool
            Whether to use the prvious target/observation as input or not. If
            yes, then the model will require 2 inputs. The first input will be
            of shape (num_examples, lookback, num_inputs) while the second input
            will be of shape (num_examples, lookback-1, 1). This second input is
            supposed to be the target variable observed at previous time step.

        kwargs :
            The keyword arguments for the ai4water.Model class

    Example
    -------
    ```python
    >>> from ai4water import DualAttentionModel
    >>> from ai4water.datasets import arg_beach
    >>> model = DualAttentionModel(data=arg_beach(),
    >>>                            use_true_prev_y=False, batch_size=4, drop_remainder=True, lookback=5)
    >>> model.fit()
    ```
    """
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

    def __init__(
            self,
            enc_config:dict=None,
            dec_config:dict=None,
            teacher_forcing:bool=True,
            **kwargs
    ):

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
        #self.use_true_prev_y = use_true_prev_y

        super(DualAttentionModel, self).__init__(teacher_forcing=teacher_forcing, **kwargs)

        setattr(self, 'category', "DL")

    def build(self, input_shape=None):

        self.config['dec_config'] = self.dec_config
        self.config['enc_config'] = self.enc_config

        self.de_LSTM_cell = layers.LSTM(self.dec_config['p'], return_state=True, name='decoder_LSTM')
        self.de_densor_We = layers.Dense(self.enc_config['m'])

        if self.config['drop_remainder']:
            h_de0 = tf.zeros((self.config['batch_size'], self.dec_config['n_hde0']), name='dec_1st_hidden_state')
            s_de0 = tf.zeros((self.config['batch_size'], self.dec_config['n_sde0']), name='dec_1st_cell_state')
        else:
            h_de0 = layers.Input(shape=(self.dec_config['n_hde0'],), name='dec_1st_hidden_state')
            s_de0 = layers.Input(shape=(self.dec_config['n_sde0'],), name='dec_1st_cell_state')

        input_y = None
        if self.config['teacher_forcing'] and self.config['drop_remainder']:
            input_y = layers.Input(batch_shape=(self.config['batch_size'], self.lookback - 1, self.num_outs), name='input_y')
        elif not self.config['drop_remainder']:
            input_y = layers.Input(shape=(self.lookback - 1, self.num_outs), name='input_y')

        if self.config['drop_remainder']:
            enc_input = keras.layers.Input(batch_shape=(self.config['batch_size'], self.lookback, self.num_ins), name='enc_input')
        else:
            enc_input = keras.layers.Input(shape=(self.lookback, self.num_ins), name='enc_input')

        enc_lstm_out, s0, h0 = self._encoder(enc_input, self.config['enc_config'])

        # originally the last dimentions was -1 but I put it equal to 'm'
        # eq 11 in paper
        enc_out = layers.Reshape((self.lookback, self.enc_config['m']), name='enc_out_eq_11')(enc_lstm_out)

        h, context = self.decoder_attention(enc_out, input_y, s_de0, h_de0)
        h = layers.Reshape((self.num_outs, self.dec_config['p']))(h)
        # concatenation of decoder hidden state and the context vector.
        last_concat = layers.Concatenate(axis=2, name='last_concat')([h, context])  # (None, 1, 50)

        sec_dim = self.enc_config['m'] + self.dec_config['p']  # original it was not defined but in tf-keras we need to define it
        last_reshape = layers.Reshape((sec_dim,), name='last_reshape')(last_concat)  # (None, 50)

        result = layers.Dense(self.dec_config['p'], name='eq_22')(last_reshape)  # (None, 30)  # equation 22

        output = layers.Dense(self.num_outs)(result)
        output = layers.Reshape(target_shape=(self.num_outs, self.forecast_len))(output)

        initial_input = [enc_input]

        if input_y is not None:
            initial_input.append(input_y)

        if self.config['drop_remainder']:
            self._model = self.compile(model_inputs=initial_input, outputs=output)
        else:
            self._model = self.compile(model_inputs=initial_input + [s0, h0, s_de0, h_de0], outputs=output)

        return

    def _encoder(self, enc_inputs, config, lstm2_seq=True, suf: str = '1', s0=None, h0=None, num_ins=None):

        if num_ins is None:
            num_ins = self.num_ins

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

        enc_attn_out = self.encoder_attention(enc_inputs, s0, h0, num_ins, suf)

        enc_lstm_in = layers.Reshape((self.lookback, num_ins), name='enc_lstm_input_'+suf)(enc_attn_out)

        _config, act_str = check_act_fn({'activation': config['enc_lstm2_act']})
        enc_lstm_out = layers.LSTM(config['m'], return_sequences=lstm2_seq, activation=_config['activation'],
                                   name='LSTM_after_encoder_'+suf)(enc_lstm_in)  # h_en_all
        config['enc_lstm2_act'] = act_str

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

    def encoder_attention(self, _input, _s0, _h0, num_ins, suf: str = '1'):

        s = _s0
        _h = _h0
        # initialize empty list of outputs
        attention_weight_t = None
        for t in range(self.lookback):

            _context = self.one_encoder_attention_step(_h, s, _input, t, suf=suf)  # (none,1,n)
            x = layers.Lambda(lambda x: _input[:, t, :])(_input)
            x = layers.Reshape((1, num_ins))(x)

            _h, _, s = self.en_LSTM_cell(x, initial_state=[_h, s])
            if t != 0:
                # attention_weight_t = layers.Merge(mode='concat', concat_axis=1,
                #                                    name='attn_weight_'+str(t))([attention_weight_t,
                #                                                                _context])
                attention_weight_t = layers.Concatenate(
                    axis=1,
                    name='attn_weight_'+str(t)+'_'+suf)([attention_weight_t, _context])
            else:
                attention_weight_t = _context

        # get the driving input series
        enc_output = layers.Multiply(name='enc_output_'+suf)([attention_weight_t, _input])  # equation 10 in paper

        return enc_output

    def one_decoder_attention_step(self, _h_de_prev, _s_de_prev, _h_en_all, t):
        """
        :param _h_de_prev: previous hidden state
        :param _s_de_prev: previous cell state
        :param _h_en_all: (None,T,m),n is length of input series at time t,T is length of time series
        :param t: int, timestep
        :return: x_t's attention weights,total n numbers,sum these are 1
        """
        # concatenation of the previous hidden state and cell state of the LSTM unit in eq 12
        _concat = layers.Concatenate(name='eq_12_'+str(t))([_h_de_prev, _s_de_prev])  # (None,1,2p)
        result1 = self.de_densor_We(_concat)   # (None,1,m)
        result1 = layers.RepeatVector(self.lookback)(result1)  # (None,T,m)
        result2 = MyDot(self.config['enc_config']['m'])(_h_en_all)

        result3 = layers.Add()([result1, result2])  # (None,T,m)
        result4 = layers.Activation(activation='tanh')(result3)  # (None,T,m)
        result5 = MyDot(1)(result4)

        beta = layers.Activation(activation='softmax', name='eq_13_'+str(t))(result5)    # equation 13
        _context = layers.Dot(axes=1, name='eq_14_'+str(t))([beta, _h_en_all])  # (1,m)  # equation 14 in paper
        return _context

    def decoder_attention(self, _h_en_all, _y, _s0, _h0):
        s = _s0
        _h = _h0

        for t in range(self.lookback-1):
            _context = self.one_decoder_attention_step(_h, s, _h_en_all, t)  # (batch_size, 1, 20)

            # if we want to use the true value of target of previous timestep as input then we will use _y
            if self.config['teacher_forcing']:
                y_prev = layers.Lambda(lambda y_prev: _y[:, t, :])(_y)  # (batch_size, lookback, 1) -> (batch_size, 1)
                y_prev = layers.Reshape((1, self.num_outs))(y_prev)   # -> (batch_size, 1, 1)

                # concatenation of decoder input and computed context vector  # ??
                y_prev = layers.Concatenate(axis=2)([y_prev, _context])   # (None,1,21)
            else:
                y_prev = _context

            y_prev = layers.Dense(self.num_outs, name='eq_15_'+str(t))(y_prev)       # (None,1,1),                   Eq 15  in paper

            _h, _, s = self.de_LSTM_cell(y_prev, initial_state=[_h, s])   # eq 16  ??

        _context = self.one_decoder_attention_step(_h, s, _h_en_all, 'final')
        return _h, _context

    def fetch_data(self, source, **kwargs):

        if self.config['teacher_forcing']:
            self.dh.teacher_forcing = True
            x, prev_y, labels = getattr(self.dh, f'{source}_data')(**kwargs)
            self.dh.teacher_forcing = False
        else:
            x, labels = getattr(self.dh, f'{source}_data')(**kwargs)
            prev_y = None

        n_s_feature_dim = self.config['enc_config']['n_s']
        n_h_feature_dim = self.config['enc_config']['n_h']
        p_feature_dim = self.config['dec_config']['p']

        if kwargs.get('use_datetime_index', False):  # during deindexification, first feature will be removed.
            n_s_feature_dim += 1
            n_h_feature_dim += 1
            p_feature_dim += 1
            idx = np.expand_dims(x[:, 1:, 0], axis=-1)   # extract the index from x

            if self.use_true_prev_y:
                prev_y = np.concatenate([prev_y, idx], axis=2)  # insert index in prev_y

        other_inputs = []
        if not self.config['drop_remainder']:
            s0 = np.zeros((x.shape[0], n_s_feature_dim))
            h0 = np.zeros((x.shape[0], n_h_feature_dim))

            h_de0 = s_de0 = np.zeros((x.shape[0], p_feature_dim))
            other_inputs = [s0, h0, s_de0, h_de0]

        if self.config['teacher_forcing']:
            return [x, prev_y] + other_inputs, prev_y, labels
        else:
            return [x] + other_inputs, labels


    def training_data(self, **kwargs):
        return self.fetch_data('training', **kwargs)

    def validation_data(self, **kwargs):
        return self.fetch_data('validation', **kwargs)

    def test_data(self,  **kwargs):
        return self.fetch_data('test', **kwargs)

    def interpret(self, data='training', **kwargs):
        import matplotlib
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)

        self.plot_act_along_inputs(f'attn_weight_{self.lookback - 1}_1',
                                   data=data,
                                   **kwargs)
        return

    def plot_act_along_inputs(
            self,
            layer_name: str,
            name: str = None,
            vmin=None,
            vmax=None,
            data='training',
            show=False
    ):

        data_name = name or data

        assert isinstance(layer_name, str), "layer_name must be a string, not of {} type".format(layer_name.__class__.__name__)

        predictions, observations = self.predict(process_results=False, data=data, return_true=True)

        from ai4water.postprocessing.visualize import Visualize

        activation = Visualize(model=self).get_activations(layer_names=layer_name, data=data)

        data, _, _ = getattr(self, f'{data}_data')()
        lookback = self.config['lookback']

        activation = activation[layer_name]  # (num_examples, lookback, num_ins)

        act_avg_over_examples = np.mean(activation, axis=0)  # (lookback, num_ins)

        plt.close('all')

        fig, axis = plt.subplots()
        im = plt.imshow(act_avg_over_examples, aspect='auto')
        ytick_labels = [f"t-{int(i)}" for i in np.linspace(lookback - 1, 0, lookback)]
        axis.set_ylabel('lookback steps')
        axis.set_yticks(np.arange(len(ytick_labels)))
        axis.set_yticklabels(ytick_labels)
        axis.set_xticks(np.arange(len(self.in_cols)))
        axis.set_xticklabels(self.in_cols, rotation=90)
        fig.colorbar(im, orientation='horizontal', pad=0.3)
        plt.savefig(os.path.join(self.act_path, f'acts_avg_over_examples_{data_name}') ,
                    dpi=400, bbox_inches='tight')

        data = self.inputs_for_attention(data)

        plot_activations_along_inputs(
            data=data,
            activations=activation,
            observations=observations,
            predictions=predictions,
            in_cols=self.in_cols,
            out_cols=self.out_cols,
            lookback=lookback,
            name=name,
            path=self.act_path,
            vmin=vmin,
            vmax=vmax,
            show=show
        )
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
        """ returns the inputs for attention mechanism """
        if isinstance(inputs, list):
            inputs = inputs[0]

        inputs = inputs[:, -1, :]  # why 0, why not -1

        assert inputs.shape[1] == self.num_ins

        return inputs


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