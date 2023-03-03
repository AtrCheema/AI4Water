__all__ = ["DualAttentionModel", "InputAttentionModel"]

from typing import Union, List

from easy_mpl import imshow

from .backend import tf, plt, np, os
from .backend import keras
from .functional import Model as FModel
from ai4water.utils.utils import print_something
from .utils.utils import DataNotFound
from ai4water.nn_tools import check_act_fn
from ai4water.preprocessing import DataSet
from ai4water.models._tensorflow.layer_definition import MyTranspose, MyDot
from ai4water.utils.utils import plot_activations_along_inputs

layers = keras.layers
KModel = keras.models.Model


class DALSTM(keras.layers.Layer):

    def __init__(
            self,
            enc_config: dict = None,
            dec_config: dict = None,
            drop_remainder: bool = True,
            teacher_forcing: bool = False,
            **kwargs
    ):
        self.enc_config = enc_config
        self.dec_config = dec_config
        self.drop_remainder = drop_remainder
        self.teacher_forcing = teacher_forcing

        super().__init__(**kwargs)
        raise NotImplementedError


class DualAttentionModel(FModel):
    """
    This is Dual-Attention LSTM model of Qin_ et al., 2017. The code is adopted
    from this_ repository

    Example:
        >>> from ai4water import DualAttentionModel
        >>> from ai4water.datasets import busan_beach
        >>> data = busan_beach()
        >>> model = DualAttentionModel(lookback=5,
        ...                            input_features=data.columns.tolist()[0:-1],
        ...                            output_features=data.columns.tolist()[-1:])
        ... #If you do not wish to feed previous output as input to the model, you
        ... #can set teacher forcing to False. The drop_remainder argument must be
        ... #set to True in such a case.
        >>> model = DualAttentionModel(teacher_forcing=False, batch_size=4,
        ...                            drop_remainder=True, ts_args={'lookback':5})
        >>> model.fit(data=data)

    .. _Qin:
        https://arxiv.org/abs/1704.02971

    .. _this:
        https://github.com/chensvm/A-Dual-Stage-Attention-Based-Recurrent-Neural-Network-for-Time-Series-Prediction
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
            enc_config: dict = None,
            dec_config: dict = None,
            teacher_forcing: bool = True,
            **kwargs
    ):
        """

        Arguments:
            enc_config:
                dictionary defining configuration of encoder/input attention. It must
                have following three keys

                    - n_h: 20
                    - n_s: 20
                    - m: 20
                    - enc_lstm1_act: None
                    - enc_lstm2_act: None

            dec_config:
                dictionary defining configuration of decoder/output attention. It must have
                following three keys

                    - p: 30
                    - n_hde0: None
                    - n_sde0: None

            teacher_forcing:
                Whether to use the prvious target/observation as input or not. If
                yes, then the model will require 2 inputs. The first input will be
                of shape (num_examples, lookback, num_inputs) while the second input
                will be of shape (num_examples, lookback-1, 1). This second input is
                supposed to be the target variable observed at previous time step.

            kwargs :
                The keyword arguments for the [ai4water's Model][ai4water.Model] class
        """
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

        super(DualAttentionModel, self).__init__(teacher_forcing=teacher_forcing, **kwargs)

    def build(self, input_shape=None):

        self.config['dec_config'] = self.dec_config
        self.config['enc_config'] = self.enc_config
        setattr(self, 'batch_size', self.config['batch_size'])
        setattr(self, 'drop_remainder', self.config['drop_remainder'])

        self.de_LSTM_cell = layers.LSTM(self.dec_config['p'], return_state=True, name='decoder_LSTM')
        self.de_densor_We = layers.Dense(self.enc_config['m'])

        if self.config['drop_remainder']:
            h_de0 = tf.zeros((self.batch_size, self.dec_config['n_hde0']), name='dec_1st_hidden_state')
            s_de0 = tf.zeros((self.batch_size, self.dec_config['n_sde0']), name='dec_1st_cell_state')
        else:
            h_de0 = layers.Input(shape=(self.dec_config['n_hde0'],), name='dec_1st_hidden_state')
            s_de0 = layers.Input(shape=(self.dec_config['n_sde0'],), name='dec_1st_cell_state')

        input_y = None
        if self.teacher_forcing and self.drop_remainder:
            input_y = layers.Input(batch_shape=(self.batch_size, self.lookback - 1, self.num_outs), name='input_y')
        elif not self.drop_remainder:
            input_y = layers.Input(shape=(self.lookback - 1, self.num_outs), name='input_y')

        if self.drop_remainder:
            enc_input = keras.layers.Input(batch_shape=(self.batch_size, self.lookback, self.num_ins), name='enc_input')
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

        # original it was not defined but in tf-keras we need to define it
        sec_dim = self.enc_config['m'] + self.dec_config['p']
        last_reshape = layers.Reshape((sec_dim,), name='last_reshape')(last_concat)  # (None, 50)

        result = layers.Dense(self.dec_config['p'], name='eq_22')(last_reshape)  # (None, 30)  # equation 22

        output = layers.Dense(self.num_outs)(result)
        if self.forecast_len>1:
            output = layers.Reshape(target_shape=(self.num_outs, self.forecast_len))(output)

        initial_input = [enc_input]

        if input_y is not None:
            initial_input.append(input_y)

        if self.config['drop_remainder']:
            self._model = self.compile(model_inputs=initial_input, outputs=output)
        else:
            self._model = self.compile(model_inputs=initial_input + [s0, h0, s_de0, h_de0], outputs=output)

        self.config['model'] = "DualAttentionModel"
        self.config['category'] = "DL"

        if not getattr(self, 'from_check_point', False) and self.verbosity>=0:
            # fit may fail so better to save config before as well.
            # This will be overwritten once the fit is complete
            self.save_config()

        return

    def _encoder(self, enc_inputs, config, lstm2_seq=True, suf: str = '1', s0=None,
                 h0=None, num_ins=None):

        if num_ins is None:
            num_ins = self.num_ins

        self.en_densor_We = layers.Dense(self.lookback, name='enc_We_'+suf)
        _config, act_str = check_act_fn({'activation': config['enc_lstm1_act']})
        self.en_LSTM_cell = layers.LSTM(config['n_h'], return_state=True, activation=_config['activation'],
                                        name='encoder_LSTM_'+suf)
        config['enc_lstm1_act'] = act_str

        # initialize the first cell state
        if s0 is None:
            if self.drop_remainder:
                s0 = tf.zeros((self.batch_size, config['n_s']), name=f'enc_first_cell_state_{suf}')
            else:
                s0 = layers.Input(shape=(config['n_s'],), name='enc_first_cell_state_' + suf)
        # initialize the first hidden state
        if h0 is None:
            if self.drop_remainder:
                h0 = tf.zeros((self.batch_size, config['n_h']), name=f'enc_first_hidden_state_{suf}')
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
        # (none,n,T) Ue(T,T), Ue * Xk in eq 8 of paper
        result2 = MyDot(self.lookback, name='eq_8_mul_'+str(t)+'_'+suf)(x_temp)
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
        result2 = MyDot(self.enc_config['m'])(_h_en_all)

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
            if self.teacher_forcing:
                y_prev = layers.Lambda(lambda y_prev: _y[:, t, :])(_y)  # (batch_size, lookback, 1) -> (batch_size, 1)
                y_prev = layers.Reshape((1, self.num_outs))(y_prev)   # -> (batch_size, 1, 1)

                # concatenation of decoder input and computed context vector  # ??
                y_prev = layers.Concatenate(axis=2)([y_prev, _context])   # (None,1,21)
            else:
                y_prev = _context

            y_prev = layers.Dense(self.num_outs, name='eq_15_'+str(t))(y_prev)  # (None,1,1),  Eq 15  in paper

            _h, _, s = self.de_LSTM_cell(y_prev, initial_state=[_h, s])   # eq 16  ??

        _context = self.one_decoder_attention_step(_h, s, _h_en_all, 'final')
        return _h, _context

    def fetch_data(self, x, y, source, data=None, **kwargs):

        if self.teacher_forcing:
            x, prev_y, labels = getattr(self.dh_, f'{source}_data')(**kwargs)
        else:
            x, labels = getattr(self.dh_, f'{source}_data')(**kwargs)
            prev_y = None

        n_s_feature_dim = self.enc_config['n_s']
        n_h_feature_dim = self.enc_config['n_h']
        p_feature_dim = self.dec_config['p']

        if kwargs.get('use_datetime_index', False):  # during deindexification, first feature will be removed.
            n_s_feature_dim += 1
            n_h_feature_dim += 1
            p_feature_dim += 1
            idx = np.expand_dims(x[:, 1:, 0], axis=-1)   # extract the index from x

            if self.use_true_prev_y:
                prev_y = np.concatenate([prev_y, idx], axis=2)  # insert index in prev_y

        other_inputs = []
        if not self.drop_remainder:
            s0 = np.zeros((x.shape[0], n_s_feature_dim))
            h0 = np.zeros((x.shape[0], n_h_feature_dim))

            h_de0 = s_de0 = np.zeros((x.shape[0], p_feature_dim))
            other_inputs = [s0, h0, s_de0, h_de0]

        if self.teacher_forcing:
            return [x, prev_y] + other_inputs, labels
        else:
            return [x] + other_inputs, labels

    def training_data(self, x=None, y=None, data='training', key=None):

        self._maybe_dh_not_set(data=data)
        return self.fetch_data(x=x, y=y, source='training', data=data, key=key)

    def validation_data(self, x=None, y=None, data='validation', **kwargs):
        self._maybe_dh_not_set(data=data)

        return self.fetch_data(x=x, y=y, source='validation', data=data, **kwargs)

    def test_data(self, x=None, y=None, data='test',  **kwargs):
        self._maybe_dh_not_set(data=data)
        return self.fetch_data(x=x, y=y, source='test', data=data, **kwargs)

    def _maybe_dh_not_set(self, data):
        """if dh_ has not been set yet, try to create it using data argument if
        possible"""
        if isinstance(data, str) and data not in ['training', 'test', 'validation']:
            self.dh_ = DataSet(data=data, **self.data_config)
        elif not isinstance(data, str):
            self.dh_ = DataSet(data=data, **self.data_config)
        return

    def interpret(
            self,
            data=None,
            data_type='training',
            **kwargs):

        return self.plot_act_along_inputs(
            data=data,
            layer_name=f'attn_weight_{self.lookback - 1}_1',
            data_type=data_type,
            **kwargs)

    def get_attention_weights(
            self,
            layer_name: str=None,
            x = None,
            data = None,
            data_type = 'training',
    )->np.ndarray:
        """
        Parameters
        ----------
            layer_name : str, optional
                the name of attention layer. If not given, the final attention
                layer will be used.
            x : optional
                input data, if given, then ``data`` must not be given
            data :
                The data which will be passed to DataSet class to get ``x``
            data_type : str, optional
                the data to make forward pass to get attention weghts. Possible
                values are

                - ``training``
                - ``validation``
                - ``test``
                - ``all``

        Returns
        -------
            a numpy array of shape (num_examples, lookback, num_ins)
        """
        if x is not None:
            # default value
            assert data_type in ("training", "test", "validation", "all")

        layer_name = layer_name or f'attn_weight_{self.lookback - 1}_1'

        assert isinstance(layer_name, str), f"""
            layer_name must be a string, not of {layer_name.__class__.__name__} type
            """

        from ai4water.postprocessing.visualize import Visualize

        kwargs = {}
        if self.config['drop_remainder']:
            kwargs['batch_size'] = self.config['batch_size']

        activation = Visualize(model=self).get_activations(
            layer_names=layer_name,
            x=x,
            data=data,
            data_type=data_type,
            **kwargs)

        activation = activation[layer_name]  # (num_examples, lookback, num_ins)

        return activation

    def plot_avg_attentions_along_inputs(
            self,
            data,
            data_type:str="training",
            layer_name:str = None,
            show:bool=True,
            save:bool=False,
            **kwargs
    ):
        """plots averaged activations along inputs
        """
        activation = self.get_attention_weights(
            layer_name=layer_name,
            data=data,
            data_type=data_type,
        )

        lookback = self.config['ts_args']['lookback']

        act_avg_over_examples = np.mean(activation, axis=0)  # (lookback, num_ins)

        fig, axis = plt.subplots()

        ytick_labels = [f"t-{int(i)}" for i in np.linspace(lookback - 1, 0, lookback)]
        im = imshow(act_avg_over_examples,
                       ax=axis,
                       aspect="auto",
                       yticklabels=ytick_labels,
                       ax_kws=dict(ylabel='lookback steps'),
                       show=False,
                    xticklabels=self.input_features,
                    **kwargs
                       )

        if save:
            plt.savefig(
            os.path.join(self.act_path, f'acts_avg_over_examples_{data_type}'),
                    dpi=400, bbox_inches='tight')

        if show:
            plt.show()
        return im

    def plot_act_along_inputs(
            self,
            data=None,
            x:np.ndarray = None,
            y:np.ndarray = None,
            layer_name: str=None,
            data_type='training',
            feature:Union[str, List[str]]=None,
            vmin=None,
            vmax=None,
            show=False,
            **kwargs
    ):
        """

        parameters
        -----------
        data :
            raw data from which x,y pairs will be extracted. If it is not given
            then, ``x`` and ``y`` must be given.
        data_type :
            type of data to use. It can be ``training``, ``validation``, ``test`` or ``all``.
            It is only valid if ``data`` is provided.
        x :
            input data. Only valid if ``data`` is not given
        y :
            observations/labels corresponding to ``x``. If ``x`` is given, then this
            value must also be provided
        layer_name :
            the layer name to extract attention weights
        feature :
            feature with respect to which attention maps are to be shown
        vmin :
            vmin for imshow
        vmax :
            vmax for imshow
        show :
            whether to show the plot or not
        **kwargs
            keyword arguments for imshow
        """
        if not os.path.exists(self.act_path):
            os.makedirs(self.act_path)

        activation = self.get_attention_weights(
            layer_name=layer_name,
            data=data,
            data_type=data_type,
            x=x,
        )

        lookback = self.config['ts_args']['lookback']
        if x is None:
            x, observations = getattr(self, f'{data_type}_data')(data=data)

            if len(x) == 0 or (isinstance(x, list) and len(x[0]) == 0):
                raise ValueError(f"no {data_type} data found.")
        else:
            assert y is not None, f"""
            if x is given, y must also be given"""
            observations = y

        predictions = self.predict(x=x, process_results=False)

        x = self.inputs_for_attention(x)

        if feature is None:
            features = self.input_features
        else:
            index = self.input_features.index(feature)
            features = [feature]
            activation = np.expand_dims(activation[:, :, index], axis=-1)
            x = np.expand_dims(x[:, index], axis=-1)

        return plot_activations_along_inputs(
            data=x,
            activations=activation,
            observations=observations,
            predictions=predictions,
            in_cols=features,
            out_cols=self.output_features,
            lookback=lookback,
            name=data_type,
            path=self.act_path,
            vmin=vmin,
            vmax=vmax,
            show=show,
            **kwargs
        )

    def plot_act_along_lookback(self, activations, sample=0):

        assert isinstance(activations, np.ndarray)

        activation = activations[sample, :, :]
        act_t = activation.transpose()

        fig, axis = plt.subplots()

        for idx, _name in enumerate(self.input_features):
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

    def _fit_transform_x(self, x):
        """transforms x and puts the transformer in config witht he key name"""
        feature_names = [
            self.input_features,
            [f"{i}" for i in range(self.enc_config['n_s'])],
            [f"{i}" for i in range(self.enc_config['n_h'])],
            [f"{i}" for i in range(self.dec_config['n_hde0'])],
            [f"{i}" for i in range(self.dec_config['n_sde0'])],
        ]
        transformation = [self.config['x_transformation'], None, None, None, None]
        if self.teacher_forcing:
            feature_names.insert(1, self.output_features)
            transformation.insert(1, self.config['y_transformation'])
        return self._fit_transform(x, 'x_transformer_', transformation, feature_names)

    def _fetch_data(self, source:str, x=None, y=None, data=None):
        """The main idea is that the user should be able to fully customize
        training/test data by overwriting training_data and test_data methods.
        However, if x is given or data is DataSet then the training_data/test_data
        methods of this(Model) class will not be called."""

        x, y, prefix, key, user_defined_x = super()._fetch_data(source, x, y, data)
        if isinstance(x, np.ndarray):
            if not self.config['drop_remainder']:
                n_s_feature_dim = self.config['enc_config']['n_s']
                n_h_feature_dim = self.config['enc_config']['n_h']

                s0 = np.zeros((x.shape[0], n_s_feature_dim))
                h0 = np.zeros((x.shape[0], n_h_feature_dim))

                if self.__class__.__name__ == "DualAttentionModel":
                    p_feature_dim = self.dec_config['p']
                    h_de0 = s_de0 = np.zeros((x.shape[0], p_feature_dim))
                    x = [x, s0, h0, h_de0, s_de0]
                else:
                    x = [x, s0, h0]

        return x, y, prefix, key, user_defined_x


class InputAttentionModel(DualAttentionModel):
    """
    InputAttentionModel is same as DualAttentionModel with output attention/decoder part
    removed.

    Example:
        >>> from ai4water import InputAttentionModel
        >>> from ai4water.datasets import busan_beach
        >>> model = InputAttentionModel(
        ... input_features=busan_beach().columns.tolist()[0:-1],
        ... output_features=busan_beach().columns.tolist()[-1:])
        >>> model.fit(data=busan_beach())
    """
    def __init__(self, *args, teacher_forcing=False, **kwargs):
        super(InputAttentionModel, self).__init__(*args, teacher_forcing=teacher_forcing, **kwargs)

    def build(self, input_shape=None):

        self.config['enc_config'] = self.enc_config
        setattr(self, 'batch_size', self.config['batch_size'])
        setattr(self, 'drop_remainder', self.config['drop_remainder'])

        setattr(self, 'method', 'input_attention')
        if self.verbosity>0:
            print('building input attention model')

        enc_input = keras.layers.Input(shape=(self.lookback, self.num_ins), name='enc_input1')
        lstm_out, h0, s0 = self._encoder(enc_input, self.enc_config, lstm2_seq=False)

        act_out = layers.LeakyReLU()(lstm_out)
        predictions = layers.Dense(self.num_outs)(act_out)
        if self.forecast_len>1:
            predictions = layers.Reshape(target_shape=(self.num_outs, self.forecast_len))(predictions)
        if self.verbosity > 2:
            print('predictions: ', predictions)

        inputs = [enc_input]
        if not self.drop_remainder:
            inputs = inputs + [s0, h0]
        self._model = self.compile(model_inputs=inputs, outputs=predictions)

        self.config['model'] = "InputAttentionModel"
        self.config['category'] = "DL"
        if not getattr(self, 'from_check_point', False) and self.verbosity>=0:
            # fit may fail so better to save config before as well.
            # This will be overwritten once the fit is complete
            self.save_config()
        return

    def fetch_data(self, source, x=None, y=None, data=None, **kwargs):

        if x is None:
            if isinstance(data, str):
                if data in ("training", "test", "validation"):
                    if hasattr(self, 'dh_'):
                        data = getattr(self.dh_, f'{data}_data')(**kwargs)
                    else:
                        raise DataNotFound(source)
                else:
                    raise ValueError
            else:
                dh = DataSet(data=data, **self.data_config)
                setattr(self, 'dh_', dh)
                data = getattr(dh, f'{source}_data')(**kwargs)
        else:
            data = x, y

        if self.teacher_forcing:
            x, prev_y, labels = data
        else:
            x, labels = data

        n_s_feature_dim = self.config['enc_config']['n_s']
        n_h_feature_dim = self.config['enc_config']['n_h']

        if kwargs.get('use_datetime_index', False):  # during deindexification, first feature will be removed.
            n_s_feature_dim += 1
            n_h_feature_dim += 1
            idx = np.expand_dims(x[:, 1:, 0], axis=-1)   # extract the index from x
            if self.teacher_forcing:
                prev_y = np.concatenate([prev_y, idx], axis=2)  # insert index in prev_y

        if not self.config['drop_remainder']:
            s0 = np.zeros((x.shape[0], n_s_feature_dim))
            h0 = np.zeros((x.shape[0], n_h_feature_dim))
            x = [x, s0, h0]

        if self.verbosity > 0:
            print_something(x, "input_x")
            print_something(labels, "target")

        if self.teacher_forcing:
            return [x, prev_y], labels
        else:
            return x, labels

    def _fit_transform_x(self, x):
        """transforms x and puts the transformer in config witht he key name
        for conformity we need to add feature names of initial states and their transformations
        will always be None.
        """
        # x can be array when the user does not provide input conditions!
        if isinstance(x, list):
            assert len(x) == 3
            feature_names = [
                self.input_features,
                [f"{i}" for i in range(self.enc_config['n_s'])],
                [f"{i}" for i in range(self.enc_config['n_h'])]
            ]
            transformation = [self.config['x_transformation'], None, None]
            return self._fit_transform(x, 'x_transformer_', transformation, feature_names)
        else:
            transformation = self.config['x_transformation']
            return self._fit_transform(x, 'x_transformer_', transformation,
                                       self.input_features)

