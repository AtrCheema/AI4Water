__all__ = ["attn_layers", "SelfAttention", "AttentionLSTM"]


from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

import math

from ai4water.backend import np, tf, keras

initializers = tf.keras.initializers
regularizers = tf.keras.regularizers
constraints = tf.keras.constraints
Layer = tf.keras.layers.Layer
layers = tf.keras.layers
K = tf.keras.backend
Dense = tf.keras.layers.Dense
Lambda = tf.keras.layers.Lambda
Activation = tf.keras.layers.Activation
Softmax = tf.keras.layers.Softmax
dot = tf.keras.layers.dot
concatenate = tf.keras.layers.concatenate


# A review of different attention mechanisms is given at following link
# https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
# Raffel, SeqWeightedSelfAttention and HierarchicalAttention appear to be very much similar.


class SelfAttention(Layer):
    """
    SelfAttention is originally proposed by Cheng et al., 2016 [1]_
    Here using the implementation of Philipperemy from
    [2]_ with modification
    that `attn_units` and `attn_activation` attributes can be changed.
    The default values of these attributes are same as used by the auther.
    However, there is another implementation of SelfAttention at [3]_
    but the author have cited a different paper i.e. Zheng et al., 2018 [4]_ and
    named it as additive attention.
    A useful discussion about this (in this class) implementation can be found at [5]_

    Examples
    --------
    >>> from ai4water.models.tensorflow import SelfAttention
    >>> from tensorflow.keras.layers import Input, LSTM, Dense
    >>> from tensorflow.keras.models import Model
    >>> import numpy as np
    >>> inp = Input(shape=(10, 1))
    >>> lstm = LSTM(2, return_sequences=True)(inp)
    >>> sa, _ = SelfAttention()(lstm)
    >>> out = Dense(1)(sa)
    ...
    >>> model = Model(inputs=inp, outputs=out)
    >>> model.compile(loss="mse")
    ...
    >>> print(model.summary())
    ...
    >>> x = np.random.random((100, 10, 1))
    >>> y = np.random.random((100, 1))
    >>> h = model.fit(x=x, y=y)

    # using with ai4water's Model
    >>> from ai4water import Model
    >>> seq_len = 20
    >>> model = Model(
    ...    model = {"layers": {
    ...         "Input_1": {"shape": (seq_len, 1)},
    ...         "LSTM_1": {"config": {"units": 16, "return_sequences": True}, "inputs": "Input_1"},
    ...         "SelfAttention_1": {"config": {},
    ...                             "outputs": ["attention_vector1", "attention_weights1"]},
    ...         "Dense": {"config": {"units": 1}, "inputs": "attention_vector1"}
    ...     }})

    References
    ----------
    .. [1] https://arxiv.org/pdf/1601.06733.pdf
    .. [2] https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention/attention.py
    .. [3] https://github.com/CyberZHG/keras-self-attention/blob/master/keras_self_attention/seq_self_attention.py
    .. [4] https://arxiv.org/pdf/1806.01264.pdf
    .. [5] https://github.com/philipperemy/keras-attention-mechanism/issues/14
    """
    def __init__(
            self,
            units:int = 128,
            activation:str = 'tanh',
            return_attention_weights:bool = True,
            **kwargs
    ):
        """
        Parameters
        ----------
            units : int, optional (default=128)
                number of units for attention mechanism
            activation : str, optional (default="tanh")
                activation function to use in attention mechanism
            return_attention_weights : bool, optional (default=True)
                if True, then it returns two outputs, first is attention vector
                of shape (batch_size, units) and second is of shape (batch_size, time_steps)
                If False, then returns only attention vector.
            **kwargs :
                any additional keyword arguments for keras Layer.
        """
        self.units = units
        self.attn_activation = activation
        self.return_attention_weights = return_attention_weights
        super().__init__(**kwargs)

    def build(self, input_shape):
        hidden_size = int(input_shape[-1])
        self.d1 = Dense(hidden_size, use_bias=False)
        self.act = Activation('softmax')
        self.d2 = Dense(self.units, use_bias=False, activation=self.attn_activation)
        return

    def call(self, hidden_states, *args, **kwargs):
        """
        Many-to-one attention mechanism for Keras.
        @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
        @return: 2D tensor with shape (batch_size, 128)
        @author: felixhao28.
        The original code which has here been modified had Apache Licence 2.0.
        """
        hidden_size = int(hidden_states.shape[2])
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = self.d1(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,))(hidden_states)
        score = dot([score_first_part, h_t], [2, 1])
        attention_weights =self.act(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = dot([hidden_states, attention_weights], [1, 1])
        pre_activation = concatenate([context_vector, h_t])
        attention_vector = self.d2(pre_activation)

        if self.return_attention_weights:
            return attention_vector, attention_weights
        return attention_vector


class BahdanauAttention(Layer):
    """
    Also known as Additive attention.
    https://github.com/thushv89/attention_keras/blob/master/src/layers/attention.py
    This can be implemented in encoder-decoder model as shown here.
    https://github.com/thushv89/attention_keras/blob/master/src/examples/nmt/model.py

    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).

    There are three sets of weights introduced W_a, U_a, and V_a

    The original code had following MIT licence
     ----------------------------------------------------------
     MIT License

    Copyright (c) 2019 Thushan Ganegedara

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

     ----------------------------------------------------------
     """

    def __init__(self, **kwargs):

        super(BahdanauAttention, self).__init__(**kwargs)

    def build(self, input_shape):

        assert isinstance(input_shape, list)

        # Create a trainable weight variable for this layer.


        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)

        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)

        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(BahdanauAttention, self).build(input_shape)


    def __call__(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """

        if not self.built:
            self._maybe_build(inputs)

        assert type(inputs) == list

        encoder_out_seq, decoder_out_seq = inputs

        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):

            """ Step function for computing energy for a single decoder state """

            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))

            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""

            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]

            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # batch_size*en_seq_len, latent_dim
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))

            # batch_size*en_seq_len, latent_dim

            W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))

            if verbose:
                print('wa.s>',W_a_dot_s.shape)

            """ Computing hj.Ua """
            # batch_size, 1, latent_dim
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)

            if verbose:
                print('Ua.h>',U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # batch_size*en_seq_len, latent_dim
            reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))

            if verbose:
                print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # batch_size, en_seq_len
            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))

            # batch_size, en_seq_len
            e_i = K.softmax(e_i, name='softmax')

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """

            # batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)

            if verbose:
                print('ci>', c_i.shape)

            return c_i, [c_i]

        def create_inital_state(inputs, hidden_size):

            # We are not using initial states, but need to pass something to K.rnn funciton

            fake_state = K.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim)

            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)

            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)

            fake_state = K.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim)

            return fake_state

        fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])

        fake_state_e = create_inital_state(encoder_out_seq, encoder_out_seq.shape[1])  # <= (batch_size, enc_seq_len, latent_dim)


        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )
        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):

        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class HierarchicalAttention(Layer):
    """
    Used from https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al., 2016 [https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(HierarchicalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(HierarchicalAttention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def __call__(self, x, mask=None):
        # TODO different result when using call()
        if not self.built:
            self._maybe_build(x)

        uit = dot_product(x, self.W)  # eq 5 in paper

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)  # eq 5 in paper
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        # eq 6 in paper
        a /= math_ops.cast(math_ops.reduce_sum(a, axis=1, keepdims=True, name="HA_sum") + K.epsilon(), K.floatx(),
                           name="HA_cast")

        a = array_ops.expand_dims(a, axis=-1, name="HA_expand_dims")
        weighted_input = tf.math.multiply(x,a,name="HA_weighted_input")  #x * a
        return K.sum(weighted_input, axis=1)  # eq 7 in paper

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class SeqSelfAttention(Layer):

    ATTENTION_TYPE_ADD = 'additive'
    ATTENTION_TYPE_MUL = 'multiplicative'

    def __init__(self,
                 units=32,
                 attention_width=None,
                 attention_type=ATTENTION_TYPE_ADD,
                 return_attention=False,
                 history_only=False,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_additive_bias=True,
                 use_attention_bias=True,
                 attention_activation=None,
                 attention_regularizer_weight=0.0,
                 **kwargs):
        """
        using implementation of
        https://github.com/CyberZHG/keras-self-attention/blob/master/keras_self_attention/seq_self_attention.py
        Layer initialization.

        For additive attention, see: https://arxiv.org/pdf/1806.01264.pdf

        :param units: The dimension of the vectors that used to calculate the attention weights.
        :param attention_width: The width of local attention.
        :param attention_type: 'additive' or 'multiplicative'.
        :param return_attention: Whether to return the attention weights for visualization.
        :param history_only: Only use historical pieces of data.
        :param kernel_initializer: The initializer for weight matrices.
        :param bias_initializer: The initializer for biases.
        :param kernel_regularizer: The regularization for weight matrices.
        :param bias_regularizer: The regularization for biases.
        :param kernel_constraint: The constraint for weight matrices.
        :param bias_constraint: The constraint for biases.
        :param use_additive_bias: Whether to use bias while calculating the relevance of inputs features
                                  in additive mode.
        :param use_attention_bias: Whether to use bias while calculating the weights of attention.
        :param attention_activation: The activation used for calculating the weights of attention.
        :param attention_regularizer_weight: The weights of attention regularizer.
        :param kwargs: Parameters for parent class.

        The original code which has here been slightly modified came with following licence.
        ----------------------------------
        MIT License

        Copyright (c) 2018 PoW

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        -----------------------------------------------------------------------
        """
        super(SeqSelfAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.return_attention = return_attention
        self.history_only = history_only
        if history_only and attention_width is None:
            self.attention_width = int(1e9)

        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.attention_activation = keras.activations.get(attention_activation)
        self.attention_regularizer_weight = attention_regularizer_weight
        self._backend = keras.backend.backend()

        if attention_type.upper().startswith('ADD'):
            self.Wx, self.Wt, self.bh = None, None, None
            self.Wa, self.ba = None, None
        elif attention_type.upper().startswith('MUL'):
            self.Wa, self.ba = None, None
        else:
            raise NotImplementedError('No implementation for attention type : ' + attention_type)

    def get_config(self):
        config = {
            'units': self.units,
            'attention_width': self.attention_width,
            'attention_type': self.attention_type,
            'return_attention': self.return_attention,
            'history_only': self.history_only,
            'use_additive_bias': self.use_additive_bias,
            'use_attention_bias': self.use_attention_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'attention_activation': keras.activations.serialize(self.attention_activation),
            'attention_regularizer_weight': self.attention_regularizer_weight,
        }
        base_config = super(SeqSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.attention_type.upper().startswith('ADD'):
            self._build_additive_attention(input_shape)
        elif self.attention_type.upper().startswith('MUL'):
            self._build_multiplicative_attention(input_shape)
        super(SeqSelfAttention, self).build(input_shape)

    def _build_additive_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wx'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_additive_bias:
            self.bh = self.add_weight(shape=(self.units,),
                                      name='{}_Add_bh'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Add_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def _build_multiplicative_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wa = self.add_weight(shape=(feature_dim, feature_dim),
                                  name='{}_Mul_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Mul_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def __call__(self, inputs, mask=None, **kwargs):
        # TODO different result when using call()
        if not self.built:
            self._maybe_build(inputs)

        input_len = K.shape(inputs)[1]

        if self.attention_type.upper().startswith('ADD'):
            e = self._call_additive_emission(inputs)
        else:
            e = self._call_multiplicative_emission(inputs)

        if self.attention_activation is not None:
            e = self.attention_activation(e)
        if self.attention_width is not None:
            if self.history_only:
                lower = K.arange(0, input_len) - (self.attention_width - 1)
            else:
                lower = K.arange(0, input_len) - self.attention_width // 2
            lower = K.expand_dims(lower, axis=-1)
            upper = lower + self.attention_width
            indices = K.expand_dims(K.arange(0, input_len), axis=0)
            e -= 10000.0 * (1.0 - K.cast(lower <= indices, K.floatx()) * K.cast(indices < upper, K.floatx()))
        if mask is not None:
            mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)
            e -= 10000.0 * ((1.0 - mask) * (1.0 - K.permute_dimensions(mask, (0, 2, 1))))

        # a_{t} = \text{softmax}(e_t)
        a = Softmax(axis=-1, name='SeqSelfAttention_Softmax')(e)

        # l_t = \sum_{t'} a_{t, t'} x_{t'}
        v = K.batch_dot(a, inputs)
        if self.attention_regularizer_weight > 0.0:
            self.add_loss(self._attention_regularizer(a))

        if self.return_attention:
            return [v, a]
        return v

    def _call_additive_emission(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, input_len = input_shape[0], input_shape[1]

        # h_{t, t'} = \tanh(x_t^T W_t + x_{t'}^T W_x + b_h)
        q = K.expand_dims(K.dot(inputs, self.Wt), 2)
        k = K.expand_dims(K.dot(inputs, self.Wx), 1)
        if self.use_additive_bias:
            h = K.tanh(q + k + self.bh)
        else:
            h = K.tanh(q + k)

        # e_{t, t'} = W_a h_{t, t'} + b_a
        if self.use_attention_bias:
            e = K.reshape(K.dot(h, self.Wa) + self.ba, (batch_size, input_len, input_len))
        else:
            e = K.reshape(K.dot(h, self.Wa), (batch_size, input_len, input_len))
        return e

    def _call_multiplicative_emission(self, inputs):
        # e_{t, t'} = x_t^T W_a x_{t'} + b_a
        e = K.batch_dot(K.dot(inputs, self.Wa), K.permute_dimensions(inputs, (0, 2, 1)))
        if self.use_attention_bias:
            e += self.ba[0]
        return e

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if self.return_attention:
            return [mask, None]
        return mask

    def _attention_regularizer(self, attention):
        batch_size = K.cast(K.shape(attention)[0], K.floatx())
        input_len = K.shape(attention)[-1]
        indices = K.expand_dims(K.arange(0, input_len), axis=0)
        diagonal = K.expand_dims(K.arange(0, input_len), axis=-1)
        eye = K.cast(K.equal(indices, diagonal), K.floatx())
        return self.attention_regularizer_weight * K.sum(K.square(K.batch_dot(
            attention,
            K.permute_dimensions(attention, (0, 2, 1))) - eye)) / batch_size

    @staticmethod
    def get_custom_objects():
        return {'SeqSelfAttention': SeqSelfAttention}


class SeqWeightedAttention(keras.layers.Layer):
    r"""Y = \text{softmax}(XW + b) X
    See: https://arxiv.org/pdf/1708.00524.pdf
    using implementation of https://github.com/CyberZHG/keras-self-attention/blob/master/keras_self_attention/seq_weighted_attention.py
            The original code which has here been slightly modified came with following licence.
        -----------------------------------------------------------------------
        MIT License

        Copyright (c) 2018 PoW

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        -----------------------------------------------------------------------

    """

    def __init__(self, use_bias=True, return_attention=False, **kwargs):
        super(SeqWeightedAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.use_bias = use_bias
        self.return_attention = return_attention
        self.W, self.b = None, None

    def get_config(self):
        config = {
            'use_bias': self.use_bias,
            'return_attention': self.return_attention,
        }
        base_config = super(SeqWeightedAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.W = self.add_weight(shape=(int(input_shape[2]), 1),
                                 name='{}_W'.format(self.name),
                                 initializer=keras.initializers.get('uniform'))
        if self.use_bias:
            self.b = self.add_weight(shape=(1,),
                                     name='{}_b'.format(self.name),
                                     initializer=keras.initializers.get('zeros'))
        super(SeqWeightedAttention, self).build(input_shape)

    def __call__(self, x, mask=None):
        # TODO different result when using call()
        if not self.built:
            self._maybe_build(x)

        logits = K.dot(x, self.W)
        if self.use_bias:
            logits += self.b
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            logits -= 10000.0 * (1.0 - mask)
        ai = math_ops.exp(logits - K.max(logits, axis=-1, keepdims=True), name="SeqWeightedAttention_exp")
        #att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        att_weights = tf.math.divide(ai, (math_ops.reduce_sum(ai, axis=1, keepdims=True,
                                                              name="SeqWeightedAttention_sum") + K.epsilon()),
                                     name="SeqWeightedAttention_weights")
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return input_shape[0], output_len

    def compute_mask(self, _, input_mask=None):
        if self.return_attention:
            return [None, None]
        return None

    @staticmethod
    def get_custom_objects():
        return {'SeqWeightedAttention': SeqWeightedAttention}


class SnailAttention(Layer):
    """
    Based on work of Mishra et al., 2018 https://openreview.net/pdf?id=B1DmUzWAW
    Adopting code from https://github.com/philipperemy/keras-snail-attention/blob/master/attention.py
    """

    def __init__(self, dims, k_size, v_size, seq_len=None, **kwargs):
        self.k_size = k_size
        self.seq_len = seq_len
        self.v_size = v_size
        self.dims = dims
        self.sqrt_k = math.sqrt(k_size)
        self.keys_fc = None
        self.queries_fc = None
        self.values_fc = None
        super(SnailAttention, self).__init__(**kwargs)


    def build(self, input_shape):
        # https://stackoverflow.com/questions/54194724/how-to-use-keras-layers-in-custom-keras-layer
        self.keys_fc = Dense(self.k_size,  name="Keys_SnailAttn")
        self.keys_fc.build((None, self.dims))
        self._trainable_weights.extend(self.keys_fc.trainable_weights)

        self.queries_fc = Dense(self.k_size, name="Queries_SnailAttn")
        self.queries_fc.build((None, self.dims))
        self._trainable_weights.extend(self.queries_fc.trainable_weights)

        self.values_fc = Dense(self.v_size,  name="Values_SnailAttn")
        self.values_fc.build((None, self.dims))
        self._trainable_weights.extend(self.values_fc.trainable_weights)
        #super(SnailAttention, self).__init__(**kwargs)

    def __call__(self, inputs, **kwargs):

        if not self.built:
            self._maybe_build(inputs)

        # check that the implementation matches exactly py torch.
        keys = self.keys_fc(inputs)
        queries = self.queries_fc(inputs)
        values = self.values_fc(inputs)
        logits = K.batch_dot(queries, K.permute_dimensions(keys, (0, 2, 1)))
        mask = K.ones_like(logits) * np.triu((-np.inf) * np.ones(logits.shape.as_list()[1:]), k=1)
        logits = mask + logits
        probs = Softmax(axis=-1, name="Softmax_SnailAttn")(logits / self.sqrt_k)
        read = K.batch_dot(probs, values)
        output = K.concatenate([inputs, read], axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] += self.v_size
        return tuple(output_shape)


def regularized_padded_conv(conv_dim, *args, **kwargs):
    if conv_dim == "1d":
        return layers.Conv1D(*args, **kwargs, padding='same', use_bias=False,
                             kernel_initializer='he_normal',
                             kernel_regularizer=regularizers.l2(5e-4))
    elif conv_dim == "2d":
        return layers.Conv2D(*args, **kwargs, padding='same', use_bias=False,
                             kernel_initializer='he_normal',
                             kernel_regularizer=regularizers.l2(5e-4))
    else:
        raise ValueError(f"conv_dim must be either 1d or 2d but it is {conv_dim}")

class ChannelAttention(layers.Layer):
    """Code adopted from https://github.com/zhangkaifang/CBAM-TensorFlow2.0.
    This feature attention generates time step context descriptors self.avg and self. max by using both average and
    max pooling operations along the time step axis and then fowards to a shared multi-layer perception (MLP) to produce
    the feature (channel) attention map."""
    def __init__(self, conv_dim, in_planes, ratio=16, **kwargs):

        if conv_dim not in ["1d", "2d"]:
            raise ValueError(f" conv_dim must be either 1d or 2d but it is {conv_dim}")

        super(ChannelAttention, self).__init__(**kwargs)
        if conv_dim == "1d":
            self.axis = (1,)
            self.avg= layers.GlobalAveragePooling1D()
            self.max= layers.GlobalMaxPooling1D()
            self.conv1 = layers.Conv1D(in_planes//ratio, kernel_size=1, strides=1, padding='same',
                                       kernel_regularizer=regularizers.l2(5e-4),
                                       use_bias=True, activation=tf.nn.relu, name='channel_attn1')
            self.conv2 = layers.Conv1D(in_planes, kernel_size=1, strides=1, padding='same',
                                       kernel_regularizer=regularizers.l2(5e-4),
                                       use_bias=True, name='channel_attn2')
        elif conv_dim == "2d":
            self.axis = (1,1)
            self.avg= layers.GlobalAveragePooling2D()
            self.max= layers.GlobalMaxPooling2D()
            self.conv1 = layers.Conv2D(in_planes//ratio, kernel_size=1, strides=1, padding='same',
                                       kernel_regularizer=regularizers.l2(5e-4),
                                       use_bias=True, activation=tf.nn.relu, name='channel_attn1')
            self.conv2 = layers.Conv2D(in_planes, kernel_size=1, strides=1, padding='same',
                                       kernel_regularizer=regularizers.l2(5e-4),
                                       use_bias=True, name='channe2_attn1')

    def __call__(self, inputs, *args):
        avg = self.avg(inputs)  # [256, 32, 32, 64] -> [256, 64]
        max_pool = self.max(inputs) # [256, 32, 32, 64] -> [256, 64]
        avg = layers.Reshape((*self.axis, avg.shape[1]))(avg)   # shape (None, 1, 1 feature)  # [256, 1, 1, 64]
        max_pool = layers.Reshape((*self.axis, max_pool.shape[1]))(max_pool)   # shape (None, 1, 1 feature)  # [256, 1, 1, 64]
        avg_out = self.conv2(self.conv1(avg))  # [256, 1, 1, 64] -> [256, 1, 1, 4] -> [256, 1, 1, 64]
        max_out = self.conv2(self.conv1(max_pool))  # [256, 1, 1, 64] -> [256, 1, 1, 4] -> [256, 1, 1, 64]
        out = avg_out + max_out  # [256, 1, 1, 64]
        out = tf.nn.sigmoid(out, name="ChannelAttention_sigmoid")  # [256, 1, 1, 64]

        return out


class SpatialAttention(layers.Layer):
    """Code adopted from https://github.com/zhangkaifang/CBAM-TensorFlow2.0 .
    The time step (spatial) attention module generates a concatenated feature
    descriptor [F'Tavg;F'Tmax]∈R2×T by applying average pooling and max pooling
    along the feature axis, followed by a standard convolution layer.[6].

    .. [6] Cheng, Y., Liu, Z., & Morimoto, Y. (2020). Attention-Based SeriesNet:
        An Attention-Based Hybrid Neural Network Model
        for Conditional Time Series Forecasting. Information, 11(6), 305.
    """
    def __init__(self, conv_dim,  kernel_size=7, **kwargs):

        if conv_dim not in ["1d", "2d"]:
            raise ValueError(f" conv_dim must be either 1d or 2d but it is {conv_dim}")

        super(SpatialAttention, self).__init__(**kwargs)
        if conv_dim == "1d":
            self.axis = 2
        elif conv_dim == "2d":
            self.axis = 3
        self.conv1 = regularized_padded_conv(conv_dim,
                                             1, kernel_size=kernel_size, strides=1, activation=tf.nn.sigmoid, name="spatial_attn")

    def __call__(self, inputs, *args):
        avg_out = tf.reduce_mean(inputs, axis=self.axis, name="SpatialAttention_mean")  # [256, 32, 32, 64] -> [256, 32, 32]
        max_out = tf.reduce_max(inputs, axis=self.axis, name="SpatialAttention_max")  # [256, 32, 32, 64] -> [256, 32, 32]
        out = tf.stack([avg_out, max_out], axis=self.axis, name="SpatialAttention_stack")             # concat。 -> [256, 32, 32, 2]
        out = self.conv1(out)  # -> [256, 32, 32, 1]

        return out


class AttentionLSTM(Layer):
    """
    This layer combines Self Attention [7] mechanism with LSTM. It uses one separate
    LSTM+SelfAttention block for each input feature. The output from each
    LSTM+SelfAttention block is concatenated and returned. The layer expects
    same input dimension as by LSTM i.e. (batch_size, time_steps, input_features).
    For usage see example [8]

    References
    ----------
    .. [7] https://ai4water.readthedocs.io/en/dev/models/layers.html#selfattention

    .. [8] https://ai4water.readthedocs.io/en/dev/auto_examples/attention_lstm.html#
    """
    def __init__(
            self,
            num_inputs: int,
            lstm_units: int,
            attn_units: int = 128,
            attn_activation: str = "tanh",
            lstm_kwargs:dict = None,
            **kwargs
    ):
        """
        Parameters
        ----------
            num_inputs: int
                number of inputs
            lstm_units : int
                number of units in LSTM layers
            attn_units : int, optional (default=128)
                number of units in SelfAttention layers
            attn_activation : str, optional (default="tanh")
                activation function in SelfAttention layers
            lstm_kwargs : dict, optional (default=None)
                any keyword arguments for LSTM layer.

        Example
        -------
        >>> import numpy as np
        >>> from tensorflow.keras.models import Model
        >>> from tensorflow.keras.layers import Input, Dense
        >>> from ai4water.models.tensorflow import AttentionLSTM
        >>> seq_len = 20
        >>> n_inputs = 2
        >>> inp = Input(shape=(10, n_inputs))
        >>> outs = AttentionLSTM(n_inputs, 16)(inp)
        >>> outs = Dense(1)(outs)
        ...
        >>> model = Model(inputs=inp, outputs=outs)
        >>> model.compile(loss="mse")
        ...
        >>> print(model.summary())
        ... # define input
        >>> x = np.random.random((100, seq_len, num_inputs))
        >>> y = np.random.random((100, 1))
        >>> h = model.fit(x=x, y=y)

        # using with ai4water's Modle

        >>> from ai4water import Model
        >>> model = Model(
        ...    model = {"layers": {
        ...        "Input_1": {"shape": (seq_len, num_inputs)},
        ...        "AttentionLSTM": {"num_inputs": num_inputs, "lstm_units": 16},
        ...        "Dense": 1 }})
        >>> model.fit(x=x, y=y)

        """
        super(AttentionLSTM, self).__init__(**kwargs)
        self.num_inputs = num_inputs
        self.lstm_units = lstm_units
        self.attn_units = attn_units
        self.attn_activation = attn_activation

        if lstm_kwargs is None:
            lstm_kwargs = {}

        assert isinstance(lstm_kwargs, dict)
        self.lstm_kwargs = lstm_kwargs

        self.lstms = []
        self.sas = []
        for i in range(self.num_inputs):
            self.lstms.append(tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, **self.lstm_kwargs))
            self.sas.append(SelfAttention(self.attn_units, self.attn_activation))

    def __call__(self, inputs, *args, **kwargs):

        assert self.num_inputs == inputs.shape[-1], f"""
        num_inputs {self.num_inputs} does not match with input features.
        Inputs are of shape {inputs.shape}"""

        outs = []
        for i in range(inputs.shape[-1]):
            lstm = self.lstms[i](tf.expand_dims(inputs[..., i], axis=-1))
            out, _ = self.sas[i](lstm)
            outs.append(out)

        return tf.concat(outs, axis=-1)


class attn_layers(object):

    SelfAttention = SelfAttention
    SeqSelfAttention = SeqSelfAttention
    SnailAttention = SnailAttention
    SeqWeightedAttention = SeqWeightedAttention
    BahdanauAttention = BahdanauAttention
    HierarchicalAttention = HierarchicalAttention
    SpatialAttention = SpatialAttention
    ChannelAttention = ChannelAttention
    AttentionLSTM = AttentionLSTM