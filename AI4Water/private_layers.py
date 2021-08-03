import tensorflow as tf
from tensorflow.keras import layers

from ai4water.models.attention_layers import ChannelAttention, SpatialAttention, regularized_padded_conv

def _get_tensor_shape(t):
    return t.shape


class ConditionalRNN(tf.keras.layers.Layer):

    # Arguments to the RNN like return_sequences, return_state...
    def __init__(self, units,
                 activation = 'tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 dropout = 0.0,
                 recurrent_dropout = 0.0,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 cell=tf.keras.layers.LSTMCell, *args,
                 **kwargs):
        """
        Conditional RNN. Conditions time series on categorical data.
        :param units: int, The number of units in the RNN Cell
        :param cell: string, cell class or object (pre-instantiated). In the case of string, 'GRU',
        'LSTM' and 'RNN' are supported.
        :param args: Any parameters of the tf.keras.layers.RNN class, such as return_sequences,
        return_state, stateful, unroll...
        """
        super().__init__()
        self.units = units
        self.final_states = None
        self.init_state = None
        if isinstance(cell, str):
            if cell.upper() == 'GRU':
                cell = tf.keras.layers.GRUCell
            elif cell.upper() == 'LSTM':
                cell = tf.keras.layers.LSTMCell
            elif cell.upper() == 'RNN':
                cell = tf.keras.layers.SimpleRNNCell
            else:
                raise Exception('Only GRU, LSTM and RNN are supported as cells.')
        self._cell = cell if hasattr(cell, 'units') else cell(units=units,
                                                              activation=activation,
                                                              dropout=dropout,
                                                              recurrent_dropout=recurrent_dropout,
                                                              recurrent_activation=recurrent_activation,
                                                              kernel_initializer=kernel_regularizer,
                                                              recurrent_regularizer=recurrent_regularizer,
                                                              use_bias=use_bias
                                                              )
        self.rnn = tf.keras.layers.RNN(cell=self._cell, *args, **kwargs)

        # single cond
        self.cond_to_init_state_dense_1 = tf.keras.layers.Dense(units=self.units)

        # multi cond
        max_num_conditions = 10
        self.multi_cond_to_init_state_dense = []
        for i in range(max_num_conditions):
            self.multi_cond_to_init_state_dense.append(tf.keras.layers.Dense(units=self.units))
        self.multi_cond_p = tf.keras.layers.Dense(1, activation=None, use_bias=True)

    def _standardize_condition(self, initial_cond):
        initial_cond_shape = initial_cond.shape
        if len(initial_cond_shape) == 2:
            initial_cond = tf.expand_dims(initial_cond, axis=0)
        first_cond_dim = initial_cond.shape[0]
        if isinstance(self._cell, tf.keras.layers.LSTMCell):
            if first_cond_dim == 1:
                initial_cond = tf.tile(initial_cond, [2, 1, 1])
            elif first_cond_dim != 2:
                raise Exception('Initial cond should have shape: [2, batch_size, hidden_size] '
                                'or [batch_size, hidden_size]. Shapes do not match.', initial_cond_shape)
        elif isinstance(self._cell, tf.keras.layers.GRUCell) or isinstance(self._cell, tf.keras.layers.SimpleRNNCell):
            if first_cond_dim != 1:
                raise Exception('Initial cond should have shape: [1, batch_size, hidden_size] '
                                'or [batch_size, hidden_size]. Shapes do not match.', initial_cond_shape)
        else:
            raise Exception('Only GRU, LSTM and RNN are supported as cells.')
        return initial_cond

    def __call__(self, inputs, *args, **kwargs):
        """
        :param inputs: List of n elements:
                    - [0] 3-D Tensor with shape [batch_size, time_steps, input_dim]. The inputs.
                    - [1:] list of tensors with shape [batch_size, cond_dim]. The conditions.
        In the case of a list, the tensors can have a different cond_dim.
        :return: outputs, states or outputs (if return_state=False)
        """
        assert (isinstance(inputs, list) or isinstance(inputs, tuple)) and len(inputs) >= 2
        x = inputs[0]
        cond = inputs[1:]
        if len(cond) > 1:  # multiple conditions.
            init_state_list = []
            for ii, c in enumerate(cond):
                init_state_list.append(self.multi_cond_to_init_state_dense[ii](self._standardize_condition(c)))
            multi_cond_state = self.multi_cond_p(tf.stack(init_state_list, axis=-1))
            multi_cond_state = tf.squeeze(multi_cond_state, axis=-1)
            self.init_state = tf.unstack(multi_cond_state, axis=0)
        else:
            cond = self._standardize_condition(cond[0])
            if cond is not None:
                self.init_state = self.cond_to_init_state_dense_1(cond)
                self.init_state = tf.unstack(self.init_state, axis=0)
        out = self.rnn(x, initial_state=self.init_state, *args, **kwargs)
        if self.rnn.return_state:
            outputs, h, c = out
            final_states = tf.stack([h, c])
            return outputs, final_states
        else:
            return out


class BasicBlock(layers.Layer):
    """
    The official implementation is at https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py
    The implementation of [1] does not have two conv and bn paris. They just applied channel attention followed by
    spatial attention on inputs.

    [1] https://github.com/kobiso/CBAM-tensorflow/blob/master/attention_module.py#L39
    """
    expansion = 1

    def __init__(self, conv_dim, out_channels=32, stride=1, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)

        # 1. BasicBlock模块中的共有2个卷积;BasicBlock模块中的第1个卷积层;
        self.conv1 = regularized_padded_conv(conv_dim, out_channels, kernel_size=3, strides=stride)
        self.bn1 = layers.BatchNormalization()

        # 2. 第2个；第1个卷积如果做stride就会有一个下采样，在这个里面就不做下采样了。这一块始终保持size一致，把stride固定为1
        self.conv2 = regularized_padded_conv(conv_dim, out_channels, kernel_size=3, strides=1)
        self.bn2 = layers.BatchNormalization()
        ############################### 注意力机制 ###############################
        self.ca = ChannelAttention(conv_dim=conv_dim, in_planes=out_channels)
        self.sa = SpatialAttention(conv_dim=conv_dim)

        # # 3. 判断stride是否等于1,如果为1就是没有降采样。
        # if stride != 1 or in_channels != self.expansion * out_channels:
        #     self.shortcut = Sequential([regularized_padded_conv(self.expansion * out_channels,
        #                                                         kernel_size=1, strides=stride),
        #                                 layers.BatchNormalization()])
        # else:
        #     self.shortcut = lambda x, _: x

    def call(self, inputs, training=False):
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        ############################### 注意力机制 ###############################
        out = self.ca(out) * out
        out = self.sa(out) * out

        # out = out + self.shortcut(inputs, training)
        # out = tf.nn.relu(out)

        return out


class scaled_dot_product_attention(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, q, k, v, mask):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
        output, attention_weights
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

            # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1, name='scaled_dot_prod_attn_weights')  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v, name='scaled_dot_prod_attn_outs')  # (..., seq_len_q, depth_v)

        return output, attention_weights

MHW_COUNTER = 0
ENC_COUNTER = 0

class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        global MHW_COUNTER
        MHW_COUNTER +=1

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, name=f"wq_{MHW_COUNTER}")
        self.wk = tf.keras.layers.Dense(d_model, name=f"wk_{MHW_COUNTER}")
        self.wv = tf.keras.layers.Dense(d_model, name=f"wv_{MHW_COUNTER}")

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def __call__(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            )(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='swish', name='swished_dense'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model, name='ffn_output')  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        global MHW_COUNTER
        MHW_COUNTER +=1

        self.mha = MultiHeadAttention(d_model, num_heads)
        #self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.swished_dense = layers.Dense(dff, activation='swish', name=f'swished_dense_{MHW_COUNTER}')
        self.ffn_output = layers.Dense(d_model, name=f'ffn_output_{MHW_COUNTER}')

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def __call__(self, x, training=True, mask=None):
        attn_output, attn_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        #ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)

        temp = self.swished_dense(out1)
        ffn_output = self.ffn_output(temp)

        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attn_weights


class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding,
                 rate=0.1, return_weights=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate
        self.return_weights=return_weights

        #         self.pos_encoding = positional_encoding(self.maximum_position_encoding,
        #                                                 self.d_model)
        #         self.embedding = tf.keras.layers.Dense(self.d_model)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=self.maximum_position_encoding,
                                                 output_dim=self.d_model)

        self.enc_layers = [EncoderLayer(self.d_model, self.num_heads, self.dff, self.rate)
                           for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(self.rate)

        self.attn_weights = None

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'maximum_position_encoding': self.maximum_position_encoding,
            'dropout': self.dropout,
        })
        return config

    #def call(self, x, training, mask=None):
    def __call__(self, x, training=True, mask=None, *args, **kwargs):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        #         x += self.pos_encoding[:, :seq_len, :]
        #         x = self.embedding(x)
        positions = tf.range(start=0, limit=seq_len, delta=1)
        x += self.pos_emb(positions)

        x = self.dropout(x, training=training)
        self.attn_weights = []

        for i in range(self.num_layers):
            x, _w = self.enc_layers[i](x, training, mask)
            self.attn_weights.append(_w)

        if self.return_weights:
            return x, self.attn_weights

        return x  # (batch_size, input_seq_len, d_model)


class PrivateLayers(object):

    class layers:

        TransformerEncoder = TransformerEncoder
        BasicBlock = BasicBlock
        CONDRNN = ConditionalRNN