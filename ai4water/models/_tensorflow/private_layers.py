
from typing import List

from ai4water.backend import tf

layers = tf.keras.layers
Dense = tf.keras.layers.Dense
Layer = tf.keras.layers.Layer
activations = tf.keras.activations
K = tf.keras.backend
constraints = tf.keras.constraints
initializers = tf.keras.initializers
regularizers = tf.keras.regularizers

from tensorflow.python.ops import array_ops

from .attention_layers import ChannelAttention, SpatialAttention, regularized_padded_conv


def _get_tensor_shape(t):
    return t.shape


class ConditionalRNN(tf.keras.layers.Layer):

    # Arguments to the RNN like return_sequences, return_state...
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 dropout=0.0,
                 recurrent_dropout=0.0,
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

        for _ in range(max_num_conditions):
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
        assert (isinstance(inputs, list) or isinstance(inputs, tuple)) and len(inputs) >= 2, f"{type(inputs)}"
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
        # ############################## 注意力机制 ###############################
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
        # ############################## 注意力机制 ###############################
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
        MHW_COUNTER += 1

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
        MHW_COUNTER += 1

        self.mha = MultiHeadAttention(d_model, num_heads)
        # self.ffn = point_wise_feed_forward_network(d_model, dff)

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

        # ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)

        temp = self.swished_dense(out1)
        ffn_output = self.ffn_output(temp)

        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attn_weights


class TransformerBlocks(tf.keras.layers.Layer):
    """
    This layer stacks Transformers on top of each other.

    Example
    -------
    >>> import numpy as np
    >>> from tensorflow.keras.models import Model
    >>> from tensorflow.keras.layers import Input
    >>> inp = Input(shape=(10, 32))
    >>> out = TransformerBlocks(4, 4, 32)(inp)
    >>> out = Dense(1)(out)
    >>> model = Model(inputs=inp, outputs=out)
    >>> model.compile(optimizer="Adam", loss="mse")
    >>> x = np.random.random((100, 10, 32))
    >>> y = np.random.random(100)
    >>> h = model.fit(x,y)
    """
    def __init__(
            self,
            num_blocks:int,
            num_heads:int,
            embed_dim:int,
            name:str = "TransformerBlocks",
            **kwargs
    ):
        """
        Parameters
        -----------
        num_blocks : int
        num_heads : int
        embed_dim :
        **kwargs :
            additional keyword arguments for :class:`ai4water.models.tensorflow.Transformer`
        """

        super(TransformerBlocks, self).__init__(name=name)
        self.num_blocks = num_blocks
        self.num_heads = num_heads

        self.blocks = []
        for n in range(num_blocks):
            self.blocks.append(Transformer(num_heads, embed_dim, **kwargs))

    def __call__(self, inputs, *args, **kwargs):

        attn_weights_list = []
        for transformer in self.blocks:
            inputs, attn_weights = transformer(inputs)
            attn_weights_list.append(tf.reduce_sum(attn_weights[:, :, 0, :]))

        importances = tf.reduce_sum(tf.stack(attn_weights_list), axis=0) / (
                self.num_blocks * self.num_heads)

        return inputs, importances


class Transformer(tf.keras.layers.Layer):
    """
    A basic transformer block consisting of
    LayerNormalization -> Add -> MultiheadAttention -> MLP ->

    Example
    -------
    >>> import numpy as np
    >>> from tensorflow.keras.models import Model
    >>> from tensorflow.keras.layers import Input
    >>> inp = Input(shape=(10, 32))
    >>> out = Transformer(4, 32)(inp)
    >>> out = Dense(1)(out)
    >>> model = Model(inputs=inp, outputs=out)
    >>> model.compile(optimizer="Adam", loss="mse")
    >>> x = np.random.random((100, 10, 32))
    >>> y = np.random.random(100)
    >>> h = model.fit(x,y)
    """
    def __init__(
            self,
            num_heads:int = 4,
            embed_dim:int=32,
            dropout=0.1,
            post_norm:bool = True,
            prenorm_mlp:bool = False,
            num_dense_lyrs:int = 1,
            seed:int = 313,
            *args,
            **kwargs
    ):
        """

        Parameters
        -----------
        num_heads : int
            number of attention heads
        embed_dim : int
            embedding dimension. This value is also used for units/neurons in MLP blocl
        dropout : float
            dropout rate in MLP blocl
        post_norm : bool (default=True)
            whether to apply LayerNormalization on the outputs or not.
        prenorm_mlp : bool
            whether to apply LayerNormalization on inputs of MLP or not
        num_dense_lyrs : int
            number of Dense layers in MLP block.
        """
        super(Transformer, self).__init__(*args, **kwargs)

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.post_norm = post_norm
        self.prenorm_mlp = prenorm_mlp
        self.seed = seed

        assert num_dense_lyrs <= 2
        self.num_dense_lyrs = num_dense_lyrs

        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim,
            dropout=dropout
        )
        self.skip1 = tf.keras.layers.Add()
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = self._make_mlp()

        self.skip2 = tf.keras.layers.Add()
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def _make_mlp(self):
        lyrs = []

        if self.prenorm_mlp:
            lyrs += [tf.keras.layers.LayerNormalization(epsilon=1e-6)]


        lyrs += [
            Dense(self.embed_dim, activation=tf.keras.activations.gelu),
            tf.keras.layers.Dropout(self.dropout, seed=self.seed),
        ]

        if self.num_dense_lyrs>1:
            lyrs += [tf.keras.layers.Dense(self.embed_dim)]

        return tf.keras.Sequential(lyrs)

    def __call__(self, inputs, *args, **kwargs):

        inputs = self.layernorm1(inputs)
        attention_output, att_weights = self.att(
            inputs, inputs, return_attention_scores=True
        )

        attention_output = self.skip1([inputs, attention_output])
        feedforward_output = self.ffn(attention_output)
        outputs = self.skip2([feedforward_output, attention_output])

        if self.post_norm:
            return self.layernorm2(outputs), att_weights

        return outputs, att_weights


class NumericalEmbeddings(layers.Layer):

    def __init__(
            self,
            num_features,
            emb_dim,
            *args,
            **kwargs
    ):

        self.num_features = num_features
        self.emb_dim = emb_dim
        super(NumericalEmbeddings, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        # features, n_bins, emb_dim
        self.linear_w = tf.Variable(
            initial_value=w_init(
                shape=(self.num_features, 1, self.emb_dim), dtype='float32'
            ), trainable=True, name="NumEmbeddingWeights")

        # features, n_bins, emb_dim
        self.linear_b = tf.Variable(
            w_init(
                shape=(self.num_features, 1), dtype='float32'
            ), trainable=True, name="NumEmbeddingBias")
        return

    def call(self, X, *args, **kwargs):
        embs = tf.einsum('f n e, b f -> bfe', self.linear_w, X)
        embs = tf.nn.relu(embs + self.linear_b)
        return embs


class CatEmbeddings(layers.Layer):
    """
    The layer to encode categorical features.

    Parameters
    -----------
    vocabulary : dict
    embed_dim : int
        dimention of embedding for each categorical feature
    lookup_kws : dict
        keyword arguments that will go to StringLookup layer

    """
    def __init__(
            self,
            vocabulary:dict,
            embed_dim:int = 32,
            lookup_kws:dict = None,
            *args,
            **kwargs
    ):
        super(CatEmbeddings, self).__init__(*args, **kwargs)

        self.embed_dim = embed_dim
        self.lookups = {}
        self.embedding_lyrs = {}
        self.feature_names = []

        _lookup_kws = dict(mask_token=None,
                num_oov_indices=0,
                output_mode="int")

        if lookup_kws is not None:
            _lookup_kws.update(lookup_kws)

        for feature_name, vocab in vocabulary.items():

            lookup = layers.StringLookup(
                vocabulary=vocab,
                **_lookup_kws
            )

            self.lookups[feature_name] = lookup

            embedding = layers.Embedding(
                input_dim=len(vocab), output_dim=embed_dim
            )

            self.embedding_lyrs[feature_name] = embedding

            self.feature_names.append(feature_name)

    def call(self, inputs, *args, **kwargs):
        """
        The tensors in `inputs` list must be in same
        order as in the `vocabulary` dictionary.

        Parameters
        -------------
        inputs : list
            a list of tensors of shape (None,)

        Returns
        -------
            a tensor of shape (None, num_cat_features, embed_dim)
        """

        encoded_features = []
        for idx, feat_name in enumerate(self.feature_names):
            feat_input = inputs[:, idx]
            lookup = self.lookups[feat_name]
            encoded_feature = lookup(feat_input)

            embedding = self.embedding_lyrs[feat_name]
            encoded_categorical_feature = embedding(encoded_feature)
            encoded_features.append(encoded_categorical_feature)

        cat_embeddings = tf.stack(encoded_features, axis=1)

        return cat_embeddings


class TabTransformer(layers.Layer):
    """
    tensorflow/keras layer which implements logic of TabTransformer model.

    It is available only in tensorflow >= 2.6
    """
    def __init__(
            self,
            num_numeric_features: int,
            cat_vocabulary: dict,
            hidden_units=32,
            lookup_kws:dict=None,
            num_heads: int = 4,
            depth: int = 4,
            dropout: float = 0.1,
            num_dense_lyrs: int = 2,
            prenorm_mlp: bool = True,
            post_norm: bool = True,
            final_mlp_units = 16,
            final_mpl_activation:str = "selu",
            seed: int = 313,
            *args, **kwargs
    ):
        """
        Parameters
        ----------
        num_numeric_features : int
            number of numeric features to be used as input.
        cat_vocabulary : dict
            a dictionary whose keys are names of categorical features and values
            are lists which consist of unique values of categorical features.
            You can use the function :fun:`ai4water.models.utils.gen_cat_vocab` to create this for your
            own data. The length of dictionary should be equal to number of
            categorical features.
        hidden_units : int, optional (default=32)
            number of hidden units
        num_heads : int, optional (default=4)
            number of attention heads
        depth : int (default=4)
            number of transformer blocks to be stacked on top of each other
        dropout : int, optional (default=0.1)
            droput rate in transformer
        post_norm : bool (default=True)
        prenorm_mlp : bool (default=True)
        num_dense_lyrs : int (default=2)
            number of dense layers in MLP block
        final_mlp_units : int (default=16)
            number of units/neurons in final MLP layer i.e. the MLP layer
            after Transformer block
        """
        super(TabTransformer, self).__init__(*args, **kwargs)

        self.cat_vocabulary = cat_vocabulary
        self.num_numeric_inputs = num_numeric_features
        self.hidden_units = hidden_units
        self.lookup_kws = lookup_kws
        self.num_heads = num_heads
        self.depth = depth
        self.dropout = dropout
        self.final_mlp_units = final_mlp_units
        self.final_mpl_activation = final_mpl_activation
        self.seed = seed

        self.cat_embs = CatEmbeddings(
            vocabulary=cat_vocabulary,
            embed_dim=hidden_units,
            lookup_kws=lookup_kws
        )

        # layer normalization of numerical features
        self.lyr_norm = layers.LayerNormalization(epsilon=1e-6)

        self.transformers = TransformerBlocks(
            embed_dim=hidden_units,
            num_heads=num_heads,
            num_blocks=depth,
            num_dense_lyrs=num_dense_lyrs,
            post_norm=post_norm,
            prenorm_mlp=prenorm_mlp,
            dropout=dropout,
            seed=seed
        )

        self.flatten = layers.Flatten()

        self.concat = layers.Concatenate()

        self.mlp = self.create_mlp(
            activation=self.final_mpl_activation,
            normalization_layer=layers.BatchNormalization(),
            name="MLP",
        )

    # Implement an MLP block
    def create_mlp(
            self,
            activation,
            normalization_layer,
            name=None
    ):
        if isinstance(self.final_mlp_units, int):
            hidden_units = [self.final_mlp_units]
        else:
            assert isinstance(self.final_mlp_units, list)
            hidden_units = self.final_mlp_units

        mlp_layers = []
        for units in hidden_units:
            mlp_layers.append(normalization_layer),
            mlp_layers.append(layers.Dense(units, activation=activation))
            mlp_layers.append(layers.Dropout(self.dropout, seed=self.seed))

        return tf.keras.Sequential(mlp_layers, name=name)

    def __call__(self, inputs:list, *args, **kwargs):
        """
        inputs :
            list of 2. The first tensor is numerical inputs and second
            tensor is categorical inputs
        """
        num_inputs = inputs[0]
        cat_inputs = inputs[1]

        cat_embs = self.cat_embs(cat_inputs)
        transformer_outputs, imp = self.transformers(cat_embs)
        flat_transformer_outputs = self.flatten(transformer_outputs)

        num_embs = self.lyr_norm(num_inputs)

        x = self.concat([num_embs, flat_transformer_outputs])

        return self.mlp(x), imp


class FTTransformer(layers.Layer):
    """
    tensorflow/keras layer which implements logic of FTTransformer model.
    """
    def __init__(
            self,
            num_numeric_features: int,
            cat_vocabulary: dict,
            hidden_units=32,
            num_heads: int = 4,
            depth: int = 4,
            dropout: float = 0.1,
            lookup_kws:dict = None,
            num_dense_lyrs: int = 2,
            post_norm: bool = True,
            final_mlp_units: int = 16,
            with_cls_token:bool = False,
            seed: int = 313,
            *args,
            **kwargs
    ):
        """
        Parameters
        ----------
        num_numeric_features : int
            number of numeric features to be used as input.
        cat_vocabulary : dict
            a dictionary whose keys are names of categorical features and values
            are lists which consist of unique values of categorical features.
            You can use the function :fun:`ai4water.models.utils.gen_cat_vocab` to create this for your
            own data. The length of dictionary should be equal to number of
            categorical features.
        hidden_units : int, optional (default=32)
            number of hidden units
        num_heads : int, optional (default=4)
            number of attention heads
        depth : int (default=4)
            number of transformer blocks to be stacked on top of each other
        dropout : float, optional (default=0.1)
            droput rate in transformer
        lookup_kws : dict
        post_norm : bool (default=True)
        num_dense_lyrs : int (default=2)
            number of dense layers in MLP block
        final_mlp_units : int (default=16)
            number of units/neurons in final MLP layer i.e. the MLP layer
            after Transformer block
        with_cls_token : bool (default=False)
            whether to use cls token or not
        seed : int
            seed for reproducibility
        """
        super(FTTransformer, self).__init__(*args, **kwargs)

        self.cat_vocabulary = cat_vocabulary
        self.num_numeric_inputs = num_numeric_features
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.depth = depth
        self.dropout = dropout
        self.final_mlp_units = final_mlp_units
        self.with_cls_token = with_cls_token
        self.seed = seed

        self.cat_embs = CatEmbeddings(
            vocabulary=cat_vocabulary,
            embed_dim=hidden_units,
            lookup_kws=lookup_kws
        )

        self.num_embs = NumericalEmbeddings(
            num_features=num_numeric_features,
            emb_dim=hidden_units
        )

        self.concat =  layers.Concatenate(axis=1)

        self.transformers = TransformerBlocks(
            embed_dim=hidden_units,
            num_heads=num_heads,
            num_blocks=depth,
            num_dense_lyrs=num_dense_lyrs,
            post_norm=post_norm,
            dropout=dropout,
            seed=seed
        )

        self.lmbda = tf.keras.layers.Lambda(lambda x: x[:, 0, :])

        self.lyr_norm = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = layers.Dense(final_mlp_units)

    def build(self, input_shape):
        if self.with_cls_token:
            # CLS token
            w_init = tf.random_normal_initializer()
            self.cls_weights = tf.Variable(
                initial_value=w_init(shape=(1, self.hidden_units), dtype="float32"),
                trainable=True,
            )
        return

    def __call__(self, inputs:list, *args, **kwargs):
        """
        inputs :
            list of 2. The first tensor is numerical inputs and second
            tensor is categorical inputs
        """

        num_inputs = inputs[0]
        cat_inputs = inputs[1]

        # cls_tokens = tf.repeat(self.cls_weights, repeats=tf.shape(inputs[self.numerical[0]])[0], axis=0)
        # cls_tokens = tf.expand_dims(cls_tokens, axis=1)

        num_embs = self.num_embs(num_inputs)
        cat_embs = self.cat_embs(cat_inputs)

        embs = self.concat([num_embs, cat_embs])

        x, imp = self.transformers(embs)

        x = self.lmbda(x)

        x = self.lyr_norm(x)

        return self.mlp(x), imp


class Conditionalize(tf.keras.layers.Layer):
    """Mimics the behaviour of cond_rnn of Philipperemy but puts the logic
    of condition in a separate layer so that it becomes easier to use it.

    Example
    --------
    >>> from ai4water.models._tensorflow import Conditionalize
    >>> from tensorflow.keras.layers import Input, LSTM
    >>> i = Input(shape=(10, 3))
    >>> raw_conditions = Input(shape=(14,))
    >>> processed_conds = Conditionalize(32)([raw_conditions, raw_conditions, raw_conditions])
    >>> rnn = LSTM(32)(i, initial_state=[processed_conds, processed_conds])
    """
    def __init__(self, units,
                 max_num_cond=10,
                 use_bias:bool = True,
                 **kwargs):
        self.units = units
        super().__init__(**kwargs)

        # single cond
        self.cond_to_init_state_dense_1 = tf.keras.layers.Dense(units=self.units,
                                                                use_bias=use_bias,
                                                                name="conditional_dense")

        # multi cond
        self.multi_cond_to_init_state_dense = []

        for i in range(max_num_cond):
            self.multi_cond_to_init_state_dense.append(tf.keras.layers.Dense(
                units=self.units,
                use_bias=use_bias,
                name=f"conditional_dense{i}"))

        self.multi_cond_p = tf.keras.layers.Dense(1, activation=None, use_bias=True, name="conditional_dense_out")

    @staticmethod
    def _standardize_condition(initial_cond):

        assert len(initial_cond.shape) == 2

        return initial_cond

    def __call__(self, inputs, *args, **kwargs):

        if args or kwargs:
            raise ValueError(f"Unrecognized input arguments\n args: {args} \nkwargs: {kwargs}")

        if inputs.__class__.__name__ == "Tensor":
            inputs = [inputs]

        assert (isinstance(inputs, list) or isinstance(inputs, tuple)) and len(inputs) >= 1, f"{type(inputs)}"

        cond = inputs
        if len(cond) > 1:  # multiple conditions.
            init_state_list = []
            for idx, c in enumerate(cond):
                init_state_list.append(self.multi_cond_to_init_state_dense[idx](self._standardize_condition(c)))

            multi_cond_state = tf.stack(init_state_list, axis=-1)  # -> (?, units, num_conds)
            multi_cond_state = self.multi_cond_p(multi_cond_state)  # -> (?, units, 1)
            cond_state = tf.squeeze(multi_cond_state, axis=-1)  # -> (?, units)
        else:

            cond = self._standardize_condition(cond[0])
            cond_state = self.cond_to_init_state_dense_1(cond)    # -> (?, units)

        return cond_state


class _NormalizedGate(Layer):

    _Normalizers = {
        'relu': tf.nn.relu,
        'sigmoid': tf.nn.sigmoid
    }

    def __init__(self, in_features, out_shape, normalizer="relu"):

        super(_NormalizedGate, self).__init__()

        self.in_features = in_features
        self.out_shape = out_shape
        self.normalizer = self._Normalizers[normalizer]

        self.fc = Dense(out_shape[0]*out_shape[1],
                        use_bias=True,
                        kernel_initializer="Orthogonal",
                        bias_initializer="zeros")

    def call(self, inputs):

        h = self.fc(inputs)
        h = tf.reshape(h, (-1, *self.out_shape))
        h =  self.normalizer(h)
        normalized, _ = tf.linalg.normalize(h, axis=-1)

        return normalized


class _MCLSTMCell(Layer):
    """

    Examples
    --------
        m_inp = tf.range(50, dtype=tf.float32)
        m_inp = tf.reshape(m_inp, (5, 10, 1))
        aux_inp = tf.range(150, dtype=tf.float32)
        aux_inp = tf.reshape(aux_inp, (5, 10, 3))
        cell = _MCLSTMCell(1, 3, 8)
        m_out_, ct_ = cell(m_inp, aux_inp)
    """
    def __init__(
            self,
            mass_input_size,
            aux_input_size,
            units,
            time_major:bool = False,
    ):
        super(_MCLSTMCell, self).__init__()

        self.units = units
        self.time_major = time_major

        gate_inputs = aux_input_size + self.units + mass_input_size

        self.output_gate = Dense(self.units,
                                 activation="sigmoid",
                                 kernel_initializer="Orthogonal",
                                 bias_initializer="zeros",
                                 name="sigmoid_gate")

        self.input_gate = _NormalizedGate(gate_inputs,
                                          (mass_input_size, self.units),
                                          "sigmoid")

        self.redistribution = _NormalizedGate(gate_inputs,
                                              (self.units, self.units),
                                              "relu")

    def call(self, x_m, x_a, ct=None):

        if not self.time_major:
            # (batch_size, lookback, input_features) -> (lookback, batch_size, input_features)
            x_m = tf.transpose(x_m, [1, 0, 2])
            x_a = tf.transpose(x_a, [1, 0, 2])

        lookback_steps, batch_size, _ = x_m.shape

        if ct is None:
            ct = tf.zeros((batch_size, self.units))

        m_out, c = [], []

        for time_step in range(lookback_steps):
            mt_out, ct = self._step(x_m[time_step], x_a[time_step], ct)

            m_out.append(mt_out)
            c.append(ct)

        m_out, c = tf.stack(m_out), tf.stack(c)  # (lookback, batch_size, units)

        return m_out, c

    def _step(self, xt_m, xt_a, c):

        features = tf.concat([xt_m, xt_a, c / (tf.norm(c) + 1e-5)], axis=-1)  # (examples, ?)

        # compute gate activations
        i = self.input_gate(features)  # (examples, 1, units)
        r = self.redistribution(features)  # (examples, units, units)
        o = self.output_gate(features)  # (examples, units)

        m_in = tf.squeeze(tf.matmul(tf.expand_dims(xt_m, axis=-2), i), axis=-2)

        m_sys = tf.squeeze(tf.matmul(tf.expand_dims(c, axis=-2), r), axis=-2)

        m_new = m_in + m_sys

        return tf.multiply(o, m_new), tf.multiply(tf.subtract(1.0, o),  m_new)


class MCLSTM(Layer):
    """Mass-Conserving LSTM model from Hoedt et al. [1]_.

    This implementation follows of NeuralHydrology's implementation of MCLSTM
    with some changes:
    1) reduced sum is not performed for over the units
    2) time_major argument is added
    3) no implementation of Embedding

    Examples
    --------
    >>> inputs = tf.range(150, dtype=tf.float32)
    >>> inputs = tf.reshape(inputs, (10, 5, 3))
    >>> mc = MCLSTM(1, 2, 8, 1)
    >>> h = mc(inputs)  # (batch, units)
    ...
    >>> mc = MCLSTM(1, 2, 8, 1, return_sequences=True)
    >>> h = mc(inputs)  # (batch, lookback, units)
    ...
    >>> mc = MCLSTM(1, 2, 8, 1, return_state=True)
    >>> _h, _o, _c = mc(inputs)  # (batch, lookback, units)
    ...
    >>> mc = MCLSTM(1, 2, 8, 1, return_state=True, return_sequences=True)
    >>> _h, _o, _c = mc(inputs)  # (batch, lookback, units)
    ...
    ... # with time_major as True
    >>> inputs = tf.range(150, dtype=tf.float32)
    >>> inputs = tf.reshape(inputs, (5, 10, 3))
    >>> mc = MCLSTM(1, 2, 8, 1, time_major=True)
    >>> _h = mc(inputs)  # (batch, units)
    ...
    >>> mc = MCLSTM(1, 2, 8, 1, time_major=True, return_sequences=True)
    >>> _h = mc(inputs)  # (lookback, batch, units)
    ...
    >>> mc = MCLSTM(1, 2, 8, 1, time_major=True, return_state=True)
    >>> _h, _o, _c = mc(inputs)  # (batch, units), ..., (lookback, batch, units)
    ...
    ... # end to end keras Model
    >>> from tensorflow.keras.layers import Dense, Input
    >>> from tensorflow.keras.models import Model
    >>> import numpy as np
    ...
    >>> inp = Input(batch_shape=(32, 10, 3))
    >>> lstm = MCLSTM(1, 2, 8)(inp)
    >>> out = Dense(1)(lstm)
    ...
    >>> model = Model(inputs=inp, outputs=out)
    >>> model.compile(loss='mse')
    ...
    >>> x = np.random.random((320, 10, 3))
    >>> y = np.random.random((320, 1))
    >>> y = model.fit(x=x, y=y)

    References
    ----------
    .. [1] https://arxiv.org/abs/2101.05186
    """
    def __init__(
            self,
            num_mass_inputs,
            dynamic_inputs,
            units,
            num_targets=1,
            time_major:bool = False,
            return_sequences:bool = False,
            return_state:bool = False,
            name="MCLSTM",
            **kwargs
    ):
        """
        Parameters
        ----------
        num_targets : int
            number of inputs for which mass balance is to be reserved.
        dynamic_inputs :
            number of inpts other than mass_targets
        units :
            hidden size, determines the size of weight matrix
        time_major : bool, optional (default=True)
            if True, the data is expected to be of shape (lookback, batch_size, input_features)
            otherwise, data is expected of shape (batch_size, lookback, input_features)
        """
        super(MCLSTM, self).__init__(name=name, **kwargs)

        assert num_mass_inputs ==1
        assert units>1
        assert num_targets==1

        self.n_mass_inputs = num_mass_inputs
        self.units = units
        self.n_aux_inputs = dynamic_inputs
        self.time_major = time_major
        self.return_sequences = return_sequences
        self.return_state = return_state

        self.mclstm = _MCLSTMCell(
            self.n_mass_inputs,
            self.n_aux_inputs,
            self.units,
            self.time_major,
        )

    def call(self, inputs):

        x_m = inputs[:, :, :self.n_mass_inputs]  # (batch, lookback, 1)
        x_a = inputs[:, :, self.n_mass_inputs:]  # (batch, lookback, dynamic_inputs)

        output, c = self.mclstm(x_m, x_a)  # (lookback, batch, units)

        # unlike NeuralHydrology, we don't preform reduced sum over units
        # to keep with the convention in keras/lstm
        #output = tf.math.reduce_sum(output[:, :, 1:], axis=-1, keepdims=True)

        if self.time_major:
            h, m_out, c = output, output, c
            if not self.return_sequences:
                h = h[-1]
        else:
            h = tf.transpose(output, [1, 0, 2])   # -> (batch_size, lookback, 1)
            #m_out = tf.transpose(output, [1, 0, 2])  # -> (batch_size, lookback, 1)
            c = tf.transpose(c, [1, 0, 2])  # -> (batch_size, lookback, units)

            if not self.return_sequences:
                h = h[:, -1]

        if self.return_state:
            return h, h, c

        return h


class EALSTM(Layer):
    """Entity Aware LSTM as proposed by Kratzert et al., 2019 [1]_

    The difference here is that a Dense layer is not applied on cell state as done in
    original implementation in NeuralHydrology [2]_. This is left to user's discretion.

    Examples
    --------
    >>> from ai4water.models._tensorflow import EALSTM
    >>> import tensorflow as tf
    >>> batch_size, lookback, num_dyn_inputs, num_static_inputs, units = 10, 5, 3, 2, 8
    >>> inputs = tf.range(batch_size*lookback*num_dyn_inputs, dtype=tf.float32)
    >>> inputs = tf.reshape(inputs, (batch_size, lookback, num_dyn_inputs))
    >>> stat_inputs = tf.range(batch_size*num_static_inputs, dtype=tf.float32)
    >>> stat_inputs = tf.reshape(stat_inputs, (batch_size, num_static_inputs))
    >>> lstm = EALSTM(units, num_static_inputs)
    >>> h_n = lstm(inputs, stat_inputs)  # -> (batch_size, units)
    ...
    ... # with return sequences
    >>> lstm = EALSTM(units, num_static_inputs, return_sequences=True)
    >>> h_n = lstm(inputs, stat_inputs)  # -> (batch, lookback, units)
    ...
    ... # with return sequences and return_state
    >>> lstm = EALSTM(units, num_static_inputs, return_sequences=True)
    >>> h_n, [c_n, y_hat] = lstm(inputs, stat_inputs)  # -> (batch, lookback, units), [(), ()]
    ...
    ... # end to end Keras model
    >>> from tensorflow.keras.models import Model
    >>> from tensorflow.keras.layers import Input, Dense
    >>> import numpy as np
    >>> inp_dyn = Input(batch_shape=(batch_size, lookback, num_dyn_inputs))
    >>> inp_static = Input(batch_shape=(batch_size, num_static_inputs))
    >>> lstm = EALSTM(units, num_static_inputs)(inp_dyn, inp_static)
    >>> out = Dense(1)(lstm)
    >>> model = Model(inputs=[inp_dyn, inp_static], outputs=out)
    >>> model.compile(loss='mse')
    >>> print(model.summary())
    ... # generate hypothetical data and train it
    >>> dyn_x = np.random.random((100, lookback, num_dyn_inputs))
    >>> static_x = np.random.random((100, num_static_inputs))
    >>> y = np.random.random((100, 1))
    >>> h = model.fit(x=[dyn_x, static_x], y=y, batch_size=batch_size)

    References
    ----------
    .. [1] https://doi.org/10.5194/hess-23-5089-2019

    .. [2] https://github.com/neuralhydrology/neuralhydrology
    """

    def __init__(
            self,
            units:int,
            num_static_inputs:int,
            use_bias:bool=True,

            activation = "tanh",
            recurrent_activation="sigmoid",
            static_activation="sigmoid",

            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            static_initializer = "glorot_uniform",

            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            static_constraint=None,

            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            static_regularizer=None,

            return_state=False,
            return_sequences=False,
            time_major=False,
            **kwargs
    ):
        """

        Parameters
        ----------
            units : int
                number of units
            num_static_inputs : int
                number of static features
            static_activation :
                activation function for static input gate
            static_regularizer :
            static_constraint :
            static_initializer :
        """

        super(EALSTM, self).__init__(**kwargs)

        self.units = units
        self.num_static_inputs = num_static_inputs

        self.activation = activations.get(activation)
        self.rec_activation = activations.get(recurrent_activation)
        self.static_activation = static_activation

        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.static_initializer = initializers.get(static_initializer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.static_constraint = static_constraint

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.static_regularizer = static_regularizer

        self.return_state = return_state
        self.return_sequences = return_sequences
        self.time_major=time_major

        self.input_gate = Dense(units,
                                use_bias=self.use_bias,
                                kernel_initializer=self.static_initializer,
                                bias_initializer=self.bias_initializer,
                                activation=self.static_activation,
                                kernel_constraint=self.static_constraint,
                                bias_constraint=self.bias_constraint,
                                kernel_regularizer=self.static_regularizer,
                                bias_regularizer=self.bias_regularizer,
                                name="input_gate")

    def call(self, inputs, static_inputs, initial_state=None, **kwargs):
        """
        static_inputs :
            of shape (batch, num_static_inputs)
        """
        if not self.time_major:
            inputs = tf.transpose(inputs, [1, 0, 2])

        lookback, batch_size, _ = inputs.shape

        if initial_state is None:
            initial_state = tf.zeros((batch_size, self.units))  # todo
            state = [initial_state, initial_state]
        else:
            state = initial_state

        # calculate input gate only once because inputs are static
        inp_g = self.input_gate(static_inputs)  # (batch, num_static_inputs) -> (batch, units)

        outputs, states = [], []
        for time_step in range(lookback):

            _out, state = self.cell(inputs[time_step], inp_g, state)

            outputs.append(_out)
            states.append(state)

        outputs = tf.stack(outputs)
        h_s = tf.stack([states[i][0] for i in range(lookback)])
        c_s = tf.stack([states[i][1] for i in range(lookback)])

        if not self.time_major:
            outputs = tf.transpose(outputs, [1, 0, 2])
            h_s = tf.transpose(h_s, [1, 0, 2])
            c_s = tf.transpose(c_s, [1, 0, 2])
            states = [h_s, c_s]
            last_output = outputs[:, -1]
        else:
            states = [h_s, c_s]
            last_output = outputs[-1]

        h = last_output

        if self.return_sequences:
            h = outputs

        if self.return_state:
            return h, states

        return h

    def cell(self, inputs, i, states):

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        k_f, k_c, k_o = array_ops.split(self.kernel, num_or_size_splits=3, axis=1)

        x_f = K.dot(inputs, k_f)
        x_c = K.dot(inputs, k_c)
        x_o = K.dot(inputs, k_o)

        if self.use_bias:
            b_f, b_c, b_o = array_ops.split(
                self.bias, num_or_size_splits=3, axis=0)

            x_f = K.bias_add(x_f, b_f)
            x_c = K.bias_add(x_c, b_c)
            x_o = K.bias_add(x_o, b_o)

        # forget gate
        f = self.rec_activation(x_f + K.dot(h_tm1, self.rec_kernel[:, :self.units]))
        # cell state
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1, self.rec_kernel[:, self.units:self.units * 2]))
        # output gate
        o = self.rec_activation(x_o + K.dot(h_tm1, self.rec_kernel[:, self.units * 2:]))

        h = o * self.activation(c)

        return h, [h, c]

    def build(self, input_shape):
        """
        kernel, recurrent_kernel and bias are initiated for 3 gates instead
        of 4 gates as in original LSTM
        """
        input_dim = input_shape[-1]

        self.bias = self.add_weight(
            shape=(self.units * 3,),
            name='bias',
            initializer=self.bias_initializer,
            constraint=self.bias_constraint,
            regularizer=self.bias_regularizer
        )

        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 3),
            name='kernel',
            initializer=self.kernel_initializer,
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularizer
        )

        self.rec_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            constraint=self.recurrent_constraint,
            regularizer=self.recurrent_regularizer
        )

        self.built = True
        return


class PrivateLayers(object):

    class layers:

        BasicBlock = BasicBlock
        CONDRNN = ConditionalRNN
        Conditionalize = Conditionalize
        MCLSTM = MCLSTM
        EALSTM = EALSTM
        CatEmbeddings = CatEmbeddings
        TransformerBlocks = TransformerBlocks
        NumericalEmbeddings = NumericalEmbeddings
        TabTransformer = TabTransformer
        FTTransformer = FTTransformer
