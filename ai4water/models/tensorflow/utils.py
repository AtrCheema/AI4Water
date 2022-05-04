
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops import array_ops
import tensorflow.keras.backend as K
from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, Add, Activation, Lambda, Multiply

from ai4water.backend import tf

LayerNorm = tf.keras.layers.LayerNormalization


def concatenate(tensors, axis=-1, name="concat"):
  """Concatenates a list of tensors alongside the specified axis.
  Args:
      tensors: list of tensors to concatenate.
      axis: concatenation axis.
      name: str,
  Returns:
      A tensor.
  Example:
      >>>a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      >>>b = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
      >>>tf.keras.backend.concatenate((a, b), axis=-1)
      <tf.Tensor: shape=(3, 6), dtype=int32, numpy=
      array([[ 1,  2,  3, 10, 20, 30],
             [ 4,  5,  6, 40, 50, 60],
             [ 7,  8,  9, 70, 80, 90]], dtype=int32)>
  """
  if axis < 0:
    rank = K.ndim(tensors[0])
    if rank:
      axis %= rank
    else:
      axis = 0

  if all(K.is_sparse(x) for x in tensors):
    return sparse_ops.sparse_concat(axis, tensors, name=name)
  elif all(isinstance(x, ragged_tensor.RaggedTensor) for x in tensors):
    return array_ops.concat(tensors, axis, name=name)
  else:
    return array_ops.concat([K.to_dense(x) for x in tensors], axis, name=name)


# Layer utility functions.
def linear_layer(size,
                 activation=None,
                 use_time_distributed=False,
                 use_bias=True,
                 kernel_constraint=None,
                 name=None
                 ):
  """Returns simple Keras linear layer.

  Args:
    size: Output size
    activation: Activation function to apply if required
    use_time_distributed: Whether to apply layer across time
    use_bias: Whether bias should be included in layer
    kernel_constraint:
    name:
  """
  linear = tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias, name=name,
                                 kernel_constraint=kernel_constraint)
  if use_time_distributed:
    linear = TimeDistributed(linear, name=name)
  return linear


def apply_gating_layer(x,
                       hidden_layer_size,
                       dropout_rate=None,
                       use_time_distributed=False,
                       activation=None,
                       activation_kernel_constraint=None,
                       gating_kernel_constraint=None,
                       name=None):
  """Applies a Gated Linear Unit (GLU) to an input.

  Args:
    x: Input to gating layer having shape (num_examples, hidden_units)
    hidden_layer_size: Dimension of GLU
    dropout_rate: Dropout rate to apply if any
    use_time_distributed: Whether to apply across time
    activation: Activation function to apply to the linear feature transform if
      necessary
    name: name of encompassing layers
    activation_kernel_constraint:
    gating_kernel_constraint

  Returns:
    Tuple of tensors for: (GLU output, gate) where GLU output has the shape (num_examples, hidden_units)
  """

  if dropout_rate is not None:
    x = Dropout(dropout_rate, name=f'Dropout_{name}')(x)

  if use_time_distributed:
    activation_layer = TimeDistributed(Dense(hidden_layer_size,
                                             kernel_constraint=activation_kernel_constraint,
                                             activation=activation,
                                             name=f'gating_act_{name}'),
                                       name=f'gating_act_{name}')(x)

    gated_layer = TimeDistributed(
        Dense(hidden_layer_size,
              activation='sigmoid',
              kernel_constraint=gating_kernel_constraint,
              name=f'gating_{name}'),
    name=f'gating_{name}')(x)
  else:
    activation_layer = Dense(
        hidden_layer_size, activation=activation,
        kernel_constraint=activation_kernel_constraint,
        name=f'gating_act_{name}')(x)
    gated_layer = Dense(
        hidden_layer_size,
        activation='sigmoid',
        kernel_constraint=gating_kernel_constraint,
        name=f'gating_{name}')(
            x)

  return Multiply(name=f'MulGating_{name}')([activation_layer, gated_layer]), gated_layer


def add_and_norm(x_list, name=None, norm=True):
    """Applies skip connection followed by layer normalisation.

    Args:
    x_list: List of inputs to sum for skip connection
    name:

    Returns:
    Tensor output from layer.
    """
    tmp = Add(name=f'add_{name}')(x_list)
    if norm:
        tmp = LayerNorm(name=f'norm_{name}')(tmp)
    return tmp


def gated_residual_network(
        x,
        hidden_layer_size,
        output_size=None,
        dropout_rate=None,
        use_time_distributed=True,
        additional_context=None,
        return_gate=False,
        activation:str='elu',
        kernel_constraint1=None,
        kernel_constraint2=None,
        kernel_constraint3=None,
        gating_kernel_constraint1=None,
        gating_kernel_constraint2=None,
        norm=True,
        name='GRN'
):
  """Applies the gated residual network (GRN) as defined in paper.

  Args:
    x: Network inputs
    hidden_layer_size: Internal state size
    output_size: Size of output layer
    dropout_rate: Dropout rate if dropout is applied
    use_time_distributed: Whether to apply network across time dimension
    additional_context: Additional context vector to use if relevant
    return_gate: Whether to return GLU gate for diagnostic purposes
    name: name of all layers
    activation: the kind of activation function to use
    kernel_constraint1: kernel constranint to be applied on skip_connection layer
    kernel_constraint2: kernel constranint to be applied on 1st linear layer before activation
    kernel_constraint3: kernel constranint to be applied on 2nd linear layer after activation
    gating_kernel_constraint1: kernel constraint for activation in gating layer
    gating_kernel_constraint2: kernel constaint for gating layer

  Returns:
    Tuple of tensors for: (GRN output, GLU gate)
  """

  # Setup skip connection
  if output_size is None:
    output_size = hidden_layer_size
    skip = x
  else:
    linear = Dense(output_size, name=f'skip_connection_{name}', kernel_constraint=kernel_constraint1)
    if use_time_distributed:
      linear = TimeDistributed(linear, name=f'skip_connection_{name}')
    skip = linear(x)

  # Apply feedforward network
  hidden = linear_layer(
      hidden_layer_size,
      activation=None,
      use_time_distributed=use_time_distributed,
      kernel_constraint=kernel_constraint2,
      name=f"ff_{name}"
  )(x)

  if additional_context is not None:
    hidden = hidden + linear_layer(
        hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed,
        use_bias=False,
        name=f'addition_cntxt_{name}'
    )(additional_context)

  hidden = Activation(activation, name=f'{name}_{activation}')(hidden)
  hidden = linear_layer(
      hidden_layer_size,
      activation=None,
      kernel_constraint=kernel_constraint3,
      use_time_distributed=use_time_distributed,
      name=f'{name}_LastDense'
  )(
          hidden)

  gating_layer, gate = apply_gating_layer(
      hidden,
      output_size,
      dropout_rate=dropout_rate,
      use_time_distributed=use_time_distributed,
      activation=None,
      activation_kernel_constraint=gating_kernel_constraint1,
      gating_kernel_constraint=gating_kernel_constraint2,
      name=name
  )

  if return_gate:
    return add_and_norm([skip, gating_layer], name=name, norm=norm), gate
  else:
    return add_and_norm([skip, gating_layer], name=name, norm=norm)


# Attention Components.
def get_decoder_mask(self_attn_inputs):
    """Returns causal mask to apply for self-attention layer.

    Args:
    self_attn_inputs: Inputs to self attention layer to determine mask shape
    """
    len_s = tf.shape(self_attn_inputs)[1]
    bs = tf.shape(self_attn_inputs)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask


class ScaledDotProductAttention(tf.keras.layers.Layer):
  """Defines scaled dot product attention layer.

  Attributes:
    dropout: Dropout rate to use
    activation: Normalisation function for scaled dot product attention (e.g.
      softmax by default)
  """

  def __init__(self, attn_dropout=0.0, **kwargs):
      self.dropout = Dropout(attn_dropout, name="ScaledDotProdAtten_dropout")
      self.activation = Activation('softmax', name="ScaledDotProdAtten_softmax")
      super().__init__(**kwargs)

  def __call__(self, q, k, v, mask, idx):
    """Applies scaled dot product attention.

    Args:
      q: Queries
      k: Keys
      v: Values
      mask: Masking if required -- sets softmax to very large value

    Returns:
      Tuple of (layer outputs, attention weights)
    """
    temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
    attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / temper, name=f"ScaledDotProdAttenLambda{idx}")(
        [q, k])  # shape=(batch, q, k)
    if mask is not None:
      mmask = Lambda(lambda x: (-1e+9) * (1. - K.cast(x, 'float32')), name=f"ScaledDotProdAttenLambdaMask{idx}")(
          mask)  # setting to infinity
      attn = Add(name=f'SDPA_ADD_{idx}')([attn, mmask])
    attn = self.activation(attn)
    attn = self.dropout(attn)
    output = Lambda(lambda x: K.batch_dot(x[0], x[1]), name=f"ScaledDotProdAttenOutput{idx}")([attn, v])
    return output, attn


class InterpretableMultiHeadAttention(tf.keras.layers.Layer):
  """Defines interpretable multi-head attention layer.

  Attributes:
    n_head: Number of heads
    d_k: Key/query dimensionality per head
    d_v: Value dimensionality
    dropout: Dropout rate to apply
    qs_layers: List of queries across heads
    ks_layers: List of keys across heads
    vs_layers: List of values across heads
    attention: Scaled dot product attention layer
    w_o: Output weight matrix to project internal state to the original TFT
      state size
  """

  def __init__(self, n_head, d_model, dropout, **kwargs):
    """Initialises layer.

    Args:
      n_head: Number of heads
      d_model: TFT state dimensionality
      dropout: Dropout discard rate
    """

    self.n_head = n_head
    self.d_k = self.d_v = d_k = d_v = d_model // n_head
    self.dropout = dropout

    self.qs_layers = []
    self.ks_layers = []
    self.vs_layers = []

    # Use same value layer to facilitate interp
    vs_layer = Dense(d_v, use_bias=False)

    for _ in range(n_head):
      self.qs_layers.append(Dense(d_k, use_bias=False))
      self.ks_layers.append(Dense(d_k, use_bias=False))
      self.vs_layers.append(vs_layer)  # use same vs_layer

    self.attention = ScaledDotProductAttention(name="ScaledDotProdAtten")
    self.w_o = Dense(d_model, use_bias=False, name="MH_atten_output")

    super().__init__(**kwargs)

  def __call__(self, q, k, v, mask=None):
    """Applies interpretable multihead attention.

    Using T to denote the number of time steps fed into the transformer.

    Args:
      q: Query tensor of shape=(?, T, d_model)
      k: Key of shape=(?, T, d_model)
      v: Values of shape=(?, T, d_model)
      mask: Masking if required with shape=(?, T, T)

    Returns:
      Tuple of (layer outputs, attention weights)
    """
    n_head = self.n_head

    heads = []
    attns = []
    for i in range(n_head):
      qs = self.qs_layers[i](q)
      ks = self.ks_layers[i](k)
      vs = self.vs_layers[i](v)
      head, attn = self.attention(qs, ks, vs, mask, i)

      head_dropout = Dropout(self.dropout)(head)
      heads.append(head_dropout)
      attns.append(attn)
    head = array_ops.stack(heads, axis=0, name="MultiHeadAtten_heads") if n_head > 1 else heads[0]
    attn = array_ops.stack(attns, axis=0, name="MultiHeadAttention_attens")

    _outputs = K.mean(head, axis=0) if n_head > 1 else head
    _outputs = self.w_o(_outputs)
    _outputs = Dropout(self.dropout, name="MHA_output_do")(_outputs)  # output dropout

    return _outputs, attn


# Loss functions.
def tensorflow_quantile_loss(y, y_pred, quantile):
    """Computes quantile loss for tensorflow.

    Standard quantile loss as defined in the "Training Procedure" section of
    the main TFT paper

    Args:
    y: Targets
    y_pred: Predictions
    quantile: Quantile to use for loss calculations (between 0 & 1)

    Returns:
    Tensor for quantile loss.
    """

  # Checks quantile
    if quantile < 0 or quantile > 1:
        raise ValueError(
            'Illegal quantile value={}! Values should be between 0 and 1.'.format(quantile))

    prediction_underflow = y - y_pred
    q_loss = quantile * tf.maximum(prediction_underflow, 0.) + (
            1. - quantile) * tf.maximum(-prediction_underflow, 0.)

    return tf.reduce_sum(q_loss, axis=-1)


def quantile_loss(a, b, quantiles, output_size):
    """Returns quantile loss for specified quantiles.

    Args:
      a: Targets
      b: Predictions
      quantiles:
      output_size:
    """
    quantiles_used = set(quantiles)

    loss = 0.
    for i, quantile in enumerate(quantiles):
        if quantile in quantiles_used:
            loss += tensorflow_quantile_loss(
                a[Ellipsis, output_size * i:output_size * (i + 1)],
                b[Ellipsis, output_size * i:output_size * (i + 1)], quantile)
    return loss