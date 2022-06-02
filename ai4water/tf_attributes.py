__all__ = ["ACTIVATION_LAYERS", "ACTIVATION_FNS", "LOSSES", "LAYERS", "OPTIMIZERS", "tcn"]

# it is supposed that tf is available
from .backend import get_attributes, tf

try:
    import tcn
except ModuleNotFoundError:
    tcn = None

LOSSES = {}
LAYERS = {}

if tcn is not None:
    LAYERS.update({"TCN": tcn.TCN if tcn is not None else None})

if tf is not None:
    import ai4water.utils.tf_losses as tf_losses
    from ai4water.models.tensorflow import NBeats
    import ai4water.models.tensorflow.attention_layers as attns
    from ai4water.models.tensorflow import TemporalFusionTransformer

    keras = tf.keras
    LOSSES.update({
        'nse': tf_losses.tf_nse,
        'kge': tf_losses.tf_kge,
    })
    LOSSES.update(get_attributes(aus=tf.keras, what='losses', case_sensitive=True))
else:
    NBeats, TemporalFusionTransformer, attns, keras = None, None, None, None

if tf is not None:
    LAYERS.update({"TemporalFusionTransformer": TemporalFusionTransformer})
    LAYERS.update({"TFT": TemporalFusionTransformer})
    LAYERS.update(get_attributes(aus=tf.keras, what='layers', case_sensitive=True))

    from .models.tensorflow.private_layers import PrivateLayers

    # add private layers to dictionary
    LAYERS.update(get_attributes(aus=PrivateLayers, what='layers', case_sensitive=True))

if NBeats is not None:
    LAYERS.update({"NBeats": NBeats})

if attns is not None:
    LAYERS.update(get_attributes(aus=attns, what='attn_layers', case_sensitive=True))

ACTIVATION_LAYERS = {
    # https://ai.stanford.edu/%7Eamaas/papers/relu_hybrid_icml2013_final.pdf
    'LeakyReLU': lambda name='softsign': keras.layers.LeakyReLU(),
    # https://arxiv.org/pdf/1502.01852v1.pdf
    'PReLU': lambda name='prelu': keras.layers.PReLU(name=name),
    'relu': lambda name='relu': keras.layers.Activation('relu', name=name),
    'tanh': lambda name='tanh': keras.layers.Activation('tanh', name=name),
    'ELU': lambda name='elu': keras.layers.ELU(name=name),
    'ThresholdedReLU': lambda name='ThresholdRelu': keras.layers.ThresholdedReLU(name=name),
    'selu': lambda name='selu': keras.layers.Activation("selu", name=name),
    'sigmoid': lambda name='sigmoid': keras.layers.Activation('sigmoid', name=name),
    'hardsigmoid': lambda name='HardSigmoid': keras.layers.Activation('hard_sigmoid', name=name),
    'crelu': lambda name='crelu': keras.layers.Activation(tf.nn.crelu, name=name),
    'relu6': lambda name='relu6': keras.layers.Activation(tf.nn.relu6, name=name),
    'softmax': lambda name='softmax': keras.layers.Activation(tf.nn.softmax, name=name),
    'softplus': lambda name='sofplus': keras.layers.Activation(tf.nn.softplus, name=name),
    'softsign': lambda name='softsign': keras.layers.Activation(tf.nn.softsign, name=name),
    "swish": lambda name='swish': keras.layers.Activation(tf.nn.swish, name=name),
}

ACTIVATION_FNS = {
    'relu': 'relu',  # keras.layers.Activation('relu', name=name),
    'tanh': 'tanh',
    'elu': 'elu',
    "hardsigmoid": 'hard_sigmoid',
    "linear": 'linear',

}

if tf is not None:
    ACTIVATION_FNS.update({
        'leakyrelu': tf.nn.leaky_relu,
        'crelu': tf.nn.crelu,
        'selu': tf.nn.selu,  # tf.keras.activations.selu, # https://arxiv.org/pdf/1706.02515.pdf
        'relu6': tf.nn.relu6,  # http://www.cs.utoronto.ca/%7Ekriz/conv-cifar10-aug2010.pdf
        'softmax': tf.nn.softmax,
        "softsign": tf.nn.softsign,
        "softplus": tf.nn.softplus,
        'sigmoid': tf.nn.sigmoid,
        "swish": tf.nn.swish,  # https://arxiv.org/pdf/1710.05941.pdf
    })

OPTIMIZERS = {}
if tf is not None:
    OPTIMIZERS.update(get_attributes(aus=tf.keras, what='optimizers', case_sensitive=True))
