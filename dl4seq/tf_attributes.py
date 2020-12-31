
__all__ = ["ACTIVATION_LAYERS", "ACTIVATION_FNS", "LOSSES", "LAYERS", "OPTIMIZERS", "tcn"]

# it is supposed that tf is available

import tensorflow as tf

import dl4seq.utils.tf_losses as tf_losses
from dl4seq.nbeats_keras import NBeats
import dl4seq.models.attention_layers as attns
from dl4seq.utils.utils import get_attributes
try:
    from .private_layers import PrivateLayers
except ModuleNotFoundError:
    PrivateLayers = None

try:
    import tcn
except ModuleNotFoundError:
    tcn = None

keras = tf.keras

LOSSES = {
    'nse': tf_losses.tf_nse,
    'kge': tf_losses.tf_kge,
}
LOSSES.update(get_attributes(aus=tf.keras, what='losses'))

LAYERS = {
    "TCN": tcn.TCN if tcn is not None else None,
    # Concatenate and concatenate act differently, so if we want to use Concatenate, then use Concat not Concatenate
    # this is because we have made the layer names case insensitive and CONCATENATE is actually concatenate.
    "CONCAT": keras.layers.Concatenate,
    "NBEATS": NBeats,
}

LAYERS.update(get_attributes(aus=tf.keras, what='layers'))

# tf.layers.multiply is functional interface while tf.layers.Multiply is a proper layer in keras.
LAYERS["MULTIPLY"] = keras.layers.Multiply

LAYERS.update(get_attributes(aus=attns, what='attn_layers'))

if PrivateLayers is not None:
    # add private layers to dictionary
    LAYERS.update(get_attributes(aus=PrivateLayers, what='layers'))

ACTIVATION_LAYERS = {
    # https://ai.stanford.edu/%7Eamaas/papers/relu_hybrid_icml2013_final.pdf
    'LEAKYRELU': lambda name='softsign': keras.layers.LeakyReLU(),
    # https://arxiv.org/pdf/1502.01852v1.pdf
    'PRELU': lambda name='prelu': keras.layers.PReLU(name=name),
    'RELU': lambda name='relu': keras.layers.Activation('relu', name=name),
    'TANH': lambda name='tanh': keras.layers.Activation('tanh', name=name),
    'ELU': lambda name='elu': keras.layers.ELU(name=name),
    'THRESHOLDRELU': lambda name='ThresholdRelu': keras.layers.ThresholdedReLU(name=name),
    'SELU': lambda name='selu': keras.layers.Activation("selu", name=name),
    'SIGMOID': lambda name='sigmoid': keras.layers.Activation('sigmoid', name=name),
    'HARDSIGMOID': lambda name='HardSigmoid': keras.layers.Activation('hard_sigmoid', name=name),
    'CRELU': lambda name='crelu': keras.layers.Activation(tf.nn.crelu, name=name),
    'RELU6': lambda name='relu6': keras.layers.Activation(tf.nn.relu6, name=name),
    'SOFTMAX': lambda name='softmax': keras.layers.Activation(tf.nn.softmax, name=name),
    'SOFTPLUS': lambda name='sofplus': keras.layers.Activation(tf.nn.softplus, name=name),
    'SOFTSIGN': lambda name='softsign': keras.layers.Activation(tf.nn.softsign, name=name),
    "SWISH": lambda name='swish': keras.layers.Activation(tf.nn.swish, name=name),
}

ACTIVATION_FNS = {
    'RELU': 'relu',  # keras.layers.Activation('relu', name=name),
    'TANH': 'tanh',
    'ELU': 'elu',
    'LEAKYRELU': tf.nn.leaky_relu,
    'CRELU': tf.nn.crelu,
    'SELU': tf.nn.selu,  # tf.keras.activations.selu, # https://arxiv.org/pdf/1706.02515.pdf
    'RELU6': tf.nn.relu6,  # http://www.cs.utoronto.ca/%7Ekriz/conv-cifar10-aug2010.pdf
    'SOFTMAX': tf.nn.softmax,
    "SOFTSIGN": tf.nn.softsign,
    "SOFTPLUS": tf.nn.softplus,
    'SIGMOID': tf.nn.sigmoid,
    "HARDSIGMOID": 'hard_sigmoid',
    "LINEAR": 'linear',
    "SWISH": tf.nn.swish,  # https://arxiv.org/pdf/1710.05941.pdf
}


OPTIMIZERS = get_attributes(aus=tf.keras, what='optimizers')
