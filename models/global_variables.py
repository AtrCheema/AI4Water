__all__ = ["tf", "keras", "torch", "ACTIVATION_LAYERS", "ACTIVATION_FNS", "LOSSES", "tcn", "LAYERS"]

maj_version = 0
min_version = 0
try:
    from tensorflow import keras
    import tensorflow as tf
    maj_version = int(tf.__version__[0])
    min_version = int(tf.__version__[2])
    from .attention_layers import AttentionRaffel, SelfAttention, BahdanauAttention, HierarchicalAttention, SeqSelfAttention
    from .attention_layers import SnailAttention
except ModuleNotFoundError:
    keras = None
    tf = None

try:
    import tcn
except ModuleNotFoundError:
    tcn = None

try:
    import torch
except ModuleNotFoundError:
    torch = None

keras = keras
torch = torch
tf = tf


def get_keras_layers() ->dict:
    """ gets all callable attributes of tf.keras.layers and saves them in dictionary with their names all capitalized
    so that calling them becomes case insensitive. It is possible that some of the attributes of tf.keras.layers
    are callable but still not a `layer`, in that case the error will be generated from tensorflow. We are not catching
    that error right now."""
    all_lyrs = {}
    for l in dir(tf.keras.layers):
        layer = getattr(tf.keras.layers, l)
        if callable(layer):
            all_lyrs[l.upper()] = layer

    return all_lyrs

if keras is not None:
    LAYERS = {
        "TCN": tcn.TCN if tcn is not None else None,
        "ATTENTIONRAFFEL": AttentionRaffel,
        "SELFATTENTION": SelfAttention,
        "BAHDANAUATTENTION": BahdanauAttention,
        "HIERARCHICALATTENTION": HierarchicalAttention,
        "SEQSELFATTENTION": SeqSelfAttention,
        "SNAILATTENTION": SnailAttention,
    }

    LAYERS.update(get_keras_layers())

    ACTIVATION_LAYERS = {
        'LEAKYRELU': keras.layers.LeakyReLU(), # https://ai.stanford.edu/%7Eamaas/papers/relu_hybrid_icml2013_final.pdf
        'PRELU': keras.layers.PReLU(),  # https://arxiv.org/pdf/1502.01852v1.pdf
        'RELU': keras.layers.Activation('relu'),
        'TANH': keras.layers.Activation('tanh'),
        'ELU': keras.layers.ELU(),
        'THRESHOLDRELU': keras.layers.ThresholdedReLU(),
        'SELU': keras.layers.Activation("selu"),
        'SIGMOID': keras.layers.Activation('sigmoid'),
        'HARDSIGMOID': keras.layers.Activation('hard_sigmoid'),
        'CRELU': keras.layers.Activation(tf.nn.crelu),
        'RELU6': keras.layers.Activation(tf.nn.relu6),
        'SOFTMAX': keras.layers.Activation(tf.nn.softmax),
        'SOFTPLUS': keras.layers.Activation(tf.nn.softplus),
        'SOFTSIGN': keras.layers.Activation(tf.nn.softsign)
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
        'SIGMOID': tf.nn.sigmoid,
        "HARDSIGMOID": 'hard_sigmoid',
        "LINEAR": 'linear'
    }
else:
    LAYERS = None
    ACTIVATION_LAYER = None
    ACTIVATION_FNS = None


if tf is not None:
    import tf_losses as tf_losses
    LOSSES = {
        'mse': keras.losses.mse,
        'mae': keras.losses.mae,
        'mape': keras.losses.MeanAbsolutePercentageError,
        'male': keras.losses.MeanSquaredLogarithmicError,
        'nse': tf_losses.tf_nse,
        'kge': tf_losses.tf_kge,
    }
else:
    LOSSES = {
        'mse': torch.nn.MSELoss
    }
