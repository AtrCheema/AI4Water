__all__ = ["tf", "keras", "torch", "ACTIVATION_LAYERS", "ACTIVATION_FNS", "LOSSES", "tcn", "LAYERS", "OPTIMIZERS"]

maj_version = 0
min_version = 0
try:
    from tensorflow import keras
    import tensorflow as tf
    maj_version = int(tf.__version__[0])
    min_version = int(tf.__version__[2])
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


def get_attributes(aus=tf.keras, what:str='losses') ->dict:
    """ gets all callable attributes of aus e.g. from tf.keras.what and saves them in dictionary with their names all
    capitalized so that calling them becomes case insensitive. It is possible that some of the attributes of tf.keras.layers
    are callable but still not a valid `layer`, sor some attributes of tf.keras.losses are callable but still not valid
    losses, in that case the error will be generated from tensorflow. We are not catching those error right now."""
    all_attrs = {}
    for l in dir(getattr(aus, what)):
        attr = getattr(getattr(aus, what), l)
        if callable(attr) and not l.startswith('_'):
            all_attrs[l.upper()] = attr

    return all_attrs

if keras is not None:
    LAYERS = {
        "TCN": tcn.TCN if tcn is not None else None,
        # Concatenate and concatenate act differently, so if we want to use Concatenate, then use Concat not Concatenate
        # this is because we have made the layer names case insensitive and CONCATENATE is actually concatenate.
        "CONCAT": keras.layers.Concatenate,
    }

    LAYERS.update(get_attributes(what='layers'))

    # tf.layers.multiply is functional interface while tf.layers.Multiply is a proper layer in keras.
    LAYERS["MULTIPLY"] = keras.layers.Multiply

    import models.attention_layers as attns

    LAYERS.update(get_attributes(aus=attns, what='attn_layers'))

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

    OPTIMIZERS = get_attributes(what='optimizers')
else:
    LAYERS = None
    ACTIVATION_LAYER = None
    ACTIVATION_FNS = None
    OPTIMIZERS = None


if tf is not None:
    import tf_losses as tf_losses
    LOSSES = {
        'nse': tf_losses.tf_nse,
        'kge': tf_losses.tf_kge,
    }
    LOSSES.update(get_attributes(what='losses'))
else:
    LOSSES = {
        'mse': torch.nn.MSELoss
    }
