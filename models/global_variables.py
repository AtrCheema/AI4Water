__all__ = ["tf", "keras", "torch", "ACTIVATION_LAYERS", "ACTIVATION_FNS", "LOSSES", "tcn"]

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

ACTIVATION_LAYERS = {'LEAKYRELU': lambda name='leaky_relu': keras.layers.LeakyReLU(name=name), # https://ai.stanford.edu/%7Eamaas/papers/relu_hybrid_icml2013_final.pdf
               'PRelu': lambda name='p_relu': keras.layers.PReLU(name=name),  # https://arxiv.org/pdf/1502.01852v1.pdf
               'RELU': lambda name='relu': keras.layers.Activation('relu', name=name),
               'TANH': lambda name='tanh': keras.layers.Activation('tanh', name=name),
               'ELU': lambda name='elu': keras.layers.ELU(name=name),
               'TheresholdRelu': lambda name='threshold_relu': keras.layers.ThresholdedReLU(name=name),
               # 'SRelu': layers.advanced_activations.SReLU()
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

}


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
