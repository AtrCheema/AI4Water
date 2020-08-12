#

try:
    from tensorflow import keras
    import tensorflow as tf
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

ACTIVATIONS = {'LeakyRelu': lambda name='leaky_relu': keras.layers.LeakyReLU(name=name),
               'PRelu': lambda name='p_relu': keras.layers.PReLU(name=name),  # https://arxiv.org/pdf/1502.01852v1.pdf
               'relu': lambda name='relu': keras.layers.Activation('relu', name=name),
               'tanh': lambda name='tanh': keras.layers.Activation('tanh', name=name),
               'elu': lambda name='elu': keras.layers.ELU(name=name),
               'TheresholdRelu': lambda name='threshold_relu': keras.layers.ThresholdedReLU(name=name),
               # 'SRelu': layers.advanced_activations.SReLU()
               }


if tf is not None:
    import tf_losses as tf_losses
    LOSSES = {
        'mse': keras.losses.mse,
        'mae': keras.losses.mae,
        'mape': keras.losses.MeanAbsolutePercentageError,
        'male': keras.losses.MeanSquaredLogarithmicError,
        'nse': tf_losses.tf_nse,
    }
else:
    LOSSES = {
        'mse': torch.nn.MSELoss
    }