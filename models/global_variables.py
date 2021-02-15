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

ACTIVATIONS = {'LeakyRelu': keras.layers.LeakyReLU(),
               'PRelu': lambda name='p_relu': keras.layers.PReLU(name=name),  # https://arxiv.org/pdf/1502.01852v1.pdf
               'relu': keras.layers.Activation('relu', name='relu'),
               'tanh': keras.layers.Activation('tanh', name='tanh'),
               'elu': keras.layers.ELU(name='elu'),
               'TheresholdRelu': lambda name='threshold_relu': keras.layers.ThresholdedReLU(name=name),
               # 'SRelu': layers.advanced_activations.SReLU()
               None: None
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