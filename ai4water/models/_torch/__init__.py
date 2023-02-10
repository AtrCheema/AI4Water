

from ai4water.backend import torch

LAYERS, LOSSES, OPTIMIZERS = {}, {}, {}
Learner = None

if torch is not None:
    from .HARHN import HARHN
    from .pytorch_training import Learner
    from .imv_networks import IMVTensorLSTM, IMVFullLSTM
    from .pytorch_attributes import LAYERS, LOSSES, OPTIMIZERS
    from ._functions import MLP
    from ._functions import LSTM
    from ._functions import CNN
