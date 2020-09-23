try:
    from .pytorch_models import IMVLSTMModel
    from .pytorch_models import HARHNModel
except AttributeError:
    print("\n{}Pytorch models could not be imported {}\n".format(10*'*', 10*'*'))

from .tf_models import LSTMModel
from .tf_models import CNNLSTMModel
from .tf_models import AutoEncoder
from .tf_models import InputAttentionModel
from .tf_models import DualAttentionModel
from .tf_models import NBeatsModel
from .tf_models import ConvLSTMModel
from main import Model
