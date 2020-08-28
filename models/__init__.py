try:
    from .pytorch_models import IMVLSTMModel
    from .pytorch_models import HARHNModel
except AttributeError:
    print("\n{}Pytorch models could not be imported {}\n".format(10*'*', 10*'*'))

from .tf_models import CNNModel
from .tf_models import CNNLSTMModel
from .tf_models import LSTMCNNModel
from .tf_models import LSTMAutoEncoder
from .tf_models import TCNModel
from .tf_models import InputAttentionModel
from .tf_models import DualAttentionModel
from .tf_models import NBeatsModel
from .tf_models import ConvLSTMModel
from main import Model
