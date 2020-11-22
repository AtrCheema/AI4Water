from main import Model

try:
    from .pytorch_models import IMVLSTMModel
    from .pytorch_models import HARHNModel
except AttributeError:
    print("\n{}Pytorch models could not be imported {}\n".format(10*'*', 10*'*'))

try:
    from .tf_models import CNNLSTMModel
    from .tf_models import InputAttentionModel
    from .tf_models import DualAttentionModel
    from .tf_models import NBeatsModel
    from .tf_models import ConvLSTMModel
except AttributeError:
    print("\n{}Tensorflow models could not be imported {}\n".format(10 * '*', 10 * '*'))