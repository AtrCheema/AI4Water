"""
main models are pytorch based and Model.
All tensorflow based models can be implemented purely using Model.
"""
from dl4seq.main import Model

try:
    from dl4seq.pytorch_models import IMVLSTMModel
    from dl4seq.pytorch_models import HARHNModel
except AttributeError:
    print("\n{}Pytorch models could not be imported {}\n".format(10*'*', 10*'*'))

try:
    from dl4seq.tf_models import CNNLSTMModel
    from dl4seq.tf_models import InputAttentionModel
    from dl4seq.tf_models import DualAttentionModel
    from dl4seq.tf_models import NBeatsModel
    from dl4seq.tf_models import ConvLSTMModel
except AttributeError:
    print("\n{}Tensorflow models could not be imported {}\n".format(10 * '*', 10 * '*'))