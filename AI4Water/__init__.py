"""
main models are pytorch based and Model.
All tensorflow based models can be implemented purely using Model.
"""
from AI4Water.main import Model
from AI4Water._data import DataHandler, SiteDistributedDataHandler

try:
    from AI4Water.pytorch_models import IMVLSTMModel
    from AI4Water.pytorch_models import HARHNModel
except AttributeError:
    print("\n{}Pytorch models could not be imported {}\n".format(10*'*', 10*'*'))

try:
    from AI4Water.tf_models import InputAttentionModel
    from AI4Water.tf_models import DualAttentionModel
    from AI4Water.tf_models import NBeatsModel
except AttributeError:
    print("\n{}Tensorflow models could not be imported {}\n".format(10 * '*', 10 * '*'))

__version__ = '1.0'