"""
main models are pytorch based and Model.
All tensorflow based models can be implemented purely using Model.
"""
from ai4water.main import Model

try:
    from ai4water.pytorch_models import IMVModel
    from ai4water.pytorch_models import HARHNModel
except AttributeError:
    print("\n{}Pytorch models could not be imported {}\n".format(10*'*', 10*'*'))

try:
    from ai4water.tf_models import InputAttentionModel
    from ai4water.tf_models import DualAttentionModel
except AttributeError:
    print("\n{}Tensorflow models could not be imported {}\n".format(10 * '*', 10 * '*'))

__version__ = "1.04"
