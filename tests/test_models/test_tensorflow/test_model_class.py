
import os
import time
import unittest
import site
site.addsitedir(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import tensorflow as tf
import numpy as np

if 230 <= int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 250:
    from ai4water.functional import Model
    print(f"Switching to functional API due to tensorflow version {tf.__version__}")
else:
    from ai4water import Model

from ai4water.postprocessing import PermutationImportance
from ai4water.postprocessing import PartialDependencePlot
from ai4water.datasets import busan_beach
from ai4water.functional import Model as FModel
from ai4water.preprocessing import DataSet
from ai4water.models import LSTM


data = busan_beach(inputs=['tide_cm', 'pcp_mm', 'sal_psu'])
dh = DataSet(data=data, verbosity=0)
x_reg, y_reg = dh.training_data()


class TestPermImp(unittest.TestCase):

    def test_lookback(self):
        time.sleep(1)
        model = Model(model=LSTM(1, input_shape=(3, 3)),
                      ts_args={"lookback": 3},
                      verbosity=-1)
        model.fit(data=data)
        imp = model.permutation_importance(data=data, data_type="validation")
        assert isinstance(imp, PermutationImportance)
        return


class TestPDP(unittest.TestCase):

    def test_lookback(self):
        model = Model(model=LSTM(1, input_shape=(3, 3)),
                      ts_args={"lookback": 3},
                      verbosity=-1)
        model.fit(data=data)
        pdp = model.partial_dependence_plot(data=data,
                                            feature_name='tide_cm',
                                            num_points=2, show=False)
        assert isinstance(pdp, PartialDependencePlot)
        return


class TestShapValues(unittest.TestCase):

    def test_lookback(self):
        time.sleep(1) # todo
        model = FModel(model=LSTM(1, input_shape=(3, 3)),
                      ts_args={"lookback": 3},
                      verbosity=-1)
        model.fit(data=data)
        sv = model.shap_values(data=data)
        assert isinstance(sv, (np.ndarray, list)), f"{type(sv)}"
        return


if __name__ == "__main__":

    unittest.main()
