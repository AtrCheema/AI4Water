
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

from ai4water.datasets import busan_beach
from ai4water.functional import Model as FModel
from ai4water.preprocessing import DataSet


data = busan_beach(inputs=['tide_cm', 'pcp_mm', 'sal_psu'])
dh = DataSet(data=data, verbosity=0)
x_reg, y_reg = dh.training_data()


class TestPredictMethod(unittest.TestCase):
    """Tests the `predict` method of Model class"""

    def test_without_fit_with_data(self):
        """call to predict method without training/fit"""
        model = Model(model={"layers": {"Dense_0": 8,
                                        "Dense_1": 1}},
                    input_features=data.columns.tolist()[0:-1],
                    output_features=data.columns.tolist()[-1:],
                    verbosity=-1)
        p = model.predict_on_test_data(data=data)
        assert isinstance(p, np.ndarray)
        return

    def test_without_fit_with_xy(self):
        """call to predict method without training/fit by provided x and y keywords"""
        model = Model(model={"layers": {"Dense_0": 8,
                                        "Dense_1": 1}},
                    input_features=data.columns.tolist()[0:-1],
                    output_features=data.columns.tolist()[-1:],
                    verbosity=-1)
        t, p = model.predict(x=x_reg, y=y_reg, return_true=True)
        assert isinstance(t, np.ndarray)
        assert isinstance(p, np.ndarray)
        return

    def test_without_fit_with_only_x(self):
        """call to predict method without training/fit by providing only x"""
        model = Model(model={"layers": {"Dense_0": 8,
                                        "Dense_1": 1}},
                    input_features=data.columns.tolist()[0:-1],
                    output_features=data.columns.tolist()[-1:],
                    verbosity=-1)
        p = model.predict(x=x_reg)
        assert isinstance(p, np.ndarray)
        return

    def test_tf_data(self):
        """when x is tf.data.Dataset"""

        model = Model(model={"layers": {"Dense": 1}},
                      input_features=data.columns.tolist()[0:-1],
                      output_features=data.columns.tolist()[-1:],
                      verbosity=-1
                      )
        x,y = DataSet(data=data, verbosity=0).training_data()
        tr_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size=32)
        _ = model.predict(x=tr_ds)
        return

    def test_predict_without_y(self):
        # make sure that .predict method can be called without `y`.

        model = Model(model='RandomForestRegressor',
                      verbosity=0
                      )
        model.fit(data=busan_beach())
        y = model.predict(x=np.random.random((10, model.num_ins)))
        assert len(y) == 10

        # check also for functional model
        model = FModel(model='RandomForestRegressor',
                      verbosity=0
                      )
        model.fit(data=busan_beach())
        y = model.predict(x=np.random.random((10, model.num_ins)))
        assert len(y) == 10

        time.sleep(1)
        return


if __name__ == "__main__":

    unittest.main()