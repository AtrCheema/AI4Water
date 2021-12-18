
import unittest

import numpy as np

from ai4water.preprocessing.transformations import StandardScaler
from ai4water.preprocessing.transformations import MinMaxScaler
from ai4water.preprocessing.transformations import RobustScaler
from ai4water.preprocessing.transformations import QuantileTransformer
from ai4water.preprocessing.transformations import PowerTransformer
from ai4water.preprocessing.transformations import FunctionTransformer


x = np.random.randint(1, 100, (20, 2))


def test_func_transformer(array, func, inverse_func, **kwargs):
    sc = FunctionTransformer(func=func, inverse_func=inverse_func, **kwargs)
    t_x1 = sc.fit_transform(array)

    sc1 = FunctionTransformer.from_config(sc.config())
    t_x2 = sc1.fit_transform(array)
    np.testing.assert_array_almost_equal(t_x1, t_x2)

    _x = sc1.inverse_transform(t_x2)

    if sc1.check_inverse and sc1.inverse_func is not None:
        np.testing.assert_array_almost_equal(array, _x)
    return


def test_custom_scaler(Scaler, array):

    mysc = Scaler()
    _x1 = mysc.fit_transform(array)

    mysc1 = Scaler.from_config(mysc.config())

    _x2 = mysc1.fit_transform(array)

    _x = mysc1.inverse_transform(_x2)
    np.testing.assert_array_almost_equal(_x1, _x2)

    np.testing.assert_array_almost_equal(array, _x)

    return


class TestTransformations(unittest.TestCase):

    def test_minmax(self):
        test_custom_scaler(MinMaxScaler, x)

    def test_std(self):
        test_custom_scaler(StandardScaler, x)

    def test_robust(self):
        test_custom_scaler(RobustScaler, x)

    def test_power(self):
        test_custom_scaler(PowerTransformer, x)

    def test_quantile(self):
        test_custom_scaler(QuantileTransformer, x)

    def test_function(self):
        test_func_transformer(x, np.log, None)
        test_func_transformer(x, np.log, np.exp)
        test_func_transformer(x, np.log2, inverse_func="""lambda _x: 2**_x""", validate=True)
        test_func_transformer(x, np.log10, inverse_func="""lambda _x: 10**_x""", validate=True, check_inverse=False)
        test_func_transformer(x, np.tan, inverse_func=np.tanh, validate=True, check_inverse=False)
        test_func_transformer(x, np.cumsum, inverse_func=np.diff,
                              validate=True, check_inverse=False,
                              kw_args={"axis": 0},
                              inv_kw_args={"axis": 0, "append": 0})
        return

if __name__ == "__main__":
    unittest.main()