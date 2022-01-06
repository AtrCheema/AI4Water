
import unittest

import numpy as np

from ai4water.preprocessing.transformations import StandardScaler
from ai4water.preprocessing.transformations import MinMaxScaler
from ai4water.preprocessing.transformations import RobustScaler
from ai4water.preprocessing.transformations import QuantileTransformer
from ai4water.preprocessing.transformations import PowerTransformer
from ai4water.preprocessing.transformations import FunctionTransformer
from ai4water.preprocessing.transformations import SqrtScaler
from ai4water.preprocessing.transformations import TanScaler
from ai4water.preprocessing.transformations import CumsumScaler
from ai4water.preprocessing.transformations import LogScaler
from ai4water.preprocessing.transformations import Log2Scaler
from ai4water.preprocessing.transformations import Log10Scaler
from ai4water.preprocessing.transformations import Center


x = np.random.randint(1, 100, (20, 2))
x3d = np.arange(1, 51).reshape(5, 5, 2)


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


def test_custom_scaler(Scaler, array, check_inverse=True, **kwargs):

    mysc = Scaler(**kwargs)
    _x1 = mysc.fit_transform(array)

    mysc1 = Scaler.from_config(mysc.config())

    _x2 = mysc1.fit_transform(array)

    _x = mysc1.inverse_transform(_x2)
    np.testing.assert_array_almost_equal(_x1, _x2)

    if check_inverse:
        np.testing.assert_array_almost_equal(array, _x)

    return


class TestTransformations(unittest.TestCase):

    def test_minmax(self):
        test_custom_scaler(MinMaxScaler, x)
        return

    def test_std(self):
        test_custom_scaler(StandardScaler, x)
        return

    def test_robust(self):
        test_custom_scaler(RobustScaler, x)
        return

    def test_power(self):
        test_custom_scaler(PowerTransformer, x)
        return

    def test_quantile(self):
        test_custom_scaler(QuantileTransformer, x)
        return

    def test_log(self):
        test_custom_scaler(LogScaler, x)
        test_custom_scaler(Log2Scaler, x)
        test_custom_scaler(Log10Scaler, x)
        return

    def test_log_3d(self):
        test_custom_scaler(LogScaler, x3d)
        test_custom_scaler(Log2Scaler, x3d)
        test_custom_scaler(Log10Scaler, x3d)

        test_custom_scaler(LogScaler, x3d, feature_dim="1d")
        test_custom_scaler(Log2Scaler, x3d, feature_dim="1d")
        test_custom_scaler(Log10Scaler, x3d, feature_dim="1d")
        return

    def test_sqrt(self):
        test_custom_scaler(SqrtScaler, x)
        test_custom_scaler(SqrtScaler, x3d)
        test_custom_scaler(SqrtScaler, x3d, feature_dim="1d")
        return

    def test_tan(self):
        test_custom_scaler(TanScaler, x, False)
        return

    def test_cumsum(self):
        test_custom_scaler(CumsumScaler, x, False)
        return

    def test_center(self):
        test_custom_scaler(Center, x)
        return

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

    def test_power_with_custom_lambdas(self):
        xx = np.array([1, 2, 3, 4, 5, 43, 3, 2, 2, 4, 5]).reshape(-1, 1)

        tr = PowerTransformer(method="box-cox")
        x_ = tr.fit_transform(xx)
        tr_from_config = PowerTransformer.from_config(tr.config())
        _x = tr_from_config.inverse_transform(x_)

        tr1 = PowerTransformer(method="box-cox", lambdas=np.array([-0.49657307]))
        x1_ = tr1.fit_transform(xx)
        tr1_from_config = PowerTransformer.from_config(tr1.config())
        _x1 = tr1_from_config.inverse_transform(x1_)

        assert np.allclose(x_, x1_)
        assert np.allclose(_x, _x1)
        return


if __name__ == "__main__":
    unittest.main()