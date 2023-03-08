import math
import site   # so that ai4water directory is in path
import unittest
import os
import sys
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

import numpy as np
import pandas as pd

from ai4water.preprocessing.transformations import Transformation
from ai4water.preprocessing.transformations.utils import TransformerNotFittedError
from ai4water.tf_attributes import tf
from ai4water.datasets import busan_beach
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, FunctionTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from ai4water.preprocessing.transformations import ParetoTransformer
from ai4water.preprocessing.transformations import VastTransformer
from ai4water.preprocessing.transformations import MmadTransformer


if 230 <= int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 250:
    from ai4water.functional import Model
    print(f"Switching to functional API due to tensorflow version {tf.__version__}")
else:
    from ai4water import Model


df = pd.DataFrame(np.concatenate([np.arange(1, 10).reshape(-1, 1),
                                  np.arange(1001, 1010).reshape(-1, 1)],
                                 axis=1),
                  columns=['data1', 'data2'])


def test_transform(sk_tr, ai_tr, features):

    data = busan_beach()
    data = data.dropna()
    train_data = data.iloc[0:160]
    test_data = data.iloc[160:]

    _sk_test_data, _sk_train_data, sk_test_data_, sk_train_data_  = test_transform_sklearn(
        sk_tr,
        train_data,
        test_data,
        features)

    _ai_test_data, _ai_train_data, ai_test_data_, ai_train_data_ = test_transform_Transformation(
        ai_tr,
        train_data,
        test_data,
        features)

    np.testing.assert_array_almost_equal(_sk_test_data, _ai_test_data[features].values)
    np.testing.assert_array_almost_equal(_sk_train_data, _ai_train_data[features].values)

    np.testing.assert_array_almost_equal(sk_test_data_, ai_test_data_[features].values)
    np.testing.assert_array_almost_equal(sk_train_data_, ai_train_data_[features].values)

    return

def test_transform_sklearn(sk_tr, train_data, test_data, features):

    train_data_ = sk_tr.fit_transform(train_data[features].values)
    test_data_ = sk_tr.transform(test_data[features].values)
    _test_data = sk_tr.inverse_transform(test_data_)
    _train_data = sk_tr.inverse_transform(train_data_)

    np.testing.assert_array_almost_equal(_train_data, train_data[features].values)
    np.testing.assert_array_almost_equal(_test_data, test_data[features].values)

    return _test_data, _train_data, test_data_, train_data_


def test_transform_Transformation(ai_tr, train_data, test_data, features):

    train_data_ = ai_tr.fit_transform(train_data)
    test_data_ = ai_tr.transform(test_data)
    _test_data = ai_tr.inverse_transform(test_data_)
    _train_data = ai_tr.inverse_transform(train_data_)

    denorm_train = _train_data[features].values
    denorm_test = _test_data[features].values
    np.testing.assert_array_almost_equal(denorm_train, train_data[features].values)
    np.testing.assert_array_almost_equal(denorm_test, test_data[features].values)

    return _test_data, _train_data, test_data_, train_data_


def build_and_run(x_transformation, y_transformation,
                  data, inputs, outputs):

    model = Model(model="RandomForestRegressor",
                  input_features=inputs,
                  output_features=outputs,
                  x_transformation=x_transformation,
                  y_transformation=y_transformation,
                  verbosity=0)

    model.fit(data=data)
    x, pred = model.training_data(key='junk')

    #pred, pred = model.inverse_transform(y, y, key='junk')

    index = model.dh_.indexes['junk']
    pred = pd.DataFrame(pred.reshape(len(pred), model.num_outs),
                        columns=outputs, index=index).sort_index()
    return pred


def run_method2(method,
                data=None,
                index=None,
                **kwargs):

    if index:
        data.index = pd.date_range("20110101", periods=len(data), freq="D")

    scaler = Transformation(method=method,
                             **kwargs)

    normalized_df, pp = scaler.fit_transform(data, return_proc=True)

    denormalized_df = scaler.inverse_transform(data=normalized_df,
                                               postprocessor=pp)
    return data, normalized_df, denormalized_df


def run_method3(method,
                data=None,
                index=None,
                **kwargs):

    if index:
        data.index = pd.date_range("20110101", periods=len(data), freq="D")

    scaler = Transformation(method=method,
                             **kwargs)

    normalized_df3, proc = scaler(data,
                                         return_proc=True)
    denormalized_df3 = scaler(what='inverse', data=normalized_df3,
                              postprocessor=proc)

    return data, normalized_df3, denormalized_df3


def run_log_methods(method="log", index=None, insert_nans=True, insert_zeros=False,
                    assert_equality=True,
                    insert_ones=False):
    a = np.random.random((10, 4))
    a[0, 0] = np.nan
    a[0, 1] = 1.

    if insert_nans or insert_zeros:
        a[2:4, 1] = np.nan
        a[3:5, 2:3] = np.nan

    if insert_zeros:
        a[5:8, 3] = 0.0

    if insert_ones:
        a[6, 1] = 1.0
        a[9, 2:3] = 1.0

    cols = ['data1', 'data2', 'data3', 'data4']

    if index is not None:
        index = pd.date_range("20110101", periods=len(a), freq="D")

    df3 = pd.DataFrame(a, columns=cols, index=index)

    _, _, dfo2 = run_method2(method=method, data=df3.copy())

    _, _, dfo3 = run_method3(method=method, data=df3.copy())

    if assert_equality:
        assert np.allclose(df3, dfo2, equal_nan=True)
        assert np.allclose(df3, dfo3, equal_nan=True)
    return


class test_Scalers(unittest.TestCase):

    def run_method(self,
                   method,
                   cols=None,
                   index=None,
                   assert_equality=False,
                   **kwargs):

        cols = ['data1', 'data2'] if cols is None else cols

        orig_df2, norm_df2, denorm_df2 = run_method2(method,
                                                     index=index,
                                                     features=cols,
                                                     data=df.copy(),
                                                     **kwargs
                                                     )

        orig_df3, norm_df3, denorm_df3 = run_method3(method,
                                                     index=index,
                                                     data=df.copy(),
                                                     **kwargs)

        if assert_equality:
            assert np.allclose(orig_df2, denorm_df2)
            assert np.allclose(orig_df3, denorm_df3)

        else:
            np.allclose(norm_df2[cols].values, norm_df3[cols].values)
            np.allclose(denorm_df2[cols].values, denorm_df3[cols].values)


    def check_features(self, denorm):

        for idx, v in enumerate(denorm['data2']):
            self.assertEqual(v, 1001 + idx)

    def test_get_scaler_from_dict_error(self):
        normalized_df1, _ = Transformation()(df, 'fit_transform', return_proc=True)
        self.assertRaises(NotFittedError, Transformation(), what='inverse'
                          , data=normalized_df1)
        return

    def test_log_scaler_with_feat(self):
        self.run_method("log", cols=["data1"])
        return

    def test_robust_scaler_with_feat(self):
        self.run_method("robust", cols=["data1"], assert_equality=True)
        return

    def test_minmax_scaler_with_feat(self):
        self.run_method("minmax", cols=["data1"], assert_equality=True)
        return

    def test_minmax_scaler_with_feat_and_index(self):
        self.run_method("minmax", cols=["data1"], index=True, assert_equality=True)

    def test_maxabs_scaler_with_feat(self):
        self.run_method("maxabs", cols=["data1"], assert_equality=True)

    def test_zscore_scaler_with_feat(self):
        self.run_method("minmax", cols=["data1"], assert_equality=True)

    def test_power_scaler_with_feat(self):
        self.run_method("maxabs", cols=["data1"], assert_equality=True)

    def test_quantile_scaler_with_feat(self):
        self.run_method("quantile", cols=["data1"], assert_equality=True,
                        n_quantiles=5)

    def test_log_scaler(self):
        self.run_method("log", assert_equality=True)

    def test_log10_scaler(self):
        self.run_method("log10", assert_equality=True)
        return

    def test_log2_scaler(self):
        self.run_method("log2", assert_equality=True)
        return

    def test_robust_scaler(self):
        self.run_method("robust", assert_equality=True)
        return

    def test_minmax_scaler(self):
        self.run_method("minmax", assert_equality=True)
        return

    def test_maxabs_scaler(self):
        self.run_method("maxabs", assert_equality=True)

    def test_zscore_scaler(self):
        self.run_method("minmax", assert_equality=True)

    def test_power_scaler(self):
        self.run_method("maxabs", assert_equality=True)

    def test_quantile_scaler(self):
        self.run_method("quantile", assert_equality=True, n_quantiles=5)
        return

    def test_log_with_nans(self):
        run_log_methods(index=None)
        return

    def test_log_with_index(self):
        run_log_methods("log", True)
        return

    def test_log10_with_nans(self):
        run_log_methods(method='log10', index=None)
        return

    def test_log10_with_index(self):
        run_log_methods("log10", True)
        return

    def test_log2_with_nans(self):
        run_log_methods(method='log2', index=None)
        return

    def test_log2_with_index(self):
        run_log_methods("log2", True)
        return

    def test_tan_with_nans(self):
        run_log_methods("tan", index=None, assert_equality=False)
        return

    def test_tan_with_index(self):
        run_log_methods("tan", True, assert_equality=False)
        return

    def test_cumsum_with_index(self):
        run_log_methods("cumsum", True, insert_nans=False, assert_equality=False)
        return

    def test_cumsum_with_nan(self):
        run_log_methods("cumsum", True, insert_nans=True, assert_equality=False)
        return

    def test_zero_log(self):
        run_log_methods("log", True, insert_nans=True, insert_zeros=True)
        return

    def test_zero_one_log(self):
        run_log_methods("log", True, insert_nans=True, insert_zeros=True,
                        insert_ones=True)
        return

    def test_zero_log10(self):
        run_log_methods("log10", True, insert_nans=True, insert_zeros=True)
        return

    def test_zero_one_log10(self):
        run_log_methods("log10", True, insert_nans=True, insert_zeros=True,
                        insert_ones=True)
        return

    def test_zero_log2(self):
        run_log_methods("log2", True, insert_nans=True, insert_zeros=True)
        return

    def test_zero_one_log2(self):
        run_log_methods("log2", True, insert_nans=True, insert_zeros=True,
                        insert_ones=True)
        return

    def test_multiple_transformations(self):
        """Test when we want to apply multiple transformations on one or more
        features"""
        inputs = ['in1', 'inp1']
        outputs = ['out1']

        data = pd.DataFrame(np.random.random((100, 3)), columns=inputs+outputs)

        x_transformation = "minmax"
        y_transformation = ["log", "minmax"]

        pred = build_and_run(x_transformation, y_transformation, data, inputs,
                             outputs)

        for i in pred.index:
            assert np.allclose(data['out1'].loc[i], pred['out1'].loc[i])

        return

    def test_multiple_same_transformations(self):
        """Test when we want to apply multiple transformations on one or more features"""
        inputs = ['in1', 'inp1']
        outputs = ['out1']

        data = pd.DataFrame(np.random.random((100, 3)), columns=inputs+outputs)

        x_transformation = "robust"
        y_transformation = ["robust", "robust"]

        pred = build_and_run(x_transformation, y_transformation, data, inputs,outputs)

        for i in pred.index:
            assert np.allclose(data['out1'].loc[i], pred['out1'].loc[i])

        return

    def test_multiple_same_transformations_mutile_outputs(self):
        """Test when we want to apply multiple transformations on one or more features"""
        inputs = ['in1', 'inp1']
        outputs = ['out1', 'out2']

        data = pd.DataFrame(np.random.random((100, 4)), columns=inputs+outputs)

        x_transformation = "robust"
        y_transformation = ["robust", "robust"]

        pred = build_and_run(x_transformation, y_transformation, data, inputs,outputs)

        for i in pred.index:
            assert np.allclose(data['out1'].loc[i], pred['out1'].loc[i])
            assert np.allclose(data['out2'].loc[i], pred['out2'].loc[i])

        return

    def test_example(self):
        data = busan_beach()
        inputs = ['pcp6_mm', 'pcp12_mm', 'wind_dir_deg', 'wind_speed_mps', 'air_p_hpa']
        transformer = Transformation(method='minmax', features=['pcp6_mm', 'pcp12_mm'])
        new_data = transformer.fit_transform(data[inputs])
        orig_data = transformer.inverse_transform(data=new_data)
        np.allclose(data[inputs].values, orig_data.values)

        return

    def test_example2(self):
        transformer = Transformation(method='log', replace_zeros=True)
        data = [1, 2, 3, 0.0, 5, np.nan, 7]
        transformed_data = transformer.fit_transform(data)

        expected = np.array([0.0, 0.6931, 1.0986, 0.0, 1.609, None, 1.9459])
        for i, j in zip(transformed_data.values, expected):
            if math.isfinite(i):
                self.assertAlmostEqual(i.item(), j, places=3)

        detransformed_data = transformer.inverse_transform(data=transformed_data)
        np.allclose(data, detransformed_data)
        return

    def test_negative(self):
        for m in ["log", "log2", "log10", "minmax", "zscore", "robust", "quantile",
                  "power", "scale", "center", "sqrt", "yeo-johnson", "box-cox",
                  "mmad", "vast", "pareto",
                  ]:
            kwargs = {}
            if m=="quantile":
                kwargs['n_quantiles'] = 2
            x = [1.0, 2.0, -3.0, 4.0]
            tr = Transformation(method=m, treat_negatives=True, **kwargs)
            xtr, proc = tr.fit_transform(x, return_proc=True)
            _x = tr.inverse_transform(data=xtr, postprocessor=proc)
            np.testing.assert_array_almost_equal(x, _x.values.reshape(-1,))

        for m in ["log", "log2", "log10", "minmax", "zscore", "robust", "quantile",
                  "power", "scale", "center", "sqrt", "yeo-johnson",
                  "box-cox"]:
            kwargs = {}
            if m=="quantile":
                kwargs['n_quantiles'] = 2
            x1 = [1.0, -2.0, 0.0, 4.0]
            df1 = pd.DataFrame(np.column_stack([x, x1]))
            tr = Transformation(method=m, treat_negatives=True, replace_zeros=True,
                                replace_zeros_with=1, **kwargs)
            dft, proc = tr.fit_transform(df1, return_proc=True)
            _df = tr.inverse_transform(data=dft, postprocessor=proc)
            np.testing.assert_array_almost_equal(df1.values, _df.values)

        return

    def test_boxcox(self):
        t = Transformation("box-cox")
        x1 = t.fit_transform([1,2,3])
        from sklearn.preprocessing import PowerTransformer
        x2 = PowerTransformer('box-cox').fit_transform(np.array([1,2,3]).reshape(-1,1))
        np.testing.assert_array_almost_equal(x1, x2)
        return

    def test_yeojohnson(self):
        t = Transformation("yeo-johnson")
        x1 = t.fit_transform([1,2,3])
        from sklearn.preprocessing import PowerTransformer
        x2 = PowerTransformer().fit_transform(np.array([1,2,3]).reshape(-1,1))
        np.testing.assert_array_almost_equal(x1, x2)
        return

    def test_center(self):
        run_log_methods("center")
        return

    def test_scale(self):
        run_log_methods('scale')
        return

    def test_from_config_1d(self):
        for method in ["quantile", "robust", "quantile_normal",
                       "power", "box-cox", "center", "zscore", "scale",
            "yeo-johnson", "mmad", "vast", "pareto"
                       ]:
            kwargs = {}
            if method=="quantile":
                kwargs['n_quantiles'] = 5

            if method == "yeo-johnson":
                kwargs['pre_center'] = True
                kwargs['rescale'] = True

            t = Transformation(method, treat_negatives=True, replace_zeros=True,
                               **kwargs)
            x = [1., 2., 3., 0.0, 5., 6.]
            x1, p = t.fit_transform(data=x, return_proc=True)
            conf = t.config()
            t2 = Transformation.from_config(conf)
            x2 = t2.inverse_transform(data=x1, postprocessor=p)
            np.testing.assert_array_almost_equal(np.array(x), x2.values.reshape(-1,))
        return

    def test_from_config_2d(self):

        for method in ["quantile", "robust", "quantile_normal",
                       "power", "box-cox", "center", "zscore", "scale",
            "mmad", "vast", "pareto",
                       ]:
            kwargs = {}
            if method=="quantile":
                kwargs['n_quantiles'] = 5

            if method == "yeo-johnson":
                kwargs['pre_center'] = True
                kwargs['rescale'] = True

            t = Transformation(method, features=['a', 'b'],
                               treat_negatives=True, replace_zeros=True, **kwargs)
            x = np.random.randint(-2, 30, (10, 3))
            data = pd.DataFrame(x, columns=['a', 'b', 'c'])
            x1, p = t.fit_transform(data=data.copy(), return_proc=True)
            conf = t.config()
            t2 = Transformation.from_config(conf)
            x2 = t2.inverse_transform(data=x1, postprocessor=p)
            np.testing.assert_array_almost_equal(x, x2.values)
        return

    def test_without_fit(self):
        """It is possible to inverse transform some transformations without fit"""
        for m in ['log', 'log2', 'log10', 'sqrt']:
            tr = Transformation(m)
            x = [1, 2, 3]
            x_ = tr.fit_transform(x)
            _x = tr.inverse_transform(x_)

            _x1 = Transformation(m).inverse_transform(x_)
            assert np.allclose(_x, _x1)
        return


class TestTransform(unittest.TestCase):

    def test_minmax(self):

        features = ['tide_cm', 'sal_psu']
        ai_tr = Transformation(features=features)
        sk_tr = MinMaxScaler()
        test_transform(sk_tr, ai_tr, features)

        return

    def test_zscore(self):

        features = ['tide_cm', 'sal_psu']
        ai_tr = Transformation('zscore', features=features)
        sk_tr = StandardScaler()
        test_transform(sk_tr, ai_tr, features)

        return

    def test_robust(self):

        features = ['tide_cm', 'sal_psu']
        ai_tr = Transformation('robust', features=features)
        sk_tr = RobustScaler()
        test_transform(sk_tr, ai_tr, features)

        return

    def test_quantile(self):

        features = ['tide_cm', 'sal_psu']
        ai_tr = Transformation('quantile', features=features, n_quantiles=100)
        sk_tr = QuantileTransformer(n_quantiles=100)
        test_transform(sk_tr, ai_tr, features)

        return

    def test_power(self):

        features = ['tide_cm', 'sal_psu']
        ai_tr = Transformation('power', features=features)
        sk_tr = PowerTransformer()
        test_transform(sk_tr, ai_tr, features)

        return

    def test_yeo_johson(self):

        features = ['tide_cm', 'sal_psu']
        ai_tr = Transformation('yeo-johnson', features=features)
        sk_tr = PowerTransformer(method='yeo-johnson')
        test_transform(sk_tr, ai_tr, features)
        return


if __name__ == "__main__":
    unittest.main()