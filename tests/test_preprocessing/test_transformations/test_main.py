import site   # so that ai4water directory is in path
import unittest
import os
import sys
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

import numpy as np
import pandas as pd

from ai4water.preprocessing.transformations import Transformation
from ai4water.tf_attributes import tf
from ai4water.datasets import arg_beach


if 230 <= int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 250:
    from ai4water.functional import Model
    print(f"Switching to functional API due to tensorflow version {tf.__version__}")
else:
    from ai4water import Model


df = pd.DataFrame(np.concatenate([np.arange(1, 10).reshape(-1, 1), np.arange(1001, 1010).reshape(-1, 1)], axis=1),
                  columns=['data1', 'data2'])


def build_and_run(transformation, data, inputs, outputs):

    model = Model(data=data,
                  input_features=inputs,
                  output_features=outputs,
                  transformation=transformation,
                  verbosity=0)
    x, y = model.training_data(key='junk')

    pred, pred = model.inverse_transform(y, y, key='junk')

    pred, index = model.dh.deindexify(pred, key='junk')
    pred = pd.DataFrame(pred.reshape(len(pred), model.num_outs), columns=outputs, index=index).sort_index()
    return pred


def run_method1(method,
                cols=None,
                replace_nans=False,
                data=None,
                **kwargs):

    #print(f"testing: {method} with {cols} features")

    normalized_df1, scaler = Transformation(data=df if data is None else data,
                                             method=method,
                                             features=cols,
                                             replace_nans=replace_nans,
                                             **kwargs)('Transform', return_key=True)

    denormalized_df1 = Transformation(data=normalized_df1,
                                       features=cols,
                                       replace_nans=replace_nans)('inverse', scaler=scaler['scaler'])
    return normalized_df1, denormalized_df1


def run_method2(method,
                replace_nans=False,
                data=None,
                index=None,
                **kwargs):

    data = df if data is None else data

    if index:
        data.index = pd.date_range("20110101", periods=len(data), freq="D")

    scaler = Transformation(data,
                             replace_nans=replace_nans,
                             method=method,
                             **kwargs)

    normalized_df, scaler_dict = scaler.transform(return_key=True)

    denormalized_df = scaler.inverse_transform(data=normalized_df, key=scaler_dict['key'])
    return data, normalized_df, denormalized_df


def run_method3(method,
                replace_nans=False,
                data=None,
                index=None,
                **kwargs):

    data = df if data is None else data

    if index:
        data.index = pd.date_range("20110101", periods=len(data), freq="D")

    scaler = Transformation(data,
                             replace_nans=replace_nans,
                             method=method,
                             **kwargs)

    normalized_df3, scaler_dict = scaler(return_key=True)
    denormalized_df3 = scaler('inverse', data=normalized_df3, key=scaler_dict['key'])

    return data, normalized_df3, denormalized_df3


def run_method4(method,
                replace_nans=False,
                index=None,
                data=None, **kwargs):

    scaler = Transformation(data=df if data is None else data,
                             replace_nans=replace_nans, **kwargs)

    normalized_df4, scaler_dict = getattr(scaler, "transform_with_" + method)(return_key=True)
    denormalized_df4 = getattr(scaler, "inverse_transform_with_" + method)(data=normalized_df4, key=scaler_dict['key'])

    return normalized_df4, denormalized_df4


def run_log_methods(method="log", index=None, insert_nans=True, insert_zeros=False, assert_equality=True,
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

    kwargs = {}
    if insert_zeros:
        kwargs['replace_zeros'] = True

    cols = ['data1', 'data2', 'data3', 'data4']

    if index is not None:
        index = pd.date_range("20110101", periods=len(a), freq="D")

    df3 = pd.DataFrame(a, columns=cols, index=index)

    _, _ = run_method1(method=method, replace_nans=True, data=df3, **kwargs)

    _, _, dfo2 = run_method2(method=method, replace_nans=True, data=df3, **kwargs)

    _, _, dfo3 = run_method3(method=method, replace_nans=True, data=df3, **kwargs)

    _, dfo4 = run_method4(method=method, replace_nans=True, data=df3, **kwargs)

    if assert_equality:
        #assert np.allclose(df3, dfo1, equal_nan=True)
        assert np.allclose(df3, dfo2, equal_nan=True)
        assert np.allclose(df3, dfo3, equal_nan=True)
        assert np.allclose(df3, dfo4, equal_nan=True)
    return


class test_Scalers(unittest.TestCase):

    def run_method(self, method, cols=None, index=None, assert_equality=False):

        cols = ['data1', 'data2'] if cols is None else cols
        #print(f"testing: {method} with {cols} features")

        normalized_df1, denormalized_df1 = run_method1(method, cols)

        orig_data2, normalized_df2, denormalized_df2 = run_method2(method, index=index, features=cols)

        orig_data3, normalized_df3, denormalized_df3 = run_method3(method, index=index)

        normalized_df4, denormalized_df4 = run_method4(method, index=index)

        if assert_equality:
            assert np.allclose(orig_data2, denormalized_df2)
            #assert np.allclose(orig_data3, normalized_df3)  # todo

        if len(cols) < 2:
            self.check_features(denormalized_df1)
        else:
            for i,j,k,l in zip(normalized_df1[cols].values, normalized_df2[cols].values, normalized_df3[cols].values, normalized_df4[cols].values):
                for x in [0, 1]:
                    self.assertEqual(int(i[x]), int(j[x]))
                    self.assertEqual(int(j[x]), int(k[x]))
                    self.assertEqual(int(k[x]), int(l[x]))

            for a,i,j,k,l in zip(df.values, denormalized_df1[cols].values, denormalized_df2[cols].values, denormalized_df3[cols].values, denormalized_df4[cols].values):
                for x in [0, 1]:
                    self.assertEqual(int(round(a[x])), int(round(j[x])))
                    self.assertEqual(int(round(i[x])), int(round(j[x])))
                    self.assertEqual(int(round(j[x])), int(round(k[x])))
                    self.assertEqual(int(round(k[x])), int(round(l[x])))

    def check_features(self, denorm):

        for idx, v in enumerate(denorm['data2']):
            self.assertEqual(v, 1001 + idx)

    def test_get_scaler_from_dict_error(self):
        normalized_df1, _ = Transformation(data=df)('transform', return_key=True)
        self.assertRaises(ValueError, Transformation(data=normalized_df1), 'inverse')

    def test_log_scaler_with_feat(self):
        self.run_method("log", cols=["data1"])

    def test_robust_scaler_with_feat(self):
        self.run_method("robust", cols=["data1"], assert_equality=True)

    def test_minmax_scaler_with_feat(self):
        self.run_method("minmax", cols=["data1"], assert_equality=True)

    def test_minmax_scaler_with_feat_and_index(self):
        self.run_method("minmax", cols=["data1"], index=True, assert_equality=True)

    def test_maxabs_scaler_with_feat(self):
        self.run_method("maxabs", cols=["data1"], assert_equality=True)

    def test_zscore_scaler_with_feat(self):
        self.run_method("minmax", cols=["data1"], assert_equality=True)

    def test_power_scaler_with_feat(self):
        self.run_method("maxabs", cols=["data1"], assert_equality=True)

    def test_quantile_scaler_with_feat(self):
        self.run_method("quantile", cols=["data1"], assert_equality=True)

    def test_log_scaler(self):
        self.run_method("log", assert_equality=True)

    def test_log10_scaler(self):
        self.run_method("log10", assert_equality=True)

    def test_log2_scaler(self):
        self.run_method("log2", assert_equality=True)

    def test_robust_scaler(self):
        self.run_method("robust", assert_equality=True)

    def test_minmax_scaler(self):
        self.run_method("minmax", assert_equality=True)

    def test_maxabs_scaler(self):
        self.run_method("maxabs", assert_equality=True)

    def test_zscore_scaler(self):
        self.run_method("minmax", assert_equality=True)

    def test_power_scaler(self):
        self.run_method("maxabs", assert_equality=True)

    def test_quantile_scaler(self):
        self.run_method("quantile", assert_equality=True)

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
        run_log_methods("log", True, insert_nans=True, insert_zeros=True, insert_ones=True)
        return

    def test_zero_log10(self):
        run_log_methods("log10", True, insert_nans=True, insert_zeros=True)
        return

    def test_zero_one_log10(self):
        run_log_methods("log10", True, insert_nans=True, insert_zeros=True, insert_ones=True)
        return


    def test_zero_log2(self):
        run_log_methods("log2", True, insert_nans=True, insert_zeros=True)
        return

    def test_zero_one_log2(self):
        run_log_methods("log2", True, insert_nans=True, insert_zeros=True, insert_ones=True)
        return

    def test_multiple_transformations(self):
        """Test when we want to apply multiple transformations on one or more features"""
        inputs = ['in1', 'inp1']
        outputs = ['out1']

        data = pd.DataFrame(np.random.random((100, 3)), columns=inputs+outputs)

        transformation = [
            {"method": "log", "replace_nans": True, "replace_zeros": True, "features": outputs},
            {"method": "minmax", "features": inputs + outputs}
                          ]

        pred = build_and_run(transformation, data, inputs, outputs)

        for i in pred.index:
            assert np.allclose(data['out1'].loc[i], pred['out1'].loc[i])

        return

    def test_multiple_same_transformations(self):
        """Test when we want to apply multiple transformations on one or more features"""
        inputs = ['in1', 'inp1']
        outputs = ['out1']

        data = pd.DataFrame(np.random.random((100, 3)), columns=inputs+outputs)

        transformation = [
            {"method": "robust", "features": outputs},
            {"method": "robust", "features": inputs + outputs}
                          ]
        pred = build_and_run(transformation, data, inputs,outputs)

        for i in pred.index:
            assert np.allclose(data['out1'].loc[i], pred['out1'].loc[i])

        return

    def test_multiple_same_transformations_mutile_outputs(self):
        """Test when we want to apply multiple transformations on one or more features"""
        inputs = ['in1', 'inp1']
        outputs = ['out1', 'out2']

        data = pd.DataFrame(np.random.random((100, 4)), columns=inputs+outputs)

        transformation = [
            {"method": "robust", "features": outputs},
            {"method": "robust", "features": inputs + outputs}
                          ]
        pred = build_and_run(transformation, data, inputs,outputs)

        for i in pred.index:
            assert np.allclose(data['out1'].loc[i], pred['out1'].loc[i])
            assert np.allclose(data['out2'].loc[i], pred['out2'].loc[i])

        return

    def test_example(self):
        data = arg_beach()
        inputs = ['pcp6_mm', 'pcp12_mm', 'wind_dir_deg', 'wind_speed_mps', 'air_p_hpa']
        transformer = Transformation(data=data[inputs], method='minmax', features=['pcp6_mm', 'pcp12_mm'])
        new_data = transformer.transform()
        orig_data = transformer.inverse_transform(data=new_data)
        np.allclose(data[inputs].values, orig_data.values)

        return

    def test_negative(self):
        for m in ["log", "log2", "log10", "minmax", "zscore", "robust", "quantile", "power"]:
            x = [1.0, 2.0, -3.0, 4.0]
            tr = Transformation(x, method=m, treat_negatives=True)
            xtr = tr.transform()
            _x = tr.inverse_transform(data=xtr)
            np.testing.assert_array_almost_equal(x, _x.values.reshape(-1,))

        for m in ["log", "log2", "log10", "minmax", "zscore", "robust", "quantile", "power"]:
            x1 = [1.0, -2.0, 0.0, 4.0]
            df1 = pd.DataFrame(np.column_stack([x, x1]))
            tr = Transformation(df1, method=m, treat_negatives=True, replace_zeros=True)
            dft = tr.transform()
            _df = tr.inverse_transform(data=dft)
            np.testing.assert_array_almost_equal(df1.values, _df.values)

        return


if __name__ == "__main__":
    unittest.main()