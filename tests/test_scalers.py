import os
import site   # so that AI4Water directory is in path
import unittest
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

import numpy as np
import pandas as pd

from AI4Water.utils.transformations import Transformations
from AI4Water import Model
from AI4Water.utils.datasets import load_u1

df = pd.DataFrame(np.concatenate([np.arange(1, 10).reshape(-1, 1), np.arange(1001, 1010).reshape(-1, 1)], axis=1),
                  columns=['data1', 'data2'])

def build_and_run(transformation, data, inputs, outputs):
    model = Model(data=data,
                  input_features=inputs,
                  output_features=outputs,
                  transformation=transformation,
                  verbosity=0)
    tr_data, sc = model.normalize(model.data, transformation=model.config['transformation'], key='5')

    pred, true = model.denormalize_data(inputs=tr_data[inputs],
                                        true=tr_data[outputs],
                                        predicted=tr_data[outputs],
                                        scaler_key='5',
                                        in_cols=model.in_cols,
                                        out_cols=model.out_cols,
                                        transformation=model.config['transformation'])
    return pred

def run_method1(method,
                cols=None,
                replace_nans=False,
                data=None,
                **kwargs):

    print(f"testing: {method} with {cols} features")

    normalized_df1, scaler = Transformations(data=df if data is None else data,
                                             method=method,
                                             features=cols,
                                             replace_nans=replace_nans,
                                             **kwargs)('Transform', return_key=True)

    denormalized_df1 = Transformations(data=normalized_df1,
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

    scaler = Transformations(data,
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

    scaler = Transformations(data,
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

    scaler = Transformations(data=df if data is None else data,
                             replace_nans=replace_nans, **kwargs)

    normalized_df4, scaler_dict = getattr(scaler, "transform_with_" + method)(return_key=True)
    denormalized_df4 = getattr(scaler, "inverse_transform_with_" + method)(data=normalized_df4, key=scaler_dict['key'])

    return normalized_df4, denormalized_df4

def run_plot_pca(scaler, y, dim):
    scaler.plot_pca(target=y, labels=['Setosa', 'Versicolour', 'Virginica'], save=None, dim=dim)
    return


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

    _, dfo1 = run_method1(method=method, replace_nans=True, data=df3, **kwargs)

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
        print(f"testing: {method} with {cols} features")

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

    # def test_call_error(self):
    #
    #     self.assertRaises(ValueError, Transformations(data=df), 'transform')

    def test_get_scaler_from_dict_error(self):
        normalized_df1, scaler = Transformations(data=df)('transform', return_key=True)
        self.assertRaises(ValueError, Transformations(data=normalized_df1), 'inverse')

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

    def test_pca(self):
        self.run_decomposition(method="pca")

    def test_kpca(self):
        self.run_decomposition(method="kpca")

    def test_ipca(self):
        self.run_decomposition(method="ipca")

    def test_fastica(self):
        self.run_decomposition(method="fastica")

    def test_pca_with_components(self):
        self.run_decomposition_with_components(method="pca")

    def test_kpca_with_components(self):
        self.run_decomposition_with_components(method="kpca")

    def test_ipca_with_components(self):
        self.run_decomposition_with_components(method="ipca")

    def test_fastica_with_components(self):
        self.run_decomposition_with_components(method="fastica")

    def run_decomposition(self, method):
        df_len, features, components = 10, 5, 5
        for run_method in [1,2, 3]:
            _, trans_df, orig_df = self.do_decomposition(df_len, features,components, run_method, method)
            self.assertEqual(trans_df.shape, orig_df.shape)

    def run_decomposition_with_components(self, method):
        df_len, features, components = 10, 5, 4
        for run_method in [1,2,3]:
            _, trans_df, orig_df = self.do_decomposition(df_len, features,components, run_method, method)
            self.assertEqual(trans_df.shape, (df_len,components))
            self.assertEqual(orig_df.shape, (df_len, features))

    def do_decomposition(self, df_len, features, components, m, method):
        print(f"testing {method} with {features} features and {components} components with {m} call method")

        data = pd.DataFrame(np.random.random((df_len, features)),
                          columns=['data' + str(i) for i in range(features)])

        args = {"n_components": components}
        if method.upper() == "KPCA":
            args.update({"fit_inverse_transform": True})

        if m==1:
            return self.run_decomposition1(data, args, method=method)
            #return run_method1(data=data, method=method, **args)
        elif m==2:
            return run_method2(data=data, method=method, **args)
        elif m==3:
            return run_method3(data=data, method=method, **args)

    def run_decomposition1(self, data, args, method):

        scaler = Transformations(data=data, method=method, **args)
        normalized_df, scaler_dict = scaler.transform(return_key=True)
        denormalized_df = scaler.inverse_transform(data=normalized_df, key=scaler_dict['key'])

        return None, normalized_df, denormalized_df

    def test_plot_pca3d(self):
        from sklearn import datasets

        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        scaler = Transformations(X, method="pca", n_components=3)
        scaler()
        for dim in ["2d", "3D"]:
            run_plot_pca(scaler, y, dim=dim)
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

        for i,j in zip(data['out1'], pred):
            self.assertAlmostEqual(i, float(j), 5)

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

        for i,j in zip(data['out1'], pred):
            self.assertAlmostEqual(i, float(j), 5)
        return

    def test_example(self):
        data = load_u1()
        inputs = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
        transformer = Transformations(data=data[inputs], method='minmax', features=['x1', 'x2'])
        new_data = transformer.transform()
        orig_data = transformer.inverse_transform(data=new_data)
        np.allclose(data[inputs].values, orig_data.values)

        return

    # def test_multiple_transformation_multiple_inputs(self):
    #     # TODO
    #     return
    #
    # def test_multile_transformations_multile_outputs(self):
    #     # TODO
    #     return


if __name__ == "__main__":
    unittest.main()