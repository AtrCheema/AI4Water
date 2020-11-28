import unittest
import numpy as np
import pandas as pd
import os

import site   # so that dl4seq directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

from dl4seq.utils.scalers import Scalers

df = pd.DataFrame(np.concatenate([np.arange(1, 10).reshape(-1, 1), np.arange(1001, 1010).reshape(-1, 1)], axis=1),
                  columns=['data1', 'data2'])

class test_Scalers(unittest.TestCase):

    def run_method(self, method, cols=None):

        cols = ['data1', 'data2'] if cols is None else cols
        print(f"testing: {method} with {cols} features")

        normalized_df1, scaler = Scalers(data=df, method=method, features=cols)('normalize')
        denormalized_df1 = Scalers(data=normalized_df1, features=cols)('denorm', scaler=scaler['scaler'])

        scaler = Scalers(data=df, method=method)
        normalized_df2, scaler_dict = scaler.transform()
        denormalized_df2 = scaler.inverse_transform(data=normalized_df2, key=scaler_dict['key'])

        scaler = Scalers(data=df, method=method)
        normalized_df3, scaler_dict = scaler()
        denormalized_df3 = scaler('denorm', data=normalized_df2, key=scaler_dict['key'])

        scaler = Scalers(data=df)
        normalized_df4, scaler_dict = getattr(scaler, "transform_with_" + method)()
        denormalized_df4 = getattr(scaler, "inverse_transform_with_" + method)(data=normalized_df4, key=scaler_dict['key'])

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

    def test_call_error(self):

        self.assertRaises(ValueError, Scalers(data=df), 'transform')

    def test_get_scaler_from_dict_error(self):
        normalized_df1, scaler = Scalers(data=df)('normalize')
        self.assertRaises(ValueError, Scalers(data=normalized_df1), 'denorm')

    def test_log_scaler_with_feat(self):
        self.run_method("log", cols=["data1"])

    def test_robust_scaler_with_feat(self):
        self.run_method("robust", cols=["data1"])

    def test_minmax_scaler_with_feat(self):
        self.run_method("minmax", cols=["data1"])

    def test_maxabs_scaler_with_feat(self):
        self.run_method("maxabs", cols=["data1"])

    def test_zscore_scaler_with_feat(self):
        self.run_method("minmax", cols=["data1"])

    def test_power_scaler_with_feat(self):
        self.run_method("maxabs", cols=["data1"])

    def test_quantile_scaler_with_feat(self):
        self.run_method("quantile", cols=["data1"])

    def test_log_scaler(self):
        self.run_method("log")

    def test_robust_scaler(self):
        self.run_method("robust")

    def test_minmax_scaler(self):
        self.run_method("minmax")

    def test_maxabs_scaler(self):
        self.run_method("maxabs")

    def test_zscore_scaler(self):
        self.run_method("minmax")

    def test_power_scaler(self):
        self.run_method("maxabs")

    def test_quantile_scaler(self):
        self.run_method("quantile")

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
            trans_df, orig_df = self.do_decomposition(df_len, features,components, run_method, method)
            self.assertEqual(trans_df.shape, orig_df.shape)

    def run_decomposition_with_components(self, method):
        df_len, features, components = 10, 5, 4
        for run_method in [1,2,3]:
            trans_df, orig_df = self.do_decomposition(df_len, features,components, run_method, method)
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
        elif m==2:
            return self.run_decomposition2(data, args, method=method)
        elif m==3:
            return self.run_decomposition3(data, args, method=method)

    def run_decomposition1(self, data, args, method):

        scaler = Scalers(data=data, method=method, **args)
        normalized_df, scaler_dict = scaler.transform()
        denormalized_df = scaler.inverse_transform(data=normalized_df, key=scaler_dict['key'])

        return normalized_df, denormalized_df

    def run_decomposition2(self, data, args, method):

        scaler = Scalers(data=data, **args)
        normalized_df, scaler_dict = getattr(scaler, "transform_with_"+method.lower())()
        denormalized_df = getattr(scaler, "inverse_transform_with_" + method.lower())(data=normalized_df, key=scaler_dict['key'])

        return normalized_df, denormalized_df

    def run_decomposition3(self, data, args, method):

        normalized_df, scaler = Scalers(data=data, method=method, **args)('normalize')
        denormalized_df = Scalers(data=normalized_df, method=method, **args)('denorm', scaler=scaler['scaler'])

        return normalized_df, denormalized_df

    def test_plot_pca3d(self):
        from sklearn import datasets

        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        scaler = Scalers(X, method="pca", n_components=3)
        scaler()
        for dim in ["2d", "3D"]:
            self.run_plot_pca(scaler, y, dim=dim)
        return

    def run_plot_pca(self, scaler, y, dim):
        scaler.plot_pca(target=y, labels=['Setosa', 'Versicolour', 'Virginica'], save=None, dim=dim)
        return



if __name__ == "__main__":
    unittest.main()