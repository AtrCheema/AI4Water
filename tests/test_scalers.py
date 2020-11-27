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
        print("testing: ", method)

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


if __name__ == "__main__":
    unittest.main()