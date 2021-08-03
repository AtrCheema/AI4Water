import unittest

import numpy as np
import pandas as pd

np.random.seed(313)

from ai4water.utils.imputation import Imputation

df = pd.DataFrame(np.random.random((10, 2)), columns=['a', 'b'])
df.iloc[2, 0] = np.nan
df.iloc[3, 1] = np.nan
df.iloc[0, 0] = np.nan
df.iloc[-1, 1] = np.nan


def get_df_with_nans(n=1000, inputs=True, outputs=False, frac=0.8, output_cols=None, input_cols=None):
    np.random.seed(313)

    if output_cols is None:
        output_cols=['out1']
    if input_cols is None:
        input_cols = ['in1', 'in2']

    cols=[]
    if inputs:
        cols += input_cols
    if outputs:
        cols += output_cols

    df = pd.DataFrame(np.random.random((n, len(input_cols) + len(output_cols))), columns=input_cols + output_cols)
    for col in cols:
        df.loc[df.sample(frac=frac).index, col] = np.nan

    return df

class TestPandasSKLearn(unittest.TestCase):

    def test_fillna_all_features(self):
        imputer = Imputation(df, 'fillna', imputer_args={'method': 'ffill'})
        ndf = imputer()
        assert df.isna().sum().sum() == 4
        assert ndf.isna().sum().sum() == 1
        return

    def test_fillna_features(self):
        imputer = Imputation(df, 'fillna', features=['a'], imputer_args={'method': 'ffill'})
        ndf = imputer()
        assert df.isna().sum().sum() == 4
        assert ndf.isna().sum().sum() == 3
        return

    def test_fillna_bfill_features(self):
        imputer = Imputation(df, 'fillna', features=['a'], imputer_args={'method': 'bfill'})
        ndf = imputer()
        assert df.isna().sum().sum() == 4
        assert ndf.isna().sum().sum() == 2
        return

    def test_fillna_bfill_ffill(self):
        imputer = Imputation(df, 'fillna', imputer_args={'method': 'bfill'})
        ndf = imputer()
        assert ndf.isna().sum().sum() == 1
        ndf = imputer(method='ffill')
        assert ndf.isna().sum().sum() == 0
        assert df.isna().sum().sum() == 4
        return

    def test_list(self):
        imputer = Imputation([1,3,np.nan,  np.nan, 9, np.nan, 11])
        ndf = imputer()
        assert sum(ndf) == 42.0
        assert isinstance(ndf, list)

    def test_array(self):
        imputer = Imputation(np.array([1,3,np.nan,  np.nan, 9, np.nan, 11]))
        ndf = imputer()
        assert sum(ndf) == 42.0
        assert isinstance(ndf, np.ndarray)

    def test_interpolate(self):
        imputer = Imputation(df, 'interpolate', imputer_args={'method': 'spline', 'order': 2})
        ndf = imputer()
        assert ndf.isna().sum().sum() == 1
        ndf = imputer(method='bfill')
        assert ndf.isna().sum().sum() == 0
        assert df.isna().sum().sum() == 4

    def test_knnimputer(self):
        imputer = Imputation(df, 'KNNIMPUTER')
        ndf = imputer()
        assert ndf.isna().sum().sum() == 0

    def test_knn_with_features(self):
        imputer = Imputation(df, 'KNNIMPUTER', features=['a'], imputer_args={'n_neighbors': 1})
        ndf = imputer()
        assert df.isna().sum().sum() == 4
        assert ndf.isna().sum().sum() == 2
        return

    def test_raise_error(self):
        imputer = Imputation(df, 'fancyimpute')
        self.assertRaises(NotImplementedError, imputer)

    def test_ffill(self):
        """Test that filling nan by ffill method works"""
        orig_df = get_df_with_nans()
        imputer = Imputation(data=orig_df, method='fillna', imputer_args={'method': 'ffill'})
        imputed_df = imputer()
        self.assertAlmostEqual(sum(imputed_df.values[2:8, 1]), 0.1724 * 6, 4)

        return

    def test_interpolate_cubic(self):
        """Test that fill nan by interpolating using cubic method works."""
        orig_df = get_df_with_nans()
        imputer = Imputation(data=orig_df, method='interpolate', imputer_args={'method': 'cubic'})
        imputed_df = imputer()
        self.assertAlmostEqual(imputed_df.values[8, 0], 0.6530285703060589)

        return

    def test_knn_imputation(self):
        """Test that knn imputation works seamlessly"""
        orig_df = get_df_with_nans(frac=0.5)
        imputer = Imputation(data=orig_df, method='KNNImputer', imputer_args={'n_neighbors': 3})
        imputed_df = imputer()
        self.assertEqual(sum(imputed_df.isna().sum()),  0)

        return

if __name__ == "__main__":

    unittest.main()