
import unittest

import numpy as np
import pandas as pd

from ai4water.preprocessing import DataSet, DataSetUnion


def call_methods(ds):
    x,y = ds.training_data()
    x, y = ds.validation_data()
    x, y = ds.test_data()
    return

def get_df(num_exs):
    df = pd.DataFrame(np.random.random((num_exs, 10)),
                      columns=[f"Feat_{i}" for i in range(10)])
    return df

def get_ds(num_exs, **kwargs):
    df = get_df(num_exs)
    return DataSet(df, verbosity=0, **kwargs)

class TestUnion(unittest.TestCase):

    def test_basic(self):
        ds1 = get_ds(100)
        ds2 = get_ds(100+2, ts_args={'lookback': 3})
        ds = DataSetUnion(ds1, ds2)
        call_methods(ds)
        return

    def test_as_dict(self):
        ds1 = get_ds(100)
        ds2 = get_ds(100+2, ts_args={'lookback': 3})
        ds = DataSetUnion(tab_input=ds1, img_input=ds2)
        call_methods(ds)
        return

    def test_no_val_data(self):
        ds1 = get_ds(100, val_fraction=0.0)
        ds2 = get_ds(100+2, ts_args={'lookback': 3}, val_fraction=0.0)
        ds = DataSetUnion(tab_input=ds1, img_input=ds2)
        call_methods(ds)
        return

    def test_no_test_data(self):
        ds1 = get_ds(100, train_fraction=1.0)
        ds2 = get_ds(100+2, ts_args={'lookback': 3}, val_fraction=0.0)
        ds = DataSetUnion(tab_input=ds1, img_input=ds2)
        call_methods(ds)
        return

    def test_no_y(self):
        """one of the datasets does not have y data"""
        ds1 = get_ds(100)
        ds2 = get_ds(100, input_features=[f"Feat_{i}" for i in range(10)])
        ds = DataSetUnion(ds1, ds2)
        call_methods(ds)
        return

    def test_stack_y(self):
        df = get_df(100)
        zeros = pd.DataFrame(np.zeros((2, df.shape[1])), columns=df.columns)
        df2 = df.copy()
        df2 = pd.concat([zeros, df2])
        ds1 = DataSet(df, verbosity=0)
        ds2 = DataSet(df2, ts_args={"lookback": 3}, verbosity=0)
        ds = DataSetUnion(ds1, ds2, stack_y=True)
        call_methods(ds)

    def test_stack_y_no_y(self):
        df = get_df(100)
        zeros = pd.DataFrame(np.zeros((2, df.shape[1])), columns=df.columns)
        df2 = df.copy()
        df2 = pd.concat([zeros, df2])
        ds1 = DataSet(df, verbosity=0)
        ds2 = DataSet(df2, ts_args={"lookback": 3}, verbosity=0,
                      input_features=[f"Feat_{i}" for i in range(10)])
        ds = DataSetUnion(ds1, ds2, stack_y=True)
        call_methods(ds)
        return

    def test_iterator(self):
        ds1 = get_ds(100)
        ds2 = get_ds(100+2, ts_args={'lookback': 3})
        ds = DataSetUnion(ds1, ds2)
        for _ds in ds:
            assert isinstance(_ds, DataSet)
        return

    def test_getitem(self):
        ds1 = get_ds(100)
        ds2 = get_ds(100+2, ts_args={'lookback': 3})
        ds = DataSetUnion(ds1, ds2)
        assert isinstance(ds[0], DataSet)
        return

if __name__ == "__main__":
    unittest.main()