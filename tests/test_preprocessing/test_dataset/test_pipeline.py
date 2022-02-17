
import unittest

import numpy as np
import pandas as pd

from ai4water.preprocessing import DataSet
from ai4water.preprocessing import DataSetPipeline


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


class TestPipeline(unittest.TestCase):

    def test_basic(self):

        ds1 = get_ds(100)
        ds2 =  get_ds(100)

        ds = DataSetPipeline(ds1, ds2)
        call_methods(ds)

        return


    def test_diff_lengths(self):
        ds1 = get_ds(100)
        ds2 =  get_ds(200)

        ds = DataSetPipeline(ds1, ds2)

        call_methods(ds)

        return


    def test_no_val_data(self):
        ds1 = get_ds(100)
        ds2 =  get_ds(100)
        ds3 = get_ds(100, val_fraction=0.0)

        ds = DataSetPipeline(ds1, ds2, ds3)

        call_methods(ds)

        return

    def test_no_test_data(self):
        ds1 = get_ds(100)
        ds2 =  get_ds(100)
        ds3 = get_ds(100, train_fraction=1.0)

        ds = DataSetPipeline(ds1, ds2, ds3)

        call_methods(ds)

        return


if __name__ == "__main__":
    unittest.main()