import os.path
import unittest

import numpy as np
import pandas as pd

from ai4water.preprocessing import DataSet

from utils import TestAllCases


class TestMultiCases(unittest.TestCase):

    def test_two_inp_1_out_1_lookback(self):
        TestAllCases(input_features = ['a', 'b'],
                     output_features=['c'], lookback=1,
                     allow_nan_labels=2)
        return

    def test_2inp_1out_lookback3(self):
        # testing single dataframe with single output and multiple inputs
        TestAllCases(input_features = ['a', 'b'],
                   output_features=['c'], allow_nan_labels=2)
        return

    def test_1inp_2out(self):
        DataSet._from_h5 = False
        #  testing single dataframe with multiple output and sing inputs
        TestAllCases(input_features = ['a'],
                    output_features=['b', 'c'], allow_nan_labels=1)
        return

    def test_no_output(self):
        DataSet._from_h5 = False
        # testing single dataframe with all inputs and not output
        TestAllCases(input_features = ['a', 'b', 'c'],
                    output_features=None)
        return

    def test_from_config(self):
        df = pd.DataFrame(np.random.random((100, 10)),
                          columns=[f"Feat_{i}" for i in range(10)])
        ds = DataSet(df, save=True, verbosity=0)
        train_x, train_y = ds.training_data()
        val_x, val_y = ds.validation_data()
        test_x, test_y = ds.test_data()

        h5_path = os.path.join(os.getcwd(), "data.h5")
        new_ds = DataSet.from_h5(h5_path)
        train_x1, train_y1 = new_ds.training_data()
        val_x1, val_y1 = new_ds.validation_data()
        test_x1, test_y1 = new_ds.test_data()

        np.allclose(train_x, train_x1)
        np.allclose(val_x, val_x1)
        np.allclose(test_x, test_x1)

        np.allclose(train_y, train_y1)
        np.allclose(val_y, val_y1)
        np.allclose(test_y, test_y1)

        return

    def test_plot_prepared_data(self):
        DataSet._from_h5 = False
        df = pd.DataFrame(np.random.random((100, 10)),
                          columns=[f"Feat_{i}" for i in range(10)])
        ds = DataSet(df, verbosity=0)
        ds.plot_train_data()
        ds.plot_val_data()
        ds.plot_test_data()
        return


if __name__ == "__main__":
    unittest.main()