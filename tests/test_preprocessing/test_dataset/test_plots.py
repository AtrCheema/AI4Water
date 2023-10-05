
import unittest
import os
import sys
import site
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

import numpy as np
import pandas as pd

from ai4water.datasets import busan_beach
from ai4water.preprocessing import DataSet, DataSetUnion


beach_data:pd.DataFrame = busan_beach()
input_features = beach_data.columns.to_list()[0:-2]
output_features = beach_data.columns.to_list()[-2:]


def run_plots(ds):
    ds.plot_train_data()
    ds.plot_val_data()
    ds.plot_test_data()

    ds.plot_train_data(how='hist')
    ds.plot_val_data(how='hist')
    ds.plot_test_data(how='hist')
    return


class TestPlots(unittest.TestCase):

    def test_basic(self):
        model = DataSet(
            data=beach_data,
            input_features=input_features,
            output_features=output_features,
            verbosity=0)

        run_plots(model)

    def test_basic_with_2_outputs(self):
        model = DataSet(
            data=beach_data,
            input_features=input_features,
            output_features=output_features,
            verbosity=0
        )

        run_plots(model)
        return

    def test_datasetUnion(self):

        data = np.arange(int(40 * 4), dtype=np.int32).reshape(-1, 40).transpose()
        df1 = pd.DataFrame(data, columns=['a', 'b', 'c', 'd'],
                           index=pd.date_range('20110101', periods=40, freq='D'))
        df2 = pd.DataFrame(np.array([5,6]).repeat(40, axis=0).reshape(40, -1),
                           columns=['len', 'dep'],
                           index=pd.date_range('20110101', periods=40, freq='D'))

        ds1 = DataSet(df1, input_features=['a', 'b'], output_features=['c', 'd'],
                      ts_args={"lookback":4})
        ds2 = DataSet(df2, input_features=['len', 'dep'],
                      ts_args={"lookback":4})
        ds = DataSetUnion(ds1, ds2, verbosity=0)

        run_plots(ds)

        # as keyword arguments
        ds = DataSetUnion(cont_data=ds1, static_data=ds2, verbosity=0)

        run_plots(ds)
        return


if __name__ == "__main__":
    unittest.main()

