import unittest

import numpy as np
import pandas as pd

from ai4water import Model
from ai4water.utils.datasets import arg_beach

beach_data:pd.DataFrame = arg_beach()

inputs = beach_data.columns.to_list()[0:-1]

def run_plots(model):
    model.plot_train_data()
    model.plot_val_data()
    model.plot_test_data()

    model.plot_train_data(how='hist')
    model.plot_val_data(how='hist')
    model.plot_test_data(how='hist')
    return


class TestPlots(unittest.TestCase):

    def test_basic(self):
        model = Model(
            data=beach_data,
            verbosity=0
        )

        run_plots(model)

    def test_basic_with_2_outputs(self):
        model = Model(
            data=beach_data,
            input_features=beach_data.columns.to_list()[0:-2],
            output_features=beach_data.columns.to_list()[-2:],
            verbosity=0
        )

        run_plots(model)
        return

    def test_multisource(self):

        data = np.arange(int(40 * 4), dtype=np.int32).reshape(-1, 40).transpose()
        df1 = pd.DataFrame(data, columns=['a', 'b', 'c', 'd'],
                                    index=pd.date_range('20110101', periods=40, freq='D'))
        df2 = pd.DataFrame(np.array([5,6]).repeat(40, axis=0).reshape(40, -1), columns=['len', 'dep'],
                           index=pd.date_range('20110101', periods=40, freq='D'))

        model = Model(
            data=[df1, df2],
            input_features=[['a', 'b'], ['len', 'dep']],
            output_features=[['c', 'd'], []],
            lookback=4,
            verbosity=0
        )

        run_plots(model)

        # as dictionary
        model = Model(
            data={'cont_data': df1, 'static_data': df2},
            input_features={'cont_data': ['a', 'b'], 'static_data': ['len', 'dep']},
            output_features={'cont_data': ['c', 'd'], 'static_data': []},
            lookback=4,
            verbosity=0
        )

        run_plots(model)


if __name__ == "__main__":
    unittest.main()

