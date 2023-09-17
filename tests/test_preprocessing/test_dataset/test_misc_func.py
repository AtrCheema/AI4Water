
import unittest
import os
import site
a = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
package_path = os.path.join(os.path.dirname(os.path.dirname(a)))
print(package_path, 'package path')
site.addsitedir(package_path)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ai4water.datasets import busan_beach
from ai4water.preprocessing.dataset import DataSet

from utils import build_and_test_loader, make_df

beach_data = busan_beach()


class TestMiscFunctionalities(unittest.TestCase):

    def test_with_indices_and_nans(self):
        # todo, check with two output columns
        data = beach_data
        train_idx, test_idx = train_test_split(np.arange(len(data.dropna())),
                                               test_size=0.25, random_state=332898)
        out_cols = [list(data.columns)[-1]]
        config = {
            'indices': {"training": train_idx},
            'input_features': list(data.columns)[0:-1],
            'output_features': out_cols,
            'val_fraction': 0.0,
            'train_fraction': 0.7,
            'ts_args': {'lookback': 14},
        }

        build_and_test_loader(data,
                              config,
                              out_cols=out_cols, train_ex=163, val_ex=0, test_ex=55,
                              check_examples=False, save=False)

        return

    def test_with_indices_and_nans_no_test_data(self):
        # todo, check with two output columns
        data = beach_data
        train_idx, test_idx = train_test_split(np.arange(len(data.dropna())),
                                               test_size=0.25, random_state=332898)
        out_cols = [list(data.columns)[-1]]
        config = {
            'indices': {"training": train_idx, 'validation': test_idx},
            'input_features': list(data.columns)[0:-1],
            'output_features': out_cols,
            'val_fraction': 0.2,
            'train_fraction': 0.7,
            'ts_args': {'lookback': 14},
        }

        build_and_test_loader(data,
                              config,
                              out_cols=out_cols, train_ex=163, val_ex=55, test_ex=0,
                              check_examples=False, save=False)

        return

    def test_with_string_index(self):

        data = beach_data
        data.index = [f"ind_{i}" for i in range(len(data))]
        config = {
            'input_features': ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c'],
            'output_features': ['tetx_coppml'],
             'ts_args': {'lookback': 3},
            'split_random': False,
        }

        true_train_y = data['tetx_coppml'].dropna().values[0:106].reshape(-1,1)
        true_val_y = data['tetx_coppml'].dropna().values[106:152].reshape(-1, 1)
        true_test_y = data['tetx_coppml'].dropna().values[152:].reshape(-1, 1)

        build_and_test_loader(data, config, out_cols=['tetx_coppml'],
                              true_train_y=true_train_y,
                              true_val_y=true_val_y,
                              true_test_y=true_test_y,
                              train_ex=106, val_ex=46, test_ex=66)
        return

    def test_with_random_with_transformation_of_features(self):

        data = make_df(100, ['a', 'b', 'c'])
        data['date'] = data.index
        config = {'input_features':['b'],
                  'output_features': ['c'],
                  'ts_args': {'lookback': 5},
                  'split_random': True}

        dh = DataSet(data, verbosity=0, **config)

        x,y = dh.training_data()
        assert x is not None
        return


    def test_random_with_intervals(self):
        data = np.random.randint(0, 1000, (40560, 14))
        input_features = [f'input_{i}' for i in range(13)]
        output_features = ['NDX']
        data = pd.DataFrame(data, columns=input_features+output_features)

        out = data["NDX"]

        # put four chunks of missing intervals
        intervals = [(100, 200), (1000, 8000), (10000, 31000)]

        for interval in intervals:
            st, en = interval[0], interval[1]
            out[st:en] = np.nan

        data["NDX"] = out

        config = {
            'input_features': input_features,
            'output_features': output_features,
            'ts_args': {'lookback': 5},
            'split_random': True,
            'intervals': [(0, 99), (200, 999), (8000, 9999), (31000, 40560)],
        }

        build_and_test_loader(data, config, out_cols=output_features,
                              train_ex=6095, val_ex=2613, test_ex=3733,
                              assert_uniqueness=False,
                              save=False)

        return


    def test_AI4WaterDataSets(self):
        config = {'intervals': [("20000101", "20011231")],
                  'input_features': ['precipitation_AWAP',
                                     'evap_pan_SILO'],
                  'output_features': ['streamflow_MLd_inclInfilled'],
                  'dataset_args': {'stations': 1}
        }

        build_and_test_loader('CAMELS_AUS', config=config,
                              out_cols=['streamflow_MLd_inclInfilled'],
                              train_ex=357, val_ex=154, test_ex=220,
                              assert_uniqueness=False,
                              save=False)

        return

    def test_check_for_cls(self):
        data = pd.DataFrame(np.random.randint(0, 10, size=(100, 3)),
                            columns=['a', 'b', 'c'])
        data['date'] = data.index
        config = {'input_features':['b'],
                  'output_features': ['c'],
                  'split_random': True,
                  'mode': 'classification',
                  }

        ds = DataSet(data, verbosity=0, **config)
        x, y = ds.training_data()
        assert 'int' in y.dtype.name
        return


if __name__ == "__main__":
    unittest.main()