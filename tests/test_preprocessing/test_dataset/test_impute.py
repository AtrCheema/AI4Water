
import unittest
import os
import site
package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) 
site.addsitedir(package_path)

from ai4water.datasets import busan_beach
from ai4water.preprocessing.dataset import DataSet


beach_data = busan_beach()


class TestImputation(unittest.TestCase):
    """makes sure that DataHandler can interact with Imputation class"""

    def test_fillna(self):
        dh = DataSet(data=beach_data,
                         nan_filler={'method': 'fillna', 'imputer_args': {'method': 'ffill'}}
                         )
        assert dh.data.isna().sum().sum() == 66
        return

    def test_fillna_imputer_args(self):
        dh = DataSet(data=busan_beach(target=['tetx_coppml', 'blaTEM_coppml']),
                         nan_filler={'method': 'fillna', 'imputer_args': {'method': 'ffill'},
                                     'features': ['tetx_coppml']}
                         )
        assert dh.data['tetx_coppml'].isna().sum() == 66
        assert dh.data['blaTEM_coppml'].isna().sum() == 1228
        return

    def test_knnimputer(self):
        dh = DataSet(data=busan_beach(target=['tetx_coppml', 'blaTEM_coppml']),
                         nan_filler={'method': 'KNNImputer', 'imputer_args': {'n_neighbors': 3},
                                     'features': ['tetx_coppml']}
                         )
        assert dh.data['tetx_coppml'].isna().sum() == 0
        assert dh.data['blaTEM_coppml'].isna().sum() == 1228
        return


if __name__ == '__main__':
    unittest.main()