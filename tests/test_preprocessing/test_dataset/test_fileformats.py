

import os
import site
package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) 
site.addsitedir(package_path)


import unittest

import scipy
import numpy as np

from ai4water.datasets import busan_beach
from ai4water.preprocessing.dataset import DataSet


beach_data = busan_beach()


class TestFileFormats(unittest.TestCase):

    ds = DataSet(beach_data, verbosity=0)
    input_features = ds.input_features
    output_features = ds.output_features
    train_x, train_y = ds.training_data()
    val_x, val_y = ds.validation_data()
    test_x, test_y = ds.test_data()
    train_x_shape, train_y_shape = train_x.shape, train_y.shape
    val_x_shape, val_y_shape = val_x.shape, val_y.shape
    test_x_shape, test_y_shape = test_x.shape, test_y.shape

    def test_csv(self):
        csv_fname = os.path.join(os.getcwd(),  "data.csv")
        beach_data.to_csv(csv_fname)
        self.validate_shapes(csv_fname)
        return
    
    def test_xlsx(self):

        xlsx_fname = os.path.join(os.getcwd(), "data.xlsx")
        beach_data.to_excel(xlsx_fname, engine="xlsxwriter")
        self.validate_shapes(xlsx_fname)

        return

    def test_parquet(self):
        parq_fname = os.path.join(os.getcwd(), "data.parquet")
        beach_data.to_parquet(parq_fname)
        self.validate_shapes(parq_fname)
        return

    def test_feather(self):
        feather_fname = os.path.join(os.getcwd(), "data.feather")
        beach_data.reset_index().to_feather(feather_fname)
        self.validate_shapes(feather_fname)
        return

    def test_xarray(self):
        nc_fname = os.path.join(os.getcwd(),  "data.nc")
        xds = beach_data.to_xarray()
        xds.to_netcdf(nc_fname)
        self.validate_shapes(nc_fname)
        return

    def test_npz(self):
        npz_fname = os.path.join(os.getcwd(), "data.npz")
        np.savez(npz_fname, beach_data.values)
        self.validate_shapes(npz_fname)
        return

    def test_mat(self):
        mat_fname = os.path.join(os.getcwd(),  "data.mat")
        scipy.io.savemat(mat_fname, {'data': beach_data.values})
        self.validate_shapes(mat_fname)
        return

    def validate_shapes(self, fname):

        dh = DataSet(fname,
                    input_features=self.input_features,
                    output_features=self.output_features,
                    verbosity=0)

        train_x, train_y = dh.training_data()
        assert train_x.shape == self.train_x_shape
        assert train_y.shape == self.train_y_shape

        val_x, val_y = dh.validation_data()
        assert val_x.shape == self.val_x_shape
        assert val_y.shape == self.val_y_shape

        test_x, test_y = dh.test_data()
        assert test_x.shape == self.test_x_shape
        assert test_y.shape == self.test_y_shape

        return


if __name__ == '__main__':
    unittest.main()