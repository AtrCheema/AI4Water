import os
import unittest
import site   # so that dl4seq directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

from dl4seq.utils.TSErrors import FindErrors
from dl4seq.utils.TSErrors.utils import plot_errors

import numpy as np


t = np.random.random((20, 1))
p = np.random.random((20, 1))

er = FindErrors(t, p)

all_errors = er.calculate_all()


class test_errors(unittest.TestCase):

    def test_radial_pots(self):
        plot_errors(all_errors)
        return 

    def test_calculate_all(self):
        assert len(all_errors) > 1.0
        return

    def test_stats(self):
        s = er.stats()
        assert len(s) > 1.0
        return

    def test_mrae(self):
        assert er.mare() * 100.0 == er.mape()
        return

    def test_mare(self):
        # https://support.numxl.com/hc/en-us/articles/115001223363-MRAE-Mean-Relative-Absolute-Error
        data = np.array(
            [[-2.9, 	-2.95],
            [-2.83, 	-2.7],
            [-0.95, 	-1.00],
            [-0.88, 	-0.68],
            [1.21 ,	1.50],
            [-1.67, 	-1.00],
            [0.83, 	0.90],
            [-0.27, 	-0.37],
            [1.36, 	1.26],
            [-0.34, 	-0.54],
            [0.48, 	0.58],
            [-2.83, 	-2.13],
            [-0.95, 	-0.75],
            [-0.88, 	-0.89],
            [1.21, 	1.25],
            [-1.67, 	-1.65],
            [-2.99, 	-3.20],
            [1.24, 	1.29],
            [0.64, 	0.60]]
        )
        errs = FindErrors(data[:, 0], data[:, 1])
        np.testing.assert_almost_equal(0.348, errs.mrae(), 2)
        assert errs.mare() * 100.0 == errs.mape()
        return

if __name__ == "__main__":
    unittest.main()