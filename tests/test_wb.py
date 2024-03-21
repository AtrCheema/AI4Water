
# add parent directory into path
import os
import site
site.addsitedir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest import TestCase

import numpy as np

from ai4water._wb import WB


class TestWB(TestCase):

    def test_log_loss_curve(self):

        wb = WB(None, {'name': 'test_log_loss_curve', 'project':'test'})
        h = {'loss': [0.1, 0.2, 0.3], 'val_loss': [0.2, np.nan, 0.4]}
        wb.log_loss_curve(h)

        return

    def test_log_loss_curve_all_nan(self):

        wb = WB(None, {'name': 'test_log_loss_curve', 'project':'test'})
        h = {'loss': [np.nan, np.nan, np.nan], 'val_loss': [np.nan, np.nan, np.nan]}
        wb.log_loss_curve(h)

        return

if __name__ == '__main__':
    unittest.main()