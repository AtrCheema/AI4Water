
import os
import site   # so that ai4water directory is in path
cwd = os.path.dirname(os.path.abspath(__file__))
ai4_dir = os.path.dirname(os.path.dirname(cwd))
site.addsitedir(ai4_dir)

import unittest
import random

import numpy as np

from ai4water.utils.utils import prepare_data


data2 = np.arange(int(50 * 5)).reshape(-1, 50).transpose()


class TestPrepareData(unittest.TestCase):

    def test_prepare_data0a(self):
        # vanilla case of time series forecasting with 365 lookback and num_outputs 1
        n = 1000
        lookback_ = 365
        data = np.arange(int(n * 5)).reshape(-1, n).transpose()
        x, _, y = prepare_data(data,
                               num_outputs=1,
                               lookback=lookback_,
                               forecast_step=1)
        # first target value is same as last value value at 365th row of original data
        assert data[lookback_][-1] == y[0]
        self.assertEqual(len(x), len(y))
        self.assertAlmostEqual(y[0].sum(), 4365, 6)
        return

    def test_prepare_data0(self):
        # vanilla case of time series forecasting with num_outputs = 2
        x, _, y = prepare_data(data2,
                               num_outputs=2,
                               lookback=4,
                               forecast_step=1)
        self.assertEqual(len(x), len(y))
        self.assertAlmostEqual(y[0].sum(), 358.0, 6)
        return

    def test_prepare_data1(self):
        # multi_step ahead at multiple horizons with known future inputs
        x, prevy, y = prepare_data(data2,
                                   num_outputs=2,
                                   lookback=4,
                                   forecast_len=3,
                                   forecast_step=1,
                                   known_future_inputs=True)
        self.assertEqual(len(x), len(y))
        self.assertAlmostEqual(y[0].sum(), 1080.0, 6)
        return

    def test_prepare_data2(self):
        # multi_step ahead at multiple horizons without known_future_inputs
        x, prevy, y = prepare_data(data2,
                                   num_outputs=2,
                                   lookback=4,
                                   forecast_len=3,
                                   forecast_step=1,
                                   known_future_inputs=False)
        self.assertEqual(len(x), len(y))
        self.assertAlmostEqual(y[0].sum(), 1080.0, 6)
        return

    def test_prepare_data3(self):
        # multistep ahead at single horizon
        x, prevy, y = prepare_data(data2,
                                   num_outputs=2,
                                   lookback=4,
                                   forecast_len=1,
                                   forecast_step=3)
        self.assertEqual(len(x), len(y))
        self.assertAlmostEqual(y[0].sum(), 362.0, 6)
        return

    def test_prepare_data4(self):
        # multi output, with strides in input, multi step ahead at multiple horizons
        # without future inputs
        x, prevy, label = prepare_data(data2,
                                       num_outputs=2,
                                       lookback=4,
                                       input_steps=2,
                                       forecast_step=2,
                                       forecast_len=4)
        self.assertEqual(len(x), len(label))
        self.assertEqual(x.shape, (38, 4, 3))
        self.assertEqual(label.shape, (38, 2, 4))
        self.assertTrue(np.allclose(label[0], np.array([[158., 159., 160., 161.],
                                                        [208., 209., 210., 211.]])))
        return

    def test_prepare_data_no_outputs(self):
        """Test when all the columns are used as inputs and thus make_3d_batches does
        not produce any label data."""
        exs = 100
        d = np.arange(int(exs * 5)).reshape(-1, exs).transpose()
        x, prevy, label = prepare_data(d, num_inputs=5, lookback=4, input_steps=2,
                                       forecast_step=2,
                                       forecast_len=4)
        self.assertEqual(len(x), len(label))
        self.assertEqual(label.sum(), 0.0)
        return

    def test_prepare_data_with_mask(self):
        # mask -99 values
        data = np.arange(int(50 * 5)).reshape(-1, 50).transpose()
        idx = random.choices(np.arange(49), k=20)
        data[idx, -1] = -99
        x, prevy, y = prepare_data(data, num_outputs=1, lookback=4,
                                   mask=-99)
        self.assertEqual(len(x), len(y))
        self.assertGreater(len(data) - len(idx) + 4, len(x))
        return

    def test_prepare_data_with_mask1(self):
        # mask np.nan values
        data = np.arange(int(50 * 5), dtype=np.float32).reshape(-1, 50).transpose()
        idx = [9, 14, 24, 36, 36, 43, 0, 3, 24, 11, 48, 25, 46, 40, 42, 2, 42, 37, 2, 38]
        data[idx, -1] = np.nan
        lookback_ = 4
        num_outputs = 1
        x, prevy, y = prepare_data(data, num_outputs=num_outputs,
                                   lookback=lookback_,
                                   mask=np.nan)
        self.assertEqual(x.shape[1], lookback_)
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(x), 33)
        self.assertEqual(y.shape[1], num_outputs)
        return

    def test_multivariate_no_covariates(self):
        # number of input and output features are equal/same

        exs = 100
        lookback_ = 4
        num_inputs = 5
        num_outputs = 5
        forecast_len = 4
        d = np.arange(int(exs * 5)).reshape(-1, exs).transpose()
        x, prevy, label = prepare_data(d, num_inputs=num_inputs,
                                       num_outputs=num_outputs,
                                       lookback=lookback_,
                                       input_steps=2,
                                       forecast_step=2,
                                       forecast_len=forecast_len)

        self.assertEqual(len(x), len(label))
        self.assertEqual(x.shape[1], lookback_)
        self.assertEqual(x.shape[2], num_inputs)
        self.assertEqual(label.shape[1], num_outputs)
        self.assertEqual(label.shape[2], forecast_len)  # forecast_len

        return

    def test_output_steps(self):
        # output_steps are > 1
        exs = 100
        lookback_ = 4
        num_inputs = 5
        num_outputs = 5
        forecast_len = 4
        d = np.arange(int(exs * 5)).reshape(-1, exs).transpose()
        x, prevy, label = prepare_data(d, num_inputs=num_inputs,
                                       num_outputs=num_outputs,
                                       lookback=lookback_,
                                       input_steps=2,
                                       forecast_step=2,
                                       output_steps=2,
                                       forecast_len=forecast_len)
        self.assertEqual(len(x), len(label))
        self.assertEqual(x.shape[1], lookback_)
        self.assertEqual(x.shape[2], num_inputs)
        self.assertEqual(label.shape[1], num_outputs)
        self.assertEqual(label.shape[2], forecast_len)  # forecast_len
        self.assertEqual(label.shape[0], 84)

        return


if __name__ == "__main__":
    unittest.main()