import unittest

import os
import sys
import site   # so that ai4water directory is in path
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from ai4water.utils import tf_losses
from ai4water.postprocessing.SeqMetrics import RegressionMetrics

tf_losses.reset_graph()
_true = np.random.random(10)
pred = np.random.random(10)

t = tf.convert_to_tensor(_true, dtype=tf.float32)
p = tf.convert_to_tensor(pred, dtype=tf.float32)

np_errors = RegressionMetrics(_true, pred)


class test_errors(unittest.TestCase):

    def test_corr_coeff(self):
        self.assertAlmostEqual(np_errors.corr_coeff(), K.eval(tf_losses.corr_coeff(t, p)), 4)

    def test_r2(self):
        self.assertAlmostEqual(np_errors.r2(), K.eval(tf_losses.tf_r2(t, p)), 4)  # TODO why not minus 1.

    def test_nse(self):
        self.assertAlmostEqual(np_errors.nse(), 1.0 - K.eval(tf_losses.tf_nse(t, p)), 4)

    def test_kge(self):
        self.assertAlmostEqual(np_errors.kge(), 1.0 - K.eval(tf_losses.tf_kge(t, p)), 4)

    def test_r2_mod(self):
        self.assertAlmostEqual(np_errors.r2_score(), 1.0 - K.eval(tf_losses.tf_r2_mod(t, p)), 4)

    def nse_beta(self):
        self.assertAlmostEqual(np_errors.nse_beta(), 1.0 - K.eval(tf_losses.tf_nse_beta(t, p)), 4)

    def nse_alpha(self):
        self.assertAlmostEqual(np_errors.nse_alpha(), 1.0 - K.eval(tf_losses.tf_nse_alpha(t, p)), 4)

    def nse_pbias(self):
        self.assertAlmostEqual(np_errors.pbias(), K.eval(tf_losses.pbias(t, p)), 4)


if __name__ == "__main__":
    unittest.main()