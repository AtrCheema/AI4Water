
import unittest

import numpy as np

from ai4water.hyperopt.utils import to_skopt_space
from ai4water.hyperopt.utils import plot_evaluations
from ai4water.hyperopt.utils import plot_hyperparameters


class OptimizeResult:
    n = 10

    func_vals = np.random.random(n)
    space = to_skopt_space({'n_estimators': np.random.randint((10,)),
                            'lr': np.random.random(10)})
    x = []
    x_iters = []


class Test1(unittest.TestCase):

    def test_to_skopt_space(self):
        s = {'n_estimators': np.random.random((10,))}
        s = to_skopt_space(s)
        return


class TestPlot_evaluations(unittest.TestCase):

    show = False

    def test_single_categorical(self):
        return

    def test_multiple_categorical(self):
        return

    def test_sinlge_real(self):
        return

    def test_multiple_real(self):
        return
    def test_categorical_and_real(self):
        return


class TestPlot_hyperparameters(unittest.TestCase):

    show = False

    def test_single_categorical(self):
        return

    def test_multiple_categorical(self):
        return

    def test_sinlge_real(self):
        return

    def test_multiple_real(self):
        return
    def test_categorical_and_real(self):
        return



if __name__ == "__main__":
    unittest.main()