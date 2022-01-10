
import unittest

import numpy as np

from ai4water.hyperopt.utils import to_skopt_space


class Test1(unittest.TestCase):

    def test_to_skopt_space(self):
        s = {'n_estimators': np.random.random((10,))}
        s = to_skopt_space(s)
        return

if __name__ == "__main__":
    unittest.main()