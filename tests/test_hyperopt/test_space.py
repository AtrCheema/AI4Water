
import unittest

import numpy as np

from ai4water.hyperopt import Real, Integer, Categorical


class TestSpace(unittest.TestCase):

    def test_real_num_samples(self):
        r = Real(low=10, high=100, num_samples=20)
        grit = r.grid
        assert grit.shape == (20,)

    def test_real_steps(self):
        r = Real(low=10, high=100, step=20)
        grit = r.grid
        assert grit.shape == (5,)

    def test_real_grid(self):
        grit = [1,2,3,4,5]
        r = Real(grid=grit)
        np.testing.assert_array_equal(grit, r.grid)

    def test_integer_num_samples(self):
        r = Integer(low=10, high=100, num_samples=20)
        grit = r.grid
        assert len(grit) == 20

    def test_integer_steps(self):
        r = Integer(low=10, high=100, step=20)
        grit = r.grid
        assert len(grit) == 5

    def test_integer_grid(self):
        grit = [1, 2, 3, 4, 5]
        r = Integer(grid=grit)
        np.testing.assert_array_equal(grit, r.grid)

    def test_categorical(self):
        cats = ['a', 'b', 'c']
        c = Categorical(cats)
        assert len(cats) == len(c.grid)


if __name__ == "__main__":
    unittest.main()
