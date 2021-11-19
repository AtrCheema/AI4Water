import unittest

import numpy as np

from ai4water import Model
from ai4water.datasets import arg_beach
from ai4water.postprocessing.explain import PermutationImportance


class TestPermImportance(unittest.TestCase):

    def test_one_2d_input(self):
        model = Model(model="XGBRegressor", verbosity=0,
                      data=arg_beach())
        model.fit()
        x_val, y_val = model.validation_data()

        pimp = PermutationImportance(
            model.predict,
            x_val,
            y_val.reshape(-1,))
        fig = pimp.plot_as_boxplot(show=False)

        assert fig.__class__.__name__ == "Figure"

        return

    def test_one_3d_input(self):
        return

    def test_two_3d_input(self):
        return

    def test_one_2d_and_one_3d_input(self):
        return

    def test_two_2d_inputs(self):
        model = Model(model={"layers": {
            "Input_0": {"shape": (5,)},
            "Input_1": {"shape": (3,)},
            "Concatenate": {"config": {"name": "Concat"},
                            "inputs": ["Input_0", "Input_1"]},
            "Dense_0": {"config": 8,
                        "inputs": "Concat"},
            "Dense_1": 1}},
            lookback=1,
            verbosity=0
        )
        x1 = np.random.random((100, 5))
        x2 = np.random.random((100, 3))
        pimp = PermutationImportance(model.predict, [x1, x2], np.random.random((100, 1)), verbose=0)
        fig = pimp.plot_as_boxplot(show=False)

        assert fig.__class__.__name__ == "Figure"
        return

if __name__ == "__main__":

    unittest.main()
