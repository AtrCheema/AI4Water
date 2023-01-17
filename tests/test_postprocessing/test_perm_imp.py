import time
import unittest

import numpy as np
import matplotlib.pyplot as plt

from ai4water import Model
from ai4water.datasets import busan_beach, mg_photodegradation
from ai4water.postprocessing.explain import PermutationImportance


data=busan_beach()


class TestPermImportance(unittest.TestCase):

    def test_one_2d_input(self):
        model = Model(model="XGBRegressor", verbosity=0)
        model.fit(data=data)
        x_val, y_val = model.validation_data()

        pimp = PermutationImportance(
            model.predict,
            x_val,
            y_val.reshape(-1,),
            show=False,
            save=False)
        ax = pimp.plot_1d_pimp()
        assert ax.__class__.__name__ == "AxesSubplot"
        ax = pimp.plot_1d_pimp(plot_type="bar_chart")
        assert ax.__class__.__name__ == "AxesSubplot", ax

        return

    def test_one_3d_input(self):
        time.sleep(1)
        beach_data = data
        model = Model(
            model={"layers": {
                "LSTM": 32,
                "Dense": 1
            }},
            input_features=beach_data.columns.tolist()[0:-1],
            output_features=beach_data.columns.tolist()[-1:],
            ts_args={"lookback": 5},
            verbosity=0
        )

        model.fit(data=beach_data)

        x, y = model.training_data()

        pimp = PermutationImportance(model.predict, inputs=x,
                                     target=y.reshape(-1, ),
                                     n_repeats=4,
                                     verbose=False,
                                     show=False,
                                     save=False)
        axes = pimp.plot_as_heatmap(annotate=False)
        assert axes.__class__.__name__ == "AxesSubplot"

        pimp.plot_1d_pimp()

        return

    def test_two_3d_input(self):
        model = Model(
            model={"layers": {
                "Input_0": {"shape": (5, 4)},
                "Input_1": {"shape": (5, 3)},
                "Concatenate": {"config": {"name": "Concat"},
                                "inputs": ["Input_0", "Input_1"]},
                "LSTM": 32,
                "Dense": 1
            }},
            verbosity=0
        )

        x1 = np.random.random((100, 5, 4))
        x2 = np.random.random((100, 5, 3))
        pimp = PermutationImportance(model.predict, [x1, x2],
                                     np.random.random((100, 1)),
                                     verbose=0,
                                     show=False,
                                     save=False
                                     )
        assert len(pimp.importances) == 2
        assert len(pimp.importances[0]) == 5
        assert len(pimp.importances[1]) == 5
        return

    def test_one_2d_and_one_3d_input(self):
        model = Model(
            model={"layers": {
                "Input_0": {"shape": (5, 4)},
                "LSTM": {"config": 32,
                         "inputs": "Input_0"},

                "Input_1": {"shape": (4,)},
                "Dense_0": {"config": 8,
                            "inputs": "Input_1"},

                "Concatenate": {"config": {"name": "Concat"},
                                "inputs": ["LSTM", "Dense_0"]},
                "Dense": 1
            }},
            verbosity=0,
        )

        x1 = np.random.random((100, 5, 4))
        x2 = np.random.random((100, 4))
        pimp = PermutationImportance(model.predict, [x1, x2], np.random.random((100, 1)),
                                     show=False,
                                     save=False,
                                     verbose=0)
        assert len(pimp.importances) == 2
        assert len(pimp.importances[0]) == 5
        assert pimp.importances[1].shape == (4, 14)
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
            ts_args={'lookback':1},
            verbosity=0
        )
        x1 = np.random.random((100, 5))
        x2 = np.random.random((100, 3))
        pimp = PermutationImportance(model.predict, [x1, x2], np.random.random((100, 1)),
                                     show=False,
                                     save=False,
                                     verbose=0)
        fig = pimp.plot_1d_pimp()

        assert fig.__class__.__name__ == "AxesSubplot"
        return

    def test_categorical(self):
        cat_map = {'Catalyst': list(range(9, 24)), 'Anions': list(range(24, 30))}
        mg_data, cat_enc, an_enc  = mg_photodegradation(encoding="ohe")
        model = Model(model="XGBRegressor", verbosity=0)
        model.fit(data=mg_data)
        x, y = model.training_data(data=mg_data)
        pimp = PermutationImportance(
            model.predict, x, y,
            show=False,
            save=False,
            cat_map=cat_map,
            feature_names=model.input_features,
            n_repeats=2)
        pimp.plot_1d_pimp()
        plt.close('all')
        pimp.plot_1d_pimp("barchart")
        plt.close('all')
        return


if __name__ == "__main__":

    unittest.main()
