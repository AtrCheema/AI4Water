import time
import unittest

import matplotlib.pyplot as plt

from ai4water import Model
from ai4water.datasets import arg_beach
from ai4water.postprocessing.explain import PartialDependencePlot


data = arg_beach()
input_features = data.columns.tolist()[0:-1]
output_features = data.columns.tolist()[-1:]


def test_plot_1d_plots(pdp):
    pdp.plot_1d("tide_cm", show=False)
    pdp.plot_1d("tide_cm", show_dist_as="grid", show=False)
    pdp.plot_1d("tide_cm", show_dist=False, show=False)
    pdp.plot_1d("tide_cm", show_dist=False, ice=False, show=False)
    pdp.plot_1d("tide_cm", show_dist=False, ice=False, model_expected_value=True, show=False)
    pdp.plot_1d("tide_cm", show_dist=False, ice=False, feature_expected_value=True, show=False)

    return


class TestPDP(unittest.TestCase):

    def test_2d_data_single_input(self):
        model = Model(model="XGBRegressor",
                      verbosity=0)

        model.fit(data=arg_beach())
        x, _ = model.training_data()

        pdp = PartialDependencePlot(model.predict, x, model.input_features, num_points=14)

        test_plot_1d_plots(pdp)

        return

    def test_3d_single_input(self):
        time.sleep(1)
        model = Model(model={"layers": {"LSTM": 32, "Dense": 1}},
                      lookback=4,
                      input_features=input_features,
                      output_features=output_features,
                      verbosity=0,
                      )

        model.fit(data=arg_beach())

        x, _ = model.training_data()

        pdp = PartialDependencePlot(model.predict, x, model.input_features, verbose=0, num_points=14)
        test_plot_1d_plots(pdp)
        return

    def test_interactions_2d_single_data(self):
        model = Model(model="XGBRegressor", verbosity=0)

        model.fit(data=arg_beach(inputs=['tide_cm', 'wat_temp_c', 'sal_psu',
                                             'air_temp_c', 'pcp_mm', 'pcp3_mm',
                                             'rel_hum']))
        x, _ = model.training_data()

        pdp = PartialDependencePlot(model.predict, x, model.input_features, num_points=14)

        pdp.nd_interactions(show_dist=True, show=False)

        ax = pdp.plot_interaction(["tide_cm", "wat_temp_c"], show=False, save=False)
        assert isinstance(ax, plt.Axes)

        return

if __name__ == "__main__":
    unittest.main()