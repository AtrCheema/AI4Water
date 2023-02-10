import time
import unittest

import matplotlib.pyplot as plt
import pandas as pd

from ai4water import Model
from ai4water.datasets import mg_photodegradation
from ai4water.datasets import busan_beach
from ai4water.postprocessing.explain import PartialDependencePlot


data = busan_beach()
input_features = data.columns.tolist()[0:-1]
output_features = data.columns.tolist()[-1:]


def test_plot_1d_plots(pdp, feature="tide_cm"):
    pdp.plot_1d(feature)
    pdp.plot_1d(feature, show_dist_as="grid")
    pdp.plot_1d(feature, show_dist=False)
    pdp.plot_1d(feature, show_dist=False, ice=False)
    pdp.plot_1d(feature, show_dist=False, ice=False, model_expected_value=True)
    pdp.plot_1d(feature, show_dist=False, ice=False, feature_expected_value=True)
    pdp.plot_1d(feature, ice_only=True, ice_color="red")
    pdp.plot_1d(feature, ice_only=True, ice_color="Blues")

    return


class TestPDP(unittest.TestCase):

    def test_with_cat_data(self):
        data, cat_enc, an_enc  = mg_photodegradation(encoding="ohe")
        model = Model(model="XGBRegressor", verbosity=0)
        model.fit(data=data)
        x, _ = model.training_data(data=data)
        pdp = PartialDependencePlot(model.predict, x, model.input_features,
                                        num_points=14, show=False, save=False)
        feature = [f for f in model.input_features if f.startswith('Catalyst_type')]
        test_plot_1d_plots(pdp, feature)
        return

    def test_2d_data_single_input(self):
        model = Model(model="XGBRegressor",
                      verbosity=0)

        model.fit(data=data)
        x, _ = model.training_data()

        pdp = PartialDependencePlot(model.predict, x, model.input_features,
                                    num_points=14, show=False, save=False)

        test_plot_1d_plots(pdp)

        return

    def test_3d_single_input(self):
        time.sleep(1)
        model = Model(model={"layers": {"LSTM": 32, "Dense": 1}},
                      ts_args={'lookback':4},
                      input_features=input_features,
                      output_features=output_features,
                      verbosity=0,
                      )

        model.fit(data=data)

        x, _ = model.training_data()

        pdp = PartialDependencePlot(model.predict, x, model.input_features, verbose=0,
                                    num_points=14, show=False, save=False)
        test_plot_1d_plots(pdp)
        return

    def test_interactions_2d_single_data(self):
        model = Model(model="XGBRegressor", verbosity=0)

        model.fit(data=busan_beach(inputs=['tide_cm', 'wat_temp_c', 'sal_psu',
                                             'air_temp_c', 'pcp_mm', 'pcp3_mm',
                                             'rel_hum']))
        x, _ = model.training_data()

        pdp = PartialDependencePlot(model.predict, x, model.input_features,
                                    num_points=14, show=False, save=False)

        pdp.nd_interactions(show_dist=True)
        #
        ax = pdp.plot_interaction(["tide_cm", "wat_temp_c"], show=False, save=False)
        assert isinstance(ax, plt.Axes)

        model.partial_dependence_plot(x=pd.DataFrame(x, columns=model.input_features),
                                     feature_name='tide_cm', show=False)

        return


if __name__ == "__main__":
    unittest.main()