
import os
import time
import unittest

import numpy as np

from ai4water import Model
from ai4water.datasets import arg_beach
from ai4water.preprocessing import DataHandler
from ai4water.utils.utils import find_best_weight

from ai4water.functional import Model as FModel


data = arg_beach()
dh = DataHandler(data=data, verbosity=0)
x_reg, y_reg = dh.training_data()

mlp_model = {"layers": {"Dense": 8, "Dense_1": 1}}


def _test_from_config_basic(
        _model,
        models, x, y,
        find_best=False,
        config_file=False,
):

    for m in models:
        model = _model(model=m,
                       lookback=1,
                       verbosity=0,
                       input_features=data.columns.tolist()[0:-1],
                       output_features=data.columns.tolist()[-1:])
        model.fit(x, y)
        ini_y = model.predict(np.arange(13).reshape(-1, 13)).item()

        if config_file:
            m2 = Model.from_config_file(os.path.join(model.path, 'config.json'))
        else:
            m2 = Model.from_config(model.config)

        best_weight = None
        if find_best:
            best_weight = os.path.join(model.w_path, find_best_weight(model.w_path))

        m2.update_weights(best_weight)
        fin_y = m2.predict(np.arange(13).reshape(-1, 13)).item()
        assert np.allclose(ini_y, fin_y)
        time.sleep(1)

    return

class TestFromConfig(unittest.TestCase):
    models = ["RandomForestRegressor",
              "XGBRegressor",
              "CatBoostRegressor",
              "LGBMRegressor",
              mlp_model
              ]
    def test_subclassing(self):
        _test_from_config_basic(Model, self.models, x_reg, y_reg)
        return

    def test_subclassing_fn(self):
        _test_from_config_basic(FModel, self.models, x_reg, y_reg)

    def test_subclassing_with_weights(self):
        _test_from_config_basic(Model, self.models, x_reg, y_reg, find_best=True)

    def test_subclassing_fn_with_weights(self):
        # we are able to load functinoal model
        _test_from_config_basic(FModel, self.models, x_reg, y_reg, find_best=True)

    def test_subclassing_with_config_file(self):
        # we are able to load subclassing Model from config_file
        _test_from_config_basic(Model, self.models, x_reg, y_reg, config_file=True)
        return

    def test_fn_with_config_file(self):
        # we are able to load functional model from config_file
        _test_from_config_basic(FModel, self.models, x_reg, y_reg, config_file=True)
        return

if __name__ == "__main__":

    unittest.main()