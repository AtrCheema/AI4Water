
import unittest

from ai4water import Model
from ai4water.datasets import busan_beach


def get_model(model_name):
    model = Model(model=model_name,
                  verbosity=0)

    model.fit(data=busan_beach())

    p = model.predict()

    return p


class TestML(unittest.TestCase):

    def test_xgboost(self):

        p = get_model("XGBRegressor")
        assert p.sum() == 130264.75

        return

    def test_catboost(self):

        p = get_model("CatBoostRegressor")
        assert p.sum() == 2985922.097779183

        return

    def test_lgbm(self):
        p = get_model("LGBMRegressor")
        assert p.sum() == 40543045.42882621

        return

    def test_rf(self):

        p = get_model("RandomForestRegressor")
        assert p.sum() == 2706155.5142797576

        return


if __name__ == "__main__":
    unittest.main()