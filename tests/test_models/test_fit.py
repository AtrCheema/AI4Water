
import unittest

import tensorflow as tf


if 230 <= int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 250:
    from ai4water.functional import Model
    print(f"Switching to functional API due to tensorflow version {tf.__version__}")
else:
    from ai4water import Model

from ai4water.functional import Model as FModel
from ai4water.preprocessing import DataSet
from ai4water.datasets import busan_beach, MtropicsLaos

data = busan_beach()
dh = DataSet(data=data, verbosity=0)
x_reg, y_reg = dh.training_data()

laos = MtropicsLaos(path=r'/mnt/datawaha/hyex/atr/data/MtropicsLaos/')
data_cls = laos.make_classification(lookback_steps=2)
dh_cls = DataSet(data=data_cls, verbosity=0)
x_cls, y_cls = dh_cls.training_data()


def _test_fit(_model, model_name, x, y):
    model = _model(model=model_name, verbosity=-1)
    model.fit(x, y)

    model.fit(x, y=y)

    model.fit(x=x, y=y)
    return


class TestFitML(unittest.TestCase):

    def test_fit(self):
        _test_fit(Model, "RandomForestRegressor", x_reg, y_reg)
        _test_fit(Model, "RandomForestClassifier", x_cls, y_cls)
        return

    def test_fit_functional(self):
        _test_fit(FModel, "RandomForestRegressor", x_reg, y_reg)
        _test_fit(FModel, "RandomForestClassifier", x_cls, y_cls)
        return

    def test_fill_on_all_data(self):
        model = Model(model="RandomForestRegressor", verbosity=0)
        model.fit_on_all_training_data(data=data)
        return

    def test_ml_kwargs(self):
        """additional kwargs to .fit of ML models such as catboost/lgbm"""
        model = Model(model="LGBMRegressor")
        model.fit(data=data, init_model=None)
        return


if __name__ == "__main__":
    unittest.main()