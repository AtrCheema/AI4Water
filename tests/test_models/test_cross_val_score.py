
import unittest

import numpy as np

from ai4water import Model
from ai4water.datasets import arg_beach


def make_cross_validator(cv, **kwargs):

    model = Model(
        model={'RandomForestRegressor': {}},
        cross_validator=cv,
        val_metric="mse",
        verbosity=0,
        **kwargs
    )

    return model


class Testcross_val_score(unittest.TestCase):
    """cross_val_score of Model class"""
    def test_tscv(self):
        model = make_cross_validator(cv={'TimeSeriesSplit': {'n_splits': 5}})
        cv_score = model.cross_val_score(data=arg_beach())
        assert isinstance(cv_score, float)
        assert hasattr(model.cross_val_mse, '__len__')  # cross_val_scores are array like
        return

    def test_kfold(self):
        model = make_cross_validator(cv={'KFold': {'n_splits': 5}})
        cv_score = model.cross_val_score(data=arg_beach())
        assert isinstance(cv_score, float)
        assert hasattr(model.cross_val_mse, '__len__')  # cross_val_scores are array like
        return

    def test_loocv(self):
        model = make_cross_validator(cv={'LeaveOneOut': {}}, test_fraction=0.6)
        cv_score = model.cross_val_score(data=arg_beach())
        assert isinstance(cv_score, float)
        assert hasattr(model.cross_val_mse, '__len__')  # cross_val_scores are array like
        return

    def test_kfold_with_refit(self):
        model = make_cross_validator(cv={'KFold': {'n_splits': 5}})
        cv_score = model.cross_val_score(data=arg_beach(), refit=True)
        assert isinstance(cv_score, float)
        assert hasattr(model.cross_val_mse, '__len__')  # cross_val_scores are array like
        return

    def test_kfold_with_xy_paris(self):
        model = make_cross_validator(cv={'KFold': {'n_splits': 5}})
        cv_score = model.cross_val_score(x= np.random.random((50, 2)), y= np.random.random((50, 1)))
        assert isinstance(cv_score, float)
        assert hasattr(model.cross_val_mse, '__len__')  # cross_val_scores are array like
        return


if __name__ == "__main__":
    unittest.main()