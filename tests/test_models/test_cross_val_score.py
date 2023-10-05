
import unittest

import numpy as np
from SeqMetrics import ClassificationMetrics
from sklearn.model_selection import KFold

from ai4water import Model
from ai4water.datasets import busan_beach


beach_data = busan_beach()


def make_cross_validator(cv, **kwargs):

    model = Model(
        model={'RandomForestRegressor': {}},
        cross_validator=cv,
        val_metric="mse",
        verbosity=-1,
        **kwargs
    )

    return model


class Testcross_val_score(unittest.TestCase):
    """cross_val_score of Model class"""
    def test_tscv(self):
        model = make_cross_validator(cv={'TimeSeriesSplit': {'n_splits': 5}})
        cv_score = model.cross_val_score(data=beach_data)
        assert isinstance(cv_score, list)
        assert hasattr(model.cross_val_scores, '__len__')  # cross_val_scores are array like
        return

    def test_kfold(self):
        model = make_cross_validator(cv={'KFold': {'n_splits': 5}})
        cv_score = model.cross_val_score(data=beach_data)
        assert isinstance(cv_score, list)
        assert hasattr(model.cross_val_scores, '__len__')  # cross_val_scores are array like
        return

    def test_cv_arg(self):
        model = make_cross_validator(cv=None)
        cv_score = model.cross_val_score(data=beach_data, cv="KFold", cv_kws={'n_splits': 5})
        assert isinstance(cv_score, list)
        assert hasattr(model.cross_val_scores, '__len__')  # cross_val_scores are array like
        return

    def test_loocv(self):
        model = make_cross_validator(cv={'LeaveOneOut': {}}, train_fraction=0.4)
        cv_score = model.cross_val_score(data=beach_data)
        assert isinstance(cv_score, list)
        assert hasattr(model.cross_val_scores, '__len__')  # cross_val_scores are array like
        return

    def test_kfold_with_refit(self):
        model = make_cross_validator(cv={'KFold': {'n_splits': 5}})
        cv_score = model.cross_val_score(data=beach_data, refit=True)
        assert isinstance(cv_score, list)
        assert hasattr(model.cross_val_scores, '__len__')  # cross_val_scores are array like
        return

    def test_kfold_with_xy_paris(self):
        model = make_cross_validator(cv={'KFold': {'n_splits': 5}})
        cv_score = model.cross_val_score(x= np.random.random((50, 2)), y= np.random.random((50, 1)))
        assert isinstance(cv_score, list)
        assert hasattr(model.cross_val_scores, '__len__')  # cross_val_scores are array like
        return

    def test_mutiple_scoring(self):
        model = make_cross_validator(cv={'KFold': {'n_splits': 5}})
        cv_score = model.cross_val_score(data=beach_data, scoring=['r2', 'nse'])
        assert isinstance(cv_score, list) and len(cv_score) == 2
        assert hasattr(model.cross_val_scores, '__len__')  # cross_val_scores are array like
        return

    def test_callable_scoring(self):
        from ai4water.datasets import MtropicsLaos
        data = MtropicsLaos(path=r'/mnt/datawaha/hyex/atr/data/MtropicsLaos/').make_classification(lookback_steps=1)
        def f1_score_(t,p)->float:
           return ClassificationMetrics(t, p).f1_score(average="macro")
        model = Model(model="RandomForestClassifier",
                      cross_validator={"KFold": {"n_splits": 5}},)
        model.cross_val_score(data=data, scoring=f1_score_)
        return

    def test_cv_argument(self):
        model = make_cross_validator(cv=None)
        cv_score = model.cross_val_score(data=beach_data,
                                         cv="KFold", cv_kws={'n_splits': 5})
        assert isinstance(cv_score, list)
        assert hasattr(model.cross_val_scores, '__len__')  # cross_val_scores are array like
        return


if __name__ == "__main__":
    unittest.main()