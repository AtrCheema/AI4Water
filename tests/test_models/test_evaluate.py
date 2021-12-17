
import unittest

import numpy as np

from ai4water import Model
from ai4water.datasets import arg_beach


model = Model(
    model={"layers": {"Dense": 1}},
    lookback=1,
    data=arg_beach(),
    verbosity=0,
)

class TestEvaluate(unittest.TestCase):


    def test_basic(self):
        eval_scores = model.evaluate()
        assert isinstance(eval_scores, list) and len(eval_scores) == 2
        return

    def test_basic_with_metrics(self):
        # basic example with metrics
        eval_scores = model.evaluate(metrics="kge")
        assert isinstance(eval_scores, float)

    def test_basic_with_metric_groups(self):
        # basic example with metrics
        eval_scores = model.evaluate(metrics="hydro_metrics")
        assert isinstance(eval_scores, dict)
        return

    def test_custom_xy(self):
        eval_scores = model.evaluate(np.random.random((10, 13)), np.random.random((10, 1, 1)))
        assert isinstance(eval_scores, list) and len(eval_scores) == 2
        return

    def test_custom_xy0(self):
        # only y as keyword
        eval_scores = model.evaluate(np.random.random((10, 13)), y=np.random.random((10, 1, 1)))
        assert isinstance(eval_scores, list) and len(eval_scores) == 2
        return

    def test_custom_xy_as_keyword_args(self):
        eval_scores = model.evaluate(x=np.random.random((10, 13)), y=np.random.random((10, 1, 1)))
        assert isinstance(eval_scores, list) and len(eval_scores) == 2
        return

    def test_custom_xy_with_metrics(self):
        # custom data with metrics
        pbias = model.evaluate(x=np.random.random((10, 13)), y=np.random.random((10, 1, 1)),
                                    metrics='pbias')
        assert isinstance(pbias, float)
        return

    def test_custom_xy_with_metric_groups(self):
        # custom data with group of metrics
        hydro = model.evaluate(x=np.random.random((10, 13)), y=np.random.random((10, 1, 1)),
                                    metrics='hydro_metrics')
        assert isinstance(hydro, dict)
        return

if __name__ == "__main__":

    unittest.main()