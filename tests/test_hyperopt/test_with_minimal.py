import os
import unittest
import site
ai4_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(ai4_dir)

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from scipy.stats import uniform
from sklearn.svm import SVC

from ai4water.backend import np
from ai4water.hyperopt import HyperOpt, Real
from utils import run_unified_interface


class TestHyperOpt(unittest.TestCase):

    def test_random(self):
        # testing for sklearn-based model with random search
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV
        iris = load_iris()
        logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200,
                                      random_state=0)
        distributions1 = dict(C=uniform(loc=0, scale=4),
                             penalty=['l2', 'l1'])

        clf = HyperOpt('random', objective_fn=logistic, param_space=distributions1,
                       random_state=0,
                       verbosity=0)

        search = clf.fit(iris.data, iris.target)
        np.testing.assert_almost_equal(search.best_params_['C'], 2.195254015709299, 5)
        assert search.best_params_['penalty'] == 'l1'

        return clf

    def test_grid(self):
        # testing for sklearn-based model with grid search
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
        iris = load_iris()
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        svc = SVC()
        clf = HyperOpt("grid", objective_fn=svc, param_space=parameters, verbosity=0)
        search = clf.fit(iris.data, iris.target)

        sorted(clf.cv_results_.keys())

        assert search.best_params_['C'] == 1
        assert search.best_params_['kernel'] == 'linear'

        return clf

    def test_grid_custom_model(self):
        # testing grid search algorithm for custom model
        def f(x, noise_level=0.1):
            return np.sin(5 * x) * (1 - np.tanh(x ** 2)) \
                   + np.random.randn() * noise_level

        opt = HyperOpt("grid",
                       objective_fn=f,
                       param_space=[Real(low=-2.0, high=2.0, num_samples=20)],
                       n_calls=15,  # the number of evaluations of f,
                       verbosity=0
                       )

        # executes bayesian optimization
        sr = opt.fit()
        assert len(sr) == 20
        return

    def test_unified_interface(self):

        run_unified_interface('random', 'sklearn', 5, num_samples=5)
        run_unified_interface('grid', 'sklearn', None, num_samples=2)
        return

    def test_kwargs_in_objective_fn(self):

        run_unified_interface('random', 'sklearn', 5, num_samples=5, process_results=False, use_kws=True)
        run_unified_interface('grid', 'sklearn', None, num_samples=2, process_results=False, use_kws=True)
        return


if __name__ == "__main__":
    unittest.main()
