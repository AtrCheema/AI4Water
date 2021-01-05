from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import uniform
from sklearn.svm import SVC
import numpy as np
np.random.seed(313)
import os
from skopt.space import Real, Categorical, Integer
from TSErrors import FindErrors
import pandas as pd
import unittest

import site   # so that dl4seq directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

from dl4seq.hyper_opt import HyperOpt
from dl4seq import Model

from inspect import getsourcefile
from os.path import abspath


file_path = abspath(getsourcefile(lambda:0))
dpath = os.path.join(os.path.join(os.path.dirname(os.path.dirname(file_path)), "dl4seq"), "data")
fname = os.path.join(dpath, "input_target_u1.csv")

def run_dl4seq(method):
    dims = {'n_estimators': [1000,  2000],
            'max_depth': [3,  6],
            'learning_rate': [0.1,  0.0005],
            'booster': ["gbtree", "dart"]}


    data = pd.read_csv(fname)
    inputs = list(data.columns)
    inputs.remove('index')
    inputs.remove('target')
    inputs.remove('target_by_group')
    outputs = ['target']

    dl4seq_args = {"inputs": inputs,
                   "outputs": outputs,
                   "lookback": 1,
                   "batches": "2d",
                   "val_data": "same",
                   "test_fraction": 0.3,
                   "ml_model": "xgboostregressor",
                   "ml_model_args": {'objective': 'reg:squarederror'},
                   "transformation": None
                   }

    opt = HyperOpt(method,
                   param_space=dims,
                   dl4seq_args=dl4seq_args,
                   data=data,
                   use_named_args=True,
                   acq_func='EI',  # Expected Improvement.
                   n_calls=12,
                   n_iter=12,
                   # acq_optimizer='auto',
                   x0=[1000, 3, 0.01, "gbtree"],
                   n_random_starts=3,  # the number of random initialization points
                   random_state=2
                   )

    # executes bayesian optimization
    sr = opt.fit()
    return


class TestHyperOpt(unittest.TestCase):

    def test_random(self):
        # testing for sklearn-based model with random search
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV
        iris = load_iris()
        logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200,
                                      random_state=0)
        distributions1 = dict(C=uniform(loc=0, scale=4),
                             penalty=['l2', 'l1'])

        clf = HyperOpt('random', model=logistic, param_space=distributions1, random_state=0)

        search = clf.fit(iris.data, iris.target)
        np.testing.assert_almost_equal(search.best_params_['C'], 2.195254015709299, 5)
        assert search.best_params_['penalty'] == 'l1'
        print("RandomizeSearchCV test passed")
        return clf


    def test_grid(self):
        # testing for sklearn-based model with grid search
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
        iris = load_iris()
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        svc = SVC()
        clf = HyperOpt("grid", model=svc, param_space=parameters)
        search = clf.fit(iris.data, iris.target)

        sorted(clf.cv_results_.keys())

        assert search.best_params_['C'] == 1
        assert search.best_params_['kernel'] == 'linear'
        print("GridSearchCV test passed")
        return clf


    def test_bayes(self):
        # testing for sklearn-based model with gp_min
        # https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html
        X, y = load_iris(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

        opt = HyperOpt("bayes",  model=SVC(),
                       param_space={
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 8),
            'kernel': Categorical(['linear', 'poly', 'rbf']),
            },
            n_iter=32,
            random_state=0
        )
        # executes bayesian optimization
        _ = opt.fit(X_train, y_train)

        # model can be saved, used for predictions or scoring
        np.testing.assert_almost_equal(0.9736842105263158, opt.score(X_test, y_test), 5)
        print("BayesSearchCV test passed")
        return


    def test_gpmin_skopt(self):
        # testing for custom model with gp_min
        # https://github.com/scikit-optimize/scikit-optimize/blob/9334d50a1ad5c9f7c013a1c1cb95313a54b83168/examples/bayesian-optimization.py#L109
        def f(x, noise_level=0.1):
            return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) \
                   + np.random.randn() * noise_level

        opt = HyperOpt("bayes", model=f, param_space=[(-2.0, 2.0)],
                       acq_func="EI",  # the acquisition function
                       n_calls=15,  # the number of evaluations of f
                       n_random_starts=5,  # the number of random initialization points
                       noise=0.1 ** 2,  # the noise level (optional)
                       random_state=1234
                       )

        # executes bayesian optimization
        sr = opt.fit()
        np.testing.assert_almost_equal(-0.909471164417979, sr.fun, 7)  # when called from same file where hyper_opt is saved
        return

    def test_named_custom_bayes(self):
        dims = [Integer(low=1000, high=2000, name='n_estimators'),
                Integer(low=3, high=6, name='max_depth'),
                Real(low=1e-5, high=0.1, name='learning_rate'),
                Categorical(categories=["gbtree", "dart"], name="booster")
                ]

        def f(**kwargs):
            data = pd.read_csv(fname)

            inputs = list(data.columns)
            inputs.remove('index')
            inputs.remove('target')
            inputs.remove('target_by_group')
            outputs = ['target']
            kwargs['objective'] = 'reg:squarederror'

            model = Model(
                inputs=inputs,
                outputs=outputs,
                lookback=1,
                batches="2d",
                val_data="same",
                test_fraction=0.3,
                ml_model="xgboostregressor",
                ml_model_args=kwargs,
                transformation=None,
                data=data,
                prefix='testing',
                verbosity=0)

            model.fit(indices="random")

            t, p = model.predict(indices=model.test_indices, pref='test')
            mse = FindErrors(t, p).mse()
            print(f"Validation mse {mse}")

            return mse

        opt = HyperOpt("bayes",
                       model=f,
                       param_space=dims,
                       use_named_args=True,
                       acq_func='EI',  # Expected Improvement.
                       n_calls=12,
                       # acq_optimizer='auto',
                       x0=[1000, 3, 0.01, "gbtree"],
                       n_random_starts=3,  # the number of random initialization points
                       random_state=2
                       )

        res = opt.fit()
        return


    def test_dl4seq_bayes(self):
        dims = [Integer(low=1000, high=2000, name='n_estimators'),
                Integer(low=3, high=6, name='max_depth'),
                Real(low=1e-5, high=0.1, name='learning_rate'),
                Categorical(categories=["gbtree", "dart"], name="booster")
                ]

        data = pd.read_csv(fname)
        inputs = list(data.columns)
        inputs.remove('index')
        inputs.remove('target')
        inputs.remove('target_by_group')
        outputs = ['target']

        dl4seq_args = {"inputs": inputs,
                       "outputs": outputs,
                       "lookback": 1,
                       "batches": "2d",
                       "val_data": "same",
                       "test_fraction": 0.3,
                       "ml_model": "xgboostregressor",
                       "ml_model_args": {'objective': 'reg:squarederror'},
                       "transformation": None
                       }

        opt = HyperOpt("bayes",
                       param_space=dims,
                       dl4seq_args=dl4seq_args,
                       data=data,
                       use_named_args=True,
                       acq_func='EI',  # Expected Improvement.
                       n_calls=12,
                       # acq_optimizer='auto',
                       x0=[1000, 3, 0.01, "gbtree"],
                       n_random_starts=3,  # the number of random initialization points
                       random_state=2
                       )

        res = opt.fit()
        return

    def test_dl4seq_grid(self):
        run_dl4seq("grid")
        print("dl4seq for grid passing")

    def test_dl4seq_random(self):
        run_dl4seq("random")
        print("dl4seq for random passing")
        return



if __name__ == "__main__":
    unittest.main()