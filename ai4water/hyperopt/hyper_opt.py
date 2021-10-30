import os
import json
import copy
import inspect
import warnings
from typing import Union
from collections import OrderedDict

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import ParameterGrid, ParameterSampler

try:
    import plotly
except ImportError:
    plotly = None

try:
    import skopt
    from skopt import gp_minimize
    from skopt import BayesSearchCV
    from skopt.space.space import Space
    from skopt.utils import use_named_args
    from skopt.space.space import Dimension
    from skopt.plots import plot_convergence, plot_evaluations
except ImportError:
    skopt, gp_minimize, BayesSearchCV, Space, _Real, use_named_args = None, None, None, None, None, None
    Dimension, _Integer, _Categorical, plot_evaluations, plot_convergence = None, None, None, None, None

try:
    import hyperopt
    from hyperopt.pyll.base import Apply
    from hyperopt import fmin as fmin_hyperopt
    from hyperopt import tpe, STATUS_OK, Trials, rand
except ImportError:
    hyperopt, fmin_hyperopt, tpe, atpe, Trials, rand, Apply = None, None, None, None, None, None, None
    space_eval, miscs_to_idxs_vals = None, None

try:  # atpe is only available in later versions of hyperopt
    from hyperopt import atpe
except ImportError:
    atpe = None

try:
    import optuna
    from optuna.study import Study
    from optuna.trial._trial import TrialState
    from optuna.visualization import plot_edf
    from optuna.visualization import plot_parallel_coordinate, plot_contour
except ImportError:
    optuna, plot_parallel_coordinate, plot_contour, plot_edf, = None, None, None, None
    Study = None

from ai4water.backend import tf
from ai4water.utils.utils import JsonEncoder
from .utils import plot_convergences
from ai4water.postprocessing.SeqMetrics import RegressionMetrics
from .utils import get_one_tpe_x_iter
from ai4water.utils.utils import Jsonize, dateandtime_now
from .utils import skopt_space_from_hp_space
from .utils import post_process_skopt_results
from .utils import Categorical, Real, Integer
from .utils import sort_x_iters, x_iter_for_tpe
from .utils import loss_histogram, plot_hyperparameters

if tf is not None:
    if 230 <= int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 250:
        from ai4water.functional import Model
        print(f"Switching to functional API due to tensorflow version {tf.__version__} for hpo")
    else:
        from ai4water import Model
else:
    from ai4water import Model

try:
    from ai4water.hyperopt.testing import plot_param_importances
except ModuleNotFoundError:
    plot_param_importances = None

# TODO RayTune libraries under the hood https://docs.ray.io/en/master/tune/api_docs/suggestion.html#summary
# TODO add generic algorithm, deap/pygad
# TODO skopt provides functions other than gp_minimize, see if they are useful and can be used.
# todo loading gpmin_results is not consistent.

SEP = os.sep
COUNTER = 0

ALGORITHMS = {
    'gp': {'name': 'gaussian_processes', 'backend': ['skopt']},
    'bayes': {},
    'forest': {'name': 'decision_tree', 'backend': ['skopt']},
    'gbrt': {'name': 'gradient-boosted-tree regression', 'backend': ['skopt']},
    'tpe': {'name': 'Tree of Parzen Estimators', 'backend': ['hyperopt', 'optuna']},
    'atpe': {'name': 'Adaptive Tree of Parzen Estimators', 'backend': ['hyperopt']},
    'random': {'name': 'random search', 'backend': ['sklearn', 'optuna', 'hyperopt']},
    'grid': {'name': 'grid search', 'backend': ['sklearn', 'optuna']},
    'cmaes': {'name': 'Covariance Matrix Adaptation Evolution Strategy', 'backend': ['optuna']}
}

class HyperOpt(object):
    """
    The purpose of this class is to provide a uniform and simplifed interface to
    use `hyperopt`, `optuna`, `scikit-optimize` and `scikit-learn` based hyperparameter
    optimization methods. Ideally this class should provide all the functionalities of
    beforementioned libaries with a uniform interface. It however also complements
    these libraries by combining their functionalities and adding some additional
    functionalities to them. On the other hand this class should not limit or
    complicate the use of its underlying libraries. This means all the functionalities
    of underlying libraries are available in this class as well. Moreover, you can
    use this class just as you use one of its underlying library.

    The purpose here is to make a class which allows application of any of the
    available optimization methods on any type of model/classifier/regressor. If the
    classifier/regressor is of sklearn-based, then for random search, we use
    RanddomSearchCV, for grid search, we use GridSearchCV and for Bayesian, we
    use BayesSearchCV. On the other hand, if the model is not sklearn-based, you
    will still be able to implement any of the three methods. In such case, the
    bayesian will be implemented using `gp_minimize`. Random search and grid search
    will be done by simple iterating over the sample space generated as in sklearn
    based samplers. However, the post-processing of the results is (supposed to be)
    done same as is done in RandomSearchCV and GridSearchCV.

    The class is expected to pass all the tests written in sklearn or skopt for
    corresponding classes.

    For detailed use of this class see [example](https://github.com/AtrCheema/AI4Water/blob/master/examples/hyper_para_opt.ipynb)

    Attributes
    --------------
    - results dict:
    - gpmin_results dict:
    - skopt_results :
    - hp_space :
    - space
    - skopt_space :
    - space dict:
    - title str: name of the folder in which all results will be saved. By
        default this is same as name of `algorithm`. For `AI4Water` based
            models, this is more detailed, containing problem type etc.


    Methods
    -----------------
    - eval_with_best: evaluates the objective_fn on best parameters
    - best_paras(): returns the best parameters from optimization.

    The following examples illustrate how we can uniformly apply different optimization algorithms.

    Examples
    ---------
    ```python
    >>>from ai4water import Model
    >>>from ai4water.hyperopt import HyperOpt, Categorical, Integer, Real
    >>>from ai4water.datasets import load_u1
    >>>from ai4water.postprocessing.SeqMetrics import RegressionMetrics
    >>>data = load_u1()
    >>>input_features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
    >>>output_features = ['target']
    ...# We have to define an objective function which will take keyword arguments
    ...# and return a scaler value as output. This scaler value will be minized during optimzation
    >>>def objective_fn(**suggestion)->float:
    ...   # the objective function must receive new parameters as keyword arguments
    ...    model = Model(
    ...        input_features=input_features,
    ...        output_features=output_features,
    ...        model={"xgboostregressor": suggestion},
    ...        data=data,
    ...        train_data='random',
    ...        verbosity=0)
    ...
    ...    model.fit()
    ...
    ...    t, p = model.predict(prefix='test', return_true=True)
    ...    mse = RegressionMetrics(t, p).mse()
    ...    # the objective function must return a scaler value which needs to be minimized
    ...    return mse
    ```

    # Define search space
    The search splace determines pool from which parameters are chosen during optimization.
    ```python
    >>>num_samples=5   # only relavent for random and grid search
    >>>search_space = [
    ...    Categorical(['gbtree', 'dart'], name='booster'),
    ...    Integer(low=1000, high=2000, name='n_estimators', num_samples=num_samples),
    ...    Real(low=1.0e-5, high=0.1, name='learning_rate', num_samples=num_samples)
    ...]
    ```
    ```

    Using Baysian with gaussian processes
    ```python
    >>>optimizer = HyperOpt('bayes', objective_fn=objective_fn, param_space=search_space,
    ...                     num_iterations=num_iterations )
    >>>optimizer.fit()
    ```
    # Using TPE with optuna
    ```python
    >>>num_iterations = 10
    >>>optimizer = HyperOpt('tpe', objective_fn=objective_fn, param_space=search_space,
    ...                     backend='optuna',
    ...                     num_iterations=num_iterations )
    >>>optimizer.fit()
    ```

    # Using cmaes with optuna
    ```python
    >>>optimizer = HyperOpt('cmaes', objective_fn=objective_fn, param_space=search_space,
    ...                     backend='optuna',
    ...                     num_iterations=num_iterations )
    >>>optimizer.fit()
    ```

    # Using random with optuna, we can also try hyperopt and sklearn as backend for random algorithm
    ```python
    >>>optimizer = HyperOpt('random', objective_fn=objective_fn, param_space=search_space,
    ...                     backend='optuna',
    ...                     num_iterations=num_iterations )
    >>>optimizer.fit()
    ```

    # Using TPE of hyperopt
    ```python
    >>>optimizer = HyperOpt('tpe', objective_fn=objective_fn, param_space=search_space,
    ...                     backend='hyperopt',
    ...                     num_iterations=num_iterations )
    >>>optimizer.fit()

    Using grid with sklearn
    ```python
    >>>optimizer = HyperOpt('grid', objective_fn=objective_fn, param_space=search_space,
    ...                     backend='sklearn',
    ...                     num_iterations=num_iterations )
    >>>optimizer.fit()
    ```

    # Backward compatability
    The following shows some tweaks with hyperopt to make its working compatible with its underlying libraries.
    # using grid search with AI4Water
    ```python
    >>>opt = HyperOpt("grid",
    ...           param_space={'n_estimators': [1000, 1200, 1400, 1600, 1800,  2000],
    ...                        'max_depth': [3, 4, 5, 6]},
    ...           ai4water_args={'model': 'XGBoostRegressor',
    ...                        'input_features': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],
    ...                        'output_features': ['target']},
    ...           data=data,
    ...           )
    >>>opt.fit()
    ```

    # using random search with AI4Water
    ```python
    >>>opt = HyperOpt("random",
    ...           param_space={'n_estimators': [1000, 1200, 1400, 1600, 1800,  2000],
    ...                        'max_depth': [3, 4, 5, 6]},
    ...           ai4water_args={'model': 'XGBoostRegressor',
    ...                        'input_features': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],
    ...                        'output_features': ['target']},
    ...           data=data,
    ...           n_iter=100
    ...           )
    >>>sr = opt.fit()
    ```

    # using Bayesian with AI4Water
    ```python
    >>>from ai4water.hyperopt import Integer
    >>>opt = HyperOpt("bayes",
    ...           param_space=[Integer(low=1000, high=2000, name='n_estimators'),
    ...                        Integer(low=3, high=6, name='max_depth')],
    ...           ai4water_args={'model': 'xgboostRegressor'},
    ...               data=data,
    ...               n_calls=100,
    ...               x0=[1000, 3],
    ...               n_random_starts=3,  # the number of random initialization points
    ...               random_state=2)
    >>>sr = opt.fit()
    ```


    # using Bayesian with custom objective_fn
    ```python
    >>>def f(x, noise_level=0.1):
    ...      return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) + np.random.randn() * noise_level
    ...
    >>>opt = HyperOpt("bayes",
    ...           objective_fn=f,
    ...           param_space=[Categorical([32, 64, 128, 256], name='lstm_units'),
    ...                        Categorical(categories=["relu", "elu", "leakyrelu"], name="dense_actfn")
    ...                        ],
    ...           acq_func='EI',  # Expected Improvement.
    ...           n_calls=50,     #number of iterations
    ...           x0=[32, "relu"],  # inital value of optimizing parameters
    ...           n_random_starts=3,  # the number of random initialization points
    ...           )
    >>>opt_results = opt.fit()
    ```

    # using Bayesian with custom objective_fn and named args
    ```python
    >>>def f(noise_level=0.1, **kwargs):
    ...    x = kwargs['x']
    ...    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) + np.random.randn() * noise_level

    >>>opt = HyperOpt("bayes",
    ...           objective_fn=f,
    ...           param_space=[Categorical([32, 64, 128, 256], name='lstm_units'),
    ...                        Categorical(categories=["relu", "elu", "leakyrelu"], name="dense_actfn")
    ...                        ],
    ...           acq_func='EI',  # Expected Improvement.
    ...           n_calls=50,     #number of iterations
    ...           x0=[32, "relu"],  # inital value of optimizing parameters
    ...           n_random_starts=3,  # the number of random initialization points
    ...           random_state=2
    ...           )
    >>>opt_results = opt.fit()
    ```

    References
    --------------
    1 https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV

    2 https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV

    3 https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html

    4 https://github.com/scikit-optimize/scikit-optimize/blob/9334d50a1ad5c9f7c013a1c1cb95313a54b83168/examples/bayesian-optimization.py#L109

    """

    def __init__(self,
                 algorithm:str, *,
                 param_space,
                 objective_fn=None,
                 eval_on_best:bool=False,
                 backend:str=None,
                 opt_path:str = None,
                 process_results:bool = True,
                 verbosity:int = 1,
                 **kwargs
                 ):

        """
        Arguments:
            algorithm str:
                must be one of "random", "grid" "bayes" and "tpe", defining which
                optimization algorithm to use.
            objective_fn callable:
                Any callable function whose returned value is to be minimized.
                It can also be either sklearn/xgboost based regressor/classifier.
                If you are using `ai4water_args` then you don't need to define this.
            backend str:
                Defines which backend library to use for the `algorithm`. For
                example the user can specify whether to use `optuna` or `hyper_opt`
                or `sklearn` for `grid` algorithm.
            param_space list/dict:
                the space of parameters to be optimized. We recommend the use
                of Real, Integer and categorical classes from ai4water/hyper_opt
                (not from skopt.space). These classes allow a uniform way of defining
                the parameter space for all the underlying libraries. However, to
                make this class work exactly similar to its underlying libraries,
                the user can also define parameter space as is defined in its
                underlying libraries. For example, for hyperopt based method like
                'tpe' the parameter space can be specified as in the examples of
                hyperopt library. In case the code breaks, please report.
            eval_on_best bool:
                if True, then after optimization, the objective_fn will
                be evaluated on best parameters and the results will be stored in the
                folder named "best" inside `title` folder.
            opt_path : path to save the results
            verbosity : determines amount of information being printed
            kwargs dict:
                Any additional keyword arguments will for the underlying optimization
                algorithm. In case of using AI4Water model, these must be arguments
                which are passed to AI4Water's Model class.
        """
        if algorithm not in ALGORITHMS:
            raise ValueError(f"""Invalid value of algorithm provided. Allowd values for algorithm"
                                are {list(ALGORITHMS.keys())}. 
                                You provided {algorithm}""")

        self.objective_fn = objective_fn
        self.algorithm = algorithm
        self.backend=backend
        self.param_space=param_space
        self.original_space = param_space       # todo self.space and self.param_space should be combined.
        self.ai4water_args = None
        self.title = self.algorithm
        self.results = {}  # internally stored results
        self.gpmin_results = None  #
        self.data = None
        self.eval_on_best=eval_on_best
        self.opt_path = opt_path
        self.process_results=process_results
        self.objective_fn_is_dl = False
        self.verbosity = verbosity

        self.gpmin_args = self.check_args(**kwargs)

        if self.use_sklearn:
            if self.algorithm == "random":
                self.optfn = RandomizedSearchCV(estimator=objective_fn, param_distributions=param_space, **kwargs)
            else:
                self.optfn = GridSearchCV(estimator=objective_fn, param_grid=param_space, **kwargs)

        elif self.use_skopt_bayes:
            self.optfn = BayesSearchCV(estimator=objective_fn, search_spaces=param_space, **kwargs)

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, x):
        if x is not None:
            assert x in ['optuna', 'hyperopt', 'sklearn', 'skopt'], f"""
Backend must be one of hyperopt, optuna or sklearn but is is {x}"""
        if self.algorithm == 'tpe':
            if x is None:
                x = 'optuna'
            assert x in ['optuna', 'hyperopt']
        elif self.algorithm == 'cmaes':
            if x is None:
                x = 'optuna'
            assert x == 'optuna'
        elif self.algorithm == 'atpe':
            if x is None:
                x = 'hyperopt'
            assert x == 'hyperopt'
        elif self.algorithm == 'random':
            if x is None:
                x = 'sklearn'
            assert x in ['optuna', 'hyperopt', 'sklearn']
        elif self.algorithm == 'grid':
            if x is None:
                x = 'sklearn'
            assert x in ['sklearn', 'optuna']
        elif self.algorithm == 'bayes':
            if x is None:
                x = 'skopt'
        else:
            raise ValueError
        if x == 'hyperopt' and hyperopt is None:
            raise ValueError(f"You must install `hyperopt` to use it as backend for {self.algorithm} algorithm.")
        if x == 'optuna' and optuna is None:
            raise ValueError(f"You must install optuna to use `optuna` as backend for {self.algorithm} algorithm")
        self._backend = x

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, x):
        self._title = x + '_' + str(dateandtime_now())

    @property
    def objective_fn_is_dl(self):
        return self._objective_fn_is_dl

    @objective_fn_is_dl.setter
    def objective_fn_is_dl(self, x):
        self._objective_fn_is_dl = x

    def check_args(self, **kwargs):
        kwargs = copy.deepcopy(kwargs)

        self.use_ai4water_model = False
        if "ai4water_args" in kwargs:
            self.ai4water_args = kwargs.pop("ai4water_args")
            self.data = kwargs.pop("data")
            self._model = self.ai4water_args.pop("model")
            #self._model = list(_model.keys())[0]
            self.use_ai4water_model = True

        if 'n_initial_points' in kwargs:
            if int(''.join(skopt.__version__.split('.')[1])) < 8:
                raise ValueError(f"""
                        'n_initial_points' argument is not available in skopt version < 0.8.
                        However you are using skopt version {skopt.__version__} .
                        See https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html#skopt.gp_minimize
                        for more details.
                        """"")
        if 'x0' in kwargs and self.algorithm in ['tpe', 'atpe', 'random', 'grid', 'cmaes']:
            kwargs.pop('x0')
        return kwargs

    def __getattr__(self, item):
        # TODO, not sure if this is the best way but venturing since it is done by the legend
        #  here https://github.com/philipperemy/n-beats/blob/master/nbeats_keras/model.py#L166
        # Since it was not possible to inherit this class from BaseSearchCV and BayesSearchCV at the same time, this
        # hack makes sure that all the functionalities of GridSearchCV, RandomizeSearchCV and BayesSearchCV are also
        # available with class.
        return getattr(self.optfn, item)

    @property
    def param_space(self):
        return self._param_space

    @param_space.setter
    def param_space(self, x):
        if self.algorithm == "bayes":
            if isinstance(x, dict):
                _param_space = []
                for k,v in x.items():
                    assert isinstance(v, Dimension), f"""
                            space for parameter {k} is of invalid type {v.__class__.__name__}.
                            For {self.algorithm}, it must be of type {Dimension.__name__}
                            """
                    _param_space.append(v)
            else:
                assert isinstance(x, list), f"""
                        param space must be list of parameters but it is of type
                        {x.__class__.__name__}"""
                for space in x:
                    # each element in the list can be a tuple of lower and and upper bounds
                    if not isinstance(space, tuple):
                        assert isinstance(space, Dimension), f"""
                                param space must be one of Integer, Real or Categorical
                                but it is of type {space.__class__.__name__}"""
                _param_space = x

        elif self.algorithm in ["random", "grid"] and self.backend != 'optuna':
            # todo, do we also need to provide grid of sample space for random??
            if isinstance(x, dict):
                _param_space = x
            elif isinstance(x, list):
                _param_space = {}
                for _space in x:
                    assert isinstance(_space, Dimension)
                    _param_space[_space.name] = _space.grid
            else:
                raise ValueError
        elif self.algorithm in ['tpe', 'atpe', 'random'] and self.backend == 'hyperopt':
            if isinstance(x, list):
                # space is provided as list. Either all of them must be hp.space or Dimension.
                if isinstance(x[0], Dimension):
                    _param_space = {}
                    for space in x:
                        assert isinstance(space, Dimension)
                        _param_space[space.name] = space.as_hp()
                elif isinstance(x[0], Apply):
                    _param_space = []
                    for space in x:
                        assert isinstance(space, Apply), f"""invalid space type {space.__class__.__name__}"""
                        _param_space.append(space)
                else:
                    raise NotImplementedError

            elif isinstance(x, Dimension): # for single hyper-parameter optimization ?
                _param_space = x.as_hp()
            else:
                _param_space = x

        elif self.backend == 'optuna':
            if isinstance(x, list):
                _param_space = {}
                for s in x:
                    assert isinstance(s, Dimension)
                    _param_space[s.name] = s
            elif isinstance(x, dict):
                assert all([isinstance(s, Dimension) for s in x.values()])
                _param_space = x
            else:
                raise NotImplementedError(f"unknown type of space {x.__class__.__name__}")
        else:
            raise ValueError

        self._param_space = _param_space

    def skopt_space(self):
        """Tries to make skopt compatible Space object. If unsuccessful, return None"""
        x = self.original_space
        if isinstance(x, list):
            if all([isinstance(s, Dimension) for s in x]):
                _space = Space(x)
            elif len(x) == 1 and isinstance(x[0], tuple):
                if len(x[0]) == 2:
                    if 'int' in x[0][0].__class__.__name__:
                        _space = Integer(low=x[0][0], high=x[0][1])
                    elif 'float' in x[0][0].__class__.__name__:
                        _space = Integer(low=x[0][0], high=x[0][1])
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
            elif all([isinstance(s, Apply) for s in self.original_space]):
                _space = Space([skopt_space_from_hp_space(v) for v in self.original_space])
            else:
                raise NotImplementedError
        elif isinstance(x, dict):  # todo, in random, should we build Only Categorical space?
            space_ = []
            for k,v in x.items():
                if isinstance(v, list):
                    s = space_from_list(v, k)
                elif isinstance(v, Dimension):
                    # it is possible that the user has not specified the name so assign the names
                    # because we have keys.
                    if v.name is None or v.name.startswith('real_') or v.name.startswith('integer_'):
                        v.name = k
                    s = v
                elif isinstance(v, Apply) or 'rv_frozen' in v.__class__.__name__:
                    s = skopt_space_from_hp_space(v, k)
                elif isinstance(v, tuple) or isinstance(v, list):
                    s = Categorical(v, name=k)
                else:
                    raise NotImplementedError(f"unknown type {v}, {type(v)}")
                space_.append(s)

            _space = Space(space_) if len(space_)>0 else None
        elif 'rv_frozen' in x.__class__.__name__ or isinstance(x, Apply):
            _space =  Space([skopt_space_from_hp_space(x)])
        else:
            raise NotImplementedError(f"unknown type {x}, {type(x)}")
        return _space

    def space(self)->dict:
        """Returns a skopt compatible space but as dictionary"""
        if self.backend == 'hyperopt':
            if isinstance(self.original_space, Apply):
                _space = skopt_space_from_hp_space(self.original_space)
                _space = {_space.name: _space}
            elif isinstance(self.original_space, dict):
                _space = OrderedDict()
                for k, v in self.original_space.items():
                    if isinstance(v, Apply) or 'rv_frozen' in v.__class__.__name__:
                        _space[k] = skopt_space_from_hp_space(v)
                    elif isinstance(v, Dimension):
                        _space[v.name] = v
                    else:
                        raise NotImplementedError
            elif isinstance(self.original_space, list):
                if  all([isinstance(s, Dimension) for s in self.original_space]):
                    _space = OrderedDict({s.name:s for s in self.original_space})
                elif all([isinstance(s, Apply) for s in self.original_space]):
                    d = [skopt_space_from_hp_space(v) for v in self.original_space]
                    _space = OrderedDict({s.name:s for s in d})
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        elif self.backend == 'optuna':
            if isinstance(self.original_space, list):
                if all([isinstance(s, Dimension) for s in self.original_space]):
                    _space = OrderedDict({s.name: s for s in self.original_space})
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        elif self.backend == 'skopt':
            sk_space = self.skopt_space()

            if isinstance(sk_space, Dimension):
                _space = {sk_space.name: sk_space}

            elif all([isinstance(s, Dimension) for s in sk_space]):
                _space = OrderedDict()
                for s in sk_space:
                    _space[s.name] = s

            else:
                raise NotImplementedError
        elif self.backend == 'sklearn':
            if isinstance(self.original_space, list):
                if all([isinstance(s, Dimension) for s in self.original_space]):
                    _space = OrderedDict({s.name:s for s in self.original_space})
                else:
                    raise NotImplementedError
            elif isinstance(self.original_space, dict):
                _space = OrderedDict()
                for k, v in self.original_space.items():
                    if isinstance(v, list):
                        s = space_from_list(v, k)
                    elif isinstance(v, Dimension):
                        s = v
                    elif isinstance(v, tuple) or isinstance(v, list):
                        s = Categorical(v, name=k)
                    elif isinstance(v, Apply) or 'rv_frozen' in v.__class__.__name__:
                        if self.algorithm == 'random':
                            s = Real(v.kwds['loc'], v.kwds['loc'] + v.kwds['scale'], name=k, prior=v.dist.name)
                        else:
                            s = skopt_space_from_hp_space(v)
                    else:
                        raise NotImplementedError(f"unknown type {v}, {type(v)}")
                    _space[k] = s
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return _space

    @property
    def use_sklearn(self):
        # will return True if we are to use sklearn's GridSearchCV or RandomSearchCV
        if self.algorithm in ["random", "grid"] and "sklearn" in str(type(self.objective_fn)):
            return True
        return False

    @property
    def use_skopt_bayes(self):
        # will return true if we have to use skopt based BayesSearchCV
        if self.algorithm=="bayes" and "sklearn" in str(type(self.objective_fn)):
            assert not self.use_sklearn
            return True
        return False

    @property
    def use_skopt_gpmin(self):
        # will return True if we have to use skopt based gp_minimize function. This is to implement Bayesian on
        # non-sklearn based models
        if self.algorithm == "bayes" and "sklearn" not in str(type(self.objective_fn)):
            assert not self.use_sklearn
            assert not self.use_skopt_bayes
            return True
        return False

    @property
    def use_tpe(self):
        if self.algorithm in ['tpe', 'atpe', 'random'] and self.backend == 'hyperopt':
            return True
        else:
            return False

    @property
    def use_own(self):
        # return True, we have to build our own optimization method.
        if not self.use_sklearn and not self.use_skopt_bayes and not self.use_skopt_gpmin:
            return True
        return False

    @property
    def random_state(self):
        if "random_state" not in self.gpmin_args:
            return np.random.RandomState(313)
        else:
            return np.random.RandomState(self.gpmin_args['random_state'])

    @property
    def num_iterations(self):
        if self.backend == 'sklearn' and self.algorithm == 'grid':
            self.gpmin_args['num_iterations'] = len(ParameterGrid(self.param_space))
        if 'num_iterations' in self.gpmin_args:
            return self.gpmin_args['num_iterations']
        if self.algorithm in ['tpe', 'atpe', 'random'] and self.backend == 'hyperopt':
            return self.gpmin_args.get('max_evals', 9223372036854775807)
        if self.backend == 'optuna':
            return self.gpmin_args.get('n_trials', None)  # default value of n_trials is None in study.optimize()
        if 'n_calls' in self.gpmin_args:
            return self.gpmin_args['n_calls']
        return self.gpmin_args['n_iter']

    @property
    def use_named_args(self):
        if self.use_ai4water_model:
            return True
        argspec = inspect.getfullargspec(self.objective_fn)
        if argspec.varkw is None:
            return False
        elif isinstance(argspec.varkw, str):
            return True
        else:
            raise NotImplementedError

    @property
    def opt_path(self):
        return self._opt_path

    @opt_path.setter
    def opt_path(self, path):
        if path is None:
            path = os.path.join(os.getcwd(), f"results{SEP}" + self.title)
            if not os.path.exists(path):
                os.makedirs(path)
        elif not os.path.exists(path):
            os.makedirs(path)

        self._opt_path = path

    def best_paras(self, as_list=False)->Union[list, dict]:
        # returns best parameters either as dictionary or as list
        if self.use_skopt_gpmin:
            d = self.xy_of_iterations()
            k = list(dict(sorted(d.items())).keys())[0]
            paras = d[k]
        elif self.backend == 'hyperopt':
            d = get_one_tpe_x_iter(self.trials.best_trial['misc']['vals'], self.hp_space())
            if as_list:
                return list(d.values())
            else:
                return d
        elif self.backend == 'optuna':
            if as_list:
                return list(self.study.best_trial.params.values())
            return self.study.best_trial.params
        elif self.use_skopt_bayes or self.use_sklearn:
            paras = self.optfn.best_params_
        else:
            best_y = list(sorted(self.results.keys()))[0]
            paras = sort_x_iters(self.results[best_y], list(self.param_space.keys()))

        if as_list:
            return list(paras.values())
        return paras

    def fit(self, *args, **kwargs):
        """Makes and calls the underlying fit method"""

        if self.use_sklearn or self.use_skopt_bayes:
            fit_fn =  self.optfn.fit

        elif self.use_skopt_gpmin:
            fit_fn = self.own_fit

        elif self.use_own:
            self.predict = self._predict
            if self.algorithm == "grid" and self.backend != 'optuna':
                fit_fn = self.grid_search
            elif self.algorithm == 'random' and self.backend not in ['optuna', 'hyperopt']:
                fit_fn = self.random_search
            elif self.backend == 'hyperopt':
                fit_fn = self.fmin
            elif self.backend == 'optuna':
                fit_fn = self.optuna_objective
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError(f"""No fit function found for algorithm {self.algorithm}
                                          with backend {self.backend}""")
        res = fit_fn(*args, **kwargs)

        serialized = self.serialize()
        fname  = os.path.join(self.opt_path, 'serialized.json')
        with open(fname, 'w') as fp:
            json.dump(serialized, fp, sort_keys=True, indent=4, cls=JsonEncoder)

        return res

    def ai4water_model(self,
                       pp=False,
                       title=None,
                       return_model=False,
                       view_model=False,
                       interpret=False,
                     **kwargs):

        # this is for it to make json serializable.
        kwargs = Jsonize(kwargs)()

        if title is None:
            title =  self.opt_path #self.method + '_' + config.model["mode"] + '_' + config.model["ml_model"]
            self.title = title
        else:
            title = title

        _model = self._model
        if isinstance(_model, dict):
            _model = list(_model.keys())[0]
        model = Model(data=self.data,
                      prefix=title,
                      verbosity=self.verbosity,
                      model={_model: kwargs},
                      **self.ai4water_args)

        assert model.config["model"] is not None, "Currently supported only for ml models. Make your own" \
                                                               " AI4Water model and pass it as custom model."
        model.fit()

        t, p = model.predict(process_results=pp, return_true=True)
        mse = RegressionMetrics(t, p).mse()

        global COUNTER

        error = f'{round(mse, 9)}_{COUNTER+1}'  # don't convert to float now
        self.results[error] = sort_x_iters(kwargs, self.original_para_order())
        COUNTER += 1

        error = float(error)
        if self.verbosity>0:
            print(f"Validation mse {error}")

        if view_model:
            model.predict(data='training', prefix='train')
            model.predict(prefix='all')
            model.view_model()
            if interpret:
                model.interpret(save=True)

        if return_model:
            return model
        return error

    def original_para_order(self):
        if isinstance(self.param_space, dict):
            return list(self.param_space.keys())
        elif self.skopt_space() is not None:
            names = []
            for s in self.skopt_space():
                names.append(s.name)
            return names
        else:
            raise NotImplementedError


    def dims(self):
        # this will be used for gp_minimize
        return list(self.param_space)

    def model_for_gpmin(self):
        """This function can be called in two cases:
            - The user has made its own objective_fn.
            - We make objective_fn using AI4Water and return the error.
          In first case, we just return what user has provided.
          """
        if callable(self.objective_fn) and not self.use_named_args:
            # external function for bayesian but this function does not require named args.
            return self.objective_fn

        dims = self.dims()
        if self.use_named_args and self.ai4water_args is None:
            # external function and this function accepts named args.
            @use_named_args(dimensions=dims)
            def fitness(**kwargs):
                return self.objective_fn(**kwargs)
            return fitness

        if self.use_named_args and self.ai4water_args is not None:
            # using in-build ai4water_model as objective function.
            @use_named_args(dimensions=dims)
            def fitness(**kwargs):
                return self.ai4water_model(**kwargs)
            return fitness

        raise ValueError(f"used named args is {self.use_named_args}")

    def own_fit(self):
        kwargs = self.gpmin_args
        if 'num_iterations' in kwargs:
            kwargs['n_calls'] = kwargs.pop('num_iterations')

        try:
            search_result = gp_minimize(func=self.model_for_gpmin(),
                                        dimensions=self.dims(),
                                        **kwargs)
        except ValueError as e:
            if int(''.join(sklearn.__version__.split('.')[1]))>22:
                raise ValueError(f"""
                    For bayesian optimization, If your sklearn version is above 0.23,
                    then this error may be related to 
                    https://github.com/kiudee/bayes-skopt/issues/90 .
                    Try to lower the sklearn version to 0.22 and run again.
                    {e}
                    """)
            else:
                raise ValueError(e)

        # the `space` in search_results may not be in same order as originally provided.
        space = search_result['space']
        if space.__dict__.__len__()>1:
            ordered_sapce = OrderedDict()
            for k in self.space().keys():
                ordered_sapce[k] = [s for s in space if s.name == k][0]
            search_result['space'] = Space(ordered_sapce.values())

        self.gpmin_results = search_result

        if len(self.results) < 1:
            fv = search_result.func_vals
            xiters = search_result.x_iters
            self.results = {f'{round(k, 8)}_{idx}': self.to_kw(v) for idx, k, v in zip(range(self.num_iterations), fv, xiters)}

        if self.process_results:
            post_process_skopt_results(search_result, self.results, self.opt_path)

            self._plot()

        if self.eval_on_best:
            self.eval_with_best()

        return search_result

    def eval_sequence(self, params):

        if self.verbosity>0:
            print(f"total number of iterations: {len(params)}")
        for idx, para in enumerate(params):

            if self.use_ai4water_model:
                err = self.ai4water_model(**para)
            elif self.use_named_args:  # objective_fn is external but uses kwargs
                err = self.objective_fn(**para)
            else: # objective_fn is external and does not uses keywork arguments
                try:
                    err = self.objective_fn(*list(para.values()))
                except TypeError:
                    raise TypeError(f"""
                        use_named_args argument is set to {self.use_named_args}. If your
                        objective function takes key word arguments, make sure that
                        this argument is set to True during initiatiation of HyperOpt.""")
            err = round(err, 8)

            if not self.use_ai4water_model:
                self.results[err + idx] = sort_x_iters(para, self.original_para_order())

        if self.process_results:
            self._plot()

        if self.eval_on_best:
            self.eval_with_best()

        return self.results

    def grid_search(self):

        params = list(ParameterGrid(self.param_space))
        self.param_grid = params

        return self.eval_sequence(params)

    def random_search(self):

        param_list = list(ParameterSampler(self.param_space, n_iter=self.num_iterations,
                                           random_state=self.random_state))

        if len(param_list) < self.num_iterations:
            # we need to correct it so that num_iterations gets calculated correctly next time
            self.gpmin_args['n_calls'] = len(param_list)
            self.gpmin_args['n_iter'] = len(param_list)

        self.param_grid = param_list

        return self.eval_sequence(param_list)

    def optuna_objective(self, **kwargs):

        if self.verbosity==0:
            optuna.logging.set_verbosity(optuna.logging.ERROR)

        sampler = {
            'tpe': optuna.samplers.TPESampler,
            'cmaes': optuna.samplers.CmaEsSampler,
            'random': optuna.samplers.RandomSampler,
            'grid': optuna.samplers.GridSampler
        }

        def objective(trial):
            suggestion = {}
            for space_name, _space in self.param_space.items():
                    suggestion[space_name] = _space.suggest(trial)
            return self.objective_fn(**suggestion)

        if self.algorithm in ['tpe', 'cmaes', 'random']:
            study = optuna.create_study(direction='minimize', sampler=sampler[self.algorithm]())
        else:
            space = {s.name:s.grid for s in self.skopt_space()}
            study = optuna.create_study(sampler=sampler[self.algorithm](space))
        study.optimize(objective, n_trials=self.num_iterations)
        setattr(self, 'study', study)

        if self.process_results:
            self._plot()

        return study

    def fmin(self, **kwargs):

        suggest_options = {
            'tpe': tpe.suggest,
            'random': rand.suggest
        }
        if atpe is not None:
            suggest_options.update({'atpe': atpe.suggest})

        trials = Trials()
        model_kws = self.gpmin_args
        if 'num_iterations' in model_kws:
            model_kws['max_evals'] = model_kws.pop('num_iterations')

        space = self.hp_space()
        if self.use_named_args and not self.use_ai4water_model:
            def objective_fn(kws):
                # the objective function in hyperopt library receives a dictionary
                return self.objective_fn(**kws)
            objective_f = objective_fn

        elif self.use_named_args and self.use_ai4water_model:
            # make objective_fn using ai4water
            def fitness(kws):
                return self.ai4water_model(**kws)
            objective_f = fitness

        else:
            objective_f = self.objective_fn

            if len(self.space()) >1:
                space = list(self.hp_space().values())
            elif len(self.space()) == 1:
                space = list(self.hp_space().values())[0]
            else:
                raise NotImplementedError

        best = fmin_hyperopt(objective_f,
                    space=space,
                    algo=suggest_options[self.algorithm],
                    trials=trials,
                    **kwargs,
                    **model_kws)

        with open(os.path.join(self.opt_path, 'trials.json'), "w") as fp:
            json.dump(Jsonize(trials.trials)(), fp, sort_keys=True, indent=4, cls=JsonEncoder)

        setattr(self, 'trials', trials)
        #self.results = trials.results
        if self.process_results:
            self._plot()

        return best

    def _predict(self, *args, **params):

        if self.use_named_args and self.ai4water_args is not None:
            return self.ai4water_model(pp=True, **params)

        if self.use_named_args and self.ai4water_args is None:
            return self.objective_fn(**params)

        if callable(self.objective_fn) and not self.use_named_args:
            return self.objective_fn(*args)

    def hp_space(self)->dict:
        """returns a dictionary whose values are hyperopt equivalent space instances."""
        return {k:v.as_hp(False if self.algorithm=='atpe' else True) for k,v in self.space().items()}

    def xy_of_iterations(self)->dict:
        # todo, not in original order
        if self.backend == "optuna":
            num_iters = range(self.num_iterations)
            return {float(f'{round(trial.value, 5)}_{idx}'):trial.params for idx, trial in zip(num_iters, self.study.trials)}
        elif self.backend == "hyperopt":
            return x_iter_for_tpe(self.trials, self.hp_space(), as_list=False)
        elif self.backend == 'skopt':
            assert self.gpmin_results is not None, f"gpmin_results is not populated yet"
            # adding idx because sometimes the difference between two func_vals is negligible
            fv = self.gpmin_results['func_vals']
            xiters = self.gpmin_results['x_iters']
            return {f'{round(k, 5)}_{idx}':self.to_kw(v) for idx, k,v in zip(range(len(fv)), fv, xiters)}
        else:
            # for sklearn based
            return self.results

    def func_vals(self):
        if self.backend == 'hyperopt':
            return [self.trials.results[i]['loss'] for i in range(self.num_iterations)]
        elif self.backend == 'optuna':
            return [s.values for s in self.study.trials]
        else:
            return np.array(list(self.results.keys()), dtype=np.float32)

    def skopt_results(self):
        if self.use_own and self.algorithm == "bayes" and self.backend == 'skopt':
            return self.gpmin_results
        else:
            class SR:
                x_iters = [list(s.values()) for s in self.xy_of_iterations().values()]
                func_vals = self.func_vals()
                space = self.skopt_space()
                if isinstance(self.best_paras(), list):
                    x = self.best_paras
                elif isinstance(self.best_paras(), dict):
                    x = list(self.best_paras().values())
                else:
                    raise NotImplementedError

            return SR()

    def best_xy(self)->dict:
        if self.backend == 'skopt':
            d = self.xy_of_iterations()
            k = list(dict(sorted(d.items())).keys())[0]
            paras = {k:d[k]}
        else:
            raise NotImplementedError
        return paras

    def _plot(self):

        self.save_iterations_as_xy()

        if self.objective_fn_is_dl:
            plot_convergences(self.opt_path, what='val_loss', ylabel='Validation MSE')
            plot_convergences(self.opt_path, what='loss', ylabel='MSE', leg_pos="upper right")

        sr = self.skopt_results()
        plt.close('all')
        if sr.x_iters is not None and self.backend != "skopt":
            plot_convergence([sr])  #todo, should include an option to plot original evaluations instead of only minimum

            fname = os.path.join(self.opt_path, "convergence.png")
            plt.savefig(fname, dpi=300, bbox_inches='tight')

        if self.backend != 'skopt' and len(self.space())<20:# and len(self.space())>1:
            plt.close('all')
            plot_evaluations(sr, dimensions=self.best_paras(as_list=True))
            plt.savefig(os.path.join(self.opt_path, "evaluations.png"), dpi=300, bbox_inches='tight')

        self.plot_importance(raise_error=False)

        if self.backend == 'hyperopt':
            loss_histogram([y for y in self.trials.losses()],
                           save=True,
                           fname=os.path.join(self.opt_path, "loss_histogram.png")
                           )
            plot_hyperparameters(self.trials,
                                 fname=os.path.join(self.opt_path, "hyperparameters.png"),
                                 save=True
                                 )

        if plotly is not None:

            if self.backend == 'optuna':

                fig = plot_parallel_coordinate(self.study)
                plotly.offline.plot(fig, filename=os.path.join(self.opt_path, 'parallel_coordinates.html'),auto_open=False)

                fig = plot_contour(self.study)
                plotly.offline.plot(fig, filename=os.path.join(self.opt_path, 'contours.html'),auto_open=False)

                fig = plot_edf(self.study)
                plotly.offline.plot(fig, filename=os.path.join(self.opt_path, 'edf.html'),auto_open=False)

        return

    def plot_importance(self, raise_error=True):

        msg = "You must optuna and plotly installed to get hyper-parameter importance."
        if plotly is None or optuna is None:
            if raise_error:
                raise ModuleNotFoundError(msg)
            else:
                warnings.warn(msg)

        else:
            importances, importance_paras, fig = plot_param_importances(self.optuna_study())
            if importances is not None:
                plotly.offline.plot(fig, filename=os.path.join(self.opt_path, 'fanova_importance.html'),
                                    auto_open=False)

                plt.close('all')
                df = pd.DataFrame.from_dict(importance_paras)
                axis = df.boxplot(rot=70, return_type="axes")
                axis.set_ylabel("Relative Importance")
                plt.savefig(os.path.join(self.opt_path, "fanova_importance_hist.png"), dpi=300, bbox_inches='tight')

                with open(os.path.join(self.opt_path, "importances.json"), 'w') as fp:
                    json.dump(importances, fp, indent=4, sort_keys=True, cls=JsonEncoder)

                with open(os.path.join(self.opt_path, "fanova_importances.json"), 'w') as fp:
                    json.dump(importance_paras, fp, indent=4, sort_keys=True, cls=JsonEncoder)

        return

    def to_kw(self, x):
        names = []
        if isinstance(self.space(), dict):
            for key in self.space().keys():
                names.append(key)
        else:
            raise NotImplementedError

        xkv = {}
        if names is not None:
            for name, val in zip(names, x):
                xkv[name] = val
        else:
            xkv = x

        return xkv

    def eval_with_best(self,
                       return_model=False,
                       view_model=True,
                       interpret=False):
        """
        Find the best parameters and evaluate the objective_fn with them.
        Arguments:
            return_model bool: If True, then then the built objective_fn will be returned
            view_model bool: only relevent if ai4Water_args are given.
            interpret bool:  only relevent if ai4water_args are given
        """

        if self.use_named_args:
            x = self.best_paras()
        else:
            x = self.best_paras(True)

        if self.use_named_args and self.ai4water_args is not None:
            return self.ai4water_model(pp=True,
                                       view_model=view_model,
                                       return_model=return_model,
                                       interpret=interpret,
                                       title=os.path.join(self.opt_path, "best"),
                                     **x)

        if self.use_named_args and self.ai4water_args is None:
            return self.objective_fn(**x)

        if callable(self.objective_fn) and not self.use_named_args:
            if isinstance(x, list) and self.backend == 'hyperopt':  # when x = [x]
                if len(x) == 1:
                    x = x[0]
            return self.objective_fn(x)

        raise NotImplementedError

    @classmethod
    def from_gp_parameters(cls, fpath:str, objective_fn):
        """loads results saved from bayesian optimization"""
        opt_path = os.path.dirname(fpath)
        with open(fpath, 'r') as fp:
            gpmin_results = json.load(fp)
        space = gpmin_results['space']
        spaces = []
        for sp_name, sp_paras in space.items():
            if sp_paras['type'] ==  'Categorical':
                spaces.append(Categorical(sp_paras['categories'], name=sp_name))
            elif sp_paras['type'] == 'Integer':
                spaces.append(Integer(low=sp_paras['low'], high=sp_paras['high'], name=sp_name, prior=sp_paras['prior']))
            elif sp_paras['type'] == 'Real':
                spaces.append(Real(low=sp_paras['low'], high=sp_paras['high'], name=sp_name, prior=sp_paras['prior']))
            else:
                raise NotImplementedError

        optimizer = cls('bayes',
                        param_space=spaces,
                        objective_fn=objective_fn,
                        opt_path=opt_path,
                        backend='skopt')
        optimizer.gpmin_results = gpmin_results

        return optimizer

    def pre_calculated_results(self, resutls, from_gp_parameters=True):
        """Loads the pre-calculated results i.e. x and y values which
         have been already evaluated."""
        with open(resutls, 'r') as fp:
            results = json.load(fp)
        return

    def serialize(self):
        return {'fun': '',
                'x': '',
                "best_paras": Jsonize(self.best_paras())(),
                'space': {k: v.serialize() for k,v in self.space().items()},
                'fun_vals': self.func_vals(),
                #'iters': self.xy_of_iterations(), # todo, for BayesSearchCVs, not getting ys
                'algorithm': self.algorithm,
                'backend': self.backend,
                'opt_path': self.opt_path
                }

    def optuna_study(self)->Study:
        """
        Attempts to create an optuna Study instance so that
        optuna based plots can be generated.
        Returns None, if not possible."""

        if self.backend == 'optuna':
            return self.study

        class _Trial:
            state = TrialState.COMPLETE
            def __init__(self, number:int, values:list, params:dict, distributions:dict):
                values = Jsonize(values)()
                self._number = number
                self._values = values
                if isinstance(values, list):
                    assert len(values) == 1
                    self.value = values[0]
                elif isinstance(values, float) or isinstance(values, int):
                    self.value = values
                else:
                    try:  # try to convert it to float if possible
                        self.value = float(values)
                    except Exception as e:
                        raise NotImplementedError(f"""values must be convertible to list but it is {values} of type
                         {values.__class__.__name__} Actual error message was {e}""")
                self.params = params
                self._distributions = distributions
                self.distributions = distributions

        class _Study(Study):

            trials = []
            idx = 0

            distributions = {sn:s.to_optuna() for sn, s in self.space().items()}

            for _y, _x in self.xy_of_iterations().items():

                assert isinstance(_x, dict), f'params must of type dict but provided params are of type {_x.__class__.__name__}'

                trials.append(_Trial(number=idx,
                                     values=_y,
                                     params=_x,
                                     distributions=distributions
                                     ))
                idx += 1
            best_params = self.best_paras()
            best_trial = None
            best_value = None
            _study_id = 0
            _distributions = distributions

            def __init__(StudyObject):
                pass

            def _is_multi_objective(StudyObject):
                return False

        study = _Study()

        setattr(self, 'study', study)

        return study

    def save_iterations_as_xy(self):

        iterations = self.xy_of_iterations()

        jsonized_iterations = Jsonize(iterations)()

        fname = os.path.join(self.opt_path, "iterations.json")
        with open(fname, "w") as fp:
            json.dump(jsonized_iterations, fp, sort_keys=True, indent=4, cls=JsonEncoder)

        fname = os.path.join(self.opt_path, "iterations_sorted.json")
        with open(fname, "w") as fp:
            json.dump(dict(sorted(jsonized_iterations.items())), fp, sort_keys=True, indent=4, cls=JsonEncoder)


def space_from_list(v:list, k:str)->Dimension:
    if len(v) > 2:
        if isinstance(v[0], int):
            s = Integer(grid=v, name=k)
        elif isinstance(v[0], float):
            s = Real(grid=v, name=k)
        else:
            s = Categorical(v, name=k)
    else:
        if isinstance(v[0], int):
            s = Integer(low=np.min(v), high=np.max(v), name=k)
        elif isinstance(v[0], float):
            s = Real(low=np.min(v), high=np.max(v), name=k)
        elif isinstance(v[0], str):
            s = Categorical(v, name=k)
        else:
            raise NotImplementedError
    return s