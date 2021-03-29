import os
import json
import traceback

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import ParameterGrid, ParameterSampler

import numpy as np
import matplotlib.pyplot as plt
import sklearn

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
    from hyperopt import fmin, tpe, atpe, STATUS_OK, Trials, rand
except ImportError:
    hyperopt, fmin, tpe, atpe, Trials, rand, Apply = None, None, None, None, None, None, None
    space_eval, miscs_to_idxs_vals = None, None

try:
    import optuna
    from optuna.visualization import plot_param_importances, plot_edf
    from optuna.visualization import plot_parallel_coordinate, plot_contour, plot_slice
except ImportError:
    optuna, plot_parallel_coordinate, plot_contour, plot_edf, plot_param_importances = None, None, None, None, None

from dl4seq import Model
from dl4seq.utils.TSErrors import FindErrors
from dl4seq.hyper_opt.utils import get_one_tpe_x_iter
from dl4seq.utils.utils import Jsonize, dateandtime_now
from dl4seq.hyper_opt.utils import skopt_space_from_hp_space
from dl4seq.hyper_opt.utils import post_process_skopt_results
from dl4seq.hyper_opt.utils import Categorical, Real, Integer
from dl4seq.hyper_opt.utils import sort_x_iters, x_iter_for_tpe
from dl4seq.hyper_opt.utils import loss_histogram, plot_hyperparameters

# TODO RayTune libraries under the hood
# TODO add generic algorithm, deap/pygad
# TODO skopt provides functions other than gp_minimize, see if they are useful and can be used.


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
    The purpose of this class is to provide a uniform and simplifed interface to use hyperopt, optuna, scikit-optimize
    and scikit-learn based RandomizeSearchCV, GridSearchCV. Thus this class sits on top of hyperopt, optuna,
    scikit-optimize and scikit-learn. Ideally this class should provide all the functionalities
    of beforementioned libaries with a uniform interface. It however also complements these libraries by combining
    their functionalities and adding some additional functionalities to them. On the other hand this class should not
    limit or complicate the use of its underlying libraries. This means all the functionalities of underlying libraries
    are available in this class as well. Moreover, you can use this class just as you use one of its underlying library.

    Sklearn is great but
      - sklearn based SearchCVs cna be applied only on sklearn based models and not on external models such as on NNs
      - sklearn does not provide Bayesian optimization
    On the other hand BayesSearchCV of skopt library
      - extends sklearn such that the sklearn-based regressors/classifiers could be used for Bayesian but then it can be
        used only for sklearn-based regressors/classifiers
      - The gp_minimize function from skopt allows application of Bayesian on any regressor/classifier/model, but in that
        case this will only be Bayesian

    We wish to make a class which allows application of any of the three optimization methods on any type of
    model/classifier/regressor. If the classifier/regressor is of sklearn-based, then for random search,
    we use RanddomSearchCV, for grid search, we use GridSearchCV and for Bayesian, we use BayesSearchCV. On the other
    hand, if the model is not sklearn-based, you will still be able to implement any of the three methods. In such case,
    the bayesian will be implemented using gp_minimize. Random search and grid search will be done by simple iterating
    over the sample space generated as in sklearn based samplers. However, the post-processing of the results is
    (supposed to be) done same as done in RandomSearchCV and GridSearchCV.

    The class should pass all the tests written in sklearn or skopt for corresponding classes.

    For detailed use of this class see [example](https://github.com/AtrCheema/dl4seq/blob/master/examples/hyper_para_opt.ipynb)
    :Scenarios
    ---------------
    Use scenarios of this class can be one of the following:
      1) Apply grid/random/bayesian search for sklearn based regressor/classifier
      2) Apply grid/random/bayesian search for custom regressor/classifier/model/function
      3) Apply grid/random/bayesian search for dl4seq. This may be the easierst one, if user is familier with dl4seq. Only
         supported for ml models and not for dl models. For dl based dl4eq's models, consider scenario 2.


    :parameters
    --------------
    algorithm: str, must be one of "random", "grid" "bayes" and "tpe", defining which optimization algorithm to use.
    objective_fn: callable, It can be either sklearn/xgboost based regressor/classifier or any function whose returned
                  values can act as objective function for the optimization problem.
    param_space: list/dict, the space parameters to be optimized. We recommend the use of Real, Integer and categorical
                 classes from dl4seq/hyper_opt (not from skopt.space). These classes allow a uniform way of defining
                 the parameter space for all the underlying libraries. However, to make this class work exactly similar
                 to its underlying libraries, the user can also define parameter space as is defined in its underlying
                 libraries. For example, for hyperopt based method like 'tpe' the parameter space can be specified as
                 in the examples of hyperopt library. In case the code breaks, please report.
                  Based upon above scenarios

    eval_on_best: bool, if True, then after optimization, the objective_fn will be evaluated on best parameters and the results
                  will be stored in the folder named "best" inside `title` folder.
    kwargs: dict, For scenario 3, you must provide `dl4seq_args` as dictionary for additional arguments which  are to be
                  passed to initialize dl4seq's Model class. The choice of kwargs depends whether you are using this class
                  For scenario 1 ,the kwargs will be passed to either GridSearchCV, RandomizeSearchCV or BayesSearchCV.
                  For scenario 2, if the `method` is Bayes, then kwargs will be passed to `gp_minimize`.
                  For scenario 2, f your custom objective_fn/function accepts named arguments, then an argument `use_named_args`
                  must be passed as True. This must also be passed if you are using in-built `dl4seq_model` as objective
                  function.


    Attributes
    --------------
    For scenario 1, all attributes of corresponding classes of skopt and sklean as available from HyperOpt.
    For scenario 2 and 3, some additional attributes are available.

    - best_paras: returns the best parameters from optimization.
    - results: dict
    - gpmin_results: dict
    - paam_grid: dict, only for scenario 3.
    - title: str, name of the folder in which all results will be saved. By default this is same as name of `algorithm`. For
             `dl4seq` based models, this is more detailed, containing problem type etc.


    Methods
    -----------------
    best_paras_kw: returns the best parameters as dictionary.
    eval_with_best: evaluates the objective_fn on best parameters


    Examples
    ---------------
    ```python
    The following examples illustrate how we can uniformly apply different optimization algorithms.
    >>>from dl4seq import Model
    >>>from dl4seq.hyper_opt import HyperOpt
    >>>from dl4seq.data import load_u1
    # We have to define an objective function which will take keyword arguments. If the objective
    # function does not take keyword arguments, make sure to set use_named_args=False
    >>>data = load_u1()
    >>>inputs = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
    >>>outputs = ['target']
    >>>def objective_fn(**suggestion):
    ...    model = Model(
    ...        inputs=inputs,
    ...        outputs=outputs,
    ...        model={"xgboostregressor": suggestion},
    ...        data=data,
    ...        verbosity=0)
    ...
    ...    model.fit(indices="random")
    ...
    ...    t, p = model.predict(indices=model.test_indices, pref='test')
    ...    mse = FindErrors(t, p).mse()
    ...
    ...    return mse
    # Define search space
    >>>num_samples=5   # only relavent for random and grid search
    >>>    search_space = [
    ...    Categorical(['gbtree', 'dart'], name='booster'),
    ...    Integer(low=1000, high=2000, name='n_estimators', num_samples=num_samples),
    ...    Real(low=1.0e-5, high=0.1, name='learning_rate', num_samples=num_samples)
    ...]
    # Using TPE with optuna
    >>>optimizer = HyperOpt('tpe', objective_fn=objective_fn, param_space=search_space,
    ...                     backend='optuna',
    ...                     num_iterations=num_iterations,
    ...                     use_named_args=True)
    >>>optimizer.fit()
    # Using cmaes with optuna
    >>>optimizer = HyperOpt('cmaes', objective_fn=objective_fn, param_space=search_space,
    ...                     backend='optuna',
    ...                     num_iterations=num_iterations,
    ...                     use_named_args=True)
    >>>optimizer.fit()

    # Using random with optuna, we can also try hyperopt and sklearn as backend for random algorithm
    >>>optimizer = HyperOpt('random', objective_fn=objective_fn, param_space=search_space,
    ...                     backend='optuna',
    ...                     num_iterations=num_iterations,
    ...                     use_named_args=True)
    >>>optimizer.fit()

    # Using TPE of hyperopt
    >>>optimizer = HyperOpt('tpe', objective_fn=objective_fn, param_space=search_space,
    ...                     backend='hyperopt',
    ...                     num_iterations=num_iterations,
    ...                     use_named_args=True)
    >>>optimizer.fit()

    Using Baysian with gaussian processes
    >>>optimizer = HyperOpt('bayes', objective_fn=objective_fn, param_space=search_space,
    ...                     backend='skopt',
    ...                     num_iterations=num_iterations,
    ...                     use_named_args=True)
    >>>optimizer.fit()

    Using grid with sklearn
    >>>optimizer = HyperOpt('grid', objective_fn=objective_fn, param_space=search_space,
    ...                     backend='sklearn',
    ...                     num_iterations=num_iterations,
    ...                     use_named_args=True)
    >>>optimizer.fit()
    # Backward compatability
    The following shows some tweaks with hyperopt to make its working compatible with its underlying libraries.
    # using grid search with dl4seq
    >>>opt = HyperOpt("grid",
    ...           param_space={'n_estimators': [1000, 1200, 1400, 1600, 1800,  2000],
    ...                        'max_depth': [3, 4, 5, 6]},
    ...           dl4seq_args={'model': 'XGBoostRegressor',
    ...                        'inputs': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],
    ...                        'outputs': ['target']},
    ...           data=data,
    ...           use_named_args=True,
    ...           )
    >>>opt.fit()

    #using random search with dl4seq
    >>>opt = HyperOpt("random",
    ...           param_space={'n_estimators': [1000, 1200, 1400, 1600, 1800,  2000],
    ...                        'max_depth': [3, 4, 5, 6]},
    ...           dl4seq_args={'model': 'XGBoostRegressor',
    ...                        'inputs': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],
    ...                        'outputs': ['target']},
    ...           data=data,
    ...           use_named_args=True,
    ...           n_iter=100
    ...           )
    >>>sr = opt.fit()

    # using Bayesian with dl4seq
    >>>from dl4seq.hyper_opt import Integer
    >>>opt = HyperOpt("bayes",
    ...           param_space=[Integer(low=1000, high=2000, name='n_estimators'),
    ...                        Integer(low=3, high=6, name='max_depth')],
    ...           dl4seq_args={'model': 'xgboostRegressor'},
    ...               data=data,
    ...               use_named_args=True,
    ...               n_calls=100,
    ...               x0=[1000, 3],
    ...               n_random_starts=3,  # the number of random initialization points
    ...               random_state=2)
    >>>sr = opt.fit()


    # using Bayesian with custom objective_fn
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

    # using Bayesian with custom objective_fn and named args
    >>>
    >>>def f(noise_level=0.1, **kwargs):
    ...    x = kwargs['x']
    ...    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) + np.random.randn() * noise_level

    >>>opt = HyperOpt("bayes",
    ...           objective_fn=f,
    ...           param_space=[Categorical([32, 64, 128, 256], name='lstm_units'),
    ...                        Categorical(categories=["relu", "elu", "leakyrelu"], name="dense_actfn")
    ...                        ],
    ...           use_named_args=True,
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
                 eval_on_best=False,
                 backend=None,
                 **kwargs
                 ):

        if algorithm not in ALGORITHMS:
            raise ValueError(f"""Invalid value of algorithm provided. Allowd values for algorithm"
                                are {list(ALGORITHMS.keys())}. 
                                You provided {algorithm}""")

        self.objective_fn = objective_fn
        self.algorithm = algorithm
        self.backend=backend
        self.param_space=param_space
        self.original_space = param_space       # todo self.space and self.param_space should be combined.
        self.dl4seq_args = None
        self.use_named_args = False
        self.title = self.algorithm
        self.results = {}  # internally stored results
        self.gpmin_results = None  #
        self.data = None
        self.eval_on_best=eval_on_best
        self.opt_path = kwargs.pop('opt_path') if 'opt_path' in kwargs else None

        self.gpmin_args = self.check_args(**kwargs)

        if self.use_sklearn:
            if self.algorithm == "random":
                self.optfn = RandomizedSearchCV(estimator=objective_fn, param_distributions=param_space, **kwargs)
            else:
                self.optfn = GridSearchCV(estimator=objective_fn, param_grid=param_space, **kwargs)

        elif self.use_skopt_bayes:
            self.optfn = BayesSearchCV(estimator=objective_fn, search_spaces=param_space, **kwargs)

        elif self.use_skopt_gpmin:
            self.fit = self.own_fit

        elif self.use_own:
            self.predict = self._predict and self.backend != 'optuna'
            if self.algorithm == "grid" and self.backend != 'optuna':
                self.fit = self.grid_search
            elif self.algorithm == 'random' and self.backend not in ['optuna', 'hyperopt']:
                self.fit = self.random_search
            elif self.backend == 'hyperopt':
                self.fit = self.fmin
            elif self.backend == 'optuna':
                self.fit = self.optuna_objective
        else:
            raise NotImplementedError(f"""No fit function found for algorithm {self.algorithm}
                                          with backend {self.backend}""")

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
        elif self.algorithm == 'atpe':
            if x is None:
                x = 'hyperopt'
            assert x == 'hyperopt'
        elif self.algorithm == 'random':
            if x is None:
                x = 'sklearn'
            assert x in ['optuna', 'hyperopt', 'sklearn']
        self._backend = x

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, x):
        self._title = x + '_' + str(dateandtime_now())

    def check_args(self, **kwargs):
        kwargs = kwargs.copy()
        if "use_named_args" in kwargs:
            self.use_named_args = kwargs.pop("use_named_args")

        self.use_dl4seq_model = False
        if "dl4seq_args" in kwargs:
            self.dl4seq_args = kwargs.pop("dl4seq_args")
            self.data = kwargs.pop("data")
            self._model = self.dl4seq_args.pop("model")
            #self._model = list(_model.keys())[0]
            self.use_dl4seq_model = True

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
        # TODO, not sure if this is the best way but venturing since it is done by the legend here https://github.com/philipperemy/n-beats/blob/master/nbeats_keras/model.py#L166
        # Since it was not possible to inherit this class from BaseSearchCV and BayesSearchCV at the same time, this
        # hack makes sure that all the functionalities of GridSearchCV, RandomizeSearchCV and BayesSearchCV are also
        # available with class.
        if hasattr(self.optfn, item):
            return getattr(self.optfn, item)
        else:
            raise AttributeError(f"Attribute {item} not found")

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
                    for idx, space in enumerate(x):
                        assert isinstance(space, Dimension)
                        _param_space[space.name] = space.as_hp()
                elif isinstance(x[0], Apply):
                    _param_space = []
                    for idx, space in enumerate(x):
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
                return  Space(x)
            elif len(x) == 1 and isinstance(x[0], tuple):
                if len(x[0]) == 2:
                    if 'int' in x[0][0].__class__.__name__:
                        self._space = Integer(low=x[0][0], high=x[0][1])
                    elif 'float' in x[0][0].__class__.__name__:
                        self._space = Integer(low=x[0][0], high=x[0][1])
                self._space = Categorical(x[0])
            else:
                return None
        elif isinstance(x, dict):  # todo, in random, should we build Only Categorical space?
            _space = []
            for k,v in x.items():
                if isinstance(v, list):
                    if len(v)>2:
                        if isinstance(v[0], int):
                            s = Integer(grid=v, name=k)
                        elif isinstance(v[0], float):
                            s = Real(grid=v, name=k)
                        else:
                            s = Categorical(v,name=k)
                    else:
                        if isinstance(v[0], int):
                            s = Integer(low=np.min(v), high=np.max(v), name=k)
                        elif isinstance(v[0], float):
                            s = Real(low=np.min(v), high=np.max(v), name=k)
                        elif isinstance(v[0], str):
                            s = Categorical(v, name=k)
                        else:
                            raise NotImplementedError
                elif isinstance(v, Dimension):
                    s = v
                elif isinstance(v, Apply) or 'rv_frozen' in v.__class__.__name__:
                    s = skopt_space_from_hp_space(v, k)
                elif isinstance(v, tuple) or isinstance(v, list):
                    s = Categorical(v, name=k)
                else:
                    raise NotImplementedError(f"unknown type {v}, {type(v)}")
                _space.append(s)

            return Space(_space) if len(_space)>0 else None
        elif 'rv_frozen' in x.__class__.__name__ or isinstance(x, Apply):
            return None
        else:
            raise NotImplementedError(f"unknown type {x}, {type(x)}")


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
    def best_paras(self):
        if self.use_skopt_gpmin:
            x_iters = self.gpmin_results['x_iters']
            func_vals = self.gpmin_results['func_vals']
            idx = np.argmin(func_vals)
            paras = x_iters[idx]
        elif self.backend == 'hyperopt':
            if isinstance(self.param_space, dict):
                return get_one_tpe_x_iter(self.trials.best_trial['misc']['vals'], self.param_space)
            else:
                return self.trials.best_trial['misc']['vals']
        elif self.backend == 'optuna':
            return self.study.best_trial.params
        else:
            best_y = list(sorted(self.results.keys()))[0]
            paras = sort_x_iters(self.results[best_y], list(self.param_space.keys()))

        return paras

    @property
    def opt_path(self):
        return self._opt_path

    @opt_path.setter
    def opt_path(self, path):
        if path is None:
            path = os.path.join(os.getcwd(), "results\\" + self.title)
            if not os.path.exists(path):
                os.makedirs(path)
        elif not os.path.exists(path):
            os.makedirs(path)

        self._opt_path = path

    def dl4seq_model(self,
                     pp=False,
                     title=None,
                     return_model=False,
                     view_model=False,
                     **kwargs):

        # this is for it to make json serializable.
        kwargs = Jsonize(kwargs)()

        if title is None:
            title =  self.opt_path #self.method + '_' + config.model["problem"] + '_' + config.model["ml_model"]
            self.title = title
        else:
            title = title

        _model = self._model
        if isinstance(_model, dict):
            _model = list(_model.keys())[0]
        model = Model(data=self.data,
                      prefix=title,
                      verbosity=1 if pp else 0,
                      model={_model: kwargs},
                      **self.dl4seq_args)

        assert model.config["model"] is not None, "Currently supported only for ml models. Make your own" \
                                                               " dl4seq model and pass it as custom model."
        model.fit(indices="random")

        t, p = model.predict(indices=model.test_indices, pp=pp)
        mse = FindErrors(t, p).mse()

        error = round(mse, 6)
        self.results[str(error)] = sort_x_iters(kwargs, self.original_para_order())

        print(f"Validation mse {error}")

        if view_model:
            model.predict(indices=model.train_indices, pref='train')
            model.predict(pref='all')
            model.view_model()

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
            - We make objective_fn using dl4seq and return the error.
          In first case, we just return what user has provided.
          """
        if callable(self.objective_fn) and not self.use_named_args:
            # external function for bayesian but this function does not require named args.
            return self.objective_fn

        dims = self.dims()
        if self.use_named_args and self.dl4seq_args is None:
            # external function and this function accepts named args.
            @use_named_args(dimensions=dims)
            def fitness(**kwargs):
                return self.objective_fn(**kwargs)
            return fitness

        if self.use_named_args and self.dl4seq_args is not None:
            # using in-build dl4seq_model as objective function.
            @use_named_args(dimensions=dims)
            def fitness(**kwargs):
                return self.dl4seq_model(**kwargs)
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
        except ValueError:
            if int(''.join(sklearn.__version__.split('.')[1]))>22:
                raise ValueError(f"""
                    For bayesian optimization, If your sklearn version is above 0.23,
                    then this error may be related to 
                    https://github.com/kiudee/bayes-skopt/issues/90 .
                    Try to lower the sklearn version to 0.22 and run again.
                    {traceback.print_stack()}
                    """)
            else:
                raise ValueError(traceback.print_stack())

        self.gpmin_results = search_result

        if len(self.results) < 1:
            self.results = {str(round(k, 8)): self.to_kw(v) for k, v in zip(search_result.func_vals, search_result.x_iters)}

        post_process_skopt_results(search_result, self.results, self.opt_path)

        if self.eval_on_best:
            self.eval_with_best()

        return search_result

    def eval_sequence(self, params):

        print(f"total number of iterations: {len(params)}")
        for idx, para in enumerate(params):

            if self.use_dl4seq_model:
                err = self.dl4seq_model(**para)
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

            if not self.use_dl4seq_model:
                self.results[f'{err}_{idx}'] = sort_x_iters(para, self.original_para_order())

        fname = os.path.join(self.opt_path, "eval_results.json")
        jsonized_results = {}
        for res, val in self.results.items():
            jsonized_results[res] = Jsonize(val)()
        with open(fname, "w") as fp:
            json.dump(jsonized_results, fp, sort_keys=True, indent=4)

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
        self.param_grid = param_list

        return self.eval_sequence(param_list)

    def optuna_objective(self, **kwargs):

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

        study = optuna.create_study(direction='minimize', sampler=sampler[self.algorithm]())
        study.optimize(objective, n_trials=self.num_iterations)
        setattr(self, 'study', study)

        self._plot()

        return study

    def fmin(self, **kwargs):

        suggest_options = {
            'tpe': tpe.suggest,
            'atpe': atpe.suggest,
            'random': rand.suggest
        }

        trials = Trials()
        model_kws = self.gpmin_args
        if 'num_iterations' in model_kws:
            model_kws['max_evals'] = model_kws.pop('num_iterations')

        if self.use_named_args and not self.use_dl4seq_model:
            def objective_fn(kws):
                # the objective function in hyperopt library receives a dictionary
                return self.objective_fn(**kws)
            objective_f = objective_fn

        elif self.use_named_args and self.use_dl4seq_model:
            # make objective_fn using dl4seq
            def fitness(kws):
                return self.dl4seq_model(**kws)
            objective_f = fitness

        else:
            objective_f = self.objective_fn

        best = fmin(objective_f,
                    space=self.param_space,
                    algo=suggest_options[self.algorithm],
                    trials=trials,
                    **kwargs,
                    **model_kws)

        with open(os.path.join(self.opt_path, 'trials.json'), "w") as fp:
            json.dump(Jsonize(trials.trials)(), fp, sort_keys=True, indent=4)

        setattr(self, 'trials', trials)
        self.results = trials.results
        self._plot()

        return best

    def _predict(self, *args, **params):

        if self.use_named_args and self.dl4seq_args is not None:
            return self.dl4seq_model(pp=True, **params)

        if self.use_named_args and self.dl4seq_args is None:
            return self.objective_fn(**params)

        if callable(self.objective_fn) and not self.use_named_args:
            return self.objective_fn(*args)

    def x_iters(self, as_list=True):
        if self.backend == 'hyperopt':
            # we have to extract x values from trail.trials.
            if isinstance(self.param_space, dict):

                return x_iter_for_tpe(self.trials, self.param_space, as_list=as_list)

        elif self.backend == 'optuna':

            return [list(s.params.values()) for s in self.study.trials]
        else:
            return [list(_iter.values()) for _iter in self.results.values()]

    def func_vals(self):
        if self.backend == 'hyperopt':
            return [self.trials.results[i]['loss'] for i in range(self.num_iterations)]
        elif self.backend == 'optuna':
            return [s.values for s in self.study.trials]
        else:
            return np.array(list(self.results.keys()), dtype=np.float32)

    def _plot(self):

        class sr:
            x_iters = self.x_iters() if not isinstance(self.param_space, Apply) else None
            func_vals = self.func_vals()
            space = self.skopt_space()
            x = list(self.best_paras.values())

        res = sr()
        plt.close('all')
        if sr.x_iters is not None:
            plot_convergence([res])  #todo, should include an option to plot original evaluations instead of only minimum

            fname = os.path.join(self.opt_path, "convergence.png")
            plt.savefig(fname, dpi=300, bbox_inches='tight')

        if self.skopt_space() is not None and res.x_iters is not None:
            plt.close('all')
            plot_evaluations(res, dimensions=list(self.best_paras.keys()))
            plt.savefig(os.path.join(self.opt_path, "evaluations.png"), dpi=300, bbox_inches='tight')

        if self.backend == 'hyperopt':
            loss_histogram([y for y in self.trials.losses()],
                           save=True,
                           fname=os.path.join(self.opt_path, "loss_histogram.png")
                           )
            plot_hyperparameters(self.trials,
                                 fname=os.path.join(self.opt_path, "loss_histogram.png"),
                                 save=True
                                 )

        if self.backend == 'optuna':

            if plotly is not None:
                fig = plot_parallel_coordinate(self.study)
                plotly.offline.plot(fig, filename=os.path.join(self.opt_path, 'parallel_coordinates.html'),auto_open=False)

                fig = plot_contour(self.study)
                plotly.offline.plot(fig, filename=os.path.join(self.opt_path, 'contours.html'),auto_open=False)

                fig = plot_param_importances(self.study)
                plotly.offline.plot(fig, filename=os.path.join(self.opt_path, 'parameter_importance.html'),auto_open=False)

                fig = plot_edf(self.study)
                plotly.offline.plot(fig, filename=os.path.join(self.opt_path, 'edf.html'),auto_open=False)

        return

    def best_paras_kw(self)->dict:
        """Returns a dictionary consisting of best parameters with their names as keys and their values as keys."""
        x = self.best_paras
        if isinstance(x, dict):
            return x

        return self.to_kw(x)

    def to_kw(self, x):
        names = []
        if isinstance(self.param_space, list):

            for para in self.param_space:

                if isinstance(para, dict):
                    names.append(list(para.keys())[0])
                    # each dictionary must contain only one key in this case
                    assert len(para) == 1
                else:
                    if hasattr(para, 'name'):
                        names.append(para.name)
                    else:
                        # self.param_space was not named rather it was just a list of tuples for example
                        names = None
                        break

        elif isinstance(self.param_space, dict):
            for key in self.param_space.keys():
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
                       view_model=True,
                       return_model=False):
        """Find the best parameters and evaluate the objective_fn on them."""
        print("Evaluting objective_fn on best set of parameters.")
        x = self.best_paras

        if self.use_named_args:
            x = self.best_paras_kw()

        if self.use_named_args and self.dl4seq_args is not None:
            return self.dl4seq_model(pp=True,
                                     view_model=view_model,
                                     return_model=return_model,
                                     title=os.path.join(self.opt_path, "best"),
                                     **x)

        if self.use_named_args and self.dl4seq_args is None:
            return self.objective_fn(**x)

        if callable(self.objective_fn) and not self.use_named_args:
            return self.objective_fn(x)

        raise NotImplementedError
