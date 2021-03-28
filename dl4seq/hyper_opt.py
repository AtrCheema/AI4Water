import os
import json
import traceback

import skopt
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_evaluations
from sklearn.model_selection import ParameterGrid, ParameterSampler
from skopt.space import Real as _Real
from skopt.space import Categorical as _Categorical
from skopt.space import Integer as _Integer
from skopt.space.space import Dimension
from skopt.space.space import Space
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import matplotlib as mpl

try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    from hyperopt.pyll.base import Apply
    from hyperopt import space_eval
    from hyperopt.plotting import main_plot_history, main_plot_histogram  # todo
except ImportError:
    hyperopt = None

from dl4seq import Model
from dl4seq.utils.TSErrors import FindErrors
from dl4seq.utils.utils import post_process_skopt_results, Jsonize, dateandtime_now


# TODO incorporate hyper_opt, optuna and RayTune libraries under the hood
# TODO add generic algorithm, deap/pygad
# TODO skopt provides functions other than gp_minimize, see if they are useful and can be used.
# TODO add post processing results for all optimizations

class Counter:
    counter = 0

class Real(_Real, Counter):
    """Extends the Real class of Skopt so that it has an attribute grid which then can be fed to optimization
    algorithm to create grid space.
    num_samples: int, if given, it will be used to create grid space using the formula"""
    def __init__(self,
                 low=None,
                 high=None,
                 num_samples:int=None,
                 step:int=None,
                 grid=None,
                 *args,
                 **kwargs
                 ):

        if low is None:
            assert grid is not None
            assert hasattr(grid, '__len__')
            low = grid[0]
            high = grid[-1]

        self.counter += 1
        if 'name' not in kwargs:
            kwargs['name'] = f'real_{self.counter}'

        self.num_samples = num_samples
        self.step = step
        super().__init__(low=low, high=high, *args, **kwargs)
        self.grid = grid

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, x):
        if x is None:
            if self.num_samples:
                self._grid = np.linspace(self.low, self.high, self.num_samples)
            elif self.step:
                self._grid = np.arange(self.low, self.high, self.step)
        else:
            self._grid = np.array(x)

    def as_hp(self):

        if self.prior == 'log-uniform':
            return hp.loguniform(self.name, low=self.low, high=self.high)
        else:
            assert self.prior in ['uniform', 'loguniform', 'normal', 'lognormal',
                                  'quniform', 'qloguniform', 'qnormal', 'qlognormal']
            return getattr(hp, self.prior)(label=self.name, low=self.low, high=self.hight)



class Integer(_Integer, Counter):
    """Extends the Real class of Skopt so that it has an attribute grid which then can be fed to optimization
    algorithm to create grid space. Moreover it also generates optuna and hyperopt compatible/equivalent instances.
    num_samples: int, if given, it will be used to create grid space using the formula"""
    def __init__(self,
                 low=None,
                 high=None,
                 num_samples:int=None,
                 step:int=None,
                 grid=None,
                 *args,
                 **kwargs
                 ):

        if low is None:
            assert grid is not None
            assert hasattr(grid, '__len__')
            low = grid[0]
            high = grid[-1]

        self.counter += 1
        if 'name' not in kwargs:
            kwargs['name'] = f'integer_{self.counter}'

        self.num_samples = num_samples
        self.step = step
        super().__init__(low=low, high=high, *args, **kwargs)
        self.grid = grid

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, x):
        if x is None:
            if self.num_samples:
                self._grid = np.linspace(self.low, self.high, self.num_samples, dtype=np.int32)
            elif self.step:
                self._grid = np.arange(self.low, self.high, self.step, dtype=np.int32)
            else:
                self._grid = None
        else:
            assert hasattr(x, '__len__'), f"unacceptable type of grid {x.__class__.__name__}"
            self._grid = np.array(x)

    def as_hp(self):
        return hp.randint(self.name, low=self.low, high=self.high)


class Categorical(_Categorical):

    @property
    def grid(self):
        return self.categories

    def as_hp(self):
        return hp.choice(self.name, self.categories)


ALGORITHMS = {
    'gp': {'name': 'gaussian_processes', 'backend': ['skopt']},
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
    # using grid search with dl4seq
    >>>from dl4seq.hyper_opt import HyperOpt
    >>>from dl4seq.data import load_u1
    >>>data = load_u1()
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

        if algorithm not in ["random", "grid", "bayes", "tpe"]:
            raise ValueError("algorithm must be one of random, grid, bayes or tpe.")

        self.objective_fn = objective_fn
        self.algorithm = algorithm
        self.param_space=param_space
        self.original_space = param_space       # todo self.space and self.param_space should be combined.
        self.backend=backend
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
            self.predict = self._predict
            if self.algorithm == "grid":
                self.fit = self.grid_search
            elif self.algorithm == 'random':
                self.fit = self.random_search
            elif self.algorithm == 'tpe':
                self.fit = self.fmin
        else:
            raise NotImplementedError

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, x):
        if x is not None:
            assert x in ['optuna', 'hyperopt', 'sklearn'], f"""
Backend must be one of hyperopt, optuna or sklearn but is is {x}"""
        if self.algorithm == 'tpe':
            if x is None:
                x = 'optuna'
            assert x in ['optuna', 'hyperopt']
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

        elif self.algorithm in ["random", "grid"]:
            if isinstance(x, dict):
                _param_space = x
            elif isinstance(x, list):
                _param_space = {}
                for _space in x:
                    assert isinstance(_space, Dimension)
                    _param_space[_space.name] = _space.grid
            else:
                raise ValueError
        elif self.algorithm == 'tpe':
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
        if self.algorithm == 'tpe':
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
        if self.algorithm == 'tpe':
            return self.gpmin_args.get('max_evals', 9223372036854775807)
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
        elif self.use_tpe:
            if isinstance(self.param_space, dict):
                return get_one_tep_x_iter(self.trials.best_trial['misc']['vals'], self.param_space)
            else:
                return self.trials.best_trial['misc']['vals']
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

    def fmin(self, **kwargs):

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
                    algo=tpe.suggest,
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
        if self.algorithm == 'tpe':
            # we have to extract x values from trail.trials.
            if isinstance(self.param_space, dict):

                return x_iter_for_tpe(self.trials, self.param_space, as_list=as_list)
        else:
            return [list(_iter.values()) for _iter in self.results.values()]

    def func_vals(self):
        if self.algorithm == 'tpe':
            return [self.trials.results[i]['loss'] for i in range(self.num_iterations)]
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
            plot_convergence([res])

            fname = os.path.join(self.opt_path, "convergence.png")
            plt.savefig(fname, dpi=300, bbox_inches='tight')

        if self.skopt_space() is not None and res.x_iters is not None:
            plot_evaluations(res, dimensions=list(self.best_paras.keys()))
            plt.savefig(os.path.join(self.opt_path, "evaluations.png"), dpi=300, bbox_inches='tight')
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


def is_choice(space):
    """checks if an hp.space is hp.choice or not"""
    if 'switch' in space.name:
        return True
    return False


def x_iter_for_tpe(trials, param_space:dict, as_list=True):

    assert isinstance(param_space, dict)

    x_iters = []
    for t in trials.trials:

        vals = t['misc']['vals']
        _iter = get_one_tep_x_iter(vals, param_space)

        x_iters.append(_iter)

    if as_list:
        return [list(d.values()) for d in x_iters]
    return x_iters


def get_one_tep_x_iter(tpe_vals, param_space:dict, sort=True):

    x_iter = {}
    for para, para_val in tpe_vals.items():
        if is_choice(param_space[para]):
            hp_assign = {para: para_val[0]}
            cval = space_eval(param_space[para], hp_assign)
            x_iter[para] = cval
        else:
            x_iter[para] = para_val[0]

    if sort:
        x_iter = sort_x_iters(x_iter, list(param_space.keys()))

    return x_iter


def sort_x_iters(x_iter:dict, original_order:list):
    # the values in x_iter may not be sorted as the parameters provided in original order

    new_x_iter = {}
    for s in original_order:
        new_x_iter[s] = x_iter[s]

    return new_x_iter


def skopt_space_from_hp_spaces(hp_space:dict):
    """given a dictionary of hp spaces where keys are names, this function
    converts it into skopt space."""
    new_spaces = []

    for k,s in hp_space.items():

        new_spaces.append(skopt_space_from_hp_space(s, k))

    return


def skopt_space_from_hp_space(hp_space, prior_name=None):
    """Converts on hp space into a corresponding skopt space."""
    if is_choice(hp_space):
        skopt_space = to_categorical(hp_space, prior_name=prior_name)
    elif any([i in hp_space.__str__() for i in ['loguniform', 'quniform', 'qloguniform', 'uniform']]):
        skopt_space = to_real(hp_space, prior_name=prior_name)
    elif 'randint' in hp_space.__str__():
        skopt_space = to_int(hp_space, prior_name=prior_name)
    else:
        raise NotImplementedError

    return skopt_space


def to_categorical(_space, prior_name=None):
    """converts an hp space into a Dimension object."""
    cats = []
    inferred_name = None
    for arg in _space.pos_args:
        if hasattr(arg, '_obj') and arg.pure:
            cats.append(arg._obj)
        elif arg.name == 'hyperopt_param' and len(arg.pos_args)>0:
            for a in arg.pos_args:
                if a.name == 'literal' and a.pure:
                    inferred_name = a._obj

    prior_name = verify_name(prior_name, inferred_name)

    if len(cats)==0:
        raise NotImplementedError
    return Categorical(categories=cats, name=prior_name)


def verify_name(prior_name, inferred_name):
    """Verfies that the given/prior name matches with the inferred name.
    """
    if prior_name is None:
        prior_name = inferred_name
    else:
        assert prior_name == inferred_name, f"""given name {prior_name} does not mach with 
                                                inferred name {inferred_name}"""
    return prior_name


def to_int(_space, prior_name=None):
    """converts an hp.randint into a Dimension object"""
    inferred_name = None
    limits = None
    for arg in _space.pos_args:
        if arg.name == 'literal' and len(arg.named_args)==0:
            inferred_name = arg._obj
        elif len(arg.named_args) == 2 and arg.name == 'randint':
            limits = {}
            for a in arg.named_args:
                limits[a[0]] =  a[1]._obj

        elif len(arg.pos_args) == 2 and arg.name == 'randint':
            # high and low are not named args
            _limits = []
            for a in arg.pos_args:
                _limits.append(a._obj)
            limits =  {'high': np.max(_limits), 'low': np.min(_limits)}
        else:
            raise NotImplementedError

    prior_name = verify_name(prior_name, inferred_name)

    if limits is None:
        raise NotImplementedError
    return Integer(low=limits['low'], high=limits['high'], name=prior_name)


def to_real(_space, prior_name=None):
    """converts an an hp space to real. """
    inferred_name = None
    limits = None
    prior = None
    allowed_names = ['uniform', 'loguniform', 'quniform', 'loguniform', 'qloguniform']
    for arg in _space.pos_args:
        if len(arg.pos_args) > 0:
            for a in arg.pos_args:
                if a.name == 'literal' and len(arg.named_args) == 0:
                    inferred_name = a._obj
                elif a.name in allowed_names and len(a.named_args) == 2:
                    prior = a.name
                    limits = {}
                    for _a in a.named_args:
                        limits[_a[0]] = _a[1]._obj
                elif a.name in allowed_names and len(a.pos_args)==2:
                    prior = a.name
                    _limits = []
                    for _a in a.pos_args:
                        _limits.append(_a._obj)
                    limits = {'high': np.max(_limits), 'low': np.min(_limits)}

    prior_name = verify_name(prior_name, inferred_name)

    # prior must be inferred because hp spaces always have prior
    if limits is None or prior is None or 'high' not in limits:
        raise NotImplementedError

    return Real(low=limits['low'], high=limits['high'], prior=prior, name=prior_name)


def scatterplot_matrix_colored(params_names:list,
                               params_values:list,
                               best_accs:list,
                               blur=False,
                               save=True,
                               fname='scatter_plot.png'
                               ):
    """Scatterplot colored according to the Z values of the points.
    https://github.com/guillaume-chevalier/Hyperopt-Keras-CNN-CIFAR-100/blob/Vooban/AnalyzeTestHyperoptResults.ipynb
    """

    nb_params = len(params_values)
    best_accs = np.array(best_accs)
    norm = mpl.colors.Normalize(vmin=best_accs.min(), vmax=best_accs.max())

    fig, ax = plt.subplots(nb_params, nb_params, figsize=(16, 16))  # , facecolor=bg_color, edgecolor=fg_color)

    for i in range(nb_params):
        p1 = params_values[i]
        for j in range(nb_params):
            p2 = params_values[j]

            axes = ax[i, j]
            # Subplot:
            if blur:
                s = axes.scatter(p2, p1, s=400, alpha=.1,
                                 c=best_accs, cmap='viridis', norm=norm)
                s = axes.scatter(p2, p1, s=200, alpha=.2,
                                 c=best_accs, cmap='viridis', norm=norm)
                s = axes.scatter(p2, p1, s=100, alpha=.3,
                                 c=best_accs, cmap='viridis', norm=norm)
            s = axes.scatter(p2, p1, s=15,
                             c=best_accs, cmap='viridis', norm=norm)

            # Labels only on side subplots, for x and y:
            if j == 0:
                axes.set_ylabel(params_names[i], rotation=0)
            else:
                axes.set_yticks([])

            if i == nb_params - 1:
                axes.set_xlabel(params_names[j], rotation=90)
            else:
                axes.set_xticks([])

    fig.subplots_adjust(right=0.82, top=0.95)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(s, cax=cbar_ax)

    plt.suptitle(
        'Scatterplot matrix of tried values in the search space over different params, colored in function of best test accuracy')
    plt.show()