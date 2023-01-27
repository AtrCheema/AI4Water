import json
import copy
import inspect
import warnings
from typing import Union, Dict
from collections import OrderedDict

from .utils import plot_convergences
from .utils import get_one_tpe_x_iter
from .utils import to_skopt_as_dict
from .utils import post_process_skopt_results
from .utils import to_skopt_space
from .utils import save_skopt_results
from .utils import Dimension
from .utils import plot_convergence
from ._space import Categorical, Real, Integer
from .utils import sort_x_iters, x_iter_for_tpe
from .utils import loss_histogram, plot_hyperparameters

from ai4water.utils.utils import JsonEncoder
from ai4water.utils.utils import clear_weights
from ai4water.utils.utils import jsonize, dateandtime_now
from ai4water.utils.visualizations import edf_plot
from ai4water.backend import hyperopt as _hyperopt
from ai4water.utils.utils import create_subplots
from ai4water.backend import np, pd, plt, os, sklearn, optuna, plotly, skopt, easy_mpl


GridSearchCV = sklearn.model_selection.GridSearchCV
RandomizedSearchCV = sklearn.model_selection.RandomizedSearchCV
ParameterGrid = sklearn.model_selection.ParameterGrid
ParameterSampler = sklearn.model_selection.ParameterSampler

bar_chart = easy_mpl.bar_chart
parallel_coordinates = easy_mpl.parallel_coordinates

if skopt is None:
    pass
else:
    Space = skopt.space.space.Space
    #Dimension = skopt.space.space.Dimension
    forest_minimize = skopt.forest_minimize
    gp_minimize = skopt.gp_minimize
    BayesSearchCV = skopt.BayesSearchCV
    use_named_args = skopt.utils.use_named_args
    from skopt.plots import plot_evaluations

if _hyperopt is not None:
    space_eval = _hyperopt.space_eval
    hp = _hyperopt.hp
    miscs_to_idxs_vals = _hyperopt.base.miscs_to_idxs_vals
    Apply = _hyperopt.pyll.base.Apply
    fmin_hyperopt = _hyperopt.fmin
    tpe = _hyperopt.tpe
    STATUS_OK = _hyperopt.STATUS_OK
    Trials = _hyperopt.Trials
    rand = _hyperopt.rand
else:
    space_eval, hp = None, None
    miscs_to_idxs_vals = None
    Apply = None
    fmin_hyperopt = None
    tpe = None
    STATUS_OK = None
    Trials = None
    rand = None

if _hyperopt is not None:
    try:  # atpe is only available in later versions of hyperopt
        atpe = _hyperopt.atpe
    except AttributeError:
        atpe = None
else:
    atpe = None

if optuna is None:
    plot_contour = None
else:
    plot_contour = optuna.visualization.plot_contour

from ._fanova import fANOVA

# TODO RayTune libraries under the hood https://docs.ray.io/en/master/tune/api_docs/suggestion.html#summary
# TODO add generic algorithm, deap/pygad
# TODO skopt provides functions other than gp_minimize, see if they are useful and can be used.
# todo loading gpmin_results is not consistent.

SEP = os.sep
COUNTER = 0

ALGORITHMS = {
    'bayes': {},
    'bayes_rf': {'name': 'decision_tree', 'backend': ['skopt']},
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
    RanddomSearchCV_, for grid search, we use GridSearchCV_ and for Bayesian, we
    use BayesSearchCV_ . On the other hand, if the model is not sklearn-based, you
    will still be able to implement any of the three methods. In such case, the
    bayesian_ will be implemented using `gp_minimize`. Random search and grid search
    will be done by simple iterating over the sample space generated as in sklearn
    based samplers. However, the post-processing of the results is (supposed to be)
    done same as is done in RandomSearchCV and GridSearchCV.

    The class is expected to pass all the tests written in sklearn or skopt for
    corresponding classes.

    For detailed use of this class see this `hpo_tutorial`_

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
    -------
    - eval_with_best: evaluates the objective_fn on best parameters
    - best_paras(): returns the best parameters from optimization.

    The following examples illustrate how we can uniformly apply different optimization algorithms.

    Examples
    --------
        >>> from ai4water import Model
        >>> from ai4water.hyperopt import HyperOpt, Categorical, Integer, Real
        >>> from ai4water.datasets import busan_beach
        >>> from SeqMetrics import RegressionMetrics
        >>> data = busan_beach()
        >>> input_features = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm']
        >>> output_features = ['tetx_coppml']

        We have to define an objective function which will take keyword arguments
        and return a scaler value as output. This scaler value will be minized during optimzation

        >>> def objective_fn(**suggestion)->float:
        ...   # the objective function must receive new parameters as keyword arguments
        ...    model = Model(
        ...        input_features=input_features,
        ...        output_features=output_features,
        ...        model={"XGBRegressor": suggestion},
        ...        train_data='random',
        ...        verbosity=0)
        ...
        ...    model.fit(data=data)
        ...
        ...    t, p = model.predict(return_true=True)
        ...    mse = RegressionMetrics(t, p).mse()
        ...    # the objective function must return a scaler value which needs to be minimized
        ...    return mse

        Define search space
        The search splace determines pool from which parameters are chosen during optimization.

        >>> num_samples=5   # only relavent for random and grid search
        >>> search_space = [
        ...    Categorical(['gbtree', 'dart'], name='booster'),
        ...    Integer(low=1000, high=2000, name='n_estimators', num_samples=num_samples),
        ...    Real(low=1.0e-5, high=0.1, name='learning_rate', num_samples=num_samples)
        ...]
        ...
        ... # Using Baysian with gaussian processes
        >>> optimizer = HyperOpt('bayes', objective_fn=objective_fn, param_space=search_space,
        ...                     num_iterations=num_iterations )
        >>> optimizer.fit()

        Using TPE with optuna

        >>> num_iterations = 10
        >>> optimizer = HyperOpt('tpe', objective_fn=objective_fn, param_space=search_space,
        ...                     backend='optuna',
        ...                     num_iterations=num_iterations )
        >>> optimizer.fit()

        Using cmaes with optuna

        >>> optimizer = HyperOpt('cmaes', objective_fn=objective_fn, param_space=search_space,
        ...                     backend='optuna',
        ...                     num_iterations=num_iterations )
        >>> optimizer.fit()

        Using random with optuna, we can also try hyperopt and sklearn as backend for random algorithm

        >>> optimizer = HyperOpt('random', objective_fn=objective_fn, param_space=search_space,
        ...                     backend='optuna',
        ...                     num_iterations=num_iterations )
        >>> optimizer.fit()

        Using TPE of hyperopt

        >>> optimizer = HyperOpt('tpe', objective_fn=objective_fn, param_space=search_space,
        ...                     backend='hyperopt',
        ...                     num_iterations=num_iterations )
        >>> optimizer.fit()

        Using grid with sklearn

        >>> optimizer = HyperOpt('grid', objective_fn=objective_fn, param_space=search_space,
        ...                     backend='sklearn',
        ...                     num_iterations=num_iterations )
        >>> optimizer.fit()

        Backward compatability
        The following shows some tweaks with hyperopt to make its working
        compatible with its underlying libraries.
        using grid search with AI4Water

        using Bayesian with custom objective_fn

        >>> def f(x, noise_level=0.1):
        ...      return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) + np.random.randn() * noise_level
        ...
        >>> opt = HyperOpt("bayes",
        ...           objective_fn=f,
        ...           param_space=[Categorical([32, 64, 128, 256], name='lstm_units'),
        ...                        Categorical(categories=["relu", "elu", "leakyrelu"], name="dense_actfn")
        ...                        ],
        ...           acq_func='EI',  # Expected Improvement.
        ...           n_calls=50,     #number of iterations
        ...           x0=[32, "relu"],  # inital value of optimizing parameters
        ...           n_random_starts=3,  # the number of random initialization points
        ...           )
        >>> opt_results = opt.fit()

        using Bayesian with custom objective_fn and named args

        >>> def f(noise_level=0.1, **kwargs):
        ...    x = kwargs['x']
        ...    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) + np.random.randn() * noise_level
        >>> opt = HyperOpt("bayes",
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
        >>> opt_results = opt.fit()

    .. _hpo_tutorial:
        https://ai4water-examples.readthedocs.io/en/latest/auto_examples/index.html#hyperparameter-optimization

    .. _GridSearchCV:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    .. _RanddomSearchCV:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

    .. _BayesSearchCV:
        https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html

    .. _bayesian:
        https://github.com/scikit-optimize/scikit-optimize/blob/9334d50a1ad5c9f7c013a1c1cb95313a54b83168/examples/bayesian-optimization.py#L109

    """

    def __init__(
            self,
            algorithm: str, *,
            param_space,
            objective_fn,
            eval_on_best: bool = False,
            backend: str = None,
            opt_path: str = None,
            process_results: bool = True,
            verbosity: int = 1,
            **kwargs
    ):
        """
        Initializes the class

        Parameters
        ----------
            algorithm : str
                must be one of ``random``, ``grid``, ``bayes``, ``bayes_rf``, and ``tpe``, defining which
                optimization algorithm to use.
            objective_fn : callable
                Any callable function whose returned value is to be minimized.
                It can also be either sklearn/xgboost based regressor/classifier.
            param_space : list, dict
                the search space of parameters to be optimized. We recommend the use
                of Real, Integer and categorical classes from [ai4water.hyperopt][ai4water.hyperopt.Integer]
                (not from skopt.space). These classes allow a uniform way of defining
                the parameter space for all the underlying libraries. However, to
                make this class work exactly similar to its underlying libraries,
                the user can also define parameter space as is defined in its
                underlying libraries. For example, for hyperopt based method like
                'tpe' the parameter space can be specified as in the examples of
                hyperopt library. In case the code breaks, please report.
            eval_on_best : bool, optional
                if True, then after optimization, the objective_fn will
                be evaluated on best parameters and the results will be stored in the
                folder named "best" inside `title` folder.
            opt_path :
                path to save the results
            backend : str, optional
                Defines which backend library to use for the `algorithm`. For
                example the user can specify whether to use `optuna` or `hyper_opt`
                or `sklearn` for `grid` algorithm.
            verbosity : bool, optional
                determines amount of information being printed
            **kwargs :
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
        self.backend = backend
        self.param_space = param_space
        self.original_space = param_space       # todo self.space and self.param_space should be combined.
        self.title = self.algorithm
        self.results = OrderedDict()  # internally stored results
        self.gpmin_results = None  #
        self.data = None
        self.eval_on_best = eval_on_best
        self.opt_path = opt_path
        self._process_results = process_results
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
        elif self.algorithm in ['bayes', "bayes_rf"]:
            if x is None:
                x = 'skopt'
        else:
            raise ValueError
        if x == 'hyperopt' and _hyperopt is None:
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
        if self.use_sklearn or self.use_skopt_bayes:
            return getattr(self.optfn, item)
        else:
            raise AttributeError(f"HyperOpt does not have attribute {item}")

    @property
    def param_space(self):
        return self._param_space

    @param_space.setter
    def param_space(self, x):
        if self.algorithm in ["bayes", "bayes_rf"]:
            assert Dimension is not None, f"you must have scikit-optimize installed to use {self.algorithm}."
            if isinstance(x, dict):
                _param_space = []
                for k, v in x.items():
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

            elif isinstance(x, Dimension):  # for single hyper-parameter optimization ?
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
        return to_skopt_space(self.original_space)

    def space(self) -> dict:
        """Returns a skopt compatible space but as dictionary"""
        return to_skopt_as_dict(self.algorithm, self.backend, self.original_space)

    @property
    def use_sklearn(self):
        # will return True if we are to use sklearn's GridSearchCV or RandomSearchCV
        if self.algorithm in ["random", "grid"] and "sklearn" in str(type(self.objective_fn)):
            return True
        return False

    @property
    def use_skopt_bayes(self):
        # will return true if we have to use skopt based BayesSearchCV
        if self.algorithm in ["bayes", "bayes_rf"] and "sklearn" in str(type(self.objective_fn)):
            assert not self.use_sklearn
            return True
        return False

    @property
    def use_skopt_gpmin(self):
        # will return True if we have to use skopt based gp_minimize function. This is to implement Bayesian on
        # non-sklearn based models
        if self.algorithm in ["bayes", "bayes_rf"] and "sklearn" not in str(type(self.objective_fn)):
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

    def best_paras(self, as_list=False) -> Union[list, dict]:
        # returns best parameters either as dictionary or as list
        if self.use_skopt_gpmin:
            xys = self.xy_of_iterations()
            paras = xys[self.best_iter()]['x']

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
            # using BayesSerchCV
            paras = self.optfn.best_params_

        else:
            paras = sort_x_iters(self.results[self.best_iter()]['x'], list(self.param_space.keys()))

        if as_list:
            return list(paras.values())
        return paras

    def fit(self, *args, **kwargs):
        """Makes and calls the underlying fit method

        parameters
        ----------
        **kwargs :
            any keyword arguments for the userdefined objective function

        Example
        -------
        >>> def objective_fn(a=2, b=5, **suggestions)->float:
        ...     # do something e.g calcualte validation score
        >>>     val_score = 2.0
        >>>     return val_score
        """

        if self.use_sklearn or self.use_skopt_bayes:
            fit_fn = self.optfn.fit

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
        fname = os.path.join(self.opt_path, 'serialized.json')
        with open(fname, 'w') as fp:
            json.dump(serialized, fp, sort_keys=True, indent=4, cls=JsonEncoder)

        return res

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

    def model_for_gpmin(self, **kws):
        """
        This function can be called in two cases
            - The user has made its own objective_fn.
            - We make objective_fn using AI4Water and return the error.

        In first case, we just return what user has provided.
        """
        if callable(self.objective_fn) and not self.use_named_args:
            # external function for bayesian but this function does not require named args.
            return self.objective_fn

        dims = self.dims()
        if self.use_named_args:
            # external function and this function accepts named args.
            @use_named_args(dimensions=dims)
            def fitness(**kwargs):
                return self.objective_fn(**kwargs, **kws)
            return fitness

        raise ValueError(f"used named args is {self.use_named_args}")

    def own_fit(self, **kws):
        """kws are the keyword arguments to user objective function
        by the user
        """
        if self.algorithm == "bayes":
            minimize_func = gp_minimize
        else: # bayes_rf
            minimize_func = forest_minimize

        kwargs = self.gpmin_args
        if 'num_iterations' in kwargs:
            kwargs['n_calls'] = kwargs.pop('num_iterations')

        try:
            search_result = minimize_func(
                func=self.model_for_gpmin(**kws),
                dimensions=self.dims(),
                **kwargs)
        except ValueError as e:
            if int(''.join(sklearn.__version__.split('.')[1])) > 22:
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
        if space.__dict__.__len__() > 1:
            ordered_sapce = OrderedDict()
            for k in self.space().keys():
                ordered_sapce[k] = [s for s in space if s.name == k][0]
            search_result['space'] = Space(ordered_sapce.values())

        self.gpmin_results = search_result

        if len(self.results) < 1:
            fv = search_result.func_vals
            xiters = search_result.x_iters
            for idx, y, x in zip(range(len(fv)), fv, xiters):
                self.results[idx] = {'y': y, 'x': x}

        if self._process_results:
            post_process_skopt_results(search_result, self.results,
                                       self.opt_path, rename=True)

            if len(search_result.func_vals)<=100 and self.algorithm != "bayes_rf":
                save_skopt_results(search_result, self.opt_path)

            self.process_results()

        if self.eval_on_best:
            self.eval_with_best()

        return search_result

    def eval_sequence(self, params, **kwargs):
        """"
        kwargs :
            any additional keyword arguments for objective_fn
        """
        if self.verbosity > 0:
            print(f"total number of iterations: {len(params)}")
        for idx, para in enumerate(params):

            if self.use_named_args:  # objective_fn is external but uses kwargs
                err = self.objective_fn(**para, **kwargs)
            else:  # objective_fn is external and does not uses keywork arguments
                try:
                    err = self.objective_fn(*list(para.values()), **kwargs)
                except TypeError:
                    raise TypeError(f"""
                        use_named_args argument is set to {self.use_named_args}. If your
                        objective function takes key word arguments, make sure that
                        this argument is set to True during initiatiation of HyperOpt.""")
            err = round(err, 8)

            self.results[idx] = {'y':err, 'x':sort_x_iters(para, self.original_para_order())}

        if self._process_results:
            clear_weights(self.opt_path, self.results, rename=True)
            self.process_results()

        if self.eval_on_best:
            self.eval_with_best()

        return self.results

    def grid_search(self, **kwargs):

        params = list(ParameterGrid(self.param_space))
        self.param_grid = params

        return self.eval_sequence(params, **kwargs)

    def random_search(self, **kwargs):
        """
        objective function that will used during random search method.
        parameters
        ----------
            kwargs :
                keyword arguments in the user defined objective function.
        """
        for k, v in self.param_space.items():
            if v is None:
                grid = self.space()[k].grid
                assert grid is not None, f"""grid for parameter {k} could not be created. Inferred grid is 
                {grid}. Please either provide the `num_samples` parameter while creating space or explicitly
                provide grid for {k}"""

        param_list = list(ParameterSampler(self.param_space, n_iter=self.num_iterations,
                                           random_state=self.random_state))

        if len(param_list) < self.num_iterations:
            # we need to correct it so that num_iterations gets calculated correctly next time
            self.gpmin_args['n_calls'] = len(param_list)
            self.gpmin_args['n_iter'] = len(param_list)

        self.param_grid = param_list

        return self.eval_sequence(param_list, **kwargs)

    def optuna_objective(self, **kwargs):
        """
        objective function that will used during random search method.
        parameters
        ----------
            kwargs :
                keyword arguments in the user defined objective function.
        """
        if self.verbosity == 0:
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
            return self.objective_fn(**suggestion, **kwargs)

        if self.algorithm in ['tpe', 'cmaes', 'random']:
            study = optuna.create_study(direction='minimize', sampler=sampler[self.algorithm]())
        else:
            space = {s.name: s.grid for s in self.skopt_space()}
            study = optuna.create_study(sampler=sampler[self.algorithm](space))
        study.optimize(objective, n_trials=self.num_iterations)
        setattr(self, 'study', study)

        if self._process_results:
            self.process_results()

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
        if self.use_named_args:
            def objective_fn(kws):
                # the objective function in hyperopt library receives a dictionary
                return self.objective_fn(**kws)
            objective_f = objective_fn
        else:
            objective_f = self.objective_fn

            if len(self.space()) > 1:
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
            json.dump(jsonize(trials.trials), fp, sort_keys=True, indent=4, cls=JsonEncoder)

        setattr(self, 'trials', trials)
        # self.results = trials.results
        if self._process_results:
            self.process_results()

        return best

    def _predict(self, *args, **params):

        if self.use_named_args:
            return self.objective_fn(**params)

        if callable(self.objective_fn) and not self.use_named_args:
            return self.objective_fn(*args)

    def hp_space(self) -> dict:
        """returns a dictionary whose values are hyperopt equivalent space instances."""
        return {k: v.as_hp(False if self.algorithm == 'atpe' else True) for k,v in self.space().items()}

    def xy_of_iterations(self) -> Dict[int,Dict[str, Union[str, dict]]]:
        """returns a dictionary whose keys are iteration numbers are values are xy parirs
        at those iterations.

        Returns
        Dict[int, Dict[str, [dict,float]]]
        """
        if self.backend == "optuna":
            num_iters = range(self.num_iterations)
            results = {}
            for idx, trial in zip(num_iters, self.study.trials):
                results[idx] = {'y': trial.value, 'x': trial.params}
            return results

        elif self.backend == "hyperopt":
            return x_iter_for_tpe(self.trials, self.hp_space(), as_list=False)

        elif self.backend == 'skopt':

            if self.use_skopt_bayes:
                fv = self.optfn.cv_results_['mean_test_score']
                xiters = self.optfn.cv_results_['params']
            else:
                assert self.gpmin_results is not None, f"gpmin_results is not populated yet"
                fv = self.gpmin_results['func_vals']
                xiters = self.gpmin_results['x_iters']

            results = {}
            for idx, y, x in zip(range(len(fv)), fv, xiters):
                results[idx] = {'y': y, 'x': self.to_kw(x)}
            return results

        else:
            # for sklearn based
            return self.results

    def func_vals(self)->np.ndarray:
        """returns the value of objective function at each iteration."""
        if self.backend == 'hyperopt':
            return np.array([self.trials.results[i]['loss'] for i in range(self.num_iterations)])

        elif self.backend == 'optuna':
            return np.array([s.values for s in self.study.trials])

        elif self.use_skopt_bayes or self.use_sklearn:
            return self.optfn.cv_results_['mean_test_score']
        else:
            return np.array([v['y'] for v in self.results.values()])

    def skopt_results(self):

        class OptimizeResult:
            x_iters = [list(s['x'].values()) for s in self.xy_of_iterations().values()]
            func_vals = self.func_vals()
            space = self.skopt_space()
            if isinstance(self.best_paras(), list):
                x = self.best_paras
            elif isinstance(self.best_paras(), dict):
                x = list(self.best_paras().values())
            else:
                raise NotImplementedError

        return OptimizeResult()

    def best_iter(self)->int:
        """returns the iteration on which best/optimized parameters are obtained.
        The indexing starts from 0.
        """
        return np.nanargmin(self.func_vals()).item()

    def best_xy(self) -> dict:
        """Returns best (optimized) parameters as dictionary.
        The dictionary has two keys ``x`` and ``y``. ``x`` is the
        best hyperparameters while `y` is the corresponding objective function value.
        """
        d = self.xy_of_iterations()
        key = list(d.keys())[self.best_iter()]

        return d[key]

    def _plot_edf(self, save=True, show=False, **kwargs):
        """empirical CDF of objective function"""
        plt.close("all")
        y = np.array(list(self.xy_of_iterations().keys())).astype("float64")
        edf_plot(y, show=show, **kwargs)
        if save:
            plt.savefig(os.path.join(self.opt_path, "edf"))
        return

    def _plot_parallel_coords(self, save=True, show=False, **kwargs):
        """ parallel coordinates of hyperparameters
        """
        d = self.xy_of_iterations()

        data = pd.DataFrame([list(v['x'].values()) for v in d.values()],
                            columns=[s for s in self.space()])
        categories = np.array(list(self.xy_of_iterations().keys())).astype("float64")
        parallel_coordinates(
            data=data,
            categories=categories,
            show=False,
            **kwargs,
        )
        if save:
            fname = os.path.join(self.opt_path, "parallel_coordinates")
            plt.savefig(fname, dpi=500, bbox_inches="tight")

        if show:
            plt.show()
        return

    def _plot_evaluations(self, save=True):
        plt.close('all')
        plot_evaluations(self.skopt_results(), dimensions=self.best_paras(as_list=True))
        if save:
            plt.savefig(os.path.join(self.opt_path, "evaluations.png"),
                        dpi=300,
                        bbox_inches='tight')
        return

    def _plot_convergence(self,
                          original:bool=False,
                          ax = None,
                          save=True,
                          show=False,
                          **kwargs):
        plt.close('all')
        if original:
            ax = easy_mpl.plot(self.func_vals(), '--.',
                 ax_kws=dict(xlabel="Number of calls $n$",
                 ylabel=r"$\min f(x)$ after $n$ calls"),
                               show=False,
                               **kwargs)
        else:
            ax = plot_convergence(self.func_vals(), ax=ax, show=False, **kwargs)
        if save:
            fname = os.path.join(self.opt_path, "convergence.png")
            plt.savefig(fname, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        return ax

    def process_results(self, show=False):
        """post processing of results"""
        self.save_iterations_as_xy()

        self._plot_parallel_coords()

        # deep learning related results
        if self.objective_fn_is_dl:
            plot_convergences(
                self.opt_path,
                what='val_loss',
                ylabel='Validation MSE')
            plot_convergences(
                self.opt_path,
                what='loss',
                ylabel='MSE',
                leg_pos="upper right")

        self._plot_edf()

        # distributions/historgrams of explored hyperparameters
        self._plot_distributions(show=show)

        # convergence plot,
        #if sr.x_iters is not None and self.backend != "skopt": # todo
        self._plot_convergence(show=show)

        # plot of hyperparameter space as explored by the optimizer
        if self.backend != 'skopt' and len(self.space()) < 20 and skopt is not None:
            self._plot_evaluations()

        if len(self.best_paras(True))>1:
            plt.close('all')
            try:
                self.plot_importance()
                plt.close('all')
                self.plot_importance(plot_type="bar", show=show)
            except (RuntimeError, AttributeError):
                warnings.warn(f"Error encountered during fanova calculation")

        if self.backend == 'hyperopt':
            loss_histogram([y for y in self.trials.losses()],
                           save=True,
                           fname=os.path.join(self.opt_path, "loss_histogram.png")
                           )
            plot_hyperparameters(
                self.trials,
                fname=os.path.join(self.opt_path, "hyperparameters.png"),
                save=True)

        if plotly is not None:

            if self.backend == 'optuna':

                fig = plot_contour(self.study)
                plotly.offline.plot(fig, filename=os.path.join(self.opt_path, 'contours.html'),
                                    auto_open=False)

        return

    def plot_importance(
            self,
            save=True,
            show:bool=False,
            plot_type="box",
            with_optuna:bool = False,
            **tree_kws
    )->plt.Axes:
        """plots hyperparameter importance using fANOVA"""

        if with_optuna:
            return self._calc_importance_with_optuna(plot_type, save=save, show=show)

        X = pd.DataFrame([list(iter_xy['x'].values()) for iter_xy in self.xy_of_iterations().values()])
        Y = np.array([iter_xy['y'] for iter_xy in self.xy_of_iterations().values()])
        X.columns = list(self.xy_of_iterations()[0]['x'].keys())
        dtypes = [space.__class__.__name__ for space in self.skopt_space()]
        bounds = [(space.low, space.high) if isinstance(space, (Real, Integer)) else None for space in self.skopt_space()]

        kws = {'X': X, 'Y': Y, 'dtypes': dtypes, 'bounds': bounds}

        kws.update(tree_kws)

        if plot_type == "bar":
            try:
                importance = fANOVA(**kws).feature_importance()
            except (AttributeError, RuntimeError):
                raise ValueError(f"Error encountered during fANOVA, try setting `with_optuna` to True")

            ax = self._plot_importance_as_barchart(importance, save=save)

        else:

            try:
                mean, std = fANOVA(**kws).feature_importance(return_raw=True)
            except (AttributeError, RuntimeError):
                raise ValueError(f"Error encountered during fANOVA, try setting `with_optuna` to True")

            ax = self._plot_importance_as_boxplot(mean, std, save)

        if show:
            plt.show()

        return ax

    def _plot_importance_as_boxplot(self, mean, std, save:bool=False):
        df = pd.DataFrame([mean, std])
        plt.close('all')
        ax = df.boxplot(rot=70, return_type="axes")
        ax.set_ylabel("Relative Importance")
        if save:
            plt.savefig(os.path.join(
                self.opt_path,
                "fanova_importance_hist.png"),
                dpi=300,
                bbox_inches='tight')
        fname = "fanova_importances_raw.json"

        with open(os.path.join(self.opt_path, fname), 'w') as fp:
            json.dump(jsonize(df.to_dict()), fp, indent=4, sort_keys=True)

        return ax

    def _plot_importance_as_barchart(self, importance, save=False):
        df = pd.DataFrame.from_dict(importance, orient='index')
        ax = bar_chart(df, orient='h', show=False,
                       ax_kws={'title': "fANOVA hyperparameter importance",
                               'xlabel': "Relative Importance"})

        fname = "fanova_importances.json"

        if save:
            plt.savefig(os.path.join(self.opt_path, 'fanova_importance_bar.png'),
                        bbox_inches="tight", dpi=300)

        with open(os.path.join(self.opt_path, fname), 'w') as fp:
            json.dump(jsonize(df.to_dict()), fp, indent=4, sort_keys=True)
        return ax

    def _calc_importance_with_optuna(self, plot_type="bar", save=False, show=True):

        from ._optuna_fanova import plot_param_importances

        importances, importance_paras, ax = plot_param_importances(self.optuna_study())

        if plot_type == "bar":
            if save:
                plt.savefig(os.path.join(self.opt_path, 'fanova_importance_bar.png'),
                            bbox_inches="tight", dpi=300)
        else:
            df = pd.DataFrame.from_dict(importance_paras)
            ax = df.boxplot(rot=70, return_type="axes")
            ax.set_ylabel("Relative Importance")
            if save:
                plt.savefig(os.path.join(
                    self.opt_path,
                    "fanova_importance_hist.png"),
                    dpi=300,
                    bbox_inches='tight')

        with open(os.path.join(self.opt_path, "importances.json"), 'w') as fp:
            json.dump(importances, fp, indent=4, sort_keys=True, cls=JsonEncoder)

        with open(os.path.join(self.opt_path, "fanova_importances.json"), 'w') as fp:
            json.dump(importance_paras, fp, indent=4, sort_keys=True, cls=JsonEncoder)

        if show:
            plt.show()

        return ax

    def optuna_study(self):
        """
        Attempts to create an optuna Study instance so that
        optuna based plots can be generated.

        Returns
        None, if not possible else Study
        """
        from optuna.study import Study
        from optuna.trial import TrialState

        if self.backend == 'optuna':
            return self.study

        class _Trial:
            state = TrialState.COMPLETE
            def __init__(self,
                         number:int,
                         values:Union[list, int, float],
                         params:dict,
                         distributions:dict):

                values = jsonize(values)
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
                        raise NotImplementedError(f"""
                        values must be convertible to list but it is {values} of type
                         {values.__class__.__name__} Actual error message was {e}""")
                self.params = params
                self._distributions = distributions
                self.distributions = distributions

        XY_OF_ITERATIONS = self.xy_of_iterations()
        SPACE = self.space()
        BEST_PARAS = self.best_paras()

        class _Study(Study):

            trials = []
            idx = 0

            distributions = {sn: s.to_optuna() for sn, s in SPACE.items()}

            for xy in XY_OF_ITERATIONS.values():
                _x, _y = xy['x'], xy['y']
                assert isinstance(_x, dict), f"""
                params must of type dict but provided params are of type 
                {_x.__class__.__name__}"""

                trials.append(_Trial(number=idx,
                                     values=_y,
                                     params=_x,
                                     distributions=distributions
                                     ))
                idx += 1
            best_params = BEST_PARAS
            best_trial = None
            best_value = None
            _study_id = 0
            _distributions = distributions

            def __init__(self):
                pass

            def _is_multi_objective(self):
                return False

        study = _Study()

        setattr(self, 'study', study)

        return study

    def _plot_distributions(self, save=True, show=True, figsize=None)->plt.Figure:
        """plot distributions of explored hyperparameters"""

        # name of hyperparameters
        h_paras = list(self.best_xy()['x'].keys())

        # container with a list for each hyperparameter
        h_para_lists = {k: [] for k in h_paras}

        for xy in self.xy_of_iterations().values():
            #score = xy['y']
            x_iter = xy['x']
            for para, val in x_iter.items():
                h_para_lists[para].append(val)

        figsize = figsize or (6+len(h_paras), 6+len(h_paras))
        fig, axes = create_subplots(naxes=len(h_paras),
                                    figsize=figsize)

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for ax, col in zip(axes.flat, h_paras):

            labels, bins = np.unique(np.array(h_para_lists[col]), return_counts=True)

            if isinstance(self.space()[col], Real):
                labels = [round(label, 3) for label in labels]

            bar_chart(bins, labels, orient="v", ax=ax, rotation=90, label=col,
                      show=False)
            ax.set_ylabel("Number of iterations")
            ax.legend()

        if save:
            fname = os.path.join(self.opt_path, "distributions.png")
            plt.savefig(fname, bbox_inches="tight")

        if show:
            plt.show()

        return fig

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

    def eval_with_best(self):
        """
        Find the best parameters and evaluate the objective_fn with them.
        Arguments:
            return_model bool: If True, then then the built objective_fn will be returned
        """

        if self.use_named_args:
            x = self.best_paras()
        else:
            x = self.best_paras(True)

        if self.use_named_args:
            return self.objective_fn(**x)

        if callable(self.objective_fn) and not self.use_named_args:
            if isinstance(x, list) and self.backend == 'hyperopt':  # when x = [x]
                if len(x) == 1:
                    x = x[0]
            return self.objective_fn(x)

        raise NotImplementedError

    @classmethod
    def from_gp_parameters(cls, fpath: str, objective_fn):
        """loads results saved from bayesian optimization"""
        opt_path = os.path.dirname(fpath)
        with open(fpath, 'r') as fp:
            gpmin_results = json.load(fp)
        space = gpmin_results['space']
        spaces = []
        for sp_name, sp_paras in space.items():
            if sp_paras['type'] == 'Categorical':
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
                "best_paras": jsonize(self.best_paras()),
                'space': {k: v.serialize() for k, v in self.space().items()},
                'fun_vals': self.func_vals(),
                # 'iters': self.xy_of_iterations(), # todo, for BayesSearchCVs, not getting ys
                'algorithm': self.algorithm,
                'backend': self.backend,
                'opt_path': self.opt_path
                }

    def save_iterations_as_xy(self):

        iterations = self.xy_of_iterations()

        jsonized_iterations = jsonize(iterations)

        fname = os.path.join(self.opt_path, "iterations.json")
        with open(fname, "w") as fp:
            json.dump(jsonized_iterations, fp, sort_keys=False, indent=4, cls=JsonEncoder)

        fname = os.path.join(self.opt_path, "iterations_sorted.json")
        with open(fname, "w") as fp:
            json.dump(dict(sorted(jsonized_iterations.items())), fp, sort_keys=True, indent=4, cls=JsonEncoder)

    def add_previous_results(
            self,
            iterations: Union[dict, str] = None,
            x: list = None,
            y :list = None
    ):
        """adds results from previous iterations.

        If you have run the optimization priviously, you can make use
        of those results by appending them.

        Arguments:
            iterations:
                It can be either a dictionary whose keys are y values and values are x
                or it can be a path to a file which contains these xy values as dictioary.
            x:
                a list of lists where each sub-list is the value of hyperparameter
                at at one iteratio. The `x` and `y` arguments optional and will
                only be used if `iterations` are not provided.
            y:
                a list of float values where each value in y is the output
                of objective_fn with corresponding x. The length of `x` and `y`
                must be equal.
        """

        assert self.algorithm in ["bayes", "bayes_rf"]

        if iterations is None:
            assert isinstance(x, list) and isinstance(y, list)
            assert len(x) == len(y), f"x has {len(x)} values while y has {len(y)} values. They must be equal"
            x0 = x
            y0 = y

        elif isinstance(iterations, str):
            assert os.path.exists(iterations), f"the path {iterations} does not exist"
            # it is a path
            with open(iterations, 'r') as fp:
                iter_dict = json.load(fp)

            x0, y0 = self.dict_to_xy(iter_dict)
        else:
            if not isinstance(iterations, dict):
                raise ValueError(f"iterations must be a dictionary but it is of type {iterations.__class__.__name__}")

            x0, y0 = self.dict_to_xy(iterations)

        # todo check for inf and nan in y0

        self.gpmin_args['x0'] = x0
        self.gpmin_args['y0'] = y0

        return

    @staticmethod
    def dict_to_xy(iterations:dict):
        x0, y0 = [], []
        for y, x in iterations.items():
            y0.append(float(y))
            x0.append(list(x.values()))
        return x0, y0

    def load_results(self, fname:str):
        """loads the previously computed results. It should not
        be used after .fit()

        parameters
        ----------
        fname : str
            complete path of  hpo_results.bin file e.g.
            path/to/hpo_results.bin
        """

        from joblib import load  # some modules may not be dependent upon joblib

        assert len(self.results) == 0, f"""
        Loading results after call to .fit is not allowed.
        Create a new instance of HyperOpt and then call this function.
        """

        if not os.path.exists(fname):
            raise FileNotFoundError(f" File {fname} does not exist")

        new_results =  load(fname)

        self.gpmin_results = new_results

        fv = new_results.func_vals
        xiters = new_results.x_iters
        for idx, y, x in zip(range(len(fv)), fv, xiters):
            self.results[idx] = {'y': y, 'x': x}

        return
