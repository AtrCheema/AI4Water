from skopt import BayesSearchCV
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
from sklearn.model_selection import ParameterGrid, ParameterSampler
import numpy as np
from TSErrors import FindErrors
import json
import matplotlib.pyplot as plt

from dl4seq import Model
from dl4seq.utils import make_model
from dl4seq.utils.utils import post_process_skopt_results


class HyperOpt(object):
    """
    Combines the power of sklearn based GridSeearchCV, RandomizeSearchCV and skopt based BayeSearchCV.
    Sklearn is great but
      - sklearn based SearchCVs apply only on sklearn based models and not on such as on NNs
      - sklearn does not provide Bayesian optimization
    On the other hand BayesSearchCV of skopt library
      - extends sklearn that the sklearn-based regressors/classifiers could be used for Bayesian but then it can be used
        used for sklearn-based regressors/classifiers
      - The gp_minimize function from skopt allows application of Bayesian on any regressor/classifier/model, but in that
        case this will only be Bayesian

    We wish to make a class which allows application of any of the three optimization methods on any type of model/classifier/regressor.
    If the classifier/regressor is of sklearn-based, then for random search, we use RanddomSearchCV, for grid search,
    we use GridSearchCV and for Bayesian, we use BayesSearchCV. On the other hand, if the model is not sklearn-based,
    you will still be able to implement any of the three methods. In such case, the bayesian will be implemented using
    gp_minimize. Random search and grid search will be done by simple iterating over the sample space generated as in
    sklearn based samplers. However, the post-processing of the results is (supposed to be) done same as done in
    RandomSearchCV and GridSearchCV.

    Thus one motivation of this class is to unify GridSearchCV, RandomSearchCV and BayesSearchCV by complimenting each other.
    The second motivation is to extend their abilities.

    All of the above is done (in theory at least) without limiting the capabilities of GridSearchCV, RandomSearchCV and
    BayesSearchCV or complicating their use.

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
    method: str, must be one of "random", "grid" and "bayes", defining which optimization method to use.
    model: callable, It can be either sklearn/xgboost based regressor/classifier or any function whose returned values
                     can act as objective function for the optimization problem.
    param_space: list/dict, the parameters to be optimized. Based upon above scenarios
                   - For scenario 1, if `method` is "grid", then this argument will be passed to GridSearchCV of sklearn
                       as `param_grid`.  If `method` is "random", then these arguments will be passed to RandomizeSearchCV
                       of sklearn as `param_distribution`.  If `method` is "bayes",  then this must be a
                       dictionary of parameter spaces and this argument will be passed to `BayesSearchCV` as
                       `search_spaces`.
                   - For scenario 2, and "method` is "bayes", then this must be either  dictionary of parameter spaces or
                     a list of tuples defining upper and lower bounds of each parameter of the custom function which you
                     used as `model`. These tuples must follow the same sequence as the order of input parameters in
                     your custom model/function. This argument will then be provided to `gp_minnimize` function of skopt.
                     This case will the :ref:`<example>(4) in skopt.
                   - For scenario 3,  if method is grid or random then this argument should be same as in scenario 1. If method
                     is 'bayes', then this must be list of Integer/Categorical/Real skopt.space instances..

    eval_on_best: bool, if True, then after optimization, the model will be evaluated on best parameters and the results
                  will be stored in the folder named "best" inside `title` folder.
    kwargs: dict, For scenario 3, you must provide `dl4seq_args` as dictionary for additional arguments which  are to be
                  passed to initialize dl4seq's Model class. The choice of kwargs depends whether you are using this class
                  For scenario 1 ,the kwargs will be passed to either GridSearchCV, RandomizeSearchCV or BayesSearchCV.
                  For scenario 2, if the `method` is Bayes, then kwargs will be passed to `gp_minimize`.
                  For scenario 2, f your custom model/function accepts named arguments, then an argument `use_named_args`
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
    - title: str, name of the folder in which all results will be saved. By default this is same as name of `method`. For
             `dl4seq` based models, this is more detailed, containing problem type etc.


    Methods
    -----------------
    best_paras_kw: returns the best parameters as dictionary.
    eval_with_best: evaluates the model on best parameters



    References
    --------------
    1 https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
    2 https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV
    3 https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html
    4 https://github.com/scikit-optimize/scikit-optimize/blob/9334d50a1ad5c9f7c013a1c1cb95313a54b83168/examples/bayesian-optimization.py#L109

    """

    def __init__(self,
                 method:str, *,
                 param_space,
                 model=None,
                 eval_on_best=True,
                 **kwargs
                 ):

        if method not in ["random", "grid", "bayes"]:
            raise ValueError("method must be one of random, grid or bayes.")

        self.model = model
        self.method = method
        self.param_space=param_space
        self.dl4seq_args = None
        self.use_named_args = False
        self.title = self.method
        self.results = {}  # internally stored results
        self.gpmin_results = None  #
        self.data = None
        self.eval_on_best=eval_on_best

        self.gpmin_args = self.check_args(**kwargs)

        if self.use_sklearn:
            if self.method == "random":
                self.optfn = RandomizedSearchCV(estimator=model, param_distributions=param_space, **kwargs)
            else:
                self.optfn = GridSearchCV(estimator=model, param_grid=param_space, **kwargs)

        elif self.use_skopt_bayes:
            self.optfn = BayesSearchCV(estimator=model, search_spaces=param_space, **kwargs)

        elif self.use_skopt_gpmin:
            self.fit = self.own_fit

        elif self.use_own:
            self.predict = self._predict
            if self.method == "grid":
                self.fit = self.grid_search
            else:
                self.fit = self.random_search

    def check_args(self, **kwargs):
        kwargs = kwargs.copy()
        if "use_named_args" in kwargs:
            self.use_named_args = kwargs.pop("use_named_args")

        if "dl4seq_args" in kwargs:
            self.dl4seq_args = kwargs.pop("dl4seq_args")
            self.data = kwargs.pop("data")
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
    def use_sklearn(self):
        # will return True if we are to use sklearn's GridSearchCV or RandomSearchCV
        if self.method in ["random", "grid"] and "sklearn" in str(type(self.model)):
            return True
        return False

    @property
    def use_skopt_bayes(self):
        # will return true if we have to use skopt based BayesSearchCV
        if self.method=="bayes" and "sklearn" in str(type(self.model)):
            assert not self.use_sklearn
            return True
        return False

    @property
    def use_skopt_gpmin(self):
        # will return True if we have to use skopt based gp_minimize function. This is to implement Bayesian on
        # non-sklearn based models
        if self.method == "bayes" and "sklearn" not in str(type(self.model)):
            assert not self.use_sklearn
            assert not self.use_skopt_bayes
            return True
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
    def iters(self):
        return self.gpmin_args['n_iter']

    @property
    def best_paras(self):
        if self.use_skopt_gpmin:
            x_iters = self.gpmin_results['x_iters']
            func_vals = self.gpmin_results['func_vals']
            idx = np.argmin(func_vals)
            paras = x_iters[idx]
        else:
            fun = list(sorted(self.results.keys()))[0]
            paras = self.results[fun]

        return paras

    @property
    def opt_path(self):
        path = os.path.join(os.getcwd(), "results\\" + self.title)
        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def dl4seq_model(self, pp=False,
                     title=None,
                     **kwargs):

        # this is for it to make json serializable.
        for k,v in kwargs.items():
            if 'int' in v.__class__.__name__:
                kwargs[k] = int(v)
            if 'float' in v.__class__.__name__:
                kwargs[k] = float(v)

        config = make_model(ml_model_args=kwargs, **self.dl4seq_args)

        assert config["model_config"]["ml_model"] is not None, "Currently supported only for ml models. Make your own" \
                                                               " dl4seq model and pass it as custom model."
        if title is None:
            self.title = self.method + '_' + config["model_config"]["problem"] + '_' + config["model_config"]["ml_model"]
        else:
            self.title = title

        model = Model(config,
                      data=self.data,
                      prefix=self.title,
                      verbosity=0)
        model.train(indices="random")

        t, p = model.predict(indices=model.test_indices, pp=pp)
        mse = FindErrors(t, p).mse()

        error = round(mse, 6)
        self.results[str(error)] = kwargs

        print(f"Validation mse {error}")

        return error

    def dims(self):

        if isinstance(self.param_space, dict):
            return list(self.param_space.values())

        return list(self.param_space)

    def model_for_gpmin(self):
        """This function can be called in two cases:
            - The user has made its own model.
            - We make model using dl4seq and return the error.
          In first case, we just return what user has provided.
          """
        if callable(self.model) and not self.use_named_args:
            # external function for bayesian but this function does not require named args.
            return self.model

        dims = self.dims()
        if self.use_named_args and self.dl4seq_args is None:
            # external function and this function accepts named args.
            @use_named_args(dimensions=dims)
            def fitness(**kwargs):
                return self.model(**kwargs)
            return fitness

        if self.use_named_args and self.dl4seq_args is not None:
            # using in-build dl4seq_model as objective function.
            @use_named_args(dimensions=dims)
            def fitness(**kwargs):
                return self.dl4seq_model(**kwargs)
            return fitness

        raise ValueError(f"used named args is {self.use_named_args}")

    def own_fit(self):

        search_result = gp_minimize(func=self.model_for_gpmin(),
                                    dimensions=self.dims(),
                                    **self.gpmin_args)

        self.gpmin_results = search_result

        post_process_skopt_results(search_result, self.results, self.opt_path)

        if self.eval_on_best:
            self.eval_with_best()

        return search_result

    def eval_sequence(self, params):

        print(f"total number of iterations: {len(params)}")
        for para in params:

            err = self.dl4seq_model(**para)
            err = round(err, 6)

            if self.dl4seq_args is not None:
                self.results[str(err)] = para

        fname = os.path.join(self.opt_path, "eval_results.json")
        with open(fname, "w") as fp:
            json.dump(self.results, fp, sort_keys=True, indent=4)

        self._plot_convergence()

        if self.eval_on_best:
            self.eval_with_best()

        return self.results

    def grid_search(self):

        params = list(ParameterGrid(self.param_space))
        self.param_grid = params

        return self.eval_sequence(params)

    def random_search(self):

        param_list = list(ParameterSampler(self.param_space, n_iter=self.iters,
                                           random_state=self.random_state))
        self.param_grid = param_list

        return self.eval_sequence(param_list)

    def _predict(self, *args, **params):

        if self.use_named_args and self.dl4seq_args is not None:
            return self.dl4seq_model(pp=True, **params)

        if self.use_named_args and self.dl4seq_args is None:
            return self.model(**params)

        if callable(self.model) and not self.use_named_args:
            return self.model(*args)

    def _plot_convergence(self):
        class sr:
            def __init__(self, results):
                self.x_iters = list(results.values())
                self.func_vals = np.array(list(results.keys()), dtype=np.float32)

        res = sr(self.results)
        plot_convergence([res])

        fname = os.path.join(self.opt_path, "convergence.png")
        return plt.savefig(fname, dpi=300)

    def best_paras_kw(self)->dict:
        """Returns a dictionary consisting of best parameters with their names as keys and their values as keys."""
        x = self.best_paras
        if isinstance(x, dict):
            return x

        names = []
        if isinstance(self.param_space, list):

            for para in self.param_space:

                if isinstance(para, dict):
                    names.append(list(para.keys())[0])
                    # each dictionary must contain only one key in this case
                    assert len(para) == 1
                else:
                    names.append(para.name)

        elif isinstance(self.param_space, dict):
            for key in self.param_space.keys():
                names.append(key)
        else:
            raise NotImplementedError

        xkv = {}
        for name, val in zip(names, x):
            xkv[name] = val

        return xkv

    def eval_with_best(self):
        """Find the best parameters and evaluate the model on them."""
        x = self.best_paras

        if self.use_named_args:
            x = self.best_paras_kw()

        if self.use_named_args and self.dl4seq_args is not None:
            return self.dl4seq_model(pp=True,
                                     title=os.path.join(self.opt_path, "best"),
                                     **x)

        if self.use_named_args and self.dl4seq_args is None:
            return self.model(**x)

        if callable(self.model) and not self.use_named_args:
            return self.model(x)

        raise NotImplementedError