import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skopt.space import Real, Categorical, Integer

from dl4seq.utils.TSErrors import FindErrors
from dl4seq.utils.taylor_diagram import plot_taylor
from dl4seq import Model
from dl4seq.hyper_opt import HyperOpt
from dl4seq.utils.utils import clear_weights



class Experiments(object):
    """Base class for all the experiments.
    All the expriments must be subclasses of this class.
    The core idea of of `Experiments` is `model`. An experiment consists of one or more models. The models differ from
    each other in their structure/idea/concept. When fit() is called, each model is trained.
    """
    def __init__(self, cases=None, exp_name='Experiments'):
        self.trues = {}
        self.simulations  = {}
        self.opt_results = None
        self.optimizer = None
        self.exp_name=exp_name

        self.models = [method for method in dir(self) if callable(getattr(self, method)) if method.startswith('model_')]
        if cases is None:
            cases = {}
        self.cases = cases
        self.models = self.models + list(cases.keys())

    @property
    def build_and_run(self):
        return NotImplementedError

    @property
    def build_from_config(self):
        return NotImplementedError

    def fit(self,
            run_type="dry_run",
            opt_method="bayes",
            n_calls=12,
            include=None,
            exclude=None,
            post_optimize='eval_best',
            hpo_kws: dict = None):
        """
        :param run_type: str, One of `dry_run` or `bayes`. If `dry_run`, the all the `models` will be trained only once.
                              if `optimize`, then hyperparameters of all the models will be optimized.
        :param opt_method: str, which optimization method to use. options are `bayes`, `random`, `grid`. ONly valid if
                           `run_type` is `optimize`
        :param n_calls: int, number of iterations for optimization. Only valid if `run_type` is `optimize`.
        :param include: list, name of models to included. If None, all the models found will be trained and or optimized.
        :param exclude: list, name of `models` to be excluded
        :param post_optimize: str, one of `eval_best` or `train_best`. If eval_best, the weights from the best models
                              will be uploaded again and the model will be evaluated on train, test and all the data.
                              if `train_best`, then a new model will be built and trained using the parameters of the
                              best model.
        :param hpo_kws: keyword arguments for `HyperOpt` class.
        """

        assert run_type in ['optimize', 'dry_run']
        assert post_optimize in ['eval_best', 'train_best']

        predict = False
        if run_type == 'dry_run':
            predict = True

        if hpo_kws is None:
            hpo_kws = {}

        if include is None:
            include = self.models
        else:  # make sure that include contains same elements which are present in models
            assert all(elem in self.models for elem in include)

        if exclude is None:
            exclude = []
        else:
            assert all(elem in self.models for elem in exclude)

        self.trues = {'train': None,
                      'test': None}

        self.simulations = {'train': {},
                            'test': {}}

        for model_type in include:

            model_name = model_type.split('model_')[1]

            if model_type not in exclude:

                def objective_fn(**kwargs):

                    if model_type in self.cases:
                        config = self.cases[model_type]
                    elif model_name in self.cases:
                        config = self.cases[model_name]
                    elif hasattr(self, model_type):
                        config = getattr(self, model_type)(**kwargs)
                    else:
                        raise TypeError

                    return self.build_and_run(predict=predict, title=f"{self.exp_name}\\{model_name}", **config)

                if run_type == 'dry_run':
                    train_results, test_results = objective_fn()
                    self._populate_results(model_name, train_results, test_results)
                else:
                    # there may be attributes int the model, which needs to be loaded so run the method first.
                    if hasattr(self, model_type):
                        getattr(self, model_type)()

                    opt_dir = os.path.join(os.getcwd(), f"results\\{self.exp_name}\\{model_name}")
                    self.optimizer = HyperOpt(opt_method,
                                              model=objective_fn,
                                              param_space=self.dims,
                                              use_named_args=True,
                                              opt_path=opt_dir,
                                              n_calls=n_calls,  # number of iterations
                                              x0=self.x0,
                                              **hpo_kws
                                              )
                    self.opt_results = self.optimizer.fit()

                    if post_optimize == 'eval_best':
                        self.eval_best(model_name, opt_dir)
                    else:
                        self.train_best(model_name)
        return

    def eval_best(self, model_type, opt_dir):
        """Evaluate the best models."""
        best_models = clear_weights(opt_dir, rename=False, write=False)

        for mod, props in best_models.items():
            mod_path = os.path.join(props['path'], "config.json")
            mod_weights = props['weights']

            train_results, test_results = self.build_from_config(mod_path, mod_weights)

            if mod.startswith('1_'):
                self._populate_results(model_type, train_results, test_results)
        return

    def train_best(self, model_type):
        """Train the best model."""
        train_results, test_results = self.build_and_run(predict=True,
                                                         view=True,
                                                         model={model_type: self.optimizer.best_paras_kw()},
                                                         title=f"{self.exp_name}\\{model_type}\\best")

        self._populate_results(model_type, train_results, test_results)
        return

    def _populate_results(self, model_type, train_results, test_results):
        self.trues['train'] = train_results[0]
        self.trues['test'] = test_results[0]

        self.simulations['train'][model_type] = train_results[1]
        self.simulations['test'][model_type] = test_results[1]
        return

    def plot_taylor(self, plot_pbias=True, fig_size=(9,7), add_grid=True, **kwargs):

        plot_taylor(
            trues=self.trues,
            simulations=self.simulations,
            add_grid=add_grid,
            plot_bias=plot_pbias,
            figsize=fig_size,
            save=True,
            **kwargs
        )
        return

    def compare_errors(
            self,
            matric_name: str,
            cutoff_val:float=None,
            cutoff_type:str=None,
            save=False,
            name='ErrorComparison',
            **kwargs
    ):
        """Plots a specific performance matric for all the models which were
        run during experiment.fit().
        :param matric_name: str, performance matric whose value to plot for all the models
        :param cutoff_val: float, if provided, only those models will be plotted for whome the matric is greater/smaller
                                  than this value. This works in conjuction with `cutoff_type`.
        :param cutoff_type: str, one of `greater`, `greater_equal`, `less` or `less_equal`.
                            Criteria to determine cutoff_val. For example if we want to
                            show only those models whose r2 is > 0.5, it will be 'max'.
        :param save: bool, whether to save the plot or not
        :param name: str, name of the saved file.
        kwargs are:
            fig_height:
            fig_width:
            title_fs:
            xlabel_fs:

        Example
        -----------
        >>>experiment = Experiments()
        >>>experiment.compare_errors('mse')
        >>>experiment.compare_errors('r2', 0.5, 'greater')
        """

        def find_matric_array(true, sim):
            errors = FindErrors(true, sim)
            matric_val = getattr(errors, matric_name)()
            if matric_name in ['nse', 'kge']:
                if matric_val < 0.0:
                    matric_val = 0.0

            if cutoff_type is not None:
                assert cutoff_val is not None
                if not getattr(np, cutoff_type)(matric_val, cutoff_val):
                    return None
            return matric_val

        train_matrics = []
        test_matrics = []
        models = []

        for mod in self.models:
            mod = mod.split('model_')[1]
            if mod in self.simulations['test']:  # maybe we have not done some models by using include/exclude
                test_matric = find_matric_array(self.trues['test'], self.simulations['test'][mod])
                if test_matric is not None:
                    test_matrics.append(test_matric)
                    models.append(mod)

                    train_matric = find_matric_array(self.trues['train'], self.simulations['train'][mod])
                    train_matrics.append(train_matric)

        labels = {
            'r2': "$R^{2}$"
        }

        plt.close('all')
        fig, axis = plt.subplots(1, 2, sharey='all')
        fig.set_figheight(kwargs.get('fig_heigt', 8))
        fig.set_figwidth(kwargs.get('fig_width', 8))

        ax = sns.barplot(y=models, x=train_matrics, orient='h', ax=axis[0])
        ax.set_title("Train", fontdict={'fontsize': kwargs.get('title_fs', 20)})
        ax.set_xlabel(labels.get(matric_name, matric_name), fontdict={'fontsize': kwargs.get('xlabel_fs', 16)})

        ax = sns.barplot(y=models, x=test_matrics, orient='h', ax=axis[1])
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel(labels.get(matric_name, matric_name), fontdict={'fontsize': kwargs.get('xlabel_fs', 16)})
        ax.set_title("Test", fontdict={'fontsize': kwargs.get('title_fs', 20)})

        if save:
            fname = f'{name}_{matric_name}.png'
            plt.savefig(fname, dpi=100)
        plt.show()
        return models



class MLRegressionExperiments(Experiments):
    """
    Compares peformance of around 42 machine learning models for regression problem. The experiment consists of
    `models` which are run using `fit()` method. A `model` is one experiment. This class consists of its own
    `build_and_run` method which is run everytime for each `model` and is executed after calling  `model`. The
    `build_and_run` method takes the output of `model` and streams it to `Model` class.

    The user can define new `models` by subclassing this class. In fact any new method in the sub-class which does not
    starts with `model_` wll be considered as a new `model`. Otherwise the user has to overwite the
    attribute `models` to redefine, which methods are to be used as models and which should not. The method which is a
    `model` must only return key word arguments which will be streamed to the `Model` using `build_and_run` method.
    Inside this new method the user can define, which parameters to optimize, their dimensions/space for optimization
    and the initial values to use for optimization.

    Arguments:
        dimension: dimenstions of parameters which are to be optimized. These can be overwritten in `models`.
        x0: initial values of the parameters which are to be optimized. These can be overwritten in `models`
        data: this will be passed to `Model`.
        exp_name: name of experiment, all results will be saved within this folder
        model_kwargs: keyword arguments which are to be passed to `Model` and are not optimized.
    Examples:
    --------
    >>>import pandas as pd
    >>> # first compare the performance of all available models without optimizing their parameters
    >>>data = pd.read_csv('FileName')  # read data file
    >>>inputs, outputs = [], []  # define input and output columns in data
    >>>comparisons = MLRegressionExperiments(data=data, inputs=inputs, outputs=outputs,
    ...                                      input_nans={'SimpleImputer': {'strategy':'mean'}} )
    >>>comparisons.fit(run_type="dry_run")
    >>>comparisons.compare_errors('r2')
    >>> # find out the models which resulted in r2> 0.5
    >>>best_models = comparisons.compare_errors('r2', cutoff_type='greater', cutoff_val=0.5)
    >>> # now build a new experiment for best models and otpimize them
    >>>comparisons = MLRegressionExperiments(data=data, inputs=inputs, outputs=outputs,
    ...                                   input_nans={'SimpleImputer': {'strategy': 'mean'}}, exp_name="BestMLModels")
    >>>comparisons.fit(run_type="optimize", include=best_models)
    >>>comparisons.compare_errors('r2')
    >>>comparisons.plot_taylor()
    """

    def __init__(self, dimensions=None, x0=None, data=None, cases=None, exp_name='MLExperiments', **model_kwargs):
        self.dims = dimensions
        self.x0 = x0
        self.data = data
        self.model_kws = model_kwargs

        super().__init__(cases=cases, exp_name=exp_name)

    def build_and_run(self, predict=False, view=False, title=None, **kwargs):

        model = Model(
            data=self.data,
            prefix=title,
            **self.model_kws,
            **kwargs
        )
        model.fit(indices='random')

        t, p = model.predict(indices=model.train_indices, pref='train')

        if view:
            model.view_model()

        if predict:
            tt, tp = model.predict(indices=model.test_indices, pref='test')
            return (t,p), (tt, tp)

        return FindErrors(t, p).mse()

    def model_ADABOOSTREGRESSOR(self, **kwargs):
        self.dims = [
            Integer(low=5, high=100, name='n_estimators'),
            Real(low=0.001, high=1.0, name='learning_rate')
        ]
        self.x0 = [50, 1.0]
        return {'model': {'ADABOOSTREGRESSOR': kwargs}}

    def model_ARDREGRESSION(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html
        self.dims = [
            Real(low=1e-7, high=1e-5, name='alpha_1'),
            Real(low=1e-7, high=1e-5, name='alpha_2'),
            Real(low=1e-7, high=1e-5, name='lambda_1'),
            Real(low=1e-7, high=1e-5, name='lambda_2'),
            Real(low=1000, high=1e5, name='threshold_lambda'),
            Categorical(categories=[True, False], name='fit_intercept')
        ]
        return {'model': {'ARDREGRESSION': kwargs}}

    def model_BAGGINGREGRESSOR(self, **kwargs):

        self.dims = [
            Integer(low=5, high=50, name='n_estimators'),
            Real(low=0.1, high=1.0, name='max_samples'),
            Real(low=0.1, high=1.0, name='max_features'),
            Categorical(categories=[True, False], name='bootstrap'),
            Categorical(categories=[True, False], name='bootstrap_features'),
            #Categorical(categories=[True, False], name='oob_score'),  # linked with bootstrap
        ]
        self.x0 = [10, 1.0, 1.0, True, False]

        return {'model': {'BAGGINGREGRESSOR': kwargs}}

    def model_BayesianRidge(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html
        self.dims = [
            Integer(low=40, high=1000, name='n_iter'),
            Real(low=1e-7, high=1e-5, name='alpha_1'),
            Real(low=1e-7, high=1e-5, name='alpha_2'),
            Real(low=1e-7, high=1e-5, name='lambda_1'),
            Real(low=1e-7, high=1e-5, name='lambda_2'),
            Categorical(categories=[True, False], name='fit_intercept')
        ]
        self.x0 = [30, 1e-6, 1e-6, 1e-6, 1e-6, True]
        return {'model': {'BayesianRidge': kwargs}}

    def model_CATBOOSTREGRESSOR(self, **kwargs):
        # https://catboost.ai/docs/concepts/python-reference_parameters-list.html
        self.dims = [
            Integer(low=500, high=5000, name='iterations'),  # maximum number of trees that can be built
            Real(low=0.0001, high=0.5, name='learning_rate'), # Used for reducing the gradient step.
            Real(low=0.5, high=5.0, name='l2_leaf_reg'),   # Coefficient at the L2 regularization term of the cost function.
            Real(low=0.1, high=10, name='model_size_reg'),  # arger the value, the smaller the model size.
            Real(low=0.0, high=0.99, name='rsm'),  # percentage of features to use at each split selection, when features are selected over again at random.
            Integer(low=32, high=1032, name='border_count'),  # number of splits for numerical features
            Categorical(categories=['Median', 'Uniform', 'UniformAndQuantiles', # The quantization mode for numerical features.
                                    'MaxLogSum', 'MinEntropy', 'GreedyLogSum'], name='feature_border_type')  # The quantization mode for numerical features.
        ]
        self.x0 = [1000, 0.01, 3.0, 0.5, 0.5, 32, 'GreedyLogSum']
        return {'model': {'CATBOOSTREGRESSOR': kwargs}}

    def model_DECISIONTREEREGRESSOR(self, **kwargs):
        # TODO not converging
        self.dims = [
            Categorical(["best", "random"], name='splitter'),
            Integer(low=2, high=10, name='min_samples_split'),
            #Real(low=1, high=5, name='min_samples_leaf'),
            Real(low=0.0, high=0.5, name="min_weight_fraction_leaf"),
            Categorical(categories=['auto', 'sqrt', 'log2'], name="max_features"),
        ]
        self.x0 = ['best', 5, 0.0, 'auto']

        return {'model': {'DECISIONTREEREGRESSOR': kwargs}}

    def model_DummyRegressor(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/classes.html
        self.dims = [
            Categorical(categories=['mean', 'median', 'quantile', 'constant'], name='strategy')
        ]
        return {'model': {'DummyRegressor': kwargs}}

    def model_ElasticNet(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
        self.dims = [
            Real(low=1.0, high=5.0, name='alpha'),
            Real(low=0.1, high=1.0, name='l1_ratio'),
            Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=500, high=5000, name='max_iter'),
            Real(low=1e-5, high=1e-3, name='tol')
        ]
        return {'model': {'ElasticNet': kwargs}}

    def model_ElasticNetCV(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html
        self.dims = [
            Real(low=0.1, high=1.0, name='l1_ratio'),
            Real(low=1e-5, high=1e-2, name='eps'),
            Integer(low=10, high=1000, name='n_alphas'),
            Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=500, high=5000, name='max_iter'),
        ]
        self.x0 = [0.5, 1e-3, 100, True, 1000]
        return {'model': {'ElasticNetCV': kwargs}}

    def model_EXTRATREEREGRESSOR(self, **kwargs):
        return self.model_DECISIONTREEREGRESSOR(**kwargs)

    def model_EXTRATREESREGRESSOR(self, **kwargs):
        self.dims = [
            Integer(low=5, high=50, name='n_estimators'),
            Integer(low=3, high=30, name='max_depth'),
            Real(low=0.1, high=0.5, name='min_samples_split'),
            #Real(low=0.1, high=1.0, name='min_samples_leaf'),
            Real(low=0.0, high=0.5, name='min_weight_fraction_leaf'),
            Categorical(categories=['auto', 'sqrt', 'log2'], name='max_features')
        ]
        self.x0 = [10, 5,  0.4, #0.2,
                   0.1, 'auto']
        return {'model': {'EXTRATREESREGRESSOR': kwargs}}

    def model_GaussianProcessRegressor(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
        self.dims = [
            Real(low=1e-10, high=1e-7, name='alpha'),
            Integer(low=0, high=5, name='n_restarts_optimizer')
        ]
        return {'model': {'GaussianProcessRegressor': kwargs}}

    def model_GRADIENTBOOSTINGREGRESSOR(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
        self.dims = [
            Integer(low=5, high=500, name='n_estimators'),  # number of boosting stages to perform
            Real(low=0.001, high=1.0, name='learning_rate'),   #  shrinks the contribution of each tree
            Real(low=0.0, high=1.0, name='subsample'),  # fraction of samples to be used for fitting the individual base learners
            Real(low=0.1, high=0.9, name='min_samples_split'),
            Integer(low=2, high=30, name='max_depth'),
        ]
        return {'model': {'GRADIENTBOOSTINGREGRESSOR': kwargs}}

    def model_HISTGRADIENTBOOSTINGREGRESSOR(self, **kwargs):
        ### https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
        # TODO not hpo not converging
        self.dims = [
            Real(low=0.0001, high=0.9, name='learning_rate'),  # Used for reducing the gradient step.
            Integer(low=50, high=500, name='max_iter'),  # maximum number of trees.
            Integer(low=1, high=100, name='max_depth'),  # maximum number of trees.
            Integer(low=10, high=100, name='max_leaf_nodes'),  # maximum number of leaves for each tree
            Integer(low=10, high=100, name='min_samples_leaf'),  # minimum number of samples per leaf
            Real(low=00, high=0.5, name='l2_regularization'),  # Used for reducing the gradient step.
        ]
        self.x0 = [0.1, 100, 10, 31, 20, 0.0]
        return {'model': {'HISTGRADIENTBOOSTINGREGRESSOR':kwargs}}

    def model_HUBERREGRESSOR(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html
        self.dims = [
            Real(low=1.0, high=5.0, name='epsilon'),
            Integer(low=50, high=500, name='max_iter'),
            Real(low=1e-5, high=1e-2, name='alpha'),
            Categorical(categories=[True, False], name='fit_intercept')
        ]
        return {'model': {'HUBERREGRESSOR': kwargs}}

    def model_KernelRidge(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html
        self.dims = [
            Real(low=1.0, high=5.0, name='alpha'),
            Categorical(categories=['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine'])
        ]
        return {'model': {'KernelRidge': kwargs}}

    def model_KNEIGHBORSREGRESSOR(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
        self.dims = [
            Integer(low=3, high=len(self.data), name='n_neighbors'),
            Categorical(categories=['uniform', 'distance'], name='weights'),
            Categorical(categories=['auto', 'ball_tree', 'kd_tree', 'brute'], name='algorithm'),
            Integer(low=10, high=100, name='leaf_size'),
            Integer(low=1, high=5, name='p')
        ]
        self.x0 = [5, 'uniform', 'auto', 30, 2]
        return {'model': {'KNEIGHBORSREGRESSOR': kwargs}}

    def model_LassoLars(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html
        self.dims = [
            Real(low=1.0, high=5.0, name='alpha'),
            Categorical(categories=[True, False], name='fit_intercept')
        ]
        return {'model': {'LassoLars': kwargs}}

    def model_Lars(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html
        self.dims = [
            Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=100, high=1000, name='n_nonzero_coefs')
        ]
        return {'model': {'Lars': kwargs}}

    def model_LarsCV(self, **kwargs):
        self.dims = [
            Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=100, high=1000, name='max_iter'),
            Integer(low=100, high=5000, name='max_n_alphas')
        ]
        self.x0 = [True, 500, 1000]
        return {'model': {'LarsCV': kwargs}}

    def model_LinearSVR(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html
        self.dims = [
            Real(low=1.0, high=5.0, name='C'),
            Real(low=0.01, high=0.9, name='epsilon'),
            Real(low=1e-5, high=1e-1, name='tol'),
            Categorical(categories=[True, False], name='fit_intercept')
        ]
        return {'model': {'LinearSVR': kwargs}}

    def model_Lasso(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
        self.dims = [
            Real(low=1.0, high=5.0, name='alpha'),
            Categorical(categories=[True, False], name='fit_intercept'),
            Real(low=1e-5, high=1e-1, name='tol')
        ]
        return {'model': {'Lasso': kwargs}}

    def model_LassoCV(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html
        self.dims = [
            Real(low=1e-5, high=1e-2, name='eps'),
            Integer(low=10, high=1000, name='n_alphas'),
            Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=500, high=5000, name='max_iter')
        ]
        self.x0 = [1e-3, 100, True, 1000]
        return {'model': {'LassoCV': kwargs}}

    def model_LassoLarsCV(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html
        self.dims = [
            Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=500, high=5000, name='max_n_alphas')
        ]
        return {'model': {'LassoLarsCV': kwargs}}

    def model_LassoLarsIC(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html
        self.dims = [
            Categorical(categories=[True, False], name='fit_intercept'),
            Categorical(categories=['bic', 'aic'], name='criterion')
        ]
        return {'model': {'LassoLarsIC': kwargs}}

    def model_LGBMREGRESSOR(self, **kwargs):
     ## https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
        self.dims = [
            Categorical(categories=['gbdt', 'dart', 'goss', 'rf'], name='boosting_type'),
            Integer(low=10, high=200, name='num_leaves'),
            Real(low=0.0001, high=0.1, name='learning_rate'),
            Integer(low=20, high=500, name='n_estimators'),
        ]

        self.x0 = ['gbdt', 50, 0.1, 50]
        return {'model': {'LGBMREGRESSOR': kwargs}}

    def model_LINEARREGRESSION(self, **kwargs):
        self.dims = [
            Categorical(categories=[True, False], name='fit_intercept')
        ]
        self.x0 = [True]
        return {'model': {'LINEARREGRESSION': kwargs}}

    def model_MLPREGRESSOR(self, **kwargs):
        self.dims = [
            Integer(low=10, high=500, name='hidden_layer_sizes'),
            Categorical(categories=['identity', 'logistic', 'tanh', 'relu'], name='activation'),
            Categorical(categories=['lbfgs', 'sgd', 'adam'], name='solver'),
            Real(low=1e-6, high=1e-3, name='alpha'),
            Real(low=1e-6, high=1e-3, name='learning_rate'),
        ]
        return {'model': {'MLPREGRESSOR': kwargs}}

    def model_NuSVR(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html
        self.dims = [
            Real(low=0.1,high=0.9, name='nu'),
            Real(low=1.0, high=5.0, name='C'),
            Categorical(categories=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], name='kernel'),
        ]
        return {'model': {'NuSVR': kwargs}}

    def model_OrthogonalMatchingPursuit(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html
        self.dims = [
            Categorical(categories=[True, False], name='fit_intercept'),
            Real(low=0.1, high=10, name='tol')
        ]
        return {'model': {'OrthogonalMatchingPursuit': kwargs}}

    def model_OrthogonalMatchingPursuitCV(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuitCV.html
        self.dims = [
            # Integer(low=10, high=100, name='max_iter'),
            Categorical(categories=[True, False], name='fit_intercept')
        ]
        self.x0 = [  # 50,
            True]
        return {'model': {'OrthogonalMatchingPursuitCV': kwargs}}

    def model_ONECLASSSVM(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
        self.dims = [
            Categorical(categories=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], name='kernel'),
            Real(low=0.1, high=0.9, name='nu'),
            Categorical(categories=[True, False], name='shrinking'),
        ]
        return {'model': {'ONECLASSSVM': kwargs}}

    def model_POISSONREGRESSOR(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html
        self.dims = [
            Real(low=0.0, high=1.0, name='alpha'),
            Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=50, high=500, name='max_iter'),
        ]
        self.x0 = [1.0, True, 100]
        return {'model': {'POISSONREGRESSOR': kwargs}}

    def model_RidgeCV(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html
        self.dims = [
            Categorical(categories=[True, False], name='fit_intercept'),
            Categorical(categories=['auto', 'svd', 'eigen'], name='gcv_mode'),
        ]
        return {'model': {'RidgeCV': kwargs}}

    def model_RADIUSNEIGHBORSREGRESSOR(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsRegressor.html
        self.dims = [
            Categorical(categories=['uniform', 'distance'], name='weights'),
            Categorical(categories=['auto', 'ball_tree', 'kd_tree', 'brute'], name='algorithm'),
            Integer(low=10, high=300, name='leaf_size'),
            Integer(low=1,high=5, name='p')
        ]
        return {'model': {'RADIUSNEIGHBORSREGRESSOR': kwargs}}

    def model_RANSACREGRESSOR(self, **kwargs):
        self.dims = [
            Integer(low=10, high=1000, name='max_trials'),
            Real(low=0.01, high=0.99, name='min_samples')
        ]
        return {'model': {'RANSACREGRESSOR': kwargs}}

    def model_RANDOMFORESTREGRESSOR(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        return self.model_EXTRATREEREGRESSOR(**kwargs)

    def model_SVR(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        self.dims = [
            Categorical(categories=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], name='kernel'),
            Real(low=1.0, high=5.0, name='C'),
            Real(low=0.01, high=0.9, name='epsilon')
        ]
        return {'model': {'SVR': kwargs}}

    def model_SGDREGRESSOR(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
        self.dims = [
            Categorical(categories=['l1', 'l2', 'elasticnet'], name='penalty'),
            Real(low=0.0, high=1.0, name='alpha'),
            Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=500, high=5000, name='max_iter'),
            Categorical(categories=['constant', 'optimal', 'invscaling', 'adaptive'], name='learning_rate')
        ]
        self.x0 = ['l2', 0.00001, True, 1000, 'invscaling']
        return {'model': {'SGDREGRESSOR': kwargs}}

    def model_TransformedTargetRegressor(self, **kwargs):
        return {'model': {'TransformedTargetRegressor': kwargs}}

    def model_TWEEDIEREGRESSOR(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html
        self.dims = [
            Real(low=0.0, high=5.0, name='alpha'),
            Categorical(categories=['auto', 'identity', 'log'], name='link'),
            Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=50, high=500, name='max_iter'),
        ]
        self.x0 = [1.0, 'auto', True, 100]
        return {'model': {'TWEEDIEREGRESSOR': kwargs}}

    def model_THEILSENREGRESSOR(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html
        self.dims = [
            Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=30, high=1000, name='max_iter'),
            Real(low=1e-5, high=1e-1, name='tol'),
            ## Integer(low=self.data.shape[1]+1, high=len(self.data), name='n_subsamples')
        ]
        return {'model': {'THEILSENREGRESSOR': kwargs}}

    # TODO
    # def model_GAMMAREGRESSOR(self, **kwargs):
    #     # ValueError: Some value(s) of y are out of the valid range for family GammaDistribution
    #     return {'GAMMAREGRESSOR': {}}

    def model_XGBOOSTRFREGRESSOR(self, **kwargs):
        ## https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFRegressor
        self.dims = [
            Integer(low=5, high=50, name='n_estimators'),  #  Number of gradient boosted trees
            Integer(low=3, high=30, name='max_depth'),     # Maximum tree depth for base learners
            Real(low=0.0001, high=0.5, name='learning_rate'),     #
            Categorical(categories=['gbtree', 'gblinear', 'dart'], name='booster'),
            Real(low=0.1, high=0.9, name='gamma'),  # Minimum loss reduction required to make a further partition on a leaf node of the tree.
            Real(low=0.1, high=0.9, name='min_child_weight '),  # Minimum sum of instance weight(hessian) needed in a child.
            Real(low=0.1, high=0.9, name='max_delta_step '),  # Maximum delta step we allow each tree’s weight estimation to be.
            Real(low=0.1, high=0.9, name='subsample'),  #  Subsample ratio of the training instance.
            Real(low=0.1, high=0.9, name='colsample_bytree'),
            Real(low=0.1, high=0.9, name='colsample_bylevel'),
            Real(low=0.1, high=0.9, name='colsample_bynode'),
            Real(low=0.1, high=0.9, name='reg_alpha'),
            Real(low=0.1, high=0.9, name='reg_lambda'),
        ]
        self.x0 = [10, 5, 5, 'gbtree', 0.2, 0.2, 0.2,
                   0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        return {'model': {'XGBOOSTRFREGRESSOR': kwargs}}

    def model_XGBOOSTREGRESSOR(self, **kwargs):
        # ##https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor
        return self.model_XGBOOSTRFREGRESSOR(**kwargs)



class MLClassificationExperiments(Experiments):
    """Runs classification models for comparison, with or without optimizing hyperparameters."""

    def model_AdaBoostClassifier(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
        self.dims = [
            Integer(low=10, high=500, name='n_estimators'),
            Real(low=1.0, high=5.0, name='learning_rate'),
            Categorical(categories=['SAMME', 'SAMME.R'], name='algorithm')
        ]
        return {'model': {'AdaBoostClassifier': kwargs}}

    def model_ExtraTreeClassifier(self, **kwargs):
        return

    def model_XGBClassifier(self, **kwargs):
        return

    def model_RandomForestClassifier(self, **kwargs):
        return

    def model_LabelSpreading(self, **kwargs):
        return

    def model_(self, **kwargs):
        return

    def model_LabelPropagation(self, **kwargs):
        return

    def model_ExtraTreesClassifier(self, **kwargs):
        return

    def model_LGBMClassifier(self, **kwargs):
        return

    def model_DecisionTreeClassifier(self, **kwargs):
        return

    def model_BaggingClassifier(self, **kwargs):
        return

    def model_SVC(self, **kwargs):
        return

    def model_LinearSVC(self, **kwargs):
        return

    def model_LogisticRegression(self, **kwargs):
        return

    def model_PassiveAggressiveClassifier(self, **kwargs):
        return

    def model_SGDClassifier(self, **kwargs):
        return

    def model_Perceptron(self, **kwargs):
        return

    def model_KNeighborsClassifier(self, **kwargs):
        return

    def model_QuadraticDiscriminantAnalysis(self, **kwargs):
        return

    def model_CalibratedClassifierCV(self, **kwargs):
        return

    def model_RidgeClassifier(self, **kwargs):
        return

    def model_RidgeClassifierCV(self, **kwargs):
        return

    def model_LinearDiscriminantAnalysis(self, **kwargs):
        return

    def model_NuSVC(self, **kwargs):
        return

    def model_BernoulliNB(self, **kwargs):
        return

    def model_NearestCentroid(self, **kwargs):
        return

    def model_DummyClassifier(self, **kwargs):
        return

    def model_CheckingClassifier(self, **kwargs):
        return
