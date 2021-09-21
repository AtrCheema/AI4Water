

from .experiments import Experiments
from ai4water.hyper_opt import Real, Categorical, Integer
from ai4water.post_processing.SeqMetrics import RegressionMetrics
from ai4water.backend import VERSION_INFO, tf

if tf is not None:
    if 230 <= int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 250:
        from ai4water.functional import Model
        print(f"Switching to functional API due to tensorflow version {tf.__version__} for experiments")
    else:
        from ai4water import Model

try:
    import catboost
except ModuleNotFoundError:
    catboost = None

try:
    import lightgbm
except ModuleNotFoundError:
    lightgbm = None

try:
    import xgboost
except ModuleNotFoundError:
    xgboost = None


class MLRegressionExperiments(Experiments):
    """
    Compares peformance of 40+ machine learning models for a regression problem.
    The experiment consists of `models` which are run using `fit()` method. A `model`
    is one experiment. This class consists of its own `build_and_run` method which
    is run everytime for each `model` and is executed after calling  `model`. The
    `build_and_run` method takes the output of `model` and streams it to `Model` class.

    The user can define new `models` by subclassing this class. In fact any new
    method in the sub-class which does not starts with `model_` wll be considered
    as a new `model`. Otherwise the user has to overwite the attribute `models` to
    redefine, which methods are to be used as models and which should not. The
    method which is a `model` must only return key word arguments which will be
    streamed to the `Model` using `build_and_run` method. Inside this new method
    the user can define, which parameters to optimize, their param_space for optimization
    and the initial values to use for optimization.

    """

    def __init__(self,
                 param_space=None,
                 x0=None,
                 data=None,
                 cases=None,
                 ai4water_model=None,
                 exp_name='MLExperiments',
                 num_samples=5,
                 **model_kwargs):
        """

        Arguments:
            param_space: dimensions of parameters which are to be optimized. These
                can be overwritten in `models`.
            x0 list: initial values of the parameters which are to be optimized.
                These can be overwritten in `models`
            data: this will be passed to `Model`.
            exp_name str: name of experiment, all results will be saved within this folder
            model_kwargs dict: keyword arguments which are to be passed to `Model`
                and are not optimized.

        Examples:
        --------
        ```python
        >>>from ai4water.datasets import arg_beach
        >>>from ai4water.experiments import MLRegressionExperiments
        >>> # first compare the performance of all available models without optimizing their parameters
        >>>data = arg_beach()  # read data file, in this case load the default data
        >>>inputs = list(data.columns)[0:-1]  # define input and output columns in data
        >>>outputs = list(data.columns)[-1]
        >>>comparisons = MLRegressionExperiments(data=data, input_features=inputs, output_features=outputs,
        ...                                      nan_filler= {'method': 'KNNImputer', 'features': inputs} )
        >>>comparisons.fit(run_type="dry_run")
        >>>comparisons.compare_errors('r2')
        >>> # find out the models which resulted in r2> 0.5
        >>>best_models = comparisons.compare_errors('r2', cutoff_type='greater', cutoff_val=0.3)
        >>>best_models = [m[1] for m in best_models.values()]
        >>> # now build a new experiment for best models and otpimize them
        >>>comparisons = MLRegressionExperiments(data=data, inputs_features=inputs, output_features=outputs,
        ...                                   nan_filler= {'method': 'KNNImputer', 'features': inputs}, exp_name="BestMLModels")
        >>>comparisons.fit(run_type="optimize", include=best_models)
        >>>comparisons.compare_errors('r2')
        >>>comparisons.taylor_plot()  # see help(comparisons.taylor_plot()) to tweak the taylor plot
        ```
        """
        self.param_space = param_space
        self.x0 = x0
        self.data = data
        self.model_kws = model_kwargs
        self.ai4water_model = Model if ai4water_model is None else ai4water_model

        super().__init__(cases=cases, exp_name=exp_name, num_samples=num_samples)

        if catboost is None:
            self.models.remove('model_CATBoostRegressor')
        if lightgbm is None:
            self.models.remove('model_LGBMRegressor')
        if xgboost is None:
            self.models.remove('model_XGBoostRFRegressor')

        if int(VERSION_INFO['sklearn'].split('.')[1]) < 23:
            for m in ['model_PoissonRegressor', 'model_TweedieRegressor']:
                self.models.remove(m)

    def build_and_run(self,
                      predict=True,
                      view=False,
                      title=None,
                      fit_kws=None,
                      cross_validate=False,
                      **kwargs):

        """
        Builds and run one 'model' of the experiment.

        Since an experiment consists of many models, this method
        is also run many times.
        """
        if fit_kws is None:
            fit_kws = {}

        verbosity = 0
        if 'verbosity' in self.model_kws:
            verbosity = self.model_kws.pop('verbosity')

        model = self.ai4water_model(
            data=self.data,
            prefix=title,
            verbosity=verbosity,
            **self.model_kws,
            **kwargs
        )

        setattr(self, '_model', model)

        if cross_validate:
            val_score = model.cross_val_score(model.config['val_metric'])
        else:
            model.fit(**fit_kws)
            vt, vp = model.predict('validation')
            val_score =  getattr(RegressionMetrics(vt, vp), model.config['val_metric'])()

        tt, tp = model.predict('test')

        if view:
            model.view_model()

        if predict:
            t, p = model.predict('training')

            return (t,p), (tt, tp)

        if model.config['val_metric'] in ['r2', 'nse', 'kge', 'r2_mod']:
            val_score = 1.0 - val_score

        return val_score

    def model_ADABoostRegressor(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
        self.param_space = [
            Integer(low=5, high=100, name='n_estimators', num_samples=self.num_samples),
            Real(low=0.001, high=1.0, prior='log', name='learning_rate', num_samples=self.num_samples)
        ]
        self.x0 = [50, 1.0]
        return {'model': {'ADABOOSTREGRESSOR': kwargs}}

    def model_ARDRegressor(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html
        self.param_space = [
            Real(low=1e-7, high=1e-5, name='alpha_1', num_samples=self.num_samples),
            Real(low=1e-7, high=1e-5, name='alpha_2', num_samples=self.num_samples),
            Real(low=1e-7, high=1e-5, name='lambda_1', num_samples=self.num_samples),
            Real(low=1e-7, high=1e-5, name='lambda_2', num_samples=self.num_samples),
            Real(low=1000, high=1e5, name='threshold_lambda', num_samples=self.num_samples),
            Categorical(categories=[True, False], name='fit_intercept')
        ]
        self.x0 = [1e-7, 1e-7, 1e-7, 1e-7, 1000, True]
        return {'model': {'ARDREGRESSION': kwargs}}

    def model_BaggingRegressor(self, **kwargs):

        self.param_space = [
            Integer(low=5, high=50, name='n_estimators', num_samples=self.num_samples),
            Real(low=0.1, high=1.0, name='max_samples', num_samples=self.num_samples),
            Real(low=0.1, high=1.0, name='max_features', num_samples=self.num_samples),
            Categorical(categories=[True, False], name='bootstrap'),
            Categorical(categories=[True, False], name='bootstrap_features'),
            #Categorical(categories=[True, False], name='oob_score'),  # linked with bootstrap
        ]
        self.x0 = [10, 1.0, 1.0, True, False]

        return {'model': {'BAGGINGREGRESSOR': kwargs}}

    def model_BayesianRidge(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html
        self.param_space = [
            Integer(low=40, high=1000, name='n_iter', num_samples=self.num_samples),
            Real(low=1e-7, high=1e-5, name='alpha_1', num_samples=self.num_samples),
            Real(low=1e-7, high=1e-5, name='alpha_2', num_samples=self.num_samples),
            Real(low=1e-7, high=1e-5, name='lambda_1', num_samples=self.num_samples),
            Real(low=1e-7, high=1e-5, name='lambda_2', num_samples=self.num_samples),
            Categorical(categories=[True, False], name='fit_intercept')
        ]
        self.x0 = [40, 1e-7, 1e-7, 1e-7, 1e-7, True]
        return {'model': {'BayesianRidge': kwargs}}

    def model_CATBoostRegressor(self, **kwargs):
        # https://catboost.ai/docs/concepts/python-reference_parameters-list.html
        self.param_space = [
            Integer(low=500, high=5000, name='iterations', num_samples=self.num_samples),  # maximum number of trees that can be built
            Real(low=0.0001, high=0.5, prior='log', name='learning_rate', num_samples=self.num_samples), # Used for reducing the gradient step.
            Real(low=0.5, high=5.0, name='l2_leaf_reg', num_samples=self.num_samples),   # Coefficient at the L2 regularization term of the cost function.
            Real(low=0.1, high=10, name='model_size_reg', num_samples=self.num_samples),  # arger the value, the smaller the model size.
            Real(low=0.1, high=0.95, name='rsm', num_samples=self.num_samples),  # percentage of features to use at each split selection, when features are selected over again at random.
            Integer(low=32, high=1032, name='border_count', num_samples=self.num_samples),  # number of splits for numerical features
            Categorical(categories=['Median', 'Uniform', 'UniformAndQuantiles', # The quantization mode for numerical features.
                                    'MaxLogSum', 'MinEntropy', 'GreedyLogSum'], name='feature_border_type')  # The quantization mode for numerical features.
        ]
        self.x0 = [1000, 0.01, 3.0, 0.5, 0.5, 32, 'GreedyLogSum']
        return {'model': {'CATBOOSTREGRESSOR': kwargs}}

    def model_DecisionTreeRegressor(self, **kwargs):
        # TODO not converging
        self.param_space = [
            Categorical(["best", "random"], name='splitter'),
            Integer(low=2, high=10, name='min_samples_split', num_samples=self.num_samples),
            #Real(low=1, high=5, name='min_samples_leaf'),
            Real(low=0.0, high=0.5, name="min_weight_fraction_leaf", num_samples=self.num_samples),
            Categorical(categories=['auto', 'sqrt', 'log2'], name="max_features"),
        ]
        self.x0 = ['best', 2, 0.0, 'auto']

        return {'model': {'DECISIONTREEREGRESSOR': kwargs}}

    def model_DummyRegressor(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/classes.html
        self.param_space = [
            Categorical(categories=['mean', 'median', 'quantile'], name='strategy')
        ]

        kwargs.update({'constant': 0.2,
                'quantile': 0.2})

        self.x0 = ['quantile']
        return {'model': {'DummyRegressor': kwargs}}

    def model_ElasticNet(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
        self.param_space = [
            Real(low=1.0, high=5.0, name='alpha', num_samples=self.num_samples),
            Real(low=0.1, high=1.0, name='l1_ratio', num_samples=self.num_samples),
            Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=500, high=5000, name='max_iter', num_samples=self.num_samples),
            Real(low=1e-5, high=1e-3, name='tol', num_samples=self.num_samples)
        ]
        self.x0 = [2.0, 0.2, True, 1000, 1e-4]
        return {'model': {'ElasticNet': kwargs}}

    def model_ElasticNetCV(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html
        self.param_space = [
            Real(low=0.1, high=1.0, name='l1_ratio', num_samples=self.num_samples),
            Real(low=1e-5, high=1e-2, name='eps', num_samples=self.num_samples),
            Integer(low=10, high=1000, name='n_alphas', num_samples=self.num_samples),
            Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=500, high=5000, name='max_iter', num_samples=self.num_samples),
        ]
        self.x0 = [0.5, 1e-3, 100, True, 1000]
        return {'model': {'ElasticNetCV': kwargs}}

    def model_ExtraTreeRegressor(self, **kwargs):
        self.param_space = [
            Integer(low=3, high=30, name='max_depth', num_samples=self.num_samples),
            Real(low=0.1, high=0.5, name='min_samples_split', num_samples=self.num_samples),
            Real(low=0.0, high=0.5, name='min_weight_fraction_leaf', num_samples=self.num_samples),
            Categorical(categories=['auto', 'sqrt', 'log2'], name='max_features')
        ]
        self.x0 = [5, 0.2, 0.2, 'auto']
        return {'model': {'ExtraTreeRegressor': kwargs}}

    def model_ExtraTreesRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
        self.param_space = [
            Integer(low=5, high=500, name='n_estimators', num_samples=self.num_samples),
            Integer(low=3, high=30, name='max_depth', num_samples=self.num_samples),
            Integer(low=2, high=10, name='min_samples_split', num_samples=self.num_samples),
            Integer(low=1, high=10, num_samples=self.num_samples, name='min_samples_leaf'),
            Real(low=0.0, high=0.5, name='min_weight_fraction_leaf', num_samples=self.num_samples),
            Categorical(categories=['auto', 'sqrt', 'log2'], name='max_features')
        ]
        self.x0 = [100, 5,  2, 1,
                   0.0, 'auto']
        return {'model': {'ExtraTreesRegressor': kwargs}}


    # def model_GammaRegressor(self, **kwargs):
    #     ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html?highlight=gammaregressor
    #     self.param_space = [
    #         Real(low=0.0, high=1.0, name='alpha', num_samples=self.num_samples),
    #         Integer(low=50, high=500, name='max_iter', num_samples=self.num_samples),
    #         Real(low= 1e-6, high= 1e-2, name='tol', num_samples=self.num_samples),
    #         Categorical(categories=[True, False], name='warm_start'),
    #         Categorical(categories=[True, False], name='fit_intercept')
    #     ]
    #     self.x0 = [0.5, 100,1e-6, True, True]
    #     return {'model': {'GammaRegressor': kwargs}}


    def model_GaussianProcessRegressor(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
        self.param_space = [
            Real(low=1e-10, high=1e-7, name='alpha', num_samples=self.num_samples),
            Integer(low=0, high=5, name='n_restarts_optimizer', num_samples=self.num_samples)
        ]
        self.x0 = [1e-10, 1]
        return {'model': {'GaussianProcessRegressor': kwargs}}

    def model_GradientBoostingRegressor(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
        self.param_space = [
            Integer(low=5, high=500, name='n_estimators', num_samples=self.num_samples),  # number of boosting stages to perform
            Real(low=0.001, high=1.0, prior='log', name='learning_rate', num_samples=self.num_samples),   #  shrinks the contribution of each tree
            Real(low=0.1, high=1.0, name='subsample', num_samples=self.num_samples),  # fraction of samples to be used for fitting the individual base learners
            Real(low=0.1, high=0.9, name='min_samples_split', num_samples=self.num_samples),
            Integer(low=2, high=30, name='max_depth', num_samples=self.num_samples),
        ]
        self.x0 = [5, 0.001, 1, 0.1, 3]
        return {'model': {'GRADIENTBOOSTINGREGRESSOR': kwargs}}

    def model_HistGradientBoostingRegressor(self, **kwargs):
        ### https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
        # TODO not hpo not converging
        self.param_space = [
            Real(low=0.0001, high=0.9, prior='log', name='learning_rate', num_samples=self.num_samples),  # Used for reducing the gradient step.
            Integer(low=50, high=500, name='max_iter', num_samples=self.num_samples),  # maximum number of trees.
            Integer(low=2, high=100, name='max_depth', num_samples=self.num_samples),  # maximum number of trees.
            Integer(low=10, high=100, name='max_leaf_nodes', num_samples=self.num_samples),  # maximum number of leaves for each tree
            Integer(low=10, high=100, name='min_samples_leaf', num_samples=self.num_samples),  # minimum number of samples per leaf
            Real(low=00, high=0.5, name='l2_regularization', num_samples=self.num_samples),  # Used for reducing the gradient step.
        ]
        self.x0 = [0.1, 100, 10, 31, 20, 0.0]
        return {'model': {'HISTGRADIENTBOOSTINGREGRESSOR':kwargs}}

    def model_HuberRegressor(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html
        self.param_space = [
            Real(low=1.0, high=5.0, name='epsilon', num_samples=self.num_samples),
            Integer(low=50, high=500, name='max_iter', num_samples=self.num_samples),
            Real(low=1e-5, high=1e-2, name='alpha', num_samples=self.num_samples),
            Categorical(categories=[True, False], name='fit_intercept')
        ]
        self.x0 = [2.0, 50, 1e-5, False]
        return {'model': {'HUBERREGRESSOR': kwargs}}

    def model_KernelRidge(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html
        self.param_space = [
            Real(low=1.0, high=5.0, name='alpha', num_samples=self.num_samples)
 #           Categorical(categories=['poly', 'linear', name='kernel'])
        ]
        self.x0 = [1.0] #, 'linear']
        return {'model': {'KernelRidge': kwargs}}

    def model_KNeighborsRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
        if hasattr(self.ai4water_model, 'config'):
            train_frac = self.ai4water_model.config['train_fraction']
        else:
            train_frac = self.model_kws.get('train_fraction', 0.2)
        train_data_length = train_frac * len(self.data)
        self.param_space = [
            Integer(low=3, high=train_data_length, name='n_neighbors', num_samples=self.num_samples),
            Categorical(categories=['uniform', 'distance'], name='weights'),
            Categorical(categories=['auto', 'ball_tree', 'kd_tree', 'brute'], name='algorithm'),
            Integer(low=10, high=100, name='leaf_size', num_samples=self.num_samples),
            Integer(low=1, high=5, name='p', num_samples=self.num_samples)
        ]
        self.x0 = [5, 'uniform', 'auto', 30, 2]
        return {'model': {'KNEIGHBORSREGRESSOR': kwargs}}

    def model_LassoLars(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html
        self.param_space = [
            Real(low=1.0, high=5.0, name='alpha', num_samples=self.num_samples),
            Categorical(categories=[True, False], name='fit_intercept')
        ]
        self.x0 = [1.0, False]
        return {'model': {'LassoLars': kwargs}}

    def model_Lars(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html
        self.param_space = [
            Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=100, high=1000, name='n_nonzero_coefs', num_samples=self.num_samples)
        ]
        self.x0 = [True, 100]
        return {'model': {'Lars': kwargs}}

    def model_LarsCV(self, **kwargs):
        self.param_space = [
            Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=100, high=1000, name='max_iter', num_samples=self.num_samples),
            Integer(low=100, high=5000, name='max_n_alphas', num_samples=self.num_samples)
        ]
        self.x0 = [True, 500, 1000]
        return {'model': {'LarsCV': kwargs}}

    def model_LinearSVR(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html
        self.param_space = [
            Real(low=1.0, high=5.0, name='C', num_samples=self.num_samples),
            Real(low=0.01, high=0.9, name='epsilon', num_samples=self.num_samples),
            Real(low=1e-5, high=1e-1, name='tol', num_samples=self.num_samples),
            Categorical(categories=[True, False], name='fit_intercept')
        ]
        self.x0 = [1.0, 0.01, 1e-5, True]
        return {'model': {'LinearSVR': kwargs}}

    def model_Lasso(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
        self.param_space = [
            Real(low=1.0, high=5.0, name='alpha', num_samples=self.num_samples),
            Categorical(categories=[True, False], name='fit_intercept'),
            Real(low=1e-5, high=1e-1, name='tol', num_samples=self.num_samples)
        ]
        self.x0 = [1.0, True, 1e-5]
        return {'model': {'Lasso': kwargs}}

    def model_LassoCV(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html
        self.param_space = [
            Real(low=1e-5, high=1e-2, name='eps', num_samples=self.num_samples),
            Integer(low=10, high=1000, name='n_alphas', num_samples=self.num_samples),
            Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=500, high=5000, name='max_iter', num_samples=self.num_samples)
        ]
        self.x0 = [1e-3, 100, True, 1000]
        return {'model': {'LassoCV': kwargs}}

    def model_LassoLarsCV(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html
        self.param_space = [
            Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=500, high=5000, name='max_n_alphas', num_samples=self.num_samples)
        ]
        self.x0 = [True, 1000]
        return {'model': {'LassoLarsCV': kwargs}}

    def model_LassoLarsIC(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html
        self.param_space = [
            Categorical(categories=[True, False], name='fit_intercept'),
            Categorical(categories=['bic', 'aic'], name='criterion')
        ]
        self.x0 = [True, 'bic']
        return {'model': {'LassoLarsIC': kwargs}}

    def model_LGBMRegressor(self, **kwargs):
     ## https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
        self.param_space = [
            Categorical(categories=['gbdt', 'dart', 'goss'], name='boosting_type'),  # todo, during optimization not working with 'rf'
            Integer(low=10, high=200, name='num_leaves', num_samples=self.num_samples),
            Real(low=0.0001, high=0.1,  name='learning_rate', prior='log', num_samples=self.num_samples),
            Integer(low=20, high=500, name='n_estimators', num_samples=self.num_samples)
        ]
        self.x0 = ['gbdt', 31, 0.1, 100]
        return {'model': {'LGBMREGRESSOR': kwargs}}

    def model_LinearRegression(self, **kwargs):
        self.param_space = [
            Categorical(categories=[True, False], name='fit_intercept')
        ]
        self.x0 = [True]
        return {'model': {'LINEARREGRESSION': kwargs}}

    def model_MLPRegressor(self, **kwargs):
        self.param_space = [
            Integer(low=10, high=500, name='hidden_layer_sizes', num_samples=self.num_samples),
            Categorical(categories=['identity', 'logistic', 'tanh', 'relu'], name='activation'),
            Categorical(categories=['lbfgs', 'sgd', 'adam'], name='solver'),
            Real(low=1e-6, high=1e-3, name='alpha', num_samples=self.num_samples),
            # Real(low=1e-6, high=1e-3, name='learning_rate')
            Categorical(categories=['constant', 'invscaling', 'adaptive'], name='learning_rate'),
        ]
        self.x0 = [10, 'relu', 'adam', 1e-6,  'constant']
        return {'model': {'MLPREGRESSOR': kwargs}}

    def model_NuSVR(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html
        self.param_space = [
            Real(low=0.5,high=0.9, name='nu', num_samples=self.num_samples),
            Real(low=1.0, high=5.0, name='C', num_samples=self.num_samples),
            Categorical(categories=['linear', 'poly', 'rbf', 'sigmoid'], name='kernel')
        ]
        self.x0 = [0.5, 1.0, 'sigmoid']
        return {'model': {'NuSVR': kwargs}}

    def model_OrthogonalMatchingPursuit(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html
        self.param_space = [
            Categorical(categories=[True, False], name='fit_intercept'),
            Real(low=0.1, high=10, name='tol', num_samples=self.num_samples)
        ]
        self.x0 = [True, 0.1]
        return {'model': {'OrthogonalMatchingPursuit': kwargs}}

    def model_OrthogonalMatchingPursuitCV(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuitCV.html
        self.param_space = [
            # Integer(low=10, high=100, name='max_iter'),
            Categorical(categories=[True, False], name='fit_intercept')
        ]
        self.x0 = [  # 50,
            True]
        return {'model': {'OrthogonalMatchingPursuitCV': kwargs}}

    def model_OneClassSVM(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
        self.param_space = [
            Categorical(categories=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], name='kernel'),
            Real(low=0.1, high=0.9, name='nu', num_samples=self.num_samples),
            Categorical(categories=[True, False], name='shrinking'),
        ]
        self.x0 = ['rbf', 0.1, True]
        return {'model': {'ONECLASSSVM': kwargs}}

    def model_PoissonRegressor(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html
        self.param_space = [
            Real(low=0.0, high=1.0, name='alpha', num_samples=self.num_samples),
            #Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=50, high=500, name='max_iter', num_samples=self.num_samples),
        ]
        self.x0 = [0.5, 100]
        return {'model': {'POISSONREGRESSOR': kwargs}}

    def model_Ridge(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
        self.param_space = [
            Real(low=0.0, high=3.0, name='alpha', num_samples=self.num_samples),
            Categorical(categories=[True, False], name='fit_intercept'),
            Categorical(categories=['auto', 'svd', 'cholesky', 'saga'], name='solver'),
        ]
        self.x0 = [1.0, True, 'auto']
        return {'model': {'Ridge': kwargs}}

    def model_RidgeCV(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html
        self.param_space = [
            Categorical(categories=[True, False], name='fit_intercept'),
            Categorical(categories=['auto', 'svd', 'eigen'], name='gcv_mode'),
        ]
        self.x0 = [True, 'auto']
        return {'model': {'RidgeCV': kwargs}}

    def model_RadiusNeighborsRegressor(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsRegressor.html
        self.param_space = [
            Categorical(categories=['uniform', 'distance'], name='weights'),
            Categorical(categories=['auto', 'ball_tree', 'kd_tree', 'brute'], name='algorithm'),
            Integer(low=10, high=300, name='leaf_size', num_samples=self.num_samples),
            Integer(low=1,high=5, name='p', num_samples=self.num_samples)
        ]
        self.x0 = ['uniform', 'auto', 10, 1]
        return {'model': {'RADIUSNEIGHBORSREGRESSOR': kwargs}}

    def model_RANSACRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html
        self.param_space = [
            Integer(low=10, high=1000, name='max_trials'),
            Real(low=0.01, high=0.99, name='min_samples', num_samples=self.num_samples)
        ]
        self.x0 = [10, 0.01]
        return {'model': {'RANSACREGRESSOR': kwargs}}

    def model_RandomForestRegressor(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        self.param_space = [
            Integer(low=5, high=50, name='n_estimators', num_samples=self.num_samples),
            Integer(low=3, high=30, name='max_depth', num_samples=self.num_samples),
            Real(low=0.1, high=0.5, name='min_samples_split', num_samples=self.num_samples),
            # Real(low=0.1, high=1.0, name='min_samples_leaf'),
            Real(low=0.0, high=0.5, name='min_weight_fraction_leaf', num_samples=self.num_samples),
            Categorical(categories=['auto', 'sqrt', 'log2'], name='max_features')
        ]
        self.x0 = [10, 5, 0.4,  # 0.2,
                       0.1, 'auto']
        return {'model': {'RANDOMFORESTREGRESSOR': kwargs}}

    def model_SVR(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        self.param_space = [
            # https://stackoverflow.com/questions/60015497/valueerror-precomputed-matrix-must-be-a-square-matrix-input-is-a-500x29243-mat
            Categorical(categories=['linear', 'poly', 'rbf', 'sigmoid'], name='kernel'), # todo, optimization not working with 'precomputed'
            Real(low=1.0, high=5.0, name='C', num_samples=self.num_samples),
            Real(low=0.01, high=0.9, name='epsilon', num_samples=self.num_samples)
        ]
        self.x0 = ['rbf',1.0, 0.01]
        return {'model': {'SVR': kwargs}}

    def model_SGDRegressor(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
        self.param_space = [
            Categorical(categories=['l1', 'l2', 'elasticnet'], name='penalty'),
            Real(low=0.01, high=1.0, name='alpha', num_samples=self.num_samples),
            Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=500, high=5000, name='max_iter', num_samples=self.num_samples),
            Categorical(categories=['constant', 'optimal', 'invscaling', 'adaptive'], name='learning_rate')
        ]
        self.x0 = ['l2', 0.1, True, 1000, 'invscaling']
        return {'model': {'SGDREGRESSOR': kwargs}}

    # def model_TransformedTargetRegressor(self, **kwargs):
    #     ## https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html
    #     self.param_space = [
    #         Categorical(categories=[None], name='regressor'),
    #         Categorical(categories=[None], name='transformer'),
    #         Categorical(categories=[None], name='func')
    #     ]
    #     self.x0 = [None, None, None]
    #     return {'model': {'TransformedTargetRegressor': kwargs}}

    # def model_TPOTRegressor(self, **kwargs):
    #     ## http://epistasislab.github.io/tpot/api/#regression
    #     self.param_space = [
    #         Integer(low=10, high=100, name='generations', num_samples=self.num_samples),
    #         Integer(low=10, high=100, name='population_size', num_samples=self.num_samples),
    #         Integer(low=10, high=100, name='offspring_size', num_samples=self.num_samples),
    #         Real(low=0.01, high=0.99, name='mutation_rate', num_samples=self.num_samples),
    #         Real(low=0.01, high=0.99, name='crossover_rate', num_samples=self.num_samples),
    #         Real(low=0.1, high=1.0, name='subsample', num_samples=self.num_samples)
    #     ]
    #     self.x0 = [10, 10, 10, 0.9, 0.1, 1.0]
    #     return {'model': {'TPOTREGRESSOR': kwargs}}

    def model_TweedieRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html
        self.param_space = [
            Real(low=0.0, high=5.0, name='alpha', num_samples=self.num_samples),
            Categorical(categories=['auto', 'identity', 'log'], name='link'),
            Integer(low=50, high=500, name='max_iter', num_samples=self.num_samples)
        ]
        self.x0 = [1.0, 'auto',100]
        return {'model': {'TWEEDIEREGRESSOR': kwargs}}

    def model_TheilsenRegressor(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html
        self.param_space = [
            Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=30, high=1000, name='max_iter', num_samples=self.num_samples),
            Real(low=1e-5, high=1e-1, name='tol', num_samples=self.num_samples),
            ## Integer(low=self.data.shape[1]+1, high=len(self.data), name='n_subsamples')
        ]
        self.x0 = [True, 50, 0.001]
        return {'model': {'THEILSENREGRESSOR': kwargs}}

    # TODO
    # def model_GAMMAREGRESSOR(self, **kwargs):
    #     # ValueError: Some value(s) of y are out of the valid range for family GammaDistribution
    #     return {'GAMMAREGRESSOR': {}}

    def model_XGBoostRFRegressor(self, **kwargs):
        ## https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFRegressor
        self.param_space = [
            Integer(low=5, high=100, name='n_estimators', num_samples=self.num_samples),  #  Number of gradient boosted trees
            Integer(low=3, high=50, name='max_depth', num_samples=self.num_samples),     # Maximum tree depth for base learners
            Real(low=0.0001, high=0.5, prior='log', name='learning_rate', num_samples=self.num_samples),     #
            #Categorical(categories=['gbtree', 'gblinear', 'dart'], name='booster'),  # todo solve error
            Real(low=0.1, high=0.9, name='gamma', num_samples=self.num_samples),  # Minimum loss reduction required to make a further partition on a leaf node of the tree.
            Real(low=0.1, high=0.9, name='min_child_weight', num_samples=self.num_samples),  # Minimum sum of instance weight(hessian) needed in a child.
            Real(low=0.1, high=0.9, name='max_delta_step', num_samples=self.num_samples),  # Maximum delta step we allow each tree’s weight estimation to be.
            Real(low=0.1, high=0.9, name='subsample', num_samples=self.num_samples),  #  Subsample ratio of the training instance.
            Real(low=0.1, high=0.9, name='colsample_bytree', num_samples=self.num_samples),
            Real(low=0.1, high=0.9, name='colsample_bylevel', num_samples=self.num_samples),
            Real(low=0.1, high=0.9, name='colsample_bynode', num_samples=self.num_samples),
            Real(low=0.1, high=0.9, name='reg_alpha', num_samples=self.num_samples),
            Real(low=0.1, high=0.9, name='reg_lambda', num_samples=self.num_samples)
        ]
        self.x0 = [50, 3, 0.001, 0.1, 0.1, 0.1, 0.1,
                   0.1, 0.1, 0.1, 0.1, 0.1
                   ]
        return {'model': {'XGBOOSTRFREGRESSOR': kwargs}}

    def model_XGBoostRegressor(self, **kwargs):
        # ##https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor
        self.param_space = [
            Integer(low=5, high=200, name='n_estimators', num_samples=self.num_samples),  #  Number of gradient boosted trees
            #Integer(low=3, high=50, name='max_depth', num_samples=self.num_samples),     # Maximum tree depth for base learners
            Real(low=0.0001, high=0.5, name='learning_rate', prior='log', num_samples=self.num_samples),     #
            Categorical(categories=['gbtree', 'gblinear', 'dart'], name='booster'),
            #Real(low=0.1, high=0.9, name='gamma', num_samples=self.num_samples),  # Minimum loss reduction required to make a further partition on a leaf node of the tree.
            #Real(low=0.1, high=0.9, name='min_child_weight', num_samples=self.num_samples),  # Minimum sum of instance weight(hessian) needed in a child.
            #Real(low=0.1, high=0.9, name='max_delta_step', num_samples=self.num_samples),  # Maximum delta step we allow each tree’s weight estimation to be.
            #Real(low=0.1, high=0.9, name='subsample', num_samples=self.num_samples),  #  Subsample ratio of the training instance.
            #Real(low=0.1, high=0.9, name='colsample_bytree', num_samples=self.num_samples),
            #Real(low=0.1, high=0.9, name='colsample_bylevel', num_samples=self.num_samples),
            #Real(low=0.1, high=0.9, name='colsample_bynode', num_samples=self.num_samples),
            #Real(low=0.1, high=0.9, name='reg_alpha', num_samples=self.num_samples),
            #Real(low=0.1, high=0.9, name='reg_lambda', num_samples=self.num_samples)
        ]
        self.x0 = None #[100, 6, 0.3, 'gbtree', 0.2, 0.2, 0.2,
                   #0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                   #]
        return {'model': {'XGBOOSTREGRESSOR': kwargs}}
