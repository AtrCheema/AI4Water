

from ai4water.postprocessing.SeqMetrics import RegressionMetrics
from ai4water.utils.utils import get_version_info
from .experiments import Experiments, Model
from .utils import regression_space


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

import sklearn

VERSION_INFO = get_version_info(sklearn=sklearn)


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
                 verbosity=1,
                 **model_kwargs):
        """
        Initializes the class

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
            >>> from ai4water.datasets import arg_beach
            >>> from ai4water.experiments import MLRegressionExperiments
            >>> # first compare the performance of all available models without optimizing their parameters
            >>> data = arg_beach()  # read data file, in this case load the default data
            >>> inputs = list(data.columns)[0:-1]  # define input and output columns in data
            >>> outputs = list(data.columns)[-1]
            >>> comparisons = MLRegressionExperiments(data=data,
            ...       input_features=inputs, output_features=outputs,
            ...       nan_filler= {'method': 'KNNImputer', 'features': inputs} )
            >>> comparisons.fit(run_type="dry_run")
            >>> comparisons.compare_errors('r2')
            >>> # find out the models which resulted in r2> 0.5
            >>> best_models = comparisons.compare_errors('r2', cutoff_type='greater',
            ...                                                cutoff_val=0.3)
            >>> best_models = [m[1] for m in best_models.values()]
            >>> # now build a new experiment for best models and otpimize them
            >>> comparisons = MLRegressionExperiments(data=data,
            ...     inputs_features=inputs, output_features=outputs,
            ...     nan_filler= {'method': 'KNNImputer', 'features': inputs},
            ...     exp_name="BestMLModels")
            >>> comparisons.fit(run_type="optimize", include=best_models)
            >>> comparisons.compare_errors('r2')
            >>> comparisons.taylor_plot()  # see help(comparisons.taylor_plot()) to tweak the taylor plot
        ```
        """
        self.param_space = param_space
        self.x0 = x0
        self.data = data
        self.model_kws = model_kwargs
        self.ai4water_model = Model if ai4water_model is None else ai4water_model

        super().__init__(cases=cases, exp_name=exp_name, num_samples=num_samples, verbosity=verbosity)

        self.regression_space = regression_space(num_samples=num_samples)

        if catboost is None:
            self.models.remove('model_CATBoostRegressor')
        if lightgbm is None:
            self.models.remove('model_LGBMRegressor')
        if xgboost is None:
            self.models.remove('model_XGBRFRegressor')

        sk_maj_ver = int(sklearn.__version__.split('.')[0])
        sk_min_ver = int(sklearn.__version__.split('.')[1])
        if sk_maj_ver == 0 and sk_min_ver < 23:
            for m in ['model_PoissonRegressor', 'model_TweedieRegressor']:
                self.models.remove(m)

    @property
    def tpot_estimator(self):
        try:
            from tpot import TPOTRegressor
        except (ModuleNotFoundError, ImportError):
            TPOTRegressor = None
        return TPOTRegressor

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

        verbosity = max(self.verbosity-1, 0)
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
            vt, vp = model.predict('validation', return_true=True)
            val_score = getattr(RegressionMetrics(vt, vp), model.config['val_metric'])()

        tt, tp = model.predict('test', return_true=True)

        if view:
            model.view_model()

        if predict:
            t, p = model.predict('training', return_true=True)

            return (t, p), (tt, tp)

        if model.config['val_metric'] in ['r2', 'nse', 'kge', 'r2_mod']:
            val_score = 1.0 - val_score

        return val_score

    def model_AdaBoostRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html

        self.path = "sklearn.ensemble.AdaBoostRegressor"
        self.param_space = self.regression_space["AdaBoostRegressor"]["param_space"]
        self.x0 = self.regression_space["AdaBoostRegressor"]["x0"]

        return {'model': {'AdaBoostRegressor': kwargs}}

    def model_ARDRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html

        self.path = "sklearn.linear_model.ARDRegression"
        self.param_space = self.regression_space["ARDRegression"]["param_space"]
        self.x0 = self.regression_space["ARDRegression"]["x0"]

        return {'model': {'ARDRegression': kwargs}}

    def model_BaggingRegressor(self, **kwargs):

        self.path = "sklearn.ensemble.BaggingRegressor"
        self.param_space = self.regression_space["BaggingRegressor"]["param_space"]
        self.x0 = self.regression_space["BaggingRegressor"]["x0"]

        return {'model': {'BaggingRegressor': kwargs}}

    def model_BayesianRidge(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html

        self.path = "sklearn.linear_model.BayesianRidge"
        self.param_space = self.regression_space["BayesianRidge"]["param_space"]
        self.x0 = self.regression_space["BayesianRidge"]["x0"]

        return {'model': {'BayesianRidge': kwargs}}

    def model_CATBoostRegressor(self, **kwargs):
        # https://catboost.ai/docs/concepts/python-reference_parameters-list.html

        self.path = "catboost.CatBoostRegressor"
        self.param_space = self.regression_space["CatBoostRegressor"]["param_space"]
        self.x0 = self.regression_space["CatBoostRegressor"]["x0"]

        return {'model': {'CatBoostRegressor': kwargs}}

    def model_DecisionTreeRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

        self.path = "sklearn.tree.DecisionTreeRegressor"
        # TODO not converging
        self.param_space = self.regression_space["DecisionTreeRegressor"]["param_space"]
        self.x0 = self.regression_space["DecisionTreeRegressor"]["x0"]

        return {'model': {'DecisionTreeRegressor': kwargs}}

    def model_DummyRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html

        self.path = "sklearn.dummy.DummyRegressor"
        self.param_space = self.regression_space["DummyRegressor"]["param_space"]
        self.x0 = self.regression_space["DummyRegressor"]["x0"]
        kwargs.update({'constant': 0.2,
                       'quantile': 0.2})

        return {'model': {'DummyRegressor': kwargs}}

    def model_ElasticNet(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

        self.path = "sklearn.linear_model.ElasticNet"
        self.param_space = self.regression_space["ElasticNet"]["param_space"]
        self.x0 = self.regression_space["ElasticNet"]["x0"]

        return {'model': {'ElasticNet': kwargs}}

    def model_ElasticNetCV(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html

        self.path = "sklearn.linear_model.ElasticNetCV"
        self.param_space = self.regression_space["ElasticNetCV"]["param_space"]
        self.x0 = self.regression_space["ElasticNetCV"]["x0"]

        return {'model': {'ElasticNetCV': kwargs}}

    def model_ExtraTreeRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeRegressor.htm

        self.path = "sklearn.tree.ExtraTreeRegressor"
        self.param_space = self.regression_space["ExtraTreeRegressor"]["param_space"]
        self.x0 = self.regression_space["ExtraTreeRegressor"]["x0"]

        return {'model': {'ExtraTreeRegressor': kwargs}}

    def model_ExtraTreesRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html

        self.path = "sklearn.ensemble.ExtraTreesRegressor"
        self.param_space = self.regression_space["ExtraTreesRegressor"]["param_space"]
        self.x0 = self.regression_space["ExtraTreesRegressor"]["x0"]

        return {'model': {'ExtraTreesRegressor': kwargs}}

    # def model_GammaRegressor(self, **kwargs):
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html?highlight=gammaregressor
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
        # https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html

        self.path = "sklearn.gaussian_process.GaussianProcessRegressor"
        self.param_space = self.regression_space["GaussianProcessRegressor"]["param_space"]
        self.x0 = self.regression_space["GaussianProcessRegressor"]["x0"]

        return {'model': {'GaussianProcessRegressor': kwargs}}

    def model_GradientBoostingRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

        self.path = "sklearn.ensemble.GradientBoostingRegressor"
        self.param_space = self.regression_space["GradientBoostingRegressor"]["param_space"]
        self.x0 = self.regression_space["GradientBoostingRegressor"]["x0"]

        return {'model': {'GradientBoostingRegressor': kwargs}}

    def model_HistGradientBoostingRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html

        # TODO not hpo not converging
        self.path = "sklearn.ensemble.HistGradientBoostingRegressor"
        self.param_space = self.regression_space["HistGradientBoostingRegressor"]["param_space"]
        self.x0 = self.regression_space["HistGradientBoostingRegressor"]["x0"]

        return {'model': {'HistGradientBoostingRegressor':kwargs}}

    def model_HuberRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html

        self.path = "sklearn.linear_model.HuberRegressor"
        self.param_space = self.regression_space["HuberRegressor"]["param_space"]
        self.x0 = self.regression_space["HuberRegressor"]["x0"]

        return {'model': {'HuberRegressor': kwargs}}

    def model_KernelRidge(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html

        self.path = "sklearn.kernel_ridge.KernelRidge"
        self.param_space = self.regression_space["KernelRidge"]["param_space"]
        self.x0 = self.regression_space["KernelRidge"]["x0"]

        return {'model': {'KernelRidge': kwargs}}

    def model_KNeighborsRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

        self.path = "sklearn.neighbors.KNeighborsRegressor"
        self.param_space = self.regression_space["KNeighborsRegressor"]["param_space"]
        self.x0 = self.regression_space["KNeighborsRegressor"]["x0"]

        return {'model': {'KNeighborsRegressor': kwargs}}

    def model_LassoLars(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html

        self.path = "sklearn.linear_model.LassoLars"
        self.param_space = self.regression_space["LassoLars"]["param_space"]
        self.x0 = self.regression_space["LassoLars"]["x0"]

        return {'model': {'LassoLars': kwargs}}

    def model_Lars(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html

        self.path = "sklearn.linear_model.Lars"
        self.param_space = self.regression_space["Lars"]["param_space"]
        self.x0 = self.regression_space["Lars"]["x0"]

        return {'model': {'Lars': kwargs}}

    def model_LarsCV(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LarsCV.html

        self.path = "sklearn.linear_model.LarsCV"
        self.param_space = self.regression_space["LarsCV"]["param_space"]
        self.x0 = self.regression_space["LarsCV"]["x0"]

        return {'model': {'LarsCV': kwargs}}

    def model_LinearSVR(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html

        self.path = "sklearn.svm.LinearSVR"
        self.param_space = self.regression_space["LinearSVR"]["param_space"]
        self.x0 = self.regression_space["LinearSVR"]["x0"]

        return {'model': {'LinearSVR': kwargs}}

    def model_Lasso(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

        self.path = "sklearn.linear_model.Lasso"
        self.param_space = self.regression_space["Lasso"]["param_space"]
        self.x0 = self.regression_space["Lasso"]["x0"]

        return {'model': {'Lasso': kwargs}}

    def model_LassoCV(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html

        self.path = "sklearn.linear_model.LassoCV"
        self.param_space = self.regression_space["LassoCV"]["param_space"]
        self.x0 = self.regression_space["LassoCV"]["x0"]

        return {'model': {'LassoCV': kwargs}}

    def model_LassoLarsCV(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html

        self.path = "sklearn.linear_model.LassoLarsCV"
        self.param_space = self.regression_space["LassoLarsCV"]["param_space"]
        self.x0 = self.regression_space["LassoLarsCV"]["x0"]

        return {'model': {'LassoLarsCV': kwargs}}

    def model_LassoLarsIC(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html

        self.path = "sklearn.linear_model.LassoLarsIC"
        self.param_space = self.regression_space["LassoLarsIC"]["param_space"]
        self.x0 = self.regression_space["LassoLarsIC"]["x0"]

        return {'model': {'LassoLarsIC': kwargs}}

    def model_LGBMRegressor(self, **kwargs):
        # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html

        self.path = "lightgbm.LGBMRegressor"
        self.param_space = self.regression_space["LGBMRegressor"]["param_space"]
        self.x0 = self.regression_space["LGBMRegressor"]["x0"]

        return {'model': {'LGBMRegressor': kwargs}}

    def model_LinearRegression(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        self.path = "sklearn.linear_model.LinearRegression"
        self.param_space = self.regression_space["LinearRegression"]["param_space"]
        self.x0 = self.regression_space["LinearRegression"]["x0"]

        return {'model': {'LinearRegression': kwargs}}

    def model_MLPRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html

        self.path = "sklearn.neural_network.MLPRegressor"
        self.param_space = self.regression_space["MLPRegressor"]["param_space"]
        self.x0 = self.regression_space["MLPRegressor"]["x0"]

        return {'model': {'MLPRegressor': kwargs}}

    def model_NuSVR(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html

        self.path = "sklearn.svm.NuSVR"
        self.param_space = self.regression_space["NuSVR"]["param_space"]
        self.x0 = self.regression_space["NuSVR"]["x0"]

        return {'model': {'NuSVR': kwargs}}

    def model_OrthogonalMatchingPursuit(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html

        self.path = "sklearn.linear_model.OrthogonalMatchingPursuit"
        self.param_space = self.regression_space["OrthogonalMatchingPursuit"]["param_space"]
        self.x0 = self.regression_space["OrthogonalMatchingPursuit"]["x0"]

        return {'model': {'OrthogonalMatchingPursuit': kwargs}}

    def model_OrthogonalMatchingPursuitCV(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuitCV.html

        self.path = "sklearn.linear_model.OrthogonalMatchingPursuitCV"
        self.param_space = self.regression_space["OrthogonalMatchingPursuitCV"]["param_space"]
        self.x0 = self.regression_space["OrthogonalMatchingPursuitCV"]["x0"]

        return {'model': {'OrthogonalMatchingPursuitCV': kwargs}}

    def model_OneClassSVM(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html

        self.path = "sklearn.svm.OneClassSVM"
        self.param_space = self.regression_space["OneClassSVM"]["param_space"]
        self.x0 = self.regression_space["OneClassSVM"]["x0"]

        return {'model': {'OneClassSVM': kwargs}}

    def model_PoissonRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html

        self.path = "sklearn.linear_model.PoissonRegressor"
        self.param_space = self.regression_space["PoissonRegressor"]["param_space"]
        self.x0 = self.regression_space["PoissonRegressor"]["x0"]

        return {'model': {'PoissonRegressor': kwargs}}

    def model_Ridge(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

        self.path = "sklearn.linear_model.Ridge"
        self.param_space = self.regression_space["Ridge"]["param_space"]
        self.x0 = self.regression_space["Ridge"]["x0"]

        return {'model': {'Ridge': kwargs}}

    def model_RidgeCV(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html

        self.path = "sklearn.linear_model.RidgeCV"
        self.param_space = self.regression_space["RidgeCV"]["param_space"]
        self.x0 = self.regression_space["RidgeCV"]["x0"]

        return {'model': {'RidgeCV': kwargs}}

    def model_RadiusNeighborsRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsRegressor.html

        self.path = "sklearn.neighbors.RadiusNeighborsRegressor"
        self.param_space = self.regression_space["RadiusNeighborsRegressor"]["param_space"]
        self.x0 = self.regression_space["RadiusNeighborsRegressor"]["x0"]

        return {'model': {'RadiusNeighborsRegressor': kwargs}}

    def model_RANSACRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html

        self.path = "sklearn.linear_model.RANSACRegressor"
        self.param_space = self.regression_space["RANSACRegressor"]["param_space"]
        self.x0 = self.regression_space["RANSACRegressor"]["x0"]

        return {'model': {'RANSACRegressor': kwargs}}

    def model_RandomForestRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

        self.path = "sklearn.ensemble.RandomForestRegressor"
        self.param_space = self.regression_space["RandomForestRegressor"]["param_space"]
        self.x0 = self.regression_space["RandomForestRegressor"]["x0"]

        return {'model': {'RandomForestRegressor': kwargs}}

    def model_SVR(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

        self.path = "sklearn.svm.SVR"
        self.param_space = self.regression_space["SVR"]["param_space"]
        self.x0 = self.regression_space["SVR"]["x0"]

        return {'model': {'SVR': kwargs}}

    def model_SGDRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html

        self.path = "sklearn.linear_model.SGDRegressor"
        self.param_space = self.regression_space["SGDRegressor"]["param_space"]
        self.x0 = self.regression_space["SGDRegressor"]["x0"]

        return {'model': {'SGDRegressor': kwargs}}

    # def model_TransformedTargetRegressor(self, **kwargs):
    #     ## https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html
    #     self.param_space = [
    #         Categorical(categories=[None], name='regressor'),
    #         Categorical(categories=[None], name='transformer'),
    #         Categorical(categories=[None], name='func')
    #     ]
    #     self.x0 = [None, None, None]
    #     return {'model': {'TransformedTargetRegressor': kwargs}}

    def model_TweedieRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html

        self.path = "sklearn.linear_model.TweedieRegressor"
        self.param_space = self.regression_space["TweedieRegressor"]["param_space"]
        self.x0 = self.regression_space["TweedieRegressor"]["x0"]

        return {'model': {'TweedieRegressor': kwargs}}

    def model_TheilsenRegressor(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html

        self.path = "sklearn.linear_model.TheilSenRegressor"
        self.param_space = self.regression_space["TheilSenRegressor"]["param_space"]
        self.x0 = self.regression_space["TheilSenRegressor"]["x0"]

        return {'model': {'TheilSenRegressor': kwargs}}

    # TODO
    # def model_GAMMAREGRESSOR(self, **kwargs):
    #     # ValueError: Some value(s) of y are out of the valid range for family GammaDistribution
    #     return {'GAMMAREGRESSOR': {}}

    def model_XGBRFRegressor(self, **kwargs):
        # https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFRegressor

        self.path = "xgboost.XGBRFRegressor"
        self.param_space = self.regression_space["XGBRFRegressor"]["param_space"]
        self.x0 = self.regression_space["XGBRFRegressor"]["x0"]

        return {'model': {'XGBRFRegressor': kwargs}}

    def model_XGBRegressor(self, **kwargs):
        # ##https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor

        self.path = "xgboost.XGBRegressor"
        self.param_space = self.regression_space["XGBRegressor"]["param_space"]
        self.x0 = self.regression_space["XGBRegressor"]["x0"]

        return {'model': {'XGBRegressor': kwargs}}
