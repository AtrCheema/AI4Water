
__all__ = ["MLClassificationExperiments"]

from ._main import Experiments
from .utils import classification_space


class MLClassificationExperiments(Experiments):
    """Runs classification models for comparison, with or without
    optimization of hyperparameters. It compares around 30 classification
    algorithms from sklearn, xgboost, catboost and lightgbm.

    Examples
    --------
    >>> from ai4water.datasets import MtropicsLaos
    >>> from ai4water.experiments import MLClassificationExperiments
    >>> data = MtropicsLaos().make_classification(lookback_steps=2)
    >>> inputs = data.columns.tolist()[0:-1]
    >>> outputs = data.columns.tolist()[-1:]
    >>> exp = MLClassificationExperiments(input_features=inputs,
    >>>                                       output_features=outputs)
    >>> exp.fit(data=data, include=["CatBoostClassifier", "LGBMClassifier",
    >>>             'RandomForestClassifier', 'XGBClassifier'])
    >>> exp.compare_errors('accuracy', show=False)
    """

    def __init__(self,
                 param_space=None,
                 x0=None,
                 cases=None,
                 exp_name='MLExperiments',
                 num_samples=5,
                 **model_kwargs):
        """

        Parameters
        ----------
            param_space : list, optional
            x0 : list, optional
            cases : dict, optional
            exp_name : str, optional
                name of experiment
            num_samples :
            **model_kwargs :
                keyword arguments for :py:class:`ai4water.Model` class
        """
        self.param_space = param_space
        self.x0 = x0
        self.model_kws = model_kwargs

        self.classification_space = classification_space(num_samples=num_samples)

        super().__init__(cases=cases, exp_name=exp_name, num_samples=num_samples)

    @property
    def tpot_estimator(self):
        try:
            from tpot import TPOTClassifier
        except (ModuleNotFoundError, ImportError):
            TPOTClassifier = None
        return TPOTClassifier

    @property
    def mode(self):
        return "classification"

    def model_AdaBoostClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

        self.path = "sklearn.ensemble.AdaBoostClassifier"
        self.param_space = self.classification_space["AdaBoostClassifier"]["param_space"]
        self.x0 = self.classification_space["AdaBoostClassifier"]["x0"]

        return {'model': {'AdaBoostClassifier': kwargs}}

    def model_BaggingClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html

        self.path = "sklearn.ensemble.BaggingClassifier"
        self.param_space = self.classification_space["BaggingClassifier"]["param_space"]
        self.x0 = self.classification_space["BaggingClassifier"]["x0"]

        return {'model': {'BaggingClassifier': kwargs}}

    def model_BernoulliNB(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html

        self.path = "sklearn.naive_bayes.BernoulliNB"
        self.param_space = self.classification_space["BernoulliNB"]["param_space"]
        self.x0 = self.classification_space["BernoulliNB"]["x0"]

        return {'model': {'BernoulliNB': kwargs}}

    def model_CalibratedClassifierCV(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html

        self.path = "sklearn.calibration.CalibratedClassifierCV"
        self.param_space = self.classification_space["CalibratedClassifierCV"]["param_space"]
        self.x0 = self.classification_space["CalibratedClassifierCV"]["x0"]

        return {'model': {'CalibratedClassifierCV': kwargs}}

    # def model_CheckingClassifier(self, **kwargs):
    #     return

    def model_DecisionTreeClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

        self.path = "sklearn.tree.DecisionTreeClassifier"
        self.param_space = self.classification_space["DecisionTreeClassifier"]["param_space"]
        self.x0 = self.classification_space["DecisionTreeClassifier"]["x0"]

        return {'model': {'DecisionTreeClassifier': kwargs}}

    def model_DummyClassifier(self, **kwargs):
        #  https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html

        self.path = "sklearn.dummy.DummyClassifier"
        self.param_space = self.classification_space["DummyClassifier"]["param_space"]
        self.x0 = self.classification_space["DummyClassifier"]["x0"]

        return {'model': {'DummyClassifier': kwargs}}

    def model_ExtraTreeClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html

        self.path = "sklearn.tree.ExtraTreeClassifier"
        self.param_space = self.classification_space["ExtraTreeClassifier"]["param_space"]
        self.x0 = self.classification_space["ExtraTreeClassifier"]["x0"]

        return {'model': {'ExtraTreeClassifier': kwargs}}

    def model_ExtraTreesClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html

        self.path = "sklearn.ensemble.ExtraTreesClassifier"
        self.param_space = self.classification_space["ExtraTreesClassifier"]["param_space"]
        self.x0 = self.classification_space["ExtraTreesClassifier"]["x0"]

        return {'model': {'ExtraTreesClassifier': kwargs}}

    def model_GaussianProcessClassifier(self, **kwargs):
        #  https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html

        self.path = "sklearn.gaussian_process.GaussianProcessClassifier"
        self.param_space = self.classification_space["GaussianProcessClassifier"]["param_space"]
        self.x0 = self.classification_space["GaussianProcessClassifier"]["x0"]

        return {'model': {'GaussianProcessClassifier': kwargs}}

    def model_GradientBoostingClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html

        self.path = "sklearn.ensemble.GradientBoostingClassifier"
        self.param_space = self.classification_space["GradientBoostingClassifier"]["param_space"]
        self.x0 = self.classification_space["GradientBoostingClassifier"]["x0"]

        return {'model': {'GradientBoostingClassifier': kwargs}}

    def model_HistGradientBoostingClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html

        self.path = "sklearn.ensemble.HistGradientBoostingClassifier"
        self.param_space = self.classification_space["HistGradientBoostingClassifier"]["param_space"]
        self.x0 = self.classification_space["HistGradientBoostingClassifier"]["x0"]

        return {'model': {'HistGradientBoostingClassifier': kwargs}}

    def model_KNeighborsClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

        self.path = "sklearn.neighbors.KNeighborsClassifier"
        self.param_space = self.classification_space["KNeighborsClassifier"]["param_space"]
        self.x0 = self.classification_space["KNeighborsClassifier"]["x0"]

        return {'model': {'KNeighborsClassifier': kwargs}}

    def model_LabelPropagation(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html

        self.path = "sklearn.semi_supervised.LabelPropagation"
        self.param_space = self.classification_space["LabelPropagation"]["param_space"]
        self.x0 = self.classification_space["LabelPropagation"]["x0"]

        return {'model': {'LabelPropagation': kwargs}}

    def model_LabelSpreading(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html

        self.path = "sklearn.semi_supervised.LabelSpreading"
        self.param_space = self.classification_space["LabelSpreading"]["param_space"]
        self.x0 = self.classification_space["LabelSpreading"]["x0"]

        return {'model': {'LabelSpreading': kwargs}}

    def model_LGBMClassifier(self, **kwargs):
        # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html

        self.path = "lightgbm.LGBMClassifier"
        self.param_space = self.classification_space["LGBMClassifier"]["param_space"]
        self.x0 = self.classification_space["LGBMClassifier"]["x0"]

        return {'model': {'LGBMClassifier': kwargs}}

    def model_LinearDiscriminantAnalysis(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

        self.path = "sklearn.discriminant_analysis.LinearDiscriminantAnalysis"
        self.param_space = self.classification_space["LinearDiscriminantAnalysis"]["param_space"]
        self.x0 = self.classification_space["LinearDiscriminantAnalysis"]["x0"]

        return {'model': {'LinearDiscriminantAnalysis': kwargs}}

    def model_LinearSVC(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

        self.path = "sklearn.svm.LinearSVC"
        self.param_space = self.classification_space["LinearSVC"]["param_space"]
        self.x0 = self.classification_space["LinearSVC"]["x0"]

        return {'model': {'LinearSVC': kwargs}}

    def model_LogisticRegression(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

        self.path = "sklearn.linear_model.LogisticRegression"
        self.param_space = self.classification_space["LogisticRegression"]["param_space"]
        self.x0 = self.classification_space["LogisticRegression"]["x0"]

        return {'model': {'LogisticRegression': kwargs}}

    def model_MLPClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

        self.path = "sklearn.neural_network.MLPClassifier"
        self.param_space = self.classification_space["MLPClassifier"]["param_space"]
        self.x0 = self.classification_space["MLPClassifier"]["x0"]

        return {'model': {'MLPClassifier': kwargs}}

    def model_NearestCentroid(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html

        self.path = "sklearn.neighbors.NearestCentroid"
        self.param_space = self.classification_space["NearestCentroid"]["param_space"]
        self.x0 = self.classification_space["NearestCentroid"]["x0"]

        return {'model': {'NearestCentroid': kwargs}}

    def model_NuSVC(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html

        self.path = "sklearn.svm.NuSVC"
        self.param_space = self.classification_space["NuSVC"]["param_space"]
        self.x0 = self.classification_space["NuSVC"]["x0"]

        return {'model': {'NuSVC': kwargs}}

    def model_PassiveAggressiveClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html

        self.path = "sklearn.linear_model.PassiveAggressiveClassifier"
        self.param_space = self.classification_space["PassiveAggressiveClassifier"]["param_space"]
        self.x0 = self.classification_space["PassiveAggressiveClassifier"]["x0"]

        return {'model': {'PassiveAggressiveClassifier': kwargs}}

    def model_Perceptron(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html

        self.path = "sklearn.linear_model.Perceptron"
        self.param_space = self.classification_space["Perceptron"]["param_space"]
        self.x0 = self.classification_space["Perceptron"]["x0"]

        return {'model': {'Perceptron': kwargs}}

    def model_QuadraticDiscriminantAnalysis(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html

        self.path = "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis"
        self.param_space = self.classification_space["QuadraticDiscriminantAnalysis"]["param_space"]
        self.x0 = self.classification_space["QuadraticDiscriminantAnalysis"]["x0"]

        return {'model': {'QuadraticDiscriminantAnalysis': kwargs}}

    # def model_RadiusNeighborsClassifier(self, **kwargs):
    #     # https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html
    #
    #     self.path = "sklearn.neighbors.RadiusNeighborsClassifier"
    #     self.param_space = self.classification_space["RadiusNeighborsClassifier"]["param_space"]
    #     self.x0 = self.classification_space["RadiusNeighborsClassifier"]["x0"]
    #
    #     return {'model': {'RadiusNeighborsClassifier': kwargs}}

    def model_RandomForestClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

        self.path = "sklearn.ensemble.RandomForestClassifier"
        self.param_space = self.classification_space["RandomForestClassifier"]["param_space"]
        self.x0 = self.classification_space["RandomForestClassifier"]["x0"]

        return {'model': {'RandomForestClassifier': kwargs}}

    def model_RidgeClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html

        self.path = "sklearn.linear_model.RidgeClassifier"
        self.param_space = self.classification_space["RidgeClassifierCV"]["param_space"]
        self.x0 = self.classification_space["RidgeClassifierCV"]["x0"]

        return {'model': {'RidgeClassifier': kwargs}}

    def model_RidgeClassifierCV(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html

        self.path = "sklearn.linear_model.RidgeClassifierCV"
        self.param_space = self.classification_space["RidgeClassifierCV"]["param_space"]
        self.x0 = self.classification_space["RidgeClassifierCV"]["x0"]

        return {'model': {'RidgeClassifierCV': kwargs}}

    def model_SGDClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

        self.path = "sklearn.linear_model.SGDClassifier"
        self.param_space = self.classification_space["SGDClassifier"]["param_space"]
        self.x0 = self.classification_space["SGDClassifier"]["x0"]

        return {'model': {'SGDClassifier': kwargs}}

    def model_SVC(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

        self.path = "sklearn.svm.SVC"
        self.param_space = self.classification_space["SVC"]["param_space"]
        self.x0 = self.classification_space["SVC"]["x0"]

        return {'model': {'SVC': kwargs}}

    def model_XGBClassifier(self, **kwargs):
        # https://xgboost.readthedocs.io/en/latest/python/python_api.html

        self.path = "xgboost.XGBClassifier"
        self.param_space = self.classification_space["XGBClassifier"]["param_space"]
        self.x0 = self.classification_space["XGBClassifier"]["x0"]

        return {'model': {'XGBClassifier': kwargs}}

    def model_XGBRFClassifier(self, **kwargs):
        # https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFClassifier

        self.path = "xgboost.XGBRFClassifier"
        self.param_space = self.classification_space["XGBRFClassifier"]["param_space"]
        self.x0 = self.classification_space["XGBRFClassifier"]["x0"]

        return {'model': {'XGBRFClassifier': kwargs}}

    def model_CatBoostClassifier(self, **suggestions):
        # https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier

        self.path = "xgboost.CatBoostClassifier"
        self.param_space = self.classification_space["CatBoostClassifier"]["param_space"]
        self.x0 = self.classification_space["CatBoostClassifier"]["x0"]

        return {'model': {'CatBoostClassifier': suggestions}}
