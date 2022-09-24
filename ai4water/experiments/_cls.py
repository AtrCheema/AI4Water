
__all__ = ["MLClassificationExperiments"]

import os.path

import matplotlib.pyplot as plt

from ._main import Experiments
from .utils import classification_space
from ai4water.utils.utils import dateandtime_now
from ai4water.backend import os, sklearn, catboost, xgboost, lightgbm

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

    def __init__(
            self,
            param_space=None,
            x0=None,
            cases=None,
            exp_name='MLClassificationExperiments',
            num_samples=5,
            monitor = None,
            **model_kws
    ):
        """

        Parameters
        ----------
            param_space : list, optional
            x0 : list, optional
            cases : dict, optional
            exp_name : str, optional
                name of experiment
            num_samples : int, optional
            monitor : list/str, optional
            **model_kws :
                keyword arguments for :py:class:`ai4water.Model` class
        """
        self.param_space = param_space
        self.x0 = x0

        self.spaces = classification_space(num_samples=num_samples,)

        if catboost is None:
            self.models.remove('model_CatBoostClassifier')
        if lightgbm is None:
            self.models.remove('model_LGBMClassifier')
        if xgboost is None:
            self.models.remove('model_XGBRFClassifier')
            self.models.remove('model_XGBClassifier')

        if exp_name == "MLClassificationExperiments":
            exp_name = f"{exp_name}_{dateandtime_now()}"

        super().__init__(
            cases=cases,
            exp_name=exp_name,
            num_samples=num_samples,
            monitor=monitor,
            **model_kws
        )

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

    @property
    def category(self):
        return "ML"

    def _compare_cls_curves(self, x, y, save, show, func, name):

        model_folders = [p for p in os.listdir(self.exp_path) if os.path.isdir(os.path.join(self.exp_path,p))]

        _, ax = plt.subplots()

        # find all the model folders
        m_paths = []
        for m in model_folders:
            if any(m in m_ for m_ in self.considered_models):
                m_paths.append(m)

        # load all models from config
        for m_path in m_paths:
            print(f'here: {m_path}')
            m_path = os.path.join(self.exp_path, m_path)
            assert len(os.listdir(m_path)) == 1
            m_path = os.path.join(m_path, os.listdir(m_path)[0])
            c_path = os.path.join(m_path, 'config.json')
            model = self.build_from_config(c_path)
            # calculate pr curve for each model
            self.update_model_weight(model, m_path)

            kws = {'estimator': model, 'X': x, 'y': y.reshape(-1, ), 'ax': ax, 'name': model.model_name}
            if 'LinearSVC' in model.model_name:
                # sklearn LinearSVC does not have predict_proba but ai4water Model does have this method
                # which will only throw error
                kws['estimator'] = model._model

            if model.model_name in ['Perceptron', 'PassiveAggressiveClassifier',
                                    'NearestCentroid', 'RidgeClassifier',
                                    'RidgeClassifierCV']:
                continue

            if model.model_name in ['NuSVC' , 'SVC']:
                if not model._model.get_params()['probability']:
                    continue

            if 'SGDClassifier' in model.model_name:
                if model._model.get_params()['loss'] == 'hinge':
                    continue

            func(**kws)

        if save:
            fname = os.path.join(self.exp_path, f"{name}.png")
            plt.savefig(fname, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        return ax

    def compare_precision_recall_curves(
            self,
            x,
            y,
            save:bool = True,
            show:bool = True,
    ):
        """compares precision recall curves of the all the models.

        parameters
        ---------
        x :
            input data
        y :
            labels for the input data
        save :
            whether to save the plot or not.
        show :
            whether to show the plot or not

        Returns
        -------
        plt.Axes
            matplotlib axes on which figure is drawn

        Example
        -------
        >>> from ai4water.datasets import MtropicsLaos
        >>> data = MtropicsLaos().make_classification(lookback_steps=1)
        # define inputs and outputs
        >>> inputs = data.columns.tolist()[0:-1]
        >>> outputs = data.columns.tolist()[-1:]
        # initiate the experiment
        >>> exp = MLClassificationExperiments(
        ...     input_features=inputs,
        ...     output_features=outputs)
        # run the experiment
        >>> exp.fit(data=data, include=["model_LGBMClassifier",
        ...                             "model_XGBClassifier",
        ...                             "RandomForestClassifier"])
        ... # Compare Precision Recall curves
        >>> exp.compare_precision_recall_curves(data[inputs].values, data[outputs].values)
        """

        return self._compare_cls_curves(
            x,
            y,
            save,
            show,
            name="precision_recall_curves",
            func=sklearn.metrics.PrecisionRecallDisplay.from_estimator
        )

    def compare_roc_curves(
            self,
            x,
            y,
            save:bool = True,
            show:bool = True,
    ):
        """compares roc curves of the all the models.

        parameters
        ---------
        x :
            input data
        y :
            labels for the input data
        save :
            whether to save the plot or not.
        show :
            whether to show the plot or not

        Returns
        -------
        plt.Axes
            matplotlib axes on which figure is drawn

        Example
        -------
        >>> from ai4water.datasets import MtropicsLaos
        >>> data = MtropicsLaos().make_classification(lookback_steps=1)
        # define inputs and outputs
        >>> inputs = data.columns.tolist()[0:-1]
        >>> outputs = data.columns.tolist()[-1:]
        # initiate the experiment
        >>> exp = MLClassificationExperiments(
        ...     input_features=inputs,
        ...     output_features=outputs)
        # run the experiment
        >>> exp.fit(data=data, include=["model_LGBMClassifier",
        ...                             "model_XGBClassifier",
        ...                             "RandomForestClassifier"])
        ... # Compare ROC curves
        >>> exp.compare_roc_curves(data[inputs].values, data[outputs].values)
        """

        return self._compare_cls_curves(
            x=x,
            y=y,
            save=save,
            show=show,
            name="roc_curves",
            func=sklearn.metrics.RocCurveDisplay.from_estimator
        )

    def model_AdaBoostClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

        self.path = "sklearn.ensemble.AdaBoostClassifier"
        self.param_space = self.spaces["AdaBoostClassifier"]["param_space"]
        self.x0 = self.spaces["AdaBoostClassifier"]["x0"]

        return {'model': {'AdaBoostClassifier': kwargs}}

    def model_BaggingClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html

        self.path = "sklearn.ensemble.BaggingClassifier"
        self.param_space = self.spaces["BaggingClassifier"]["param_space"]
        self.x0 = self.spaces["BaggingClassifier"]["x0"]

        return {'model': {'BaggingClassifier': kwargs}}

    def model_BernoulliNB(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html

        self.path = "sklearn.naive_bayes.BernoulliNB"
        self.param_space = self.spaces["BernoulliNB"]["param_space"]
        self.x0 = self.spaces["BernoulliNB"]["x0"]

        return {'model': {'BernoulliNB': kwargs}}

    def model_CalibratedClassifierCV(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html

        self.path = "sklearn.calibration.CalibratedClassifierCV"
        self.param_space = self.spaces["CalibratedClassifierCV"]["param_space"]
        self.x0 = self.spaces["CalibratedClassifierCV"]["x0"]

        return {'model': {'CalibratedClassifierCV': kwargs}}

    # def model_CheckingClassifier(self, **kwargs):
    #     return

    def model_DecisionTreeClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

        self.path = "sklearn.tree.DecisionTreeClassifier"
        self.param_space = self.spaces["DecisionTreeClassifier"]["param_space"]
        self.x0 = self.spaces["DecisionTreeClassifier"]["x0"]

        return {'model': {'DecisionTreeClassifier': kwargs}}

    def model_DummyClassifier(self, **kwargs):
        #  https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html

        self.path = "sklearn.dummy.DummyClassifier"
        self.param_space = self.spaces["DummyClassifier"]["param_space"]
        self.x0 = self.spaces["DummyClassifier"]["x0"]

        return {'model': {'DummyClassifier': kwargs}}

    def model_ExtraTreeClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html

        self.path = "sklearn.tree.ExtraTreeClassifier"
        self.param_space = self.spaces["ExtraTreeClassifier"]["param_space"]
        self.x0 = self.spaces["ExtraTreeClassifier"]["x0"]

        return {'model': {'ExtraTreeClassifier': kwargs}}

    def model_ExtraTreesClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html

        self.path = "sklearn.ensemble.ExtraTreesClassifier"
        self.param_space = self.spaces["ExtraTreesClassifier"]["param_space"]
        self.x0 = self.spaces["ExtraTreesClassifier"]["x0"]

        return {'model': {'ExtraTreesClassifier': kwargs}}

    def model_GaussianProcessClassifier(self, **kwargs):
        #  https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html

        self.path = "sklearn.gaussian_process.GaussianProcessClassifier"
        self.param_space = self.spaces["GaussianProcessClassifier"]["param_space"]
        self.x0 = self.spaces["GaussianProcessClassifier"]["x0"]

        return {'model': {'GaussianProcessClassifier': kwargs}}

    def model_GradientBoostingClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html

        self.path = "sklearn.ensemble.GradientBoostingClassifier"
        self.param_space = self.spaces["GradientBoostingClassifier"]["param_space"]
        self.x0 = self.spaces["GradientBoostingClassifier"]["x0"]

        return {'model': {'GradientBoostingClassifier': kwargs}}

    def model_HistGradientBoostingClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html

        self.path = "sklearn.ensemble.HistGradientBoostingClassifier"
        self.param_space = self.spaces["HistGradientBoostingClassifier"]["param_space"]
        self.x0 = self.spaces["HistGradientBoostingClassifier"]["x0"]

        return {'model': {'HistGradientBoostingClassifier': kwargs}}

    def model_KNeighborsClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

        self.path = "sklearn.neighbors.KNeighborsClassifier"
        self.param_space = self.spaces["KNeighborsClassifier"]["param_space"]
        self.x0 = self.spaces["KNeighborsClassifier"]["x0"]

        return {'model': {'KNeighborsClassifier': kwargs}}

    def model_LabelPropagation(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html

        self.path = "sklearn.semi_supervised.LabelPropagation"
        self.param_space = self.spaces["LabelPropagation"]["param_space"]
        self.x0 = self.spaces["LabelPropagation"]["x0"]

        return {'model': {'LabelPropagation': kwargs}}

    def model_LabelSpreading(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html

        self.path = "sklearn.semi_supervised.LabelSpreading"
        self.param_space = self.spaces["LabelSpreading"]["param_space"]
        self.x0 = self.spaces["LabelSpreading"]["x0"]

        return {'model': {'LabelSpreading': kwargs}}

    def model_LGBMClassifier(self, **kwargs):
        # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html

        self.path = "lightgbm.LGBMClassifier"
        self.param_space = self.spaces["LGBMClassifier"]["param_space"]
        self.x0 = self.spaces["LGBMClassifier"]["x0"]

        return {'model': {'LGBMClassifier': kwargs}}

    def model_LinearDiscriminantAnalysis(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

        self.path = "sklearn.discriminant_analysis.LinearDiscriminantAnalysis"
        self.param_space = self.spaces["LinearDiscriminantAnalysis"]["param_space"]
        self.x0 = self.spaces["LinearDiscriminantAnalysis"]["x0"]

        return {'model': {'LinearDiscriminantAnalysis': kwargs}}

    def model_LinearSVC(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

        self.path = "sklearn.svm.LinearSVC"
        self.param_space = self.spaces["LinearSVC"]["param_space"]
        self.x0 = self.spaces["LinearSVC"]["x0"]

        return {'model': {'LinearSVC': kwargs}}

    def model_LogisticRegression(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

        self.path = "sklearn.linear_model.LogisticRegression"
        self.param_space = self.spaces["LogisticRegression"]["param_space"]
        self.x0 = self.spaces["LogisticRegression"]["x0"]

        return {'model': {'LogisticRegression': kwargs}}

    def model_MLPClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

        self.path = "sklearn.neural_network.MLPClassifier"
        self.param_space = self.spaces["MLPClassifier"]["param_space"]
        self.x0 = self.spaces["MLPClassifier"]["x0"]

        return {'model': {'MLPClassifier': kwargs}}

    def model_NearestCentroid(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html

        self.path = "sklearn.neighbors.NearestCentroid"
        self.param_space = self.spaces["NearestCentroid"]["param_space"]
        self.x0 = self.spaces["NearestCentroid"]["x0"]

        return {'model': {'NearestCentroid': kwargs}}

    def model_NuSVC(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html

        self.path = "sklearn.svm.NuSVC"
        self.param_space = self.spaces["NuSVC"]["param_space"]
        self.x0 = self.spaces["NuSVC"]["x0"]

        return {'model': {'NuSVC': kwargs}}

    def model_PassiveAggressiveClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html

        self.path = "sklearn.linear_model.PassiveAggressiveClassifier"
        self.param_space = self.spaces["PassiveAggressiveClassifier"]["param_space"]
        self.x0 = self.spaces["PassiveAggressiveClassifier"]["x0"]

        return {'model': {'PassiveAggressiveClassifier': kwargs}}

    def model_Perceptron(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html

        self.path = "sklearn.linear_model.Perceptron"
        self.param_space = self.spaces["Perceptron"]["param_space"]
        self.x0 = self.spaces["Perceptron"]["x0"]

        return {'model': {'Perceptron': kwargs}}

    def model_QuadraticDiscriminantAnalysis(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html

        self.path = "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis"
        self.param_space = self.spaces["QuadraticDiscriminantAnalysis"]["param_space"]
        self.x0 = self.spaces["QuadraticDiscriminantAnalysis"]["x0"]

        return {'model': {'QuadraticDiscriminantAnalysis': kwargs}}

    # def model_RadiusNeighborsClassifier(self, **kwargs):
    #     # https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html
    #
    #     self.path = "sklearn.neighbors.RadiusNeighborsClassifier"
    #     self.param_space = self.spaces["RadiusNeighborsClassifier"]["param_space"]
    #     self.x0 = self.spaces["RadiusNeighborsClassifier"]["x0"]
    #
    #     return {'model': {'RadiusNeighborsClassifier': kwargs}}

    def model_RandomForestClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

        self.path = "sklearn.ensemble.RandomForestClassifier"
        self.param_space = self.spaces["RandomForestClassifier"]["param_space"]
        self.x0 = self.spaces["RandomForestClassifier"]["x0"]

        return {'model': {'RandomForestClassifier': kwargs}}

    def model_RidgeClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html

        self.path = "sklearn.linear_model.RidgeClassifier"
        self.param_space = self.spaces["RidgeClassifier"]["param_space"]
        self.x0 = self.spaces["RidgeClassifierCV"]["x0"]

        return {'model': {'RidgeClassifier': kwargs}}

    def model_RidgeClassifierCV(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html

        self.path = "sklearn.linear_model.RidgeClassifierCV"
        self.param_space = self.spaces["RidgeClassifierCV"]["param_space"]
        self.x0 = self.spaces["RidgeClassifierCV"]["x0"]

        return {'model': {'RidgeClassifierCV': kwargs}}

    def model_SGDClassifier(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

        self.path = "sklearn.linear_model.SGDClassifier"
        self.param_space = self.spaces["SGDClassifier"]["param_space"]
        self.x0 = self.spaces["SGDClassifier"]["x0"]

        return {'model': {'SGDClassifier': kwargs}}

    def model_SVC(self, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

        self.path = "sklearn.svm.SVC"
        self.param_space = self.spaces["SVC"]["param_space"]
        self.x0 = self.spaces["SVC"]["x0"]

        return {'model': {'SVC': kwargs}}

    def model_XGBClassifier(self, **kwargs):
        # https://xgboost.readthedocs.io/en/latest/python/python_api.html

        self.path = "xgboost.XGBClassifier"
        self.param_space = self.spaces["XGBClassifier"]["param_space"]
        self.x0 = self.spaces["XGBClassifier"]["x0"]

        return {'model': {'XGBClassifier': kwargs}}

    def model_XGBRFClassifier(self, **kwargs):
        # https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFClassifier

        self.path = "xgboost.XGBRFClassifier"
        self.param_space = self.spaces["XGBRFClassifier"]["param_space"]
        self.x0 = self.spaces["XGBRFClassifier"]["x0"]

        return {'model': {'XGBRFClassifier': kwargs}}

    def model_CatBoostClassifier(self, **suggestions):
        # https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier

        self.path = "catboost.CatBoostClassifier"
        self.param_space = self.spaces["CatBoostClassifier"]["param_space"]
        self.x0 = self.spaces["CatBoostClassifier"]["x0"]

        return {'model': {'CatBoostClassifier': suggestions}}
