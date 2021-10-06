
from .experiments import Experiments, Model
from ai4water.hyper_opt import Real, Categorical, Integer
from ai4water.post_processing.SeqMetrics import ClassificationMetrics


class MLClassificationExperiments(Experiments):
    """Runs classification models for comparison, with or without
    optimization of hyperparameters."""

    def __init__(self,
                 param_space=None,
                 x0=None,
                 data=None,
                 cases=None,
                 exp_name='MLExperiments',
                 dl4seq_model=None,
                 num_samples=5,
                 **model_kwargs):
        self.param_space = param_space
        self.x0 = x0
        self.data = data
        self.model_kws = model_kwargs
        self.dl4seq_model = Model if dl4seq_model is None else dl4seq_model

        super().__init__(cases=cases, exp_name=exp_name, num_samples=num_samples)

    def build_and_run(self,
                      predict=False,
                      view=False,
                      title=None,
                      cross_validate=False,
                      fit_kws=None,
                      **kwargs):

        fit_kws = fit_kws or {}

        verbosity = 0
        if 'verbosity' in self.model_kws:
            verbosity = self.model_kws.pop('verbosity')

        model = self.dl4seq_model(
            data=self.data,
            prefix=title,
            verbosity=verbosity,
            **self.model_kws,
            **kwargs
        )

        setattr(self, '_model', model)

        model.fit(**fit_kws)

        t = model.predict()

        if view:
            model.view_model()

        if predict:
            model.predict('training')

        return ClassificationMetrics(t, t).mse()

    def model_AdaBoostClassifier(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
        self.param_space = [
            Integer(low=10, high=500, name='n_estimators', num_samples=self.num_samples),
            Real(low=1.0, high=5.0, name='learning_rate', num_samples=self.num_samples),
            Categorical(categories=['SAMME', 'SAMME.R'], name='algorithm')
        ]
        self.x0 = [50, 1.0, 'SAMME']
        return {'model': {'AdaBoostClassifier': kwargs}}

    def model_BaggingClassifier(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html?highlight=baggingclassifier

        self.param_space = [
            Integer(low=5, high=50, name='n_estimators', num_samples=self.num_samples),
            Real(low=0.1, high=1.0, name='max_samples', num_samples=self.num_samples),
            Real(low=0.1, high=1.0, name='max_features', num_samples=self.num_samples),
            Categorical(categories=[True, False], name='bootstrap'),
            Categorical(categories=[True, False], name='bootstrap_features')
            # Categorical(categories=[True, False], name='oob_score'),  # linked with bootstrap
        ]
        self.x0 = [10, 1.0, 1.0, True, False]
        return {'model': {'BaggingClassifier': kwargs}}

    def model_BernoulliNB(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html?highlight=bernoullinb

        self.param_space = [
            Real(low=0.1, high=1.0, name='alpha', num_samples=self.num_samples),
            Real(low=0.0, high=1.0, name='binarize', num_samples=self.num_samples)
        ]
        self.x0 = [0.5, 0.5]
        return {'model': {'BernoulliNB': kwargs}}

    def model_CalibratedClassifierCV(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html?highlight=calibratedclassifiercv
        self.param_space = [
            Categorical(categories=['sigmoid', 'isotonic'], name='method'),
            Integer(low=5, high=50, name='n_jobs', num_samples=self.num_samples)
        ]
        self.x0 = [5, 'sigmoid']
        return {'model': {'CalibratedClassifierCV': kwargs}}

    def model_CheckingClassifier(self, **kwargs):
        return

    def model_DecisionTreeClassifier(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decisiontreeclassifier#sklearn.tree.DecisionTreeClassifier
        self.param_space = [
            Categorical(["best", "random"], name='splitter'),
            Integer(low=2, high=10, name='min_samples_split', num_samples=self.num_samples),
            #Real(low=1, high=5, name='min_samples_leaf'),
            Real(low=0.0, high=0.5, name="min_weight_fraction_leaf", num_samples=self.num_samples),
            Categorical(categories=['auto', 'sqrt', 'log2'], name="max_features"),
        ]
        self.x0 = ['best', 2, 0.0, 'auto']
        return {'model': {'DecisionTreeClassifier': kwargs}}

    def model_DummyClassifier(self, **kwargs):
        ##  https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html?highlight=dummyclassifier
        self.param_space = [
            Categorical(categories=['stratified', 'most_frequent', 'prior', 'uniform', 'constant'], name='strategy')
        ]
        self.x0 = ['prior']
        return {'model': {'DummyClassifier': kwargs}}

    def model_ExtraTreeClassifier(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html?highlight=extratreeclassifier
        self.param_space = [
            Integer(low=3, high=30, name='max_depth', num_samples=self.num_samples),
            Real(low=0.1, high=0.5, name='min_samples_split', num_samples=self.num_samples),
            Real(low=0.0, high=0.5, name='min_weight_fraction_leaf', num_samples=self.num_samples),
            Categorical(categories=['auto', 'sqrt', 'log2'], name='max_features')
        ]
        self.x0 = [5, 0.2, 0.2, 'auto']
        return {'model': {'ExtraTreeClassifier': kwargs}}

    def model_ExtraTreesClassifier(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html?highlight=extratreesclassifier
        self.param_space = [
            Integer(low=5, high=50, name='n_estimators', num_samples=self.num_samples),
            Integer(low=3, high=30, name='max_depth', num_samples=self.num_samples),
            Real(low=0.1, high=0.5, name='min_samples_split', num_samples=self.num_samples),
            Real(low=0.0, high=0.5, name='min_weight_fraction_leaf', num_samples=self.num_samples),
            Categorical(categories=['auto', 'sqrt', 'log2'], name='max_features')
        ]
        self.x0 = [10, 5, 0.4, 0.1, 'auto']
        return {'model': {'ExtraTreesClassifier': kwargs}}

    def model_KNeighborsClassifier(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html?highlight=kneighborsclassifier#sklearn.neighbors.KNeighborsClassifier
        self.param_space = [
            Integer(low=3, high=5, name='n_neighbors', num_samples=self.num_samples),
            Categorical(categories=['uniform', 'distance'], name='weights'),
            Categorical(categories=['auto', 'ball_tree', 'kd_tree', 'brute'], name='algorithm'),
            Integer(low=10, high=100, name='leaf_size', num_samples=self.num_samples),
            Integer(low=1, high=5, name='p', num_samples=self.num_samples)
        ]
        self.x0 = [5, 'uniform', 'auto', 30, 2]
        return {'model': {'KNeighborsClassifier': kwargs}}

    def model_LabelPropagation(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html?highlight=labelpropagation
        self.param_space = [
            Categorical(categories=['knn', 'rbf'], name='kernel'),
            Integer(low=5, high=10, name='n_neighbors', num_samples=self.num_samples),
            Integer(low=50, high=1000, name='max_iter', num_samples=self.num_samples),
            Real(low=1e-6, high=1e-2, name='tol', num_samples=self.num_samples),
            Integer(low=2, high=10, name='n_jobs', num_samples=self.num_samples)
        ]
        self.x0 = ['knn', 5, 50, 1e-4, 5]
        return {'model': {'LabelPropagation': kwargs}}

    def model_LabelSpreading(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html?highlight=labelspreading
        self.param_space = [
            Categorical(categories=['knn', 'rbf'], name='kernel'),
            Integer(low=5, high=10, name='n_neighbors', num_samples=self.num_samples),
            Integer(low=10, high=100, name='max_iter', num_samples=self.num_samples),
            Real(low=0.1, high=1.0, name='alpha', num_samples=self.num_samples),
            Real(low=1e-6, high=1e-2, name='tol', num_samples=self.num_samples),
            Integer(low=2, high=50, name='n_jobs', num_samples=self.num_samples)
        ]
        self.x0 = ['knn', 5, 10, 0.1, 1e-4, 5]
        return {'model': {'LabelSpreading': kwargs}}

    def model_LGBMClassifier(self, **kwargs):
        ## https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
        self.param_space = [
            Categorical(categories=['gbdt', 'dart', 'goss', 'rf'], name='boosting_type'),
            Integer(low=10, high=200, name='num_leaves', num_samples=self.num_samples),
            Real(low=0.0001, high=0.1, prior='log', name='learning_rate', num_samples=self.num_samples),
            Real(low=10, high=100, name='min_child_samples', num_samples=self.num_samples),
            Integer(low=20, high=500, name='n_estimators', num_samples=self.num_samples)
        ]
        self.x0 = ['rf', 10, 0.001, 10, 20]
        return {'model': {'LGBMClassifier': kwargs}}

    def model_LinearDiscriminantAnalysis(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
        self.param_space = [
            Categorical(categories=[False, True], name='store_covariance'),
            Integer(low=2, high=100, name='n_components', num_samples=self.num_samples),
            Real(low=1e-6, high=1e-2, name='tol', num_samples=self.num_samples)
        ]
        self.x0 = [True, 2, 1e-4]
        return {'model': {'LinearDiscriminantAnalysis': kwargs}}

    def model_LinearSVC(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html?highlight=linearsvc#sklearn.svm.LinearSVC
        self.param_space = [
            Categorical(categories=[True, False], name='dual'),
            Real(low=1.0, high=5.0, name='C', num_samples=10),
            Integer(low=100, high=1000, name='max_iter', num_samples=self.num_samples),
            Real(low=1e-5, high=1e-1, name='tol', num_samples=10),
            Categorical(categories=[True, False], name='fit_intercept')
        ]
        self.x0 = [True, 1.0, 100, 1e-4, True]
        return {'model': {'LinearSVC': kwargs}}

    def model_LogisticRegression(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression
        self.param_space = [
            Categorical(categories=[True, False], name='dual'),
            Real(low=1e-5, high=1e-1, name='tol', num_samples=self.num_samples),
            Real(low=0.5, high=5.0, name='C', num_samples=self.num_samples),
            Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=100, high=1000, name='max_iter', num_samples=10)
            #Categorical(categories=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], name='solver')
        ]
        self.x0 = [True,1e-6, 1.0, True, 100]
        return {'model': {'LogisticRegression': kwargs}}

    def model_NearestCentroid(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html?highlight=nearestcentroid
        self.param_space = [
            Real(low=1, high=50, name='shrink_threshold', num_samples=self.num_samples)
        ]
        self.x0 = [5]
        return {'model': {'NearestCentroid': kwargs}}

    def model_NuSVC(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html?highlight=nusvc
        self.param_space = [
            Real(low=0.5, high=0.9, name='nu', num_samples=self.num_samples),
            Integer(low=100, high=1000, name='max_iter', num_samples=self.num_samples),
            Real(low=1e-5, high=1e-1, name='tol', num_samples=self.num_samples),
            Real(low=100, high=500, name='cache_size', num_samples=self.num_samples)
        ]
        self.x0 = [0.5, 100, 1e-5, 100]
        return {'model': {'NuSVC': kwargs}}

    def model_PassiveAggressiveClassifier(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html?highlight=passiveaggressiveclassifier
        self.param_space = [
            Real(low=1.0, high=5.0, name='C', num_samples=self.num_samples),
            Real(low=0.1, high=1.0, name='validation_fraction', num_samples=self.num_samples),
            Real(low=1e-4, high=1e-1, name='tol', num_samples=self.num_samples),
            Integer(low=100, high=1000, name='max_iter', num_samples=self.num_samples),
            Categorical(categories=[True, False], name='fit_intercept')
        ]
        self.x0 = [1.0, 0.1, 1e-4, 200, True]
        return {'model': {'PassiveAggressiveClassifier': kwargs}}

    def model_Perceptron(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html?highlight=perceptron#sklearn.linear_model.Perceptron
        self.param_space = [
            Real(low=1e-6, high=1e-2, name='alpha', num_samples=self.num_samples),
            Real(low=0.1, high=1.0, name='validation_fraction', num_samples=self.num_samples),
            Real(low=1e-4, high=1e-1, name='tol', num_samples=self.num_samples),
            Integer(low=100, high=1000, name='max_iter', num_samples=self.num_samples),
            Categorical(categories=[True, False], name='fit_intercept')
        ]
        self.x0 = [1e-4, 0.1, 1e-3, 200, True]
        return {'model': {'Perceptron': kwargs}}

    def model_QuadraticDiscriminantAnalysis(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html?highlight=quadraticdiscriminantanalysis
        self.param_space = [
            Real(low=0.0, high=1.0, name='reg_param', num_samples=self.num_samples),
            Real(low=1e-4, high=1e-1, name='tol', num_samples=self.num_samples),
            Categorical(categories=[True, False], name='store_covariance')
        ]
        self.x0 = [0.1, 1e-3, True]
        return {'model': {'QuadraticDiscriminantAnalysi': kwargs}}

    def model_RandomForestClassifier(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforestclassifier#sklearn.ensemble.RandomForestClassifier
        self.param_space = [
            Integer(low=50, high=1000, name='n_estimators', num_samples=self.num_samples),
            Integer(low=3, high=30, name='max_depth', num_samples=self.num_samples),
            Integer(low=2, high=10, name='min_samples_split', num_samples=self.num_samples),
            Real(low=0.0, high=0.5, name='min_weight_fraction_leaf', num_samples=self.num_samples),
            Categorical(categories=['auto', 'sqrt', 'log2'], name='max_features')
        ]
        self.x0 = [100, 5, 2, 0.2, 'auto']
        return {'model': {'RandomForestClassifier': kwargs}}

    def model_RidgeClassifier(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html?highlight=ridgeclassifier#sklearn.linear_model.RidgeClassifier
        self.param_space = [
            Real(low=1.0, high=5.0, name='alpha', num_samples=self.num_samples),
            Real(low=1e-4, high=1e-1, name='tol', num_samples=self.num_samples),
            Categorical(categories=[True, False], name='normalize'),
            Categorical(categories=[True, False], name='fit_intercept')
        ]
        self.x0 = [1.0, 1e-3, True, True]
        return {'model': {'RidgeClassifier': kwargs}}

    def model_RidgeClassifierCV(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html?highlight=ridgeclassifier#sklearn.linear_model.RidgeClassifierCV
        self.param_space = [
            Categorical(categories=[1e-3, 1e-2, 1e-1, 1], name='alphas'),
            Categorical(categories=[True, False], name='normalize'),
            Categorical(categories=[True, False], name='fit_intercept')
        ]
        self.x0 = [1e-3,True, True]
        return {'model': {'RidgeClassifierCV': kwargs}}

    def model_SGDClassifier(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html?highlight=sgdclassifier#sklearn.linear_model.SGDClassifier
        self.param_space = [
            Categorical(categories=['l1', 'l2', 'elasticnet'], name='penalty'),
            Real(low=1e-6, high=1e-2, name='alpha', num_samples=self.num_samples),
            Real(low=0.0, high=1.0, name='eta0', num_samples=self.num_samples),
            Categorical(categories=[True, False], name='fit_intercept'),
            Integer(low=500, high=5000, name='max_iter', num_samples=self.num_samples),
            Categorical(categories=['constant', 'optimal', 'invscaling', 'adaptive'], name='learning_rate')
        ]
        self.x0 = ['l2', 1e-4, 0.5,True, 1000, 'invscaling']
        return {'model': {'SGDClassifier': kwargs}}

    def model_SVC(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?highlight=svc#sklearn.svm.SVC
        self.param_space = [
            Real(low=1.0, high=5.0, name='C', num_samples=self.num_samples),
            Real(low=1e-5, high=1e-1, name='tol', num_samples=self.num_samples),
            Real(low=200, high=1000, name='cache_size', num_samples=self.num_samples)
        ]
        self.x0 = [1.0, 1e-3, 200]
        return {'model': {'SVC': kwargs}}

    def model_XGBClassifier(self, **kwargs):
        ## https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
        self.param_space = [
            Integer(low=5, high=50, name='n_estimators', num_samples=self.num_samples),  # Number of gradient boosted trees
            Integer(low=3, high=30, name='max_depth', num_samples=self.num_samples),  # Maximum tree depth for base learners
            Real(low=0.0001, high=0.5, prior='log', name='learning_rate', num_samples=self.num_samples),  #
            Categorical(categories=['gbtree', 'gblinear', 'dart'], name='booster'),
            Real(low=0.1, high=0.9, name='gamma', num_samples=self.num_samples),
            # Minimum loss reduction required to make a further partition on a leaf node of the tree.
            Real(low=0.1, high=0.9, name='min_child_weight', num_samples=self.num_samples),
            # Minimum sum of instance weight(hessian) needed in a child.
            Real(low=0.1, high=0.9, name='max_delta_step ', num_samples=self.num_samples),
            # Maximum delta step we allow each tree’s weight estimation to be.
            Real(low=0.1, high=0.9, name='subsample', num_samples=self.num_samples),  # Subsample ratio of the training instance.
            Real(low=0.1, high=0.9, name='colsample_bytree', num_samples=self.num_samples),
            Real(low=0.1, high=0.9, name='colsample_bylevel', num_samples=self.num_samples),
            Real(low=0.1, high=0.9, name='colsample_bynode', num_samples=self.num_samples),
            Real(low=0.1, high=0.9, name='reg_alpha', num_samples=self.num_samples),
            Real(low=0.1, high=0.9, name='reg_lambda', num_samples=self.num_samples)
        ]
        self.x0 = [10, 3, 0.0001, 'gbtree', 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        return {'model': {'XGBClassifier': kwargs}}

    # def model_TPOTCLASSIFIER(self, **kwargs):
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
    #     return {'model': {'TPOTCLASSIFIER': kwargs}}
