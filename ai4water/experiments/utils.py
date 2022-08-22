import sys

from ai4water.hyperopt import Integer, Real, Categorical


def regression_space(
        num_samples:int,
        num_examples:int=None,
        verbosity: bool = 0
)->dict:

    spaces = {
        "AdaBoostRegressor":{
            "param_space": [
                Integer(low=5, high=100, name='n_estimators', num_samples=num_samples),
                Real(low=0.001, high=1.0, prior='log', name='learning_rate', num_samples=num_samples)],
            "x0":
                [50, 1.0]},
        "ARDRegression":{
            "param_space":[
                Real(low=1e-7, high=1e-5, name='alpha_1', num_samples=num_samples),
                Real(low=1e-7, high=1e-5, name='alpha_2', num_samples=num_samples),
                Real(low=1e-7, high=1e-5, name='lambda_1', num_samples=num_samples),
                Real(low=1e-7, high=1e-5, name='lambda_2', num_samples=num_samples),
                Real(low=1000, high=1e5, name='threshold_lambda', num_samples=num_samples),
                Categorical(categories=[True, False], name='fit_intercept')],
            "x0":
                [1e-7, 1e-7, 1e-7, 1e-7, 1000, True]},
        "BaggingRegressor": {
            "param_space": [
                Integer(low=5, high=50, name='n_estimators', num_samples=num_samples),
                Real(low=0.1, high=1.0, name='max_samples', num_samples=num_samples),
                Real(low=0.1, high=1.0, name='max_features', num_samples=num_samples),
                Categorical(categories=[True, False], name='bootstrap'),
                Categorical(categories=[True, False], name='bootstrap_features')],
                #Categorical(categories=[True, False], name='oob_score')], # linked with bootstrap
            "x0":
                [10, 1.0, 1.0, True, False]},
        "BayesianRidge": {
            "param_space": [
                Integer(low=40, high=1000, name='n_iter', num_samples=num_samples),
                Real(low=1e-7, high=1e-5, name='alpha_1', num_samples=num_samples),
                Real(low=1e-7, high=1e-5, name='alpha_2', num_samples=num_samples),
                Real(low=1e-7, high=1e-5, name='lambda_1', num_samples=num_samples),
                Real(low=1e-7, high=1e-5, name='lambda_2', num_samples=num_samples),
                Categorical(categories=[True, False], name='fit_intercept')],
            "x0": [40, 1e-7, 1e-7, 1e-7, 1e-7, True]},
        "DummyRegressor": {
            "param_space": [
                Categorical(categories=['mean', 'median', 'quantile'], name='strategy')],
            "x0":
                ['quantile']},
        "KNeighborsRegressor": {
            "param_space": [
                Integer(low=3, high=num_examples or 50, name='n_neighbors', num_samples=num_samples),
                Categorical(categories=['uniform', 'distance'], name='weights'),
                Categorical(categories=['auto', 'ball_tree', 'kd_tree', 'brute'], name='algorithm'),
                Integer(low=10, high=100, name='leaf_size', num_samples=num_samples),
                Integer(low=1, high=5, name='p', num_samples=num_samples)],
            "x0":
                [5, 'uniform', 'auto', 30, 2]},
        "LassoLars":{
            "param_space": [
                Real(low=1.0, high=5.0, name='alpha', num_samples=num_samples),
                Categorical(categories=[True, False], name='fit_intercept')],
            "x0":
                [1.0, False]},
        "Lars": {
            "param_space": [
                Categorical(categories=[True, False], name='fit_intercept'),
                Integer(low=100, high=1000, name='n_nonzero_coefs', num_samples=num_samples)],
            "x0":
                [True, 100]},
        "LarsCV": {
            "param_space": [
                Categorical(categories=[True, False], name='fit_intercept'),
                Integer(low=100, high=1000, name='max_iter', num_samples=num_samples),
                Integer(low=100, high=5000, name='max_n_alphas', num_samples=num_samples)],
            "x0":
                [True, 500, 1000]},
        "LinearSVR": {
            "param_space": [
                Real(low=1.0, high=5.0, name='C', num_samples=num_samples),
                Real(low=0.01, high=0.9, name='epsilon', num_samples=num_samples),
                Real(low=1e-5, high=1e-1, name='tol', num_samples=num_samples),
                Categorical(categories=[True, False], name='fit_intercept')],
            "x0":
                [1.0, 0.01, 1e-5, True]},
        "Lasso": {
            "param_space": [
                Real(low=1.0, high=5.0, name='alpha', num_samples=num_samples),
                Categorical(categories=[True, False], name='fit_intercept'),
                Real(low=1e-5, high=1e-1, name='tol', num_samples=num_samples)],
            "x0":
                [1.0, True, 1e-5]},
        "LassoCV": {
            "param_space": [
                Real(low=1e-5, high=1e-2, name='eps', num_samples=num_samples),
                Integer(low=10, high=1000, name='n_alphas', num_samples=num_samples),
                Categorical(categories=[True, False], name='fit_intercept'),
                Integer(low=500, high=5000, name='max_iter', num_samples=num_samples)],
            "x0":
                [1e-3, 100, True, 1000]},
        "LassoLarsCV": {
            "param_space": [
                Categorical(categories=[True, False], name='fit_intercept'),
                Integer(low=500, high=5000, name='max_n_alphas', num_samples=num_samples)],
            "x0":
                [True, 1000]},
        "LassoLarsIC": {
            "param_space": [
                Categorical(categories=[True, False], name='fit_intercept'),
                Categorical(categories=['bic', 'aic'], name='criterion')],
            "x0":
                [True, 'bic']},
        "LinearRegression": {
            "param_space": [
                Categorical(categories=[True, False], name='fit_intercept')],
            "x0":
                [True]},
        "MLPRegressor": {
            "param_space": [
                Integer(low=10, high=500, name='hidden_layer_sizes', num_samples=num_samples),
                Categorical(categories=['identity', 'logistic', 'tanh', 'relu'], name='activation'),
                Categorical(categories=['lbfgs', 'sgd', 'adam'], name='solver'),
                Real(low=1e-6, high=1e-3, name='alpha', num_samples=num_samples),
                # Real(low=1e-6, high=1e-3, name='learning_rate')
                Categorical(categories=['constant', 'invscaling', 'adaptive'], name='learning_rate'),],
            "x0":
                [10, 'relu', 'adam', 1e-6,  'constant']},
        "NuSVR": {
            "param_space": [
                Real(low=0.5,high=0.9, name='nu', num_samples=num_samples),
                Real(low=1.0, high=5.0, name='C', num_samples=num_samples),
                Categorical(categories=['linear', 'poly', 'rbf', 'sigmoid'], name='kernel')],
            "x0":
                [0.5, 1.0, 'sigmoid']},
        "OrthogonalMatchingPursuit": {
            "param_space": [
                Categorical(categories=[True, False], name='fit_intercept'),
                Real(low=0.1, high=10, name='tol', num_samples=num_samples)],
            "x0":
                [True, 0.1]},
        "OrthogonalMatchingPursuitCV": {
            "param_space": [
                # Integer(low=10, high=100, name='max_iter'),
                Categorical(categories=[True, False], name='fit_intercept')],
            "x0":
                [  # 50,
                    True]},
        "OneClassSVM": {
            "param_space": [
                Categorical(categories=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], name='kernel'),
                Real(low=0.1, high=0.9, name='nu', num_samples=num_samples),
                Categorical(categories=[True, False], name='shrinking')],
            "x0":
                ['rbf', 0.1, True]},
        "PoissonRegressor": {
            "param_space": [
                Real(low=0.0, high=1.0, name='alpha', num_samples=num_samples),
                # Categorical(categories=[True, False], name='fit_intercept'),
                Integer(low=50, high=500, name='max_iter', num_samples=num_samples)],
            "x0":
                [0.5, 100]},
        "Ridge": {
            "param_space": [
                Real(low=0.0, high=3.0, name='alpha', num_samples=num_samples),
                Categorical(categories=[True, False], name='fit_intercept'),
                Categorical(categories=['auto', 'svd', 'cholesky', 'saga'], name='solver'),],
            "x0":
                [1.0, True, 'auto']},
        "RidgeCV": {
            "param_space": [
                Categorical(categories=[True, False], name='fit_intercept'),
                Categorical(categories=['auto', 'svd', 'eigen'], name='gcv_mode'),],
            "x0":
                [True, 'auto']},
        "RadiusNeighborsRegressor": {
            "param_space": [
                Categorical(categories=['uniform', 'distance'], name='weights'),
                Categorical(categories=['auto', 'ball_tree', 'kd_tree', 'brute'], name='algorithm'),
                Integer(low=10, high=300, name='leaf_size', num_samples=num_samples),
                Integer(low=1,high=5, name='p', num_samples=num_samples)],
            "x0":
                ['uniform', 'auto', 10, 1]},
        "RANSACRegressor": {
            "param_space": [
                Integer(low=10, high=1000, name='max_trials'),
                Real(low=0.01, high=0.99, name='min_samples', num_samples=num_samples)],
            "x0":
                [10, 0.01]},
        "TweedieRegressor": {
            "param_space": [
                Real(low=0.0, high=5.0, name='alpha', num_samples=num_samples),
                Categorical(categories=['auto', 'identity', 'log'], name='link'),
                Integer(low=50, high=500, name='max_iter', num_samples=num_samples)],
            "x0":
                [1.0, 'auto',100]},
        "TheilSenRegressor": {
            "param_space": [
                Categorical(categories=[True, False], name='fit_intercept'),
                Integer(low=30, high=1000, name='max_iter', num_samples=num_samples),
                Real(low=1e-5, high=1e-1, name='tol', num_samples=num_samples),
                # Integer(low=self.data.shape[1]+1, high=len(self.data), name='n_subsamples')
                ],
            "x0":
                [True, 50, 0.001]},
        "XGBRegressor": {
            "param_space": [
                #  Number of gradient boosted trees
                Integer(low=5, high=200, name='n_estimators', num_samples=num_samples),
                # Maximum tree depth for base learners
                #Integer(low=3, high=50, name='max_depth', num_samples=num_samples),
                Real(low=0.0001, high=0.5, name='learning_rate', prior='log', num_samples=num_samples),
                Categorical(categories=['gbtree', 'gblinear', 'dart'], name='booster'),
                # Minimum loss reduction required to make a further partition on a leaf node of the tree.
                # Real(low=0.1, high=0.9, name='gamma', num_samples=self.num_samples),
                # Minimum sum of instance weight(hessian) needed in a child.
                # Real(low=0.1, high=0.9, name='min_child_weight', num_samples=self.num_samples),
                # Maximum delta step we allow each tree’s weight estimation to be.
                # Real(low=0.1, high=0.9, name='max_delta_step', num_samples=self.num_samples),
                #  Subsample ratio of the training instance.
                # Real(low=0.1, high=0.9, name='subsample', num_samples=self.num_samples),
                # Real(low=0.1, high=0.9, name='colsample_bytree', num_samples=self.num_samples),
                # Real(low=0.1, high=0.9, name='colsample_bylevel', num_samples=self.num_samples),
                # Real(low=0.1, high=0.9, name='colsample_bynode', num_samples=self.num_samples),
                # Real(low=0.1, high=0.9, name='reg_alpha', num_samples=self.num_samples),
                # Real(low=0.1, high=0.9, name='reg_lambda', num_samples=self.num_samples)
            ],
            "x0":
                None},
        "RandomForestRegressor": {
            "param_space": [
                Integer(low=5, high=50, name='n_estimators', num_samples=num_samples),
                Integer(low=3, high=30, name='max_depth', num_samples=num_samples),
                Real(low=0.1, high=0.5, name='min_samples_split', num_samples=num_samples),
                # Real(low=0.1, high=1.0, name='min_samples_leaf'),
                Real(low=0.0, high=0.5, name='min_weight_fraction_leaf', num_samples=num_samples),
                Categorical(categories=['auto', 'sqrt', 'log2'], name='max_features')],
            "x0":
                [10, 5, 0.4,  # 0.2,
                 0.1, 'auto']},
        "GradientBoostingRegressor": {
            "param_space": [
                # number of boosting stages to perform
                Integer(low=5, high=500, name='n_estimators', num_samples=num_samples),
                #  shrinks the contribution of each tree
                Real(low=0.001, high=1.0, prior='log', name='learning_rate', num_samples=num_samples),
                # fraction of samples to be used for fitting the individual base learners
                Real(low=0.1, high=1.0, name='subsample', num_samples=num_samples),
                Real(low=0.1, high=0.9, name='min_samples_split', num_samples=num_samples),
                Integer(low=2, high=30, name='max_depth', num_samples=num_samples)],
            "x0":
                [5, 0.001, 1, 0.1, 3]},
        "LGBMRegressor": {
            "param_space": [
                # todo, during optimization not working with 'rf'
                Categorical(categories=['gbdt', 'dart', 'goss'], name='boosting_type'),
                Integer(low=10, high=200, name='num_leaves', num_samples=num_samples),
                Real(low=0.0001, high=0.1,  name='learning_rate', prior='log', num_samples=num_samples),
                Integer(low=20, high=500, name='n_estimators', num_samples=num_samples)],
            "x0":
                ['gbdt', 31, 0.1, 100]},
        "CatBoostRegressor": {
            "param_space": [
                # maximum number of trees that can be built
                Integer(low=500, high=5000, name='iterations', num_samples=num_samples),
                # Used for reducing the gradient step.
                Real(low=0.0001, high=0.5, prior='log', name='learning_rate', num_samples=num_samples),
                # Coefficient at the L2 regularization term of the cost function.
                Real(low=0.5, high=5.0, name='l2_leaf_reg', num_samples=num_samples),
                # arger the value, the smaller the model size.
                Real(low=0.1, high=10, name='model_size_reg', num_samples=num_samples),
                # percentage of features to use at each split selection, when features are selected over again at random.
                Real(low=0.1, high=0.95, name='rsm', num_samples=num_samples),
                # number of splits for numerical features
                Integer(low=32, high=1032, name='border_count', num_samples=num_samples),
                # The quantization mode for numerical features.  The quantization mode for numerical features.
                Categorical(categories=['Median', 'Uniform', 'UniformAndQuantiles',
                                            'MaxLogSum', 'MinEntropy', 'GreedyLogSum'], name='feature_border_type')],
            "x0":
                [1000, 0.01, 3.0, 0.5, 0.5, 32, 'GreedyLogSum']},
        "DecisionTreeRegressor": {
            "param_space": [
                Categorical(["best", "random"], name='splitter'),
                Integer(low=2, high=10, name='min_samples_split', num_samples=num_samples),
                # Real(low=1, high=5, name='min_samples_leaf'),
                Real(low=0.0, high=0.5, name="min_weight_fraction_leaf", num_samples=num_samples),
                Categorical(categories=['auto', 'sqrt', 'log2'], name="max_features")],
            "x0":
                ['best', 2, 0.0, 'auto']},
        "ElasticNet": {
            "param_space": [
                Real(low=1.0, high=5.0, name='alpha', num_samples=num_samples),
                Real(low=0.1, high=1.0, name='l1_ratio', num_samples=num_samples),
                Categorical(categories=[True, False], name='fit_intercept'),
                Integer(low=500, high=5000, name='max_iter', num_samples=num_samples),
                Real(low=1e-5, high=1e-3, name='tol', num_samples=num_samples)],
            "x0":
                [2.0, 0.2, True, 1000, 1e-4]},
        "ElasticNetCV": {
            "param_space": [
                Real(low=0.1, high=1.0, name='l1_ratio', num_samples=num_samples),
                Real(low=1e-5, high=1e-2, name='eps', num_samples=num_samples),
                Integer(low=10, high=1000, name='n_alphas', num_samples=num_samples),
                Categorical(categories=[True, False], name='fit_intercept'),
                Integer(low=500, high=5000, name='max_iter', num_samples=num_samples)],
            "x0":
                [0.5, 1e-3, 100, True, 1000]},
        "ExtraTreeRegressor": {
            "param_space": [
                Integer(low=3, high=30, name='max_depth', num_samples=num_samples),
                Real(low=0.1, high=0.5, name='min_samples_split', num_samples=num_samples),
                Real(low=0.0, high=0.5, name='min_weight_fraction_leaf', num_samples=num_samples),
                Categorical(categories=['auto', 'sqrt', 'log2'], name='max_features')],
            "x0":
                [5, 0.2, 0.2, 'auto']},
        "ExtraTreesRegressor": {
                "param_space": [
                    Integer(low=5, high=500, name='n_estimators', num_samples=num_samples),
                    Integer(low=3, high=30, name='max_depth', num_samples=num_samples),
                    Integer(low=2, high=10, name='min_samples_split', num_samples=num_samples),
                    Integer(low=1, high=10, num_samples=num_samples, name='min_samples_leaf'),
                    Real(low=0.0, high=0.5, name='min_weight_fraction_leaf', num_samples=num_samples),
                    Categorical(categories=['auto', 'sqrt', 'log2'], name='max_features')],
                "x0":
                    [100, 5, 2, 1, 0.0, 'auto']},
        "GaussianProcessRegressor": {
            "param_space": [
                Real(low=1e-10, high=1e-7, name='alpha', num_samples=num_samples),
                Integer(low=0, high=5, name='n_restarts_optimizer', num_samples=num_samples)],
            "x0":
                [1e-10, 1]},
        "HistGradientBoostingRegressor": {
            "param_space": [
                # Used for reducing the gradient step.
                Real(low=0.0001, high=0.9, prior='log', name='learning_rate', num_samples=num_samples),
                Integer(low=50, high=500, name='max_iter', num_samples=num_samples),  # maximum number of trees.
                Integer(low=2, high=100, name='max_depth', num_samples=num_samples),  # maximum number of trees.
                # maximum number of leaves for each tree
                Integer(low=10, high=100, name='max_leaf_nodes', num_samples=num_samples),
                # minimum number of samples per leaf
                Integer(low=10, high=100, name='min_samples_leaf', num_samples=num_samples),
                # Used for reducing the gradient step.
                Real(low=00, high=0.5, name='l2_regularization', num_samples=num_samples)],
            "x0":
                [0.1, 100, 10, 31, 20, 0.0]},
        "HuberRegressor": {
            "param_space": [
                Real(low=1.0, high=5.0, name='epsilon', num_samples=num_samples),
                Integer(low=50, high=500, name='max_iter', num_samples=num_samples),
                Real(low=1e-5, high=1e-2, name='alpha', num_samples=num_samples),
                Categorical(categories=[True, False], name='fit_intercept')],
            "x0":
                [2.0, 50, 1e-5, False]},
        "KernelRidge": {
            "param_space": [
                Real(low=1.0, high=5.0, name='alpha', num_samples=num_samples)
                # Categorical(categories=['poly', 'linear', name='kernel'])
        ],
            "x0":
                [1.0]},
        "SVR": {
            "param_space": [
                # https://stackoverflow.com/questions/60015497/valueerror-precomputed-matrix-must-be-a-square-matrix-input-is-a-500x29243-mat
                # todo, optimization not working with 'precomputed'
                Categorical(categories=['linear', 'poly', 'rbf', 'sigmoid'], name='kernel'),
                Real(low=1.0, high=5.0, name='C', num_samples=num_samples),
                Real(low=0.01, high=0.9, name='epsilon', num_samples=num_samples)],
            "x0":
                ['rbf',1.0, 0.01]},
        "SGDRegressor": {
            "param_space": [
                Categorical(categories=['l1', 'l2', 'elasticnet'], name='penalty'),
                Real(low=0.01, high=1.0, name='alpha', num_samples=num_samples),
                Categorical(categories=[True, False], name='fit_intercept'),
                Integer(low=500, high=5000, name='max_iter', num_samples=num_samples),
                Categorical(categories=['constant', 'optimal', 'invscaling', 'adaptive'], name='learning_rate')],
            "x0":
                ['l2', 0.1, True, 1000, 'invscaling']},
        "XGBRFRegressor": {
            "param_space": [
                #  Number of gradient boosted trees
                Integer(low=5, high=100, name='n_estimators', num_samples=num_samples),
                # Maximum tree depth for base learners
                Integer(low=3, high=50, name='max_depth', num_samples=num_samples),
                Real(low=0.0001, high=0.5, prior='log', name='learning_rate', num_samples=num_samples),
                # Categorical(categories=['gbtree', 'gblinear', 'dart'], name='booster'),  # todo solve error
                # Minimum loss reduction required to make a further partition on a leaf node of the tree.
                Real(low=0.1, high=0.9, name='gamma', num_samples=num_samples),
                # Minimum sum of instance weight(hessian) needed in a child.
                Real(low=0.1, high=0.9, name='min_child_weight', num_samples=num_samples),
                # Maximum delta step we allow each tree’s weight estimation to be.
                Real(low=0.1, high=0.9, name='max_delta_step', num_samples=num_samples),
                #  Subsample ratio of the training instance.
                Real(low=0.1, high=0.9, name='subsample', num_samples=num_samples),
                Real(low=0.1, high=0.9, name='colsample_bytree', num_samples=num_samples),
                Real(low=0.1, high=0.9, name='colsample_bylevel', num_samples=num_samples),
                Real(low=0.1, high=0.9, name='colsample_bynode', num_samples=num_samples),
                Real(low=0.1, high=0.9, name='reg_alpha', num_samples=num_samples),
                Real(low=0.1, high=0.9, name='reg_lambda', num_samples=num_samples)],
            "x0":
                [50, 3, 0.001, 0.1, 0.1, 0.1, 0.1,
                 0.1, 0.1, 0.1, 0.1, 0.1]}
        }

    # remove the estimators from those libraries which are not available/installed
    libraries_to_models = {
        'catboost': ['CatBoostRegressor'],
        'xgboost': ['XGBRFRegressor', 'XGBRegressor'],
        'lightgbm': ['LGBMRegressor']
    }
    _remove_estimator(spaces, libraries_to_models, verbosity)

    return spaces


def classification_space(num_samples:int, verbosity=0):

    spaces = {
        "AdaBoostClassifier": {
            "param_space": [
                Integer(low=10, high=500, name='n_estimators', num_samples=num_samples),
                Real(low=1.0, high=5.0, name='learning_rate', num_samples=num_samples),
                Categorical(categories=['SAMME', 'SAMME.R'], name='algorithm')],
            "x0":
                [50, 1.0, 'SAMME']},
        "BaggingClassifier": {
            "param_space": [
            Integer(low=5, high=50, name='n_estimators', num_samples=num_samples),
            Real(low=0.1, high=1.0, name='max_samples', num_samples=num_samples),
            Real(low=0.1, high=1.0, name='max_features', num_samples=num_samples),
            Categorical(categories=[True, False], name='bootstrap'),
            Categorical(categories=[True, False], name='bootstrap_features')
            # Categorical(categories=[True, False], name='oob_score'),  # linked with bootstrap
        ],
            "x0":
                [10, 1.0, 1.0, True, False]},
        "BernoulliNB": {
            "param_space": [
                Real(low=0.1, high=1.0, name='alpha', num_samples=num_samples),
                Real(low=0.0, high=1.0, name='binarize', num_samples=num_samples)],
            "x0":
                [0.5, 0.5]},
        "CalibratedClassifierCV": {
            "param_space": [
                Categorical(categories=['sigmoid', 'isotonic'], name='method'),
                #Integer(low=5, high=50, name='n_jobs', num_samples=num_samples)
            ],
            "x0":
                ['sigmoid']},
        "DecisionTreeClassifier": {
            "param_space": [
                Categorical(["best", "random"], name='splitter'),
                Integer(low=2, high=10, name='min_samples_split', num_samples=num_samples),
                # Real(low=1, high=5, name='min_samples_leaf'),
                Real(low=0.0, high=0.5, name="min_weight_fraction_leaf", num_samples=num_samples),
                Categorical(categories=['auto', 'sqrt', 'log2'], name="max_features"),],
            "x0":
                ['best', 2, 0.0, 'auto']},
        "DummyClassifier": {
            "param_space": [
                Categorical(categories=['stratified', 'most_frequent', 'prior', 'uniform', 'constant'],
                            name='strategy')],
            "x0":
                ['prior']},
        "ExtraTreeClassifier": {
            "param_space": [
                Integer(low=3, high=30, name='max_depth', num_samples=num_samples),
                Real(low=0.1, high=0.5, name='min_samples_split', num_samples=num_samples),
                Real(low=0.0, high=0.5, name='min_weight_fraction_leaf', num_samples=num_samples),
                Categorical(categories=['auto', 'sqrt', 'log2'], name='max_features')],
            "x0":
                [5, 0.2, 0.2, 'auto']},
        "ExtraTreesClassifier": {
            "param_space": [
                Integer(low=5, high=50, name='n_estimators', num_samples=num_samples),
                Integer(low=3, high=30, name='max_depth', num_samples=num_samples),
                Real(low=0.1, high=0.5, name='min_samples_split', num_samples=num_samples),
                Real(low=0.0, high=0.5, name='min_weight_fraction_leaf', num_samples=num_samples),
                Categorical(categories=['auto', 'sqrt', 'log2'], name='max_features')],
            "x0": [10, 5, 0.4, 0.1, 'auto']},
        "GaussianProcessClassifier": {
            "param_space": [
                Integer(low=0, high=5, name='n_restarts_optimizer', num_samples=num_samples)],
            "x0":
                [1]},
        "GradientBoostingClassifier": {
            "param_space": [
                # number of boosting stages to perform
                Integer(low=5, high=500, name='n_estimators', num_samples=num_samples),
                #  shrinks the contribution of each tree
                Real(low=0.001, high=1.0, prior='log', name='learning_rate', num_samples=num_samples),
                # fraction of samples to be used for fitting the individual base learners
                Real(low=0.1, high=1.0, name='subsample', num_samples=num_samples),
                Real(low=0.1, high=0.9, name='min_samples_split', num_samples=num_samples),
                Integer(low=2, high=30, name='max_depth', num_samples=num_samples)],
            "x0":
                [5, 0.001, 1, 0.1, 3]},
        "HistGradientBoostingClassifier": {
            "param_space": [
                # Used for reducing the gradient step.
                Real(low=0.0001, high=0.9, prior='log', name='learning_rate', num_samples=num_samples),
                Integer(low=50, high=500, name='max_iter', num_samples=num_samples),  # maximum number of trees.
                Integer(low=2, high=100, name='max_depth', num_samples=num_samples),  # maximum number of trees.
                # maximum number of leaves for each tree
                Integer(low=10, high=100, name='max_leaf_nodes', num_samples=num_samples),
                # minimum number of samples per leaf
                Integer(low=10, high=100, name='min_samples_leaf', num_samples=num_samples),
                # Used for reducing the gradient step.
                Real(low=00, high=0.5, name='l2_regularization', num_samples=num_samples)],
            "x0": [0.1, 100, 10, 31, 20, 0.0]},
        "KNeighborsClassifier": {
            "param_space": [
                Integer(low=3, high=5, name='n_neighbors', num_samples=num_samples),
                Categorical(categories=['uniform', 'distance'], name='weights'),
                Categorical(categories=['auto', 'ball_tree', 'kd_tree', 'brute'], name='algorithm'),
                Integer(low=10, high=100, name='leaf_size', num_samples=num_samples),
                Integer(low=1, high=5, name='p', num_samples=num_samples)],
            "x0":
                [5, 'uniform', 'auto', 30, 2]},
        "LabelPropagation": {
            "param_space": [
                Categorical(categories=['knn', 'rbf'], name='kernel'),
                Integer(low=5, high=10, name='n_neighbors', num_samples=num_samples),
                Integer(low=50, high=1000, name='max_iter', num_samples=num_samples),
                Real(low=1e-6, high=1e-2, name='tol', num_samples=num_samples),
                #Integer(low=2, high=10, name='n_jobs', num_samples=num_samples)
            ],
            "x0":
                ['knn', 5, 50, 1e-4]},
        "LabelSpreading": {
            "param_space": [
                Categorical(categories=['knn', 'rbf'], name='kernel'),
                Integer(low=5, high=10, name='n_neighbors', num_samples=num_samples),
                Integer(low=10, high=100, name='max_iter', num_samples=num_samples),
                Real(low=0.1, high=1.0, name='alpha', num_samples=num_samples),
                Real(low=1e-6, high=1e-2, name='tol', num_samples=num_samples),
                #Integer(low=2, high=50, name='n_jobs', num_samples=num_samples)
            ],
            "x0":
                ['knn', 5, 10, 0.1, 1e-4]},
        "LGBMClassifier": {
            "param_space": [
                Categorical(categories=['gbdt', 'dart', 'goss', 'rf'], name='boosting_type'),
                Integer(low=10, high=200, name='num_leaves', num_samples=num_samples),
                Real(low=0.0001, high=0.1, prior='log', name='learning_rate', num_samples=num_samples),
                Integer(low=10, high=100, name='min_child_samples', num_samples=num_samples),
                Integer(low=20, high=500, name='n_estimators', num_samples=num_samples)],
            "x0":
                ['rf', 10, 0.001, 10, 20]},
        "LinearDiscriminantAnalysis": {
            "param_space": [
                Categorical(categories=[False, True], name='store_covariance'),
                Integer(low=2, high=100, name='n_components', num_samples=num_samples),
                Real(low=1e-6, high=1e-2, name='tol', num_samples=num_samples)],
            "x0": [True, 2, 1e-4]},
        "LinearSVC": {
            "param_space": [
                Categorical(categories=[True, False], name='dual'),
                Real(low=1.0, high=5.0, name='C', num_samples=10),
                Integer(low=100, high=1000, name='max_iter', num_samples=num_samples),
                Real(low=1e-5, high=1e-1, name='tol', num_samples=10),
                Categorical(categories=[True, False], name='fit_intercept')],
            "x0":
                [True, 1.0, 100, 1e-4, True]},
        "LogisticRegression": {
            "param_space": [
                #Categorical(categories=[True, False], name='dual'),
                Real(low=1e-5, high=1e-1, name='tol', num_samples=num_samples),
                Real(low=0.5, high=5.0, name='C', num_samples=num_samples),
                Categorical(categories=[True, False], name='fit_intercept'),
                Integer(low=100, high=1000, name='max_iter', num_samples=10)
                #Categorical(categories=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], name='solver')
            ],
            "x0":
                [1e-6, 1.0, True, 100]},
        "MLPClassifier": {
            "param_space": [
                Integer(low=10, high=500, name='hidden_layer_sizes', num_samples=num_samples),
                Categorical(categories=['identity', 'logistic', 'tanh', 'relu'], name='activation'),
                Categorical(categories=['lbfgs', 'sgd', 'adam'], name='solver'),
                Real(low=1e-6, high=1e-3, name='alpha', num_samples=num_samples),
                # Real(low=1e-6, high=1e-3, name='learning_rate')
                Categorical(categories=['constant', 'invscaling', 'adaptive'], name='learning_rate'), ],
            "x0":
                [10, 'relu', 'adam', 1e-6, 'constant']},
        "NearestCentroid": {
            "param_space": [
                Real(low=1, high=50, name='shrink_threshold', num_samples=num_samples)],
            "x0":
                [5]},
        "NuSVC": {
            "param_space": [
                Real(low=0.5, high=0.9, name='nu', num_samples=num_samples),
                Integer(low=100, high=1000, name='max_iter', num_samples=num_samples),
                Real(low=1e-5, high=1e-1, name='tol', num_samples=num_samples),
                Real(low=100, high=500, name='cache_size', num_samples=num_samples)],
            "x0":
                [0.5, 100, 1e-5, 100]},
        "PassiveAggressiveClassifier": {
            "param_space": [
                Real(low=1.0, high=5.0, name='C', num_samples=num_samples),
                Real(low=0.1, high=1.0, name='validation_fraction', num_samples=num_samples),
                Real(low=1e-4, high=1e-1, name='tol', num_samples=num_samples),
                Integer(low=100, high=1000, name='max_iter', num_samples=num_samples),
                Categorical(categories=[True, False], name='fit_intercept')],
            "x0": [1.0, 0.1, 1e-4, 200, True]},
        "Perceptron": {
            "param_space": [
                Real(low=1e-6, high=1e-2, name='alpha', num_samples=num_samples),
                Real(low=0.1, high=1.0, name='validation_fraction', num_samples=num_samples),
                Real(low=1e-4, high=1e-1, name='tol', num_samples=num_samples),
                Integer(low=100, high=1000, name='max_iter', num_samples=num_samples),
                Categorical(categories=[True, False], name='fit_intercept')],
            "x0":
                [1e-4, 0.1, 1e-3, 200, True]},
        "QuadraticDiscriminantAnalysis": {
            "param_space": [
                Real(low=0.0, high=1.0, name='reg_param', num_samples=num_samples),
                Real(low=1e-4, high=1e-1, name='tol', num_samples=num_samples),
                Categorical(categories=[True, False], name='store_covariance')],
            "x0":
                [0.1, 1e-3, True]},
        "RadiusNeighborsClassifier": {
            "param_space": [
                Categorical(categories=['uniform', 'distance'], name='weights'),
                Categorical(categories=['auto', 'ball_tree', 'kd_tree', 'brute'], name='algorithm'),
                Integer(low=10, high=300, name='leaf_size', num_samples=num_samples),
                Integer(low=1, high=5, name='p', num_samples=num_samples)],
            "x0":
                ['uniform', 'auto', 10, 1]},
        "RandomForestClassifier": {
            "param_space": [
                Integer(low=50, high=1000, name='n_estimators', num_samples=num_samples),
                Integer(low=3, high=30, name='max_depth', num_samples=num_samples),
                Integer(low=2, high=10, name='min_samples_split', num_samples=num_samples),
                Real(low=0.0, high=0.5, name='min_weight_fraction_leaf', num_samples=num_samples),
                Categorical(categories=['auto', 'sqrt', 'log2'], name='max_features')],
            "x0":
                [100, 5, 2, 0.2, 'auto']},
        "RidgeClassifier": {
            "param_space": [
                Real(low=1.0, high=5.0, name='alpha', num_samples=num_samples),
                Real(low=1e-4, high=1e-1, name='tol', num_samples=num_samples),
                Categorical(categories=[True, False], name='normalize'),
                Categorical(categories=[True, False], name='fit_intercept')],
            "x0":
                [1.0, 1e-3, True, True]},
        "RidgeClassifierCV": {
            "param_space": [
                Categorical(categories=[1e-3, 1e-2, 1e-1, 1], name='alphas'),
                Categorical(categories=[True, False], name='normalize'),
                Categorical(categories=[True, False], name='fit_intercept')],
            "x0":
                [1e-3,True, True]},
        "SGDClassifier": {
            "param_space": [
                Categorical(categories=['l1', 'l2', 'elasticnet'], name='penalty'),
                Real(low=1e-6, high=1e-2, name='alpha', num_samples=num_samples),
                Real(low=0.0, high=1.0, name='eta0', num_samples=num_samples),
                Categorical(categories=[True, False], name='fit_intercept'),
                Integer(low=500, high=5000, name='max_iter', num_samples=num_samples),
                Categorical(categories=['constant', 'optimal', 'invscaling', 'adaptive'], name='learning_rate')],
            "x0":
                ['l2', 1e-4, 0.5,True, 1000, 'invscaling']},
        "SVC": {
            "param_space": [
                Real(low=1.0, high=5.0, name='C', num_samples=num_samples),
                Real(low=1e-5, high=1e-1, name='tol', num_samples=num_samples),
                Real(low=200, high=1000, name='cache_size', num_samples=num_samples)],
            "x0":
                [1.0, 1e-3, 200]},
        "XGBClassifier": {
            "param_space": [
                # Number of gradient boosted trees
                Integer(low=5, high=50, name='n_estimators', num_samples=num_samples),
                # Maximum tree depth for base learners
                Integer(low=3, high=30, name='max_depth', num_samples=num_samples),
                Real(low=0.0001, high=0.5, prior='log', name='learning_rate', num_samples=num_samples),  #
                Categorical(categories=['gbtree', 'gblinear', 'dart'], name='booster'),
                Real(low=0.1, high=0.9, name='gamma', num_samples=num_samples),
                # Minimum loss reduction required to make a further partition on a leaf node of the tree.
                Real(low=0.1, high=0.9, name='min_child_weight', num_samples=num_samples),
                # Minimum sum of instance weight(hessian) needed in a child.
                Real(low=0.1, high=0.9, name='max_delta_step', num_samples=num_samples),
                # Maximum delta step we allow each tree’s weight estimation to be.
                # Subsample ratio of the training instance.
                Real(low=0.1, high=0.9, name='subsample', num_samples=num_samples),
                Real(low=0.1, high=0.9, name='colsample_bytree', num_samples=num_samples),
                Real(low=0.1, high=0.9, name='colsample_bylevel', num_samples=num_samples),
                Real(low=0.1, high=0.9, name='colsample_bynode', num_samples=num_samples),
                Real(low=0.1, high=0.9, name='reg_alpha', num_samples=num_samples),
                Real(low=0.1, high=0.9, name='reg_lambda', num_samples=num_samples)],
            "x0":
                [10, 3, 0.0001, 'gbtree', 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
        "XGBRFClassifier": {
            "param_space": [
                # Number of gradient boosted trees
                Integer(low=5, high=50, name='n_estimators', num_samples=num_samples),
                # Maximum tree depth for base learners
                Integer(low=3, high=30, name='max_depth', num_samples=num_samples),
                Real(low=0.0001, high=0.5, prior='log', name='learning_rate', num_samples=num_samples),  #
                Categorical(categories=['gbtree', 'gblinear', 'dart'], name='booster'),
                Real(low=0.1, high=0.9, name='gamma', num_samples=num_samples),
                # Minimum loss reduction required to make a further partition on a leaf node of the tree.
                Real(low=0.1, high=0.9, name='min_child_weight', num_samples=num_samples),
                # Minimum sum of instance weight(hessian) needed in a child.
                Real(low=0.1, high=0.9, name='max_delta_step', num_samples=num_samples),
                # Maximum delta step we allow each tree’s weight estimation to be.
                # Subsample ratio of the training instance.
                Real(low=0.1, high=0.9, name='subsample', num_samples=num_samples),
                Real(low=0.1, high=0.9, name='colsample_bytree', num_samples=num_samples),
                Real(low=0.1, high=0.9, name='colsample_bylevel', num_samples=num_samples),
                Real(low=0.1, high=0.9, name='colsample_bynode', num_samples=num_samples),
                Real(low=0.1, high=0.9, name='reg_alpha', num_samples=num_samples),
                Real(low=0.1, high=0.9, name='reg_lambda', num_samples=num_samples)],
            "x0":
                [10, 3, 0.0001, 'gbtree', 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
        "CatBoostClassifier": {
            "param_space": [
                # maximum number of trees that can be built
                Integer(low=50, high=5000, name='iterations', num_samples=num_samples),
                # Used for reducing the gradient step.
                Real(low=0.0001, high=0.5, prior='log', name='learning_rate', num_samples=num_samples),
                # depth
                # https://stackoverflow.com/q/67299869/5982232
                Integer(1, 15, name="depth", num_samples=num_samples),
                # Coefficient at the L2 regularization term of the cost function.
                Real(low=0.5, high=5.0, name='l2_leaf_reg', num_samples=num_samples),
                # arger the value, the smaller the model size.
                Real(low=0.1, high=10, name='model_size_reg', num_samples=num_samples),
                # percentage of features to use at each split selection, when features are selected over again at random.
                Real(low=0.1, high=0.95, name='rsm', num_samples=num_samples),
                # number of splits for numerical features
                Integer(low=32, high=1032, name='border_count', num_samples=num_samples),
                # The quantization mode for numerical features.  The quantization mode for numerical features.
                Categorical(categories=['Median', 'Uniform', 'UniformAndQuantiles',
                                            'MaxLogSum', 'MinEntropy', 'GreedyLogSum'], name='feature_border_type')],
            "x0":
                [100, 0.01, 5, 3.0, 0.5, 0.5, 32, 'GreedyLogSum']}
    }
    # remove the estimators from those libraries which are not available/installed
    libraries_to_models = {
        'catboost': ['CatBoostClassifier'],
        'xgboost': ['XGBRFClassifier', 'XGBClassifier'],
        'lightgbm': ['LGBMClassifier']
    }
    _remove_estimator(spaces, libraries_to_models, verbosity)

    return spaces


def dl_space(
        num_samples:int=10
)->dict:
    spaces = {
        "MLP": {
            "param_space":[
                Integer(8, 128, name="units", num_samples=num_samples),
                Categorical([1, 2, 3], name="num_layers"),
                Real(0.0, 0.4, name="dropout", num_samples=num_samples),
                Categorical(["relu", "linear", "leakyrelu", "elu", "tanh", "sigmoid"],
                        name="activation")],
            'x0':
                [32, 1, 0.0, "relu"]},
        "LSTM":{
            "param_space": [
                Integer(8, 128, name="units", num_samples=num_samples),
                Categorical([1, 2, 3], name="num_layers"),
                Real(0.0, 0.4, name="dropout", num_samples=num_samples),
                Categorical(["relu",  "leakyrelu", "elu", "tanh", "sigmoid"],
                            name="activation")],
            'x0':
                [32, 1, 0.0, "relu"]},
        "CNN": {
            "param_space": [
                Integer(8, 128, name="filters", num_samples=num_samples),
                Categorical([2,3,4,5], name="kernel_size"),
                Categorical([1, 2, 3], name="num_layers"),
                Real(0.0, 0.4, name="dropout"),
                Categorical(["relu", "leakyrelu", "elu", "tanh", "sigmoid"],
                            name="activation")],
            "x0":
                [32, 2, 1, 0.0, "relu"]},
        "CNNLSTM": {
            "param_space": [
                Categorical([1,2,3], name="cnn_layers"),
                Categorical([1, 2, 3], name="lstm_layers"),
                Integer(8, 128, name="units", num_samples=num_samples),
                Integer(8, 128, name="filters", num_samples=num_samples),
                Categorical([2,3,4,5], name="kernel_size")],
            "x0":
                [2, 1, 32, 32, 2]},
        "LSTMAutoEncoder": {
            "param_space": [
                Integer(8, 128, name="encoder_units", num_samples=num_samples),
                Integer(8, 128, name="decoder_units", num_samples=num_samples),
                Categorical([1,2,3], name="encoder_layers"),
                Categorical([1,2,3], name="decoder_layers")],
            "x0":
                [32, 32, 1, 1]},
        "TCN": {
            "param_space": [
                Integer(16, 128, name="filters", num_samples=num_samples),
                Categorical([2,3,4,5], name="kernel_size")],
            "x0":
                [64, 2]},
        "TFT": {
            "param_space":[
                Integer(16, 128, name="hidden_units", num_samples=num_samples),
                Categorical([1, 2, 3, 4, 5], name="num_heads")],
            "x0":
                [64, 2]}
    }

    return spaces


def _remove_estimator(spaces, libraries_to_models, verbosity=0):
    for lib, estimators in libraries_to_models.items():
        if lib not in sys.modules:
            for estimator in estimators:
                if verbosity>0:
                    print(f"excluding {estimator} because library {lib} is not found")
                spaces.pop(estimator)
    return spaces


def regression_models()->list:
    """returns availabel regression models as list"""
    return list(regression_space(5,5).keys())


def classification_models()->list:
    """returns availabel classification models as list"""
    return list(classification_space(5,0).keys())
