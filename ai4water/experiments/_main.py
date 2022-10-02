import gc
import json
import math
import warnings
from typing import Union, Tuple, List, Callable, Optional

from SeqMetrics import RegressionMetrics, ClassificationMetrics

from ai4water.backend import create_subplots
from ai4water.backend import tf, os, np, pd, plt, easy_mpl
from ai4water.backend import xgboost, catboost, lightgbm
from ai4water.hyperopt import HyperOpt
from ai4water.preprocessing import DataSet
from ai4water.utils.utils import make_model
from ai4water.utils.utils import TrainTestSplit
from ai4water.utils.utils import jsonize, ERROR_LABELS
from ai4water.utils.utils import AttribtueSetter
from ai4water.postprocessing import ProcessPredictions
from ai4water.utils.visualizations import edf_plot
from ai4water.utils.utils import find_best_weight, dateandtime_now, dict_to_file

plot = easy_mpl.plot
bar_chart = easy_mpl.bar_chart
taylor_plot = easy_mpl.taylor_plot
dumbbell_plot = easy_mpl.dumbbell_plot
reg_plot = easy_mpl.regplot

if tf is not None:
    if 230 <= int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 250:
        from ai4water.functional import Model

        print(f"""
        Switching to functional API due to tensorflow version {tf.__version__} 
        for experiments""")
    else:
        from ai4water import Model
else:
    from ai4water import Model

SEP = os.sep

# todo plots comparing different models in following youtube videos at 6:30 and 8:00 minutes.
# https://www.youtube.com/watch?v=QrJlj0VCHys
# compare models using statistical tests wuch as Giacomini-White test or Diebold-Mariano test
# paired ttest 5x2cv


# in order to unify the use of metrics
Metrics = {
    'regression': lambda t, p, multiclass=False, **kwargs: RegressionMetrics(t, p, **kwargs),
    'classification': lambda t, p, multiclass=False, **kwargs: ClassificationMetrics(t, p,
                                                                                     multiclass=multiclass, **kwargs)
}

Monitor = {
    'regression': ['r2', 'corr_coeff', 'mse', 'rmse', 'r2_score',
                   'nse', 'kge', 'mape', 'pbias', 'bias', 'mae', 'nrmse',
                   'mase'],

    'classification': ['accuracy', 'precision', 'recall', 'mse']
}

reg_dts = ["ExtraTreeRegressor","DecisionTreeRegressor",
           "ExtraTreesRegressor", "RandomForestRegressor",
            "AdaBoostRegressor",  "BaggingRegressor",
            "HistGradientBoostingRegressor", "GradientBoostingRegressor"]

cls_dts = ["DecisionTreeClassifier", "ExtraTreeClassifier",
           "ExtraTreesClassifier", "AdaBoostClassifier","RandomForestClassifier", "BaggingClassifier"
           "GradientBoostingClassifier", "HistGradientBoostingClassifier"]

if xgboost is not None:
    reg_dts += ["XGBRegressor"]
    cls_dts += ["XGBClassifier"]

if catboost is not None:
    reg_dts += ["CatBoostRegressor"]
    cls_dts += ["CatBoostClassifier"]

if lightgbm is not None:
    reg_dts += ["LGBMRegressor"]
    cls_dts += ["LGBMClassifier"]

DTs = {"regression":
           reg_dts,
       "classification":
           cls_dts
       }

LMs = {
    "regression":
        ["LinearRegression", "Ridge", "RidgeCV", "SGDRegressor",
         "ElasticNetCV", "ElasticNet",
         "Lasso", "LassoCV", "Lars", "LarsCV", "LassoLars", "LassoLarsCV", "LassoLarsIC"],
    "classification":
        ["LogisticRegression", "LogisticRegressionCV", "PassiveAggressiveClassifier", "Perceptron",
         "RidgeClassifier", "RidgeClassifierCV", "SGDClassifier", "SGDClassifierCV"]
}


class Experiments(object):
    """
    Base class for all the experiments.

    All the experiments must be subclasses of this class.
    The core idea of ``Experiments`` is based upon ``model``. An experiment
    consists of one or more models. The models differ from each other in their
    structure/idea/concept/configuration. When :py:meth:`ai4water.experiments.Experiments.fit`
    is called, each ``model`` is built and trained. The user can customize, building
    and training process by subclassing this class and customizing
    :py:meth:`ai4water.experiments.Experiments._build` and
    :py:meth:`ai4water.experiments.Experiments._fit` methods.

    Attributes
    ------------
    - metrics
    - exp_path
    - model_
    - models

    Methods
    --------
    - fit
    - taylor_plot
    - loss_comparison
    - plot_convergence
    - from_config
    - compare_errors
    - plot_improvement
    - compare_convergence
    - plot_cv_scores
    - fit_with_tpot

    """

    def __init__(
            self,
            cases: dict = None,
            exp_name: str = None,
            num_samples: int = 5,
            verbosity: int = 1,
            monitor: Union[str, list, Callable] = None,
            **model_kws,
    ):
        """

        Arguments
        ---------
            cases :
                python dictionary defining different cases/scenarios. See TransformationExperiments
                for use case.
            exp_name :
                name of experiment, used to define path in which results are saved
            num_samples :
                only relevent when you wan to optimize hyperparameters of models
                using ``grid`` method
            verbosity : bool, optional
                determines the amount of information
            monitor : str, list, optional
                list of performance metrics to monitor. It can be any performance
                metric SeqMetrics_ library.
                By default ``r2``, ``corr_coeff``, ``mse``, ``rmse``, ``r2_score``,
                ``nse``, ``kge``, ``mape``, ``pbias``, ``bias``, ``mae``, ``nrmse``
                ``mase`` are considered for regression and ``accuracy``, ``precision``
                ``recall`` are considered for classification. The user can also put a
                custom metric to monitor. In such a case we it should be callable which
                accepts two input arguments. The first one is array of true and second is
                array of predicted values.

                >>> def f1_score(t,p)->float:
                >>>     return ClassificationMetrics(t, p).f1_score(average="macro")
                >>> monitor = [f1_score, "accuracy"]

                Here ``f1_score`` is a function which accepts two arays.
            **model_kws :
            keyword arguments which are to be passed to `Model`
                and are not optimized.

        .. _SeqMetrics:
            https://seqmetrics.readthedocs.io/en/latest/index.html
        """
        self.opt_results = None
        self.optimizer = None
        self.exp_name = 'Experiments_' + str(dateandtime_now()) if exp_name is None else exp_name
        self.num_samples = num_samples
        self.verbosity = verbosity

        self.models = [method for method in dir(self) if method.startswith('model_')]

        if cases is None:
            cases = {}
        self.cases = {'model_' + key if not key.startswith('model_') else key: val for key, val in cases.items()}
        self.models = self.models + list(self.cases.keys())

        self.exp_path = os.path.join(os.getcwd(), "results", self.exp_name)
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)

        self.update_config(models=self.models, exp_path=self.exp_path,
                           exp_name=self.exp_name, cases=self.cases,
                           model_kws=jsonize(model_kws))

        if monitor is None:
            self.monitor = Monitor[self.mode]
        else:
            if not isinstance(monitor, list):
                monitor = [monitor]

            self.monitor = monitor

        self.model_kws = model_kws

        # _run_type is actually set during call to .fit
        self._run_type = None

    def _pre_build_hook(self, **suggested_paras):
        """Anything that needs to be performed before building the model."""
        return suggested_paras

    def update_and_save_config(self, **kwargs):
        self.update_config(**kwargs)
        self.save_config()
        return

    def update_config(self, **kwargs):
        if not hasattr(self, 'config'):
            setattr(self, 'config', {})
        self.config.update(kwargs.copy())
        return

    def save_config(self):
        dict_to_file(self.exp_path, config=self.config)
        return

    def build_from_config(self, config_path:str)->Model:
        assert os.path.exists(config_path), f"{config_path} does not exist"
        model: Model = Model.from_config_file(config_path=config_path)
        setattr(self, 'model_', model)
        return model

    def update_model_weight(
            self,
            model:Model,
            config_path:str
    ):
        """updates the weight of model. """
        best_weights = find_best_weight(os.path.join(config_path, "weights"))

        assert best_weights is not None, f"Can't find weight from {config_path}"
        weight_file = os.path.join(model.w_path, best_weights)
        model.update_weights(weight_file=weight_file)
        return

    @property
    def tpot_estimator(self):
        raise NotImplementedError

    @property
    def mode(self):
        raise NotImplementedError

    @property
    def num_samples(self):
        return self._num_samples

    @num_samples.setter
    def num_samples(self, x):
        self._num_samples = x

    def _reset(self):

        self.cv_scores_ = {}

        self.config['eval_models'] = {}
        self.config['optimized_models'] = {}

        self.metrics = {}
        self.features = {}
        self.iter_metrics = {}

        self.considered_models_ = []

        return

    def _named_x0(self)->dict:
        x0 = getattr(self, 'x0', None)
        param_space = getattr(self, 'param_space', None)
        if param_space:
            names = [s.name for s in param_space]
            if x0:
                return {k: v for k, v in zip(names, x0)}
        return {}

    def _get_config(self, model_type, model_name, **suggested_paras):
        # the config must contain the suggested parameters by the hpo algorithm
        if model_type in self.cases:
            config = self.cases[model_type]
            config.update(suggested_paras)
        elif model_name in self.cases:
            config = self.cases[model_name]
            config.update(suggested_paras)
        elif hasattr(self, model_type):
            config = getattr(self, model_type)(**suggested_paras)
        else:
            raise TypeError
        return config

    def _dry_run_a_model(
            self,
            model_type,
            model_name,
            cross_validate,
            train_x, train_y, val_x, val_y):

        if self.verbosity >= 0: print(f"running  {model_name} model")

        x, y = _combine_training_validation_data(train_x, train_y, (val_x, val_y))

        config = self._get_config(model_type, model_name, **self._named_x0())

        model = self._build_fit(
            x, y,
            title=f"{self.exp_name}{SEP}{model_name}",
            cross_validate=cross_validate,
            **config)

        train_results = self._predict(model=model, x=train_x, y=train_y)

        self._populate_results(model_name, train_results)

        if cross_validate:
            cv_scoring = model.val_metric
            self.cv_scores_[model_type] = getattr(model, f'cross_val_scores')
            setattr(self, '_cv_scoring', cv_scoring)
        return

    def _optimize_a_model(
            self,
            model_type,
            model_name,opt_method,
            num_iterations,
            cross_validate,
            post_optimize,
            train_x, train_y,
            val_x, val_y,
            **hpo_kws,
    ):

        def objective_fn(**suggested_paras) -> float:
            config = self._get_config(model_type, model_name, **suggested_paras)

            return self._build_fit_eval(
                train_x=train_x,
                train_y=train_y,
                validation_data=(val_x, val_y),
                cross_validate=cross_validate,
                title=f"{self.exp_name}{SEP}{model_name}",
                **config)

        opt_dir = os.path.join(os.getcwd(),
                               f"results{SEP}{self.exp_name}{SEP}{model_name}")

        if self.verbosity > 0:
            print(f"optimizing  {model_name} using {opt_method} method")

        self.optimizer = HyperOpt(
            opt_method,
            objective_fn=objective_fn,
            param_space=self.param_space,
            opt_path=opt_dir,
            num_iterations=num_iterations,  # number of iterations
            x0=self.x0,
            verbosity=self.verbosity,
            **hpo_kws
        )

        self.opt_results = self.optimizer.fit()

        self.config['optimized_models'][model_type] = self.optimizer.opt_path

        if cross_validate:
            # if we do train_best, self.model_ will change and this
            cv_scoring = self.model_.val_metric
            self.cv_scores_[model_type] = getattr(self.model_, f'cross_val_scores')
            setattr(self, '_cv_scoring', cv_scoring)

        x, y = _combine_training_validation_data(train_x, train_y, (val_x, val_y))
        if post_optimize == 'eval_best':
            train_results = self.eval_best(x, y, model_name, opt_dir)
        else:
            train_results = self.train_best(x, y, model_name)

        self._populate_results(model_type, train_results)

        if not hasattr(self, 'model_'):  # todo asking user to define this parameter is not good
            raise ValueError(f'The `build` method must set a class level attribute named `model_`.')
        self.config['eval_models'][model_type] = self.model_.path

        self.iter_metrics[model_type] = self.model_iter_metric
        return

    def fit(
            self,
            x=None,
            y=None,
            data=None,
            validation_data: Optional[tuple] = None,
            run_type: str = "dry_run",
            opt_method: str = "bayes",
            num_iterations: int = 12,
            include: Union[None, list, str] = None,
            exclude: Union[None, list, str] = '',
            cross_validate: bool = False,
            post_optimize: str = 'eval_best',
            hpo_kws: dict = None
    ):
        """
        Runs the fit loop for all the ``models`` of experiment. The user can
        however, specify the models by making use of ``include`` and ``exclude``
        keywords.

        The data should be defined according to following four rules
        either
            - only x,y should be given (val will be taken from it according to splitting schemes)
            - or x,y and validation_data should be given
            - or only data should be given (train and validation data will be
              taken accoring to splitting schemes)

        Parameters
        ----------
            x :
                input data. When ``run_type`` is ``dry_run``, then the each model is trained
                on this data. If ``run_type`` is ``optimize``, validation_data is not given,
                then x,y pairs of validation data are extracted from this data based
                upon splitting scheme i.e. ``val_fraction`` argument.
            y :
                label/true/observed data
            data :
                Raw unprepared data from which x,y pairs for training and validation
                will be extracted.
                this will be passed to :py:meth:`ai4water.Model.fit`.
                This is is only required if ``x`` and ``y`` are not given
            validation_data :
                a tuple which consists of x,y pairs for validation data. This can only
                be given if ``x`` and ``y`` are given and ``data`` is not given.
            run_type : str, optional (default="dry_run")
                One of ``dry_run`` or ``optimize``. If ``dry_run``, then all
                the `models` will be trained only once. if ``optimize``, then
                hyperparameters of all the models will be optimized.
            opt_method : str, optional (default="bayes")
                which optimization method to use. options are ``bayes``,
                ``random``, ``grid``. Only valid if ``run_type`` is ``optimize``
            num_iterations : int, optional
                number of iterations for optimization. Only valid
                if ``run_type`` is ``optimize``.
            include : list/str optional (default="DTs")
                name of models to included. If None, all the models found
                will be trained and or optimized. Default is "DTs", which
                means all decision tree based models will be used.
            exclude :
                name of ``models`` to be excluded
            cross_validate : bool, optional (default=False)
                whether to cross validate the model or not. This
                depends upon `cross_validator` agrument to the `Model`.
            post_optimize : str, optional
                one of ``eval_best`` or ``train_best``. If eval_best,
                the weights from the best models will be uploaded again and the model
                will be evaluated on train, test and all the data. If ``train_best``,
                then a new model will be built and trained using the parameters of
                the best model.
            **hpo_kws :
                keyword arguments for :py:class:`ai4water.hyperopt.HyperOpt` class.

        Examples
        ---------
        >>> from ai4water.experiments import MLRegressionExperiments
        >>> from ai4water.datasets import busan_beach
        >>> exp = MLRegressionExperiments()
        >>> exp.fit(data=busan_beach())

        If you want to compare only RandomForest, XGBRegressor, CatBoostRegressor
        and LGBMRegressor, use the ``include`` keyword

        >>> exp.fit(data=busan_beach(), include=['RandomForestRegressor', 'XGBRegressor',
        >>>    'CatBoostRegressor', 'LGBMRegressor'])

        Similarly, if you want to exclude certain models from comparison, you can
        use ``exclude`` keyword

        >>> exp.fit(data=busan_beach(), exclude=["SGDRegressor"])

        if you want to perform cross validation for each model, we must give
        the ``cross_validator`` argument which will be passed to ai4water Model

        >>> exp = MLRegressionExperiments(cross_validator={"KFold": {"n_splits": 10}})
        >>> exp.fit(data=busan_beach(), cross_validate=True)

        Setting ``cross_validate`` to True will populate `cv_scores_` dictionary
        which can be accessed as ``exp.cv_scores_``

        if you want to optimize the hyperparameters of each model,

        >>> exp.fit(data=busan_beach(), run_type="optimize", num_iterations=20)

        """

        train_x, train_y, val_x, val_y, _, _ = self.verify_data(
            x, y, data, validation_data)

        AttribtueSetter(self, train_y)

        del x, y, data, validation_data
        gc.collect()

        assert run_type in ['optimize', 'dry_run'], f"run_type mus"
        self._run_type = run_type

        assert post_optimize in ['eval_best', 'train_best'], f"""
        post_optimize must be either 'eval_best' or 'train_best' but it is {post_optimize}"""

        if exclude == '':
            exclude = []

        if hpo_kws is None:
            hpo_kws = {}

        models_to_consider = self._check_include_arg(include)

        if exclude is None:
            exclude = []
        elif isinstance(exclude, str):
            exclude = [exclude]

        consider_exclude(exclude, self.models, models_to_consider)

        self._reset()

        setattr(self, 'considered_models_', models_to_consider)

        for model_type in models_to_consider:

            model_name = model_type.split('model_')[1]

            self.model_iter_metric = {}
            self.iter_ = 0

            # there may be attributes int the model, which needs to be loaded so run the method first.
            # such as param_space etc.
            if hasattr(self, model_type):
                getattr(self, model_type)()

            if run_type == 'dry_run':
                self._dry_run_a_model(
                    model_type,
                    model_name,
                    cross_validate,
                    train_x, train_y, val_x, val_y)

            else:
                self._optimize_a_model(
                    model_type,
                    model_name,
                    opt_method,
                    num_iterations,
                    cross_validate,
                    post_optimize,
                    train_x, train_y,
                    val_x, val_y,
                    **hpo_kws
                )

        self.update_and_save_config(considered_models_ = self.considered_models_,
                                    is_multiclass_ = self.is_multiclass_,
                                    is_binary_ = self.is_binary_,
                                    is_multilabel_ = self.is_multilabel_)

        save_json_file(os.path.join(self.exp_path, 'features.json'), self.features)
        save_json_file(os.path.join(self.exp_path, 'metrics.json'), self.metrics)

        return

    def eval_best(
            self,
            x,
            y,
            model_type:str,
            opt_dir:str,
    ):
        """Evaluate the best models."""

        folders = [path for path in os.listdir(opt_dir) if os.path.isdir(os.path.join(opt_dir, path)) and path.startswith('1_')]

        if len(folders) < 1:
            return self.train_best(x, y, model_type)

        assert len(folders) == 1, f"{folders}"

        for mod_path in folders:
            config_path = os.path.join(opt_dir, mod_path, "config.json")

            model = self.build_from_config(config_path)

            self.update_model_weight(model, os.path.join(opt_dir, mod_path))

            results = self._predict(model, x=x, y=y)

        return results

    def train_best(
            self,
            x,
            y,
            model_type,
    ):
        """Finds the best model, builts it, fits it and makes predictions from it."""

        best_paras = self.optimizer.best_paras()
        if best_paras.get('lookback', 1) > 1:
            _model = 'layers'
        else:
            _model = model_type

        title = f"{self.exp_name}{SEP}{model_type}{SEP}best"
        model = self._build_fit(x, y,
                                view=False,
                                title=title,
                                cross_validate=False,
                                model={_model: self.optimizer.best_paras()},
                                )

        results = self._predict(model, x=x, y=y)

        return results

    def _populate_results(
            self,
            model_type: str,
            train_results: Tuple[np.ndarray, np.ndarray],
            val_results: Tuple[np.ndarray, np.ndarray] = None,
            test_results: Tuple[np.ndarray, np.ndarray] = None,
    ):
        """populates self.metrics dictionary"""
        if not model_type.startswith('model_'):  # internally we always use model_ at the start.
            model_type = f'model_{model_type}'

        metrics = dict()
        features = dict()

        # save performance metrics of train and test

        if train_results is not None:
            metrics['train'] = self._get_metrics(*train_results)
            features['train'] = {
                'true': {'std': np.std(train_results[0])},
                'simulation': {'std': np.std(train_results[1])}
            }

        if val_results is not None:
            metrics['val'] = self._get_metrics(*val_results)
            features['val'] = {
                'true': {'std': np.std(val_results[0])},
                'simulation': {'std': np.std(val_results[1])}
            }

        if test_results is not None:
            self.metrics[model_type]['test'] = self._get_metrics(*test_results)
            self.features[model_type]['test'] = {
                'true': {'std': np.std(test_results[0])},
                'simulation': {'std': np.std(test_results[1])}
            }

        if metrics:
            self.metrics[model_type] = metrics
            self.features[model_type] = features
        return

    def _get_metrics(self, true:np.ndarray, predicted:np.ndarray)->dict:
        """get the performance metrics being monitored given true and predicted data"""
        metrics_inst = Metrics[self.mode](true, predicted,
                                     replace_nan=True,
                                     replace_inf=True,
                                     multiclass=self.is_multiclass_)
        metrics = {}
        for metric in self.monitor:
            if isinstance(metric, str):
                metrics[metric] = getattr(metrics_inst, metric)()
            elif callable(metric):
                # metric is a callable
                metrics[metric.__name__] = metric(true, predicted)
            else:
                raise ValueError(f"invalid metric f{metric}")

        return metrics

    def taylor_plot(
            self,
            x=None,
            y=None,
            data=None,
            include: Union[None, list] = None,
            exclude: Union[None, list] = None,
            figsize: tuple = (9, 7),
            **kwargs
    ) -> plt.Figure:
        """
        Compares the models using taylor_plot_.

        Parameters
        ----------
            x :
                input data, if not given, then ``data`` must be given.
            y :
                target data
            data :
                raw unprocessed data from which x,y pairs can be drawn. This data
                will be passed to DataSet class and DataSet.test_data() method
                will be used to draw x,y pairs.
            include : str, list, optional
                if not None, must be a list of models which will be included.
                None will result in plotting all the models.
            exclude : str, list, optional
                if not None, must be a list of models which will excluded.
                None will result in no exclusion
            figsize : tuple, optional
            **kwargs :
                all the keyword arguments for taylor_plot_ function.

        Returns
        -------
        plt.Figure

        Example
        -------
        >>> from ai4water.experiments import MLRegressionExperiments
        >>> from ai4water.datasets import busan_beach
        >>> data = busan_beach()
        >>> inputs = list(data.columns)[0:-1]
        >>> outputs = list(data.columns)[-1]
        >>> experiment = MLRegressionExperiments(input_features=inputs, output_features=outputs)
        >>> experiment.fit(data=data)
        >>> experiment.taylor_plot(data=data)

        .. _taylor_plot:
            https://easy-mpl.readthedocs.io/en/latest/plots.html#easy_mpl.taylor_plot
        """

        _, _, _, _, x, y = self.verify_data(data=data, test_data=(x, y))

        self._build_predict_from_configs(x, y)

        metrics = self.metrics.copy()

        include = self._check_include_arg(include)

        if exclude is not None:
            consider_exclude(exclude, self.models, metrics)

        if 'name' in kwargs:
            fname = kwargs.pop('name')
        else:
            fname = 'taylor'
        fname = os.path.join(os.getcwd(), f'results{SEP}{self.exp_name}{SEP}{fname}.png')

        train_std = [_model['train']['true']['std'] for _model in self.features.values()]
        train_std = list(set(train_std))[0]

        if 'test' in list(self.features.values())[0]:
            test_stds = [_model['test']['true']['std'] for _model in self.features.values()]
            test_data_type = "test"
        else:
            test_stds = [_model['val']['true']['std'] for _model in self.features.values()]
            test_data_type = "val"

        # if any value in test_stds is nan, set(test_stds)[0] will be nan
        if np.isnan(list(set(test_stds)))[0]:
            test_std = list(set(test_stds))[1]
        else:
            test_std = list(set(test_stds))[0]

        assert not np.isnan(test_std)

        observations = {'train': {'std': train_std},
                        test_data_type: {'std': test_std}}

        simulations = {'train': None, test_data_type: None}

        for scen in ['train', test_data_type]:

            scen_stats = {}
            for model, _metrics in metrics.items():
                model_stats = {'std': self.features[model][scen]['simulation']['std'],
                               'corr_coeff': _metrics[scen]['corr_coeff'],
                               'pbias': _metrics[scen]['pbias']
                               }


                if model in include:
                    key = shred_model_name(model)
                    scen_stats[key] = model_stats

            simulations[scen] = scen_stats

        ax = taylor_plot(
            observations=observations,
            simulations=simulations,
            figsize=figsize,
            show=False,
            **kwargs
        )

        plt.savefig(fname, dpi=600, bbox_inches="tight")
        return ax

    def _consider_include(self, include: Union[str, list], to_filter):

        filtered = {}

        include = self._check_include_arg(include)

        for m in include:
            if m in to_filter:
                filtered[m] = to_filter[m]

        return filtered

    def _check_include_arg(self, include):

        if isinstance(include, str):
            if include == "DTs":
                include = DTs[self.mode]
            elif include == "LMs":
                include = LMs[self.mode]
            else:
                include = [include]

        if include is None:
            include = self.models
        include = ['model_' + _model if not _model.startswith('model_') else _model for _model in include]

        # make sure that include contains same elements which are present in models
        for elem in include:
            assert elem in self.models, f"""
{elem} to `include` are not available.
Available cases are {self.models} and you wanted to include
{include}
"""
        return include

    def plot_improvement(
            self,
            metric_name: str,
            plot_type: str = 'dumbbell',
            lower_limit: Union[int, float] = -1.0,
            upper_limit: Union[int, float] = None,
            save: bool = True,
            name: str = '',
            dpi: int = 200,
            **kwargs
    ) -> pd.DataFrame:
        """
        Shows how much improvement was observed after hyperparameter
        optimization. This plot is only available if ``run_type`` was set to
        `optimize` in :py:meth:`ai4water.experiments.Experiments.fit`.

        Arguments
        ---------
            metric_name :
                the peformance metric for comparison
            plot_type : str, optional
                the kind of plot to draw. Either ``dumbbell`` or ``bar``
            lower_limit : float/int, optional (default=-1.0)
                clip the values below this value. Set this value to None to avoid clipping.
            upper_limit : float/int, optional (default=None)
                clip the values above this value
            save : bool
                whether to save the plot or not
            name : str, optional
            dpi : int, optional
            **kwargs :
                any additional keyword arguments for
                `dumbell plot <https://easy-mpl.readthedocs.io/en/latest/plots.html#easy_mpl.dumbbell_plot>`_
                 or `bar_chart <https://easy-mpl.readthedocs.io/en/latest/plots.html#easy_mpl.bar_chart>`_

        Returns
        -------
        pd.DataFrame

        Examples
        --------
        >>> from ai4water.experiments import MLRegressionExperiments
        >>> from ai4water.datasets import busan_beach
        >>> experiment = MLRegressionExperiments()
        >>> experiment.fit(data=busan_beach(), run_type="optimize", num_iterations=30)
        >>> experiment.plot_improvement('r2')
        ...
        >>>  # or draw dumbbell plot
        ...
        >>> experiment.plot_improvement('r2', plot_type='bar')

        """

        assert self._run_type == "optimize", f"""
        when run_type argument during .fit() is {self._run_type}, we can
        not have improvement plot"""

        data: str = 'test'

        assert data in ['training', 'test', 'validation']

        improvement = pd.DataFrame(columns=['start', 'end'])

        for model, model_iter_metrics in self.iter_metrics.items():
            initial = model_iter_metrics[0][metric_name]
            final = self.metrics[model]['test'][metric_name]

            key = shred_model_name(model)

            improvement.loc[key] = [initial, final]

        baseline = improvement['start']

        if lower_limit:
            baseline = np.where(baseline < lower_limit, lower_limit, baseline)

        if upper_limit:
            baseline = np.where(baseline > upper_limit, upper_limit, baseline)

        improvement['start'] = baseline

        if plot_type == "dumbbell":
            ax = dumbbell_plot(
                improvement['start'],
                improvement['end'],
                improvement.index.tolist(),
                xlabel=ERROR_LABELS.get(metric_name, metric_name),
                show=False,
                **kwargs
            )
            ax.set_xlabel(ERROR_LABELS.get(metric_name, metric_name))
        else:

            colors = {
                'start': np.array([0, 56, 104]) / 256,
                'end': np.array([126, 154, 178]) / 256
            }

            order = ['start', 'end']
            if metric_name in ['r2', 'nse', 'kge', 'corr_coeff', 'r2_mod', 'r2_score']:
                order = ['end', 'start']

            fig, ax = plt.subplots()

            for ordr in order:
                bar_chart(improvement[ordr], improvement.index.tolist(),
                          ax=ax, color=colors[ordr], show=False,
                          ax_kws={'xlabel':ERROR_LABELS.get(metric_name, metric_name),
                          'label':ordr}, **kwargs)

            ax.legend()
            plt.title('Improvement after Optimization')

        if save:
            fname = os.path.join(
                os.getcwd(),
                f'results{SEP}{self.exp_name}{SEP}{name}_improvement_{metric_name}.png')
            plt.savefig(fname, dpi=dpi, bbox_inches=kwargs.get('bbox_inches', 'tight'))

        plt.show()

        return improvement

    def compare_errors(
            self,
            matric_name: str,
            x=None,
            y=None,
            data = None,
            cutoff_val: float = None,
            cutoff_type: str = None,
            save: bool = True,
            sort_by: str = 'test',
            ignore_nans: bool = True,
            name: str = 'ErrorComparison',
            show: bool = True,
            **kwargs
    ) -> pd.DataFrame:
        """
        Plots a specific performance matric for all the models which were
        run during [fit][ai4water.experiments.Experiments.fit] call.

        Parameters
        ----------
            matric_name : str
                 performance matric whose value to plot for all the models
            x :
                input data, if not given, then ``data`` must be given.
            y :
                target data
            data :
                raw unprocessed data from which x,y pairs can be drawn. This data
                will be passed to DataSet class and DataSet.test_data() method
                will be used to draw x,y pairs.
            cutoff_val : float
                 if provided, only those models will be plotted for whome the
                 matric is greater/smaller than this value. This works in conjuction
                 with `cutoff_type`.
            cutoff_type : str
                 one of ``greater``, ``greater_equal``, ``less`` or ``less_equal``.
                 Criteria to determine cutoff_val. For example if we want to
                 show only those models whose $R^2$ is > 0.5, it will be 'max'.
            save : bool, optional (default = True)
                whether to save the plot or not
            sort_by:
                either ``test`` or ``train``. How to sort the results for plotting.
                If 'test', then test performance matrics will be sorted otherwise
                train performance matrics will be sorted.
            ignore_nans:
                default True, if True, then performance matrics with nans are ignored
                otherwise nans/empty bars will be shown to depict which models have
                resulted in nans for the given performance matric.
            name:
                name of the saved file.
            show : whether to show the plot at the end or not?

            kwargs :

                - fig_height :
                - fig_width :
                - title_fs :
                - xlabel_fs :
                - color :

        returns
        -------
        pd.DataFrame
            pandas dataframe whose index is models and has two columns with name
            'train' and 'test' These columns contain performance metrics for the
            models..

        Example
        -------
            >>> from ai4water.experiments import MLRegressionExperiments
            >>> from ai4water.datasets import busan_beach
            >>> data = busan_beach()
            >>> inputs = list(data.columns)[0:-1]
            >>> outputs = list(data.columns)[-1]
            >>> experiment = MLRegressionExperiments(input_features=inputs, output_features=outputs)
            >>> experiment.fit(data=data)
            >>> experiment.compare_errors('mse', data=data)
            >>> experiment.compare_errors('r2', data=data, cutoff_val=0.2, cutoff_type='greater')
        """

        _, _, _, _, x, y = self.verify_data(data=data, test_data=(x, y))

        self._build_predict_from_configs(x, y)

        models = self.sort_models_by_metric(matric_name, cutoff_val, cutoff_type,
                                            ignore_nans, sort_by)

        plt.close('all')
        fig, axis = plt.subplots(1, 2, sharey='all')
        fig.set_figheight(kwargs.get('fig_height', 8))
        fig.set_figwidth(kwargs.get('fig_width', 8))

        labels = [model.split('model_')[1] for model in models.index.tolist()]
        models.index = labels

        bar_chart(ax=axis[0],
                  labels=models.index.tolist(),
                  values=models['train'],
                  color=kwargs.get('color', None),
                  ax_kws={'title':"Train",
                  'xlabel':ERROR_LABELS.get(matric_name, matric_name),
                  'xlabel_kws':{'fontsize': kwargs.get('xlabel_fs', 16)},
                  'title_kws':{'fontsize': kwargs.get('title_fs', 20)}},
                  show=False,
                  )

        bar_chart(ax=axis[1],
                  labels=models.index.tolist(),
                  values=models.iloc[:, 1],
                  color=kwargs.get('color', None),
                  ax_kws={'title': models.columns.tolist()[1],
                          'xlabel':ERROR_LABELS.get(matric_name, matric_name),
                          'xlabel_kws':{'fontsize': kwargs.get('xlabel_fs', 16)},
                          'title_kws':{'fontsize': kwargs.get('title_fs', 20)},
                          'show_yaxis':False},
                  show=False
                  )

        appendix = f"{cutoff_val or ''}{cutoff_type or ''}{len(models)}"
        if save:
            fname = os.path.join(
                os.getcwd(),
                f'results{SEP}{self.exp_name}{SEP}{name}_{matric_name}_{appendix}.png')
            plt.savefig(fname, dpi=100, bbox_inches=kwargs.get('bbox_inches', 'tight'))

        if show:
            plt.show()
        return models

    def loss_comparison(
            self,
            loss_name: str = 'loss',
            include: list = None,
            save: bool = True,
            show: bool = True,
            figsize: int = None,
            start: int = 0,
            end: int = None,
            **kwargs
    ) -> plt.Axes:
        """
        Plots the loss curves of the evaluated models. This method is only available
        if the models which are being compared are deep leanring mdoels.

        Parameters
        ----------
            loss_name : str, optional
                the name of loss value, must be recorded during training
            include:
                name of models to include
            save:
                whether to save the plot or not
            show:
                whether to show the plot or now
            figsize : tuple
                size of the figure
            start : int
            end : int
            **kwargs :
                any other keyword arguments to be passed to the
                `plot <https://easy-mpl.readthedocs.io/en/latest/plots.html#easy_mpl.plot>`_
        Returns
        -------
            matplotlib axes

        Example
        -------
        >>> from ai4water.experiments import DLRegressionExperiments
        >>> from ai4water.datasets import busan_beach
        >>> data = busan_beach()
        >>> exp = DLRegressionExperiments(
        >>> input_features = data.columns.tolist()[0:-1],
        >>> output_features = data.columns.tolist()[-1:],
        >>> epochs=300,
        >>> train_fraction=1.0,
        >>> y_transformation="log",
        >>> x_transformation="minmax",
        >>> )

        >>> exp.fit(data=data)
        >>> exp.loss_comparison()

        you may wish to plot on log scale

        >>> exp.loss_comparison(logy=True)
        """

        include = self._check_include_arg(include)

        if self.model_.category == "ML":
            raise NotImplementedError(f"Non neural network models can not have loss comparison")

        loss_curves = {}
        for _model, _path in self.config['eval_models'].items():
            if _model in include:
                df = pd.read_csv(os.path.join(_path, 'losses.csv'), usecols=[loss_name])
                loss_curves[_model] = df.values
                end = end or len(df)

        _kwargs = {'linestyle': '-',
                   'xlabel': "Epochs",
                   'ylabel': 'Loss'}

        if len(loss_curves) > 5:
            _kwargs['legend_kws'] = {'bbox_to_anchor': (1.1, 0.99)}

        _, axis = plt.subplots(figsize=figsize)

        for _model, _loss in loss_curves.items():
            label = shred_model_name(_model)
            plot(_loss[start:end], ax=axis, label=label, show=False, **_kwargs, **kwargs)

        if save:
            fname = os.path.join(self.exp_path, f'loss_comparison_{loss_name}.png')
            plt.savefig(fname, dpi=100, bbox_inches='tight')

        if show:
            plt.show()

        return axis

    def compare_convergence(
            self,
            show: bool = True,
            save: bool = False,
            name: str = 'convergence_comparison',
            **kwargs
    ) -> Union[plt.Axes, None]:
        """
        Plots and compares the convergence plots of hyperparameter optimization runs.
        Only valid if `run_type=optimize` during :py:meth:`ai4water.experiments.Experiments.fit`
        call.

        Parameters
        ----------
            show :
                whether to show the plot or now
            save :
                whether to save the plot or not
            name :
                name of file to save the plot
            kwargs :
                keyword arguments to plot_ function
        Returns
        -------
            if the optimized models are >1 then it returns the maplotlib axes
            on which the figure is drawn otherwise it returns None.

        Examples
        --------
        >>> from ai4water.experiments import MLRegressionExperiments
        >>> from ai4water.datasets import busan_beach
        >>> experiment = MLRegressionExperiments()
        >>> experiment.fit(data=busan_beach(), run_type="optimize", num_iterations=30)
        >>> experiment.compare_convergence()

        .. _plot:
            https://easy-mpl.readthedocs.io/en/latest/plots.html#easy_mpl.plot
        """
        if len(self.config['optimized_models']) < 1:
            print('No model was optimized')
            return
        plt.close('all')
        fig, axis = plt.subplots()

        for _model, opt_path in self.config['optimized_models'].items():
            with open(os.path.join(opt_path, 'iterations.json'), 'r') as fp:
                iterations = json.load(fp)

            convergence = sort_array(list(iterations.keys()))

            label = shred_model_name(_model)
            plot(
                convergence,
                ax=axis,
                label=label,
                linestyle='--',
                xlabel='Number of iterations $n$',
                ylabel=r"$\min f(x)$ after $n$ calls",
                show=False,
                **kwargs
            )
        if save:
            fname = os.path.join(self.exp_path, f'{name}.png')
            plt.savefig(fname, dpi=100, bbox_inches='tight')

        if show:
            plt.show()
        return axis

    def compare_edf_plots(
            self,
            x=None,
            y=None,
            data=None,
            exclude:Union[list, str] = None,
            figsize=None,
            save: Optional[bool] = True,
            show: Optional[bool] = True,
            fname: Optional[str] = "edf"
    ):
        """compare EDF plots of all the models which have been fitted.
        This plot is only available for regression problems.

        parameters
        ----------
        x :
            input data
        y :
            target data
        data :
            raw unprocessed data from which x,y pairs of the test data are drawn
        exclude : list
            name of models to exclude from plotting
        figsize :
            figure size
        save : bool, optional (default=True)
            whether to save the plot or not?
        show : bool, optional (default=True)
            whether to show the plot or not?
        fname : str, optional

        Returns
        -------
        plt.Figure
            matplotlib

        Example
        -------
        >>> from ai4water.experiments import MLRegressionExperiments
        >>> from ai4water.datasets import busan_beach
        >>> dataset = busan_beach()
        >>> inputs = list(dataset.columns)[0:-1]
        >>> outputs = list(dataset.columns)[-1]
        >>> experiment = MLRegressionExperiments(input_features=inputs, output_features=outputs)
        >>> experiment.fit(data=dataset, include="LMs")
        >>> experiment.compare_edf_plots(data=dataset, exclude="SGDRegressor")
        """
        assert self.mode == "regression", f"This plot is not available for {self.mode} mode"

        _, _, _, _, x, y = self.verify_data(data=data, test_data=(x, y))

        model_folders = self._get_model_folders()

        if exclude is None:
            exclude = []
        elif isinstance(exclude, str):
            exclude = [exclude]

        fig, axes = plt.subplots(figsize=figsize)

        # load all models from config
        for model_name in model_folders:

            if model_name not in exclude:

                m_path = self._get_best_model_path(model_name)

                c_path = os.path.join(m_path, 'config.json')
                model = self.build_from_config(c_path)
                # calculate pr curve for each model
                self.update_model_weight(model, m_path)

                true, prediction = model.predict(x, y, return_true=True,
                                                 process_results=False)

                assert len(true) == true.size
                assert len(prediction) == prediction.size
                error = np.abs(true.reshape(-1,) - prediction.reshape(-1,))

                if model_name.endswith("Regressor"):
                    label = model_name.split("Regressor")[0]
                elif model_name.endswith("Classifier"):
                    label = model_name.split("Classifier")[0]
                else:
                    label = model_name

                edf_plot(error, xlabel="Absolute Error", ax=axes, label=label,
                         show=False)

        if len(model_folders)>7:
            axes.legend(loc=(1.05, 0.0))

        if save:
            fname = os.path.join(self.exp_path, f'{fname}.png')
            plt.savefig(fname, dpi=600, bbox_inches='tight')

        if show:
            plt.show()

        return

    def compare_regression_plots(
            self,
            x=None,
            y=None,
            data=None,
            figsize=None,
            save: Optional[bool] = True,
            show: Optional[bool] = True,
            fname: Optional[str] = "regression"
    ):
        """compare regression plots of all the models which have been fitted.
        This plot is only available for regression problems.

        parameters
        ----------
        x :
            input data
        y :
            target data
        data :
            raw unprocessed data from which x,y pairs of the test data are drawn
        figsize :
            figure size
        save : bool, optional (default=True)
            whether to save the plot or not?
        show : bool, optional (default=True)
            whether to show the plot or not?
        fname : str, optional
            name of the file to save the plot
        Returns
        -------
        plt.Figure
            matplotlib

        Example
        -------
        >>> from ai4water.experiments import MLRegressionExperiments
        >>> from ai4water.datasets import busan_beach
        >>> dataset = busan_beach()
        >>> inputs = list(dataset.columns)[0:-1]
        >>> outputs = list(dataset.columns)[-1]
        >>> experiment = MLRegressionExperiments(input_features=inputs, output_features=outputs)
        >>> experiment.fit(data=dataset)
        >>> experiment.compare_regression_plots(data=dataset)
        """
        assert self.mode == "regression", f"This plot is not available for {self.mode} mode"

        _, _, _, _, x, y = self.verify_data(data=data, test_data=(x, y))

        model_folders = self._get_model_folders()

        fig, axes = create_subplots(naxes=len(model_folders), figsize=figsize)

        if not isinstance(axes, np.ndarray):
            axes = np.array(axes)

        # load all models from config
        for model_name, ax in zip(model_folders, axes.flat):

            m_path = self._get_best_model_path(model_name)

            c_path = os.path.join(m_path, 'config.json')
            model = self.build_from_config(c_path)
            # calculate pr curve for each model
            self.update_model_weight(model, m_path)

            true, prediction = model.predict(x, y, return_true=True, process_results=False)

            if np.isnan(prediction).sum() == prediction.size:
                if self.verbosity>=0:
                    print(f"Model {model_name} only predicted nans")
                continue

            reg_plot(true, prediction, marker_size=5, ax=ax, show=False)

            if model_name.endswith("Regressor"):
               label = model_name.split("Regressor")[0]
            elif model_name.endswith("Classifier"):
                label = model_name.split("Classifier")[0]
            else:
                label = model_name
            ax.legend(labels=[label], fontsize=9,
                      numpoints=2,
                      fancybox=False, framealpha=0.0)

        fig.supxlabel("Observed")
        fig.supylabel("Predicted")

        if save:
            fname = os.path.join(self.exp_path, f'{fname}.png')
            plt.savefig(fname, dpi=600, bbox_inches='tight')

        if show:
            plt.show()

        return

    def compare_residual_plots(
            self,
            x=None,
            y=None,
            data = None,
            figsize = None,
            save: Optional[bool] = True,
            show: Optional[bool] = True,
            fname: Optional[str] = "residual"
    )->plt.Figure:
        """compare residual plots of all the models which have been fitted.
        This plot is only available for regression problems.

        parameters
        ----------
        x :
            input data
        y :
            target data
        data :
            raw unprocessed data frmm which test x,y pairs are drawn using
            :py:meth:`ai4water.preprocessing.DataSet`. class. Only valid if x and y are not given.
        figsize :
        save : bool, optional (default=True)
            whether to save the plot or not?
        show : bool, optional (default=True)
            whether to show the plot or not?
        fname : str, optional

        Returns
        -------
        plt.Figure
            matplotlib

        Example
        -------
        >>> from ai4water.experiments import MLRegressionExperiments
        >>> from ai4water.datasets import busan_beach
        >>> dataset = busan_beach()
        >>> inputs = list(dataset.columns)[0:-1]
        >>> outputs = list(dataset.columns)[-1]
        >>> experiment = MLRegressionExperiments(input_features=inputs, output_features=outputs)
        >>> experiment.fit(data=dataset)
        >>> experiment.compare_residual_plots(data=dataset)
        """
        assert self.mode == "regression", f"This plot is not available for {self.mode} mode"

        _, _, _, _, x, y = self.verify_data(data=data, test_data=(x, y))

        model_folders = self._get_model_folders()

        fig, axes = create_subplots(naxes=len(model_folders), figsize=figsize)

        if not isinstance(axes, np.ndarray):
            axes = np.array(axes)

        # load all models from config
        for model_name, ax in zip(model_folders, axes.flat):

            m_path = self._get_best_model_path(model_name)

            c_path = os.path.join(m_path, 'config.json')
            model = self.build_from_config(c_path)
            # calculate pr curve for each model
            self.update_model_weight(model, m_path)

            true, prediction = model.predict(x, y, return_true=True, process_results=False)

            plot(
                prediction,
                true - prediction,
                'o',
                show=False,
                ax=ax,
                color="darksalmon",
                markerfacecolor=np.array([225, 121, 144]) / 256.0,
                markeredgecolor="black",
                markeredgewidth=0.15,
                markersize=1.5,
            )

            # draw horizontal line on y=0
            ax.axhline(0.0)

            if model_name.endswith("Regressor"):
               label = model_name.split("Regressor")[0]
            elif model_name.endswith("Classifier"):
                label = model_name.split("Classifier")[0]
            else:
                label = model_name
            ax.legend(labels=[label], fontsize=9,
                      numpoints=2,
                      fancybox=False, framealpha=0.0)

        fig.supxlabel("Prediction")
        fig.supylabel("Residual")
        if save:
            fname = os.path.join(self.exp_path, f'{fname}.png')
            plt.savefig(fname, dpi=600, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    @classmethod
    def from_config(
            cls,
            config_path: str,
            **kwargs
    ) -> "Experiments":
        """
        Loads the experiment from the config file.

        Arguments:
            config_path : complete path of experiment
            kwargs : keyword arguments to experiment
        Returns:
            an instance of `Experiments` class
        """
        if not config_path.endswith('.json'):
            raise ValueError(f"""
{config_path} is not a json file
""")
        with open(config_path, 'r') as fp:
            config = json.load(fp)

        cv_scores = {}
        scoring = "mse"

        for model_name, model_path in config['eval_models'].items():

            with open(os.path.join(model_path, 'config.json'), 'r') as fp:
                model_config = json.load(fp)

            # if cross validation was performed, then read those results.
            cross_validator = model_config['config']['cross_validator']
            if cross_validator is not None:
                cv_name = str(list(cross_validator.keys())[0])
                scoring = model_config['config']['val_metric']
                cv_fname = os.path.join(model_path, f'{cv_name}_{scoring}' + ".json")
                if os.path.exists(cv_fname):
                    with open(cv_fname, 'r') as fp:
                        cv_scores[model_name] = json.load(fp)

        exp = cls(exp_name=config['exp_name'], cases=config['cases'], **kwargs)

        exp.config = config
        exp._from_config = True

        # following four attributes are only available if .fit was run
        exp.considered_models_ = config.get('considered_models_', [])
        exp.is_binary_ = config.get('is_binary_', None)
        exp.is_multiclass_ = config.get('is_multiclass_', None)
        exp.is_multilabel_ = config.get('is_multilabel_', None)

        exp.metrics = load_json_file(
            os.path.join(os.path.dirname(config_path), "metrics.json"))
        exp.features = load_json_file(
            os.path.join(os.path.dirname(config_path), "features.json"))
        exp.cv_scores_ = cv_scores
        exp._cv_scoring = scoring

        return exp

    def plot_cv_scores(
            self,
            show: bool = False,
            name: str = "cv_scores",
            exclude: Union[str, list] = None,
            include: Union[str, list] = None,
            **kwargs
    ) -> Union[plt.Axes, None]:
        """
        Plots the box whisker plots of the cross validation scores.

        This plot is only available if cross_validation was set to True during
        :py:meth:`ai4water.experiments.Experiments.fit`.

        Arguments
        ---------
            show : whether to show the plot or not
            name : name of the plot
            include : models to include
            exclude : models to exclude
            kwargs : any of the following keyword arguments

                - notch
                - vert
                - figsize
                - bbox_inches

        Returns
        -------
            matplotlib axes if the figure is drawn otherwise None

        Example
        -------
        >>> from ai4water.experiments import MLRegressionExperiments
        >>> from ai4water.datasets import busan_beach
        >>> exp = MLRegressionExperiments(cross_validator={"KFold": {"n_splits": 10}})
        >>> exp.fit(data=busan_beach(), cross_validate=True)
        >>> exp.plot_cv_scores()

        """
        if len(self.cv_scores_) == 0:
            return

        scoring = self._cv_scoring
        cv_scores = self.cv_scores_

        consider_exclude(exclude, self.models, cv_scores)
        cv_scores = self._consider_include(include, cv_scores)

        model_names = [m.split('model_')[1] for m in list(cv_scores.keys())]
        if len(model_names) < 5:
            rotation = 0
        else:
            rotation = 90

        plt.close()

        _, axis = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))

        d = axis.boxplot(np.array(list(cv_scores.values())).squeeze().T,
                         notch=kwargs.get('notch', None),
                         vert=kwargs.get('vert', None),
                         labels=model_names
                         )

        whiskers = d['whiskers']
        caps = d['caps']
        boxes = d['boxes']
        medians = d['medians']
        fliers = d['fliers']

        axis.set_xticklabels(model_names, rotation=rotation)
        axis.set_xlabel("Models", fontsize=16)
        axis.set_ylabel(ERROR_LABELS.get(scoring, scoring), fontsize=16)

        fname = os.path.join(os.getcwd(),
                             f'results{SEP}{self.exp_name}{SEP}{name}_{len(model_names)}.png')
        plt.savefig(fname, dpi=300, bbox_inches=kwargs.get('bbox_inches', 'tight'))

        if show:
            plt.show()

        return axis

    def sort_models_by_metric(
            self,
            metric_name,
            cutoff_val=None,
            cutoff_type=None,
            ignore_nans: bool = True,
            sort_by="test"
    ) -> pd.DataFrame:
        """returns the models sorted according to their performance"""

        idx = list(self.metrics.keys())
        metrics = dict()

        metrics['train'] = np.array([v['train'][metric_name] for v in self.metrics.values()])

        if 'test' in list(self.metrics.values())[0]:
            metrics['test'] = np.array([v['test'][metric_name] for v in self.metrics.values()])
        else:
            metrics['val'] = np.array([v['val'][metric_name] for v in self.metrics.values()])

        if 'test' not in metrics and sort_by == "test":
            sort_by = "val"

        df = pd.DataFrame(metrics,  index=idx)
        if ignore_nans:
            df = df.dropna()

        df = df.sort_values(by=[sort_by], ascending=False)

        if cutoff_type is not None:
            assert cutoff_val is not None
            if cutoff_type == "greater":
                df = df.loc[df[sort_by] > cutoff_val]
            else:
                df = df.loc[df[sort_by] < cutoff_val]

        return df

    def fit_with_tpot(
            self,
            data,
            models: Union[int, List[str], dict, str] = None,
            selection_criteria: str = 'mse',
            scoring: str = None,
            **tpot_args
    ):
        """
        Fits the tpot_'s fit  method which
        finds out the best pipline for the given data.

        Arguments
        ---------
            data :
            models :
                It can be of three types.

                - If list, it will be the names of machine learning models/
                    algorithms to consider.
                - If integer, it will be the number of top
                    algorithms to consider for tpot. In such a case, you must have
                    first run `.fit` method before running this method. If you run
                    the tpot using all available models, it will take hours to days
                    for medium sized data (consisting of few thousand examples). However,
                    if you run first .fit and see for example what are the top 5 models,
                    then you can set this argument to 5. In such a case, tpot will search
                    pipeline using only the top 5 algorithms/models that have been found
                    using .fit method.
                - if dictionary, then the keys should be the names of algorithms/models
                    and values shoudl be the parameters for each model/algorithm to be
                    optimized.
                - You can also set it to ``all`` consider all models available in
                    ai4water's Experiment module.
                - default is None, which means, the `tpot_config` argument will be None

            selection_criteria :
                The name of performance metric. If ``models`` is integer, then
                according to this performance metric the models will be choosen.
                By default the models will be selected based upon their mse values
                on test data.
            scoring : the performance metric to use for finding the pipeline.
            tpot_args :
                any keyword argument for tpot's Regressor_ or Classifier_ class.
                This can include arguments like ``generations``, ``population_size`` etc.

        Returns
        -------
            the tpot object

        Example
        -------
            >>> from ai4water.experiments import MLRegressionExperiments
            >>> from ai4water.datasets import busan_beach
            >>> exp = MLRegressionExperiments(exp_name=f"tpot_reg_{dateandtime_now()}")
            >>> exp.fit(data=busan_beach())
            >>> tpot_regr = exp.fit_with_tpot(busan_beach(), 2, generations=1, population_size=2)

        .. _tpot:
            http://epistasislab.github.io/tpot/

        .. _Regressor:
            http://epistasislab.github.io/tpot/api/#regression

        .. _Classifier:
            http://epistasislab.github.io/tpot/api/#classification
        """
        tpot_caller = self.tpot_estimator
        assert tpot_caller is not None, f"tpot must be installed"

        param_space = {}
        tpot_config = None
        for m in self.models:
            getattr(self, m)()
            ps = getattr(self, 'param_space')
            path = getattr(self, 'path')
            param_space[m] = {path: {p.name: p.grid for p in ps}}

        if isinstance(models, int):

            assert len(self.metrics) > 1, f"""
            you must first run .fit() method in order to choose top {models} models"""

            # sort the models w.r.t their performance
            sorted_models = self.sort_models_by_metric(selection_criteria)

            # get names of models
            models = sorted_models.index.tolist()[0:models]

            tpot_config = {}
            for m in models:
                c: dict = param_space[f"{m}"]
                tpot_config.update(c)

        elif isinstance(models, list):

            tpot_config = {}
            for m in models:
                c: dict = param_space[f"model_{m}"]
                tpot_config.update(c)

        elif isinstance(models, dict):

            tpot_config = {}
            for mod_name, mod_paras in models.items():

                if "." in mod_name:
                    mod_path = mod_name
                else:
                    c: dict = param_space[f"model_{mod_name}"]
                    mod_path = list(c.keys())[0]
                d = {mod_path: mod_paras}
                tpot_config.update(d)

        elif isinstance(models, str) and models == "all":
            tpot_config = {}
            for mod_name, mod_config in param_space.items():
                mod_path = list(mod_config.keys())[0]
                mod_paras = list(mod_config.values())[0]
                tpot_config.update({mod_path: mod_paras})

        fname = os.path.join(self.exp_path, "tpot_config.json")
        with open(fname, 'w') as fp:
            json.dump(jsonize(tpot_config), fp, indent=True)

        tpot = tpot_caller(
            verbosity=self.verbosity + 1,
            scoring=scoring,
            config_dict=tpot_config,
            **tpot_args
        )

        not_allowed_args = ["cross_validator", "wandb_config", "val_metric",
                            "loss", "optimizer", "lr", "epochs", "quantiles", "patience"]
        model_kws = self.model_kws
        for arg in not_allowed_args:
            if arg in model_kws:
                model_kws.pop(arg)

        dh = DataSet(data, **model_kws)
        train_x, train_y = dh.training_data()
        tpot.fit(train_x, train_y.reshape(-1, 1))

        if "regressor" in self.tpot_estimator:
            mode = "regression"
        else:
            mode = "classification"
        visualizer = ProcessPredictions(path=self.exp_path,
                                        show=bool(self.verbosity),
                                        mode=mode)

        for idx, data_name in enumerate(['training', 'test']):
            x_data, y_data = getattr(dh, f"{data_name}_data")(key=str(idx))

            pred = tpot.fitted_pipeline_.predict(x_data)
            r2 = RegressionMetrics(y_data, pred).r2()

            # todo, perform inverse transform and deindexification
            visualizer(
                pd.DataFrame(y_data.reshape(-1, )),
                pd.DataFrame(pred.reshape(-1, )),
            )
        # save the python code of fitted pipeline
        tpot.export(os.path.join(self.exp_path, "tpot_fitted_pipeline.py"))

        # save each iteration
        fname = os.path.join(self.exp_path, "evaluated_individuals.json")
        with open(fname, 'w') as fp:
            json.dump(tpot.evaluated_individuals_, fp, indent=True)
        return tpot

    def _build_fit(
            self,
            train_x=None,
            train_y=None,
            validation_data=None,
            view=False,
            title=None,
            cross_validate: bool=False,
            refit: bool=False,
            **kwargs
    )->Model:
        model: Model = self._build(title=title, **kwargs)

        self._fit(
            model,
            train_x=train_x,
            train_y=train_y,
            validation_data=validation_data,
            cross_validate=cross_validate,
            refit = refit,
        )

        if view:
            self._model.view()

        return self.model_

    def _build_fit_eval(
            self,
            train_x=None,
            train_y=None,
            validation_data:tuple=None,
            view=False,
            title=None,
            cross_validate: bool=False,
            refit: bool=False,
            **kwargs
    )->float:

        """
        Builds and run one 'model' of the experiment.

        Since an experiment consists of many models, this method
        is also run many times.
        refit : bool
            This means fit on training + validation data. This is true
            when we have optimized the hyperparameters and now we would
            like to fit on training + validation data as well.
        """
        model = self._build_fit(train_x, train_y,
                                validation_data, view, title,
                                cross_validate, refit, **kwargs)

        # return the validation score
        return self._evaluate(model, *validation_data)

    def _build(self, title=None, **suggested_paras):
        """Builds the ai4water Model class and makes it a class attribute."""

        suggested_paras = self._pre_build_hook(**suggested_paras)

        suggested_paras = jsonize(suggested_paras)

        verbosity = max(self.verbosity - 1, 0)
        if 'verbosity' in self.model_kws:
            verbosity = self.model_kws.pop('verbosity')

        model = Model(
            prefix=title,
            verbosity=verbosity,
            **self.model_kws,
            **suggested_paras
        )

        setattr(self, 'model_', model)
        return model

    def _fit(
            self,
            model:Model,
            train_x,
            train_y,
            validation_data,
            cross_validate: bool = False,
            refit: bool = False
    ):
        """Trains the model"""

        if cross_validate:

            return model.cross_val_score(*_combine_training_validation_data(
                train_x,
                train_y,
                validation_data))

        if refit:
            # we need to combine training (x,y) + validation data.
            return model.fit_on_all_training_data(*_combine_training_validation_data(
                train_x,
                train_y,
                validation_data=validation_data))

        model.fit(x=train_x, y=train_y)
        # model_ is used in the class for prediction so it must be the updated/trained model
        self.model_ = model
        return

    def _evaluate(
            self,
            model:Model,
            x,
            y,
    ) -> float:
        """Evaluates the model"""

        #if validation_data is None:
        t, p = model.predict(
            x=x, y=y,
            return_true=True,
            process_results=False)

        test_metrics = self._get_metrics(t, p)
        metrics = Metrics[self.mode](t, p,
                                     remove_zero=True,
                                     remove_neg=True,
                                     replace_nan=True,
                                     replace_inf=True,
                                     multiclass=self.is_multiclass_)

        self.model_iter_metric[self.iter_] = test_metrics
        self.iter_ += 1

        val_score_ = getattr(metrics, model.val_metric)()
        val_score = val_score_
        if model.val_metric in [
            'r2', 'nse', 'kge', 'r2_mod', 'r2_adj', 'r2_score'
        ] or self.mode == "classification":
            val_score = 1.0 - val_score_

        if not math.isfinite(val_score):
            val_score = 9999  # TODO, find a better way to handle this

        print(f"val_score: {round(val_score, 5)} {model.val_metric}: {val_score_}")
        return val_score

    def _predict(
            self,
            model: Model,
            x,
            y
    )->Tuple[np.ndarray, np.ndarray]:
        """
        Makes predictions on training and test data from the model.
        It is supposed that the model has been trained before."""

        true, predicted = model.predict(x, y, return_true=True, process_results=False)

        if np.isnan(predicted).sum() == predicted.size:
            warnings.warn(f"model {model.model_name} predicted only nans")
        else:
            ProcessPredictions(self.mode,
                               forecast_len=model.forecast_len,
                               path=model.path,
                               output_features=model.output_features,
                               show=bool(model.verbosity),
                               )(true, predicted)
        return true, predicted

    def verify_data(
            self,
            x=None,
            y=None,
            data=None,
            validation_data: tuple = None,
            test_data: tuple = None,
    ) -> tuple:
        def num_examples(samples):
            if isinstance(samples, list):
                assert len(set(len(sample) for sample in samples)) == 1
                return len(samples[0])
            return len(samples)

        """
        verifies that either
            - only x,y should be given (val will be taken from it according to splitting schemes)
            - or x,y and validation_data should be given  (means no test data)
            - or x, y and validation_data and test_data are given
            - or only data should be given (train, validation and test data will be 
                taken accoring to splitting schemes)

        """

        model_maker = make_model(**self.model_kws)
        data_config = model_maker.data_config

        if x is None:
            assert y is None, f"y must only be given if x is given. x is {type(x)}"

            if data is None:
                # x,y and data are not given, we may be given test/validation data
                train_x, train_y = None, None
                if validation_data is None:
                    val_x, val_y = None, None
                else:
                    val_x, val_y = validation_data

                if test_data is None:
                    test_x, test_y = None, None
                else:
                    test_x, test_y = test_data
            else:
                # case 4, only data is given
                assert data is not None, f"if x is given, data must not be given"
                assert validation_data is None, f"validation data must only be given if x is given"
                #assert test_data is None, f"test data must only be given if x is given"

                data_config.pop('category')

                if 'lookback' in self._named_x0() and 'ts_args' not in self.model_kws:
                    # the value of lookback has been set by model_maker which can be wrong
                    # because the user expects it to be hyperparameter
                    data_config['ts_args']['lookback'] = self._named_x0()['lookback']

                dataset = DataSet(data=data,
                                  save=data_config.pop('save') or True,
                                  category=self.category,
                                  **data_config)

                train_x, train_y = dataset.training_data()
                val_x, val_y = dataset.validation_data() # todo what if there is not validation data
                test_x, test_y = dataset.test_data()
                if len(test_x) == 0:
                    test_x, test_y = None, None

        elif test_data is None and validation_data is None:
            # case 1, only x,y are given
            assert num_examples(x) == num_examples(y)

            splitter= TrainTestSplit(data_config['val_fraction'], seed=data_config['seed'] or 313)

            if data_config['split_random']:
                train_x, val_x, train_y, val_y = splitter.split_by_random(x, y)
            else:
                train_x, val_x, train_y, val_y = splitter.split_by_slicing(x, y)

            test_x, test_y = None, None

        elif test_data is None:
            # case 2: x,y and validation_data should be given  (means no test data)

            assert num_examples(x) == num_examples(y)

            train_x, train_y = x, y
            val_x, val_y = validation_data
            test_x, test_y = None, None

        else:
            # case 3
            assert num_examples(x) == num_examples(y)

            train_x, train_y = x, y
            val_x, val_y = validation_data
            test_x, test_y = test_data

        return train_x, train_y, val_x, val_y, test_x, test_y

    def _get_model_folders(self):
        model_folders = [p for p in os.listdir(self.exp_path) if os.path.isdir(os.path.join(self.exp_path, p))]

        # find all the model folders
        m_folders = []
        for m in model_folders:
            if any(m in m_ for m_ in self.considered_models_):
                m_folders.append(m)
        return m_folders

    def _build_predict_from_configs(self, x, y):

        model_folders = self._get_model_folders()

        # load all models from config
        for model_name in model_folders:

            m_path = self._get_best_model_path(model_name)

            c_path = os.path.join(m_path, 'config.json')
            model = self.build_from_config(c_path)
            # calculate pr curve for each model
            self.update_model_weight(model, m_path)

            out = model.predict(x, y, return_true=True, process_results=False)

            self._populate_results(f"model_{model_name}", train_results=None, test_results=out)

        return

    def _get_best_model_path(self, model_name):
        m_path = os.path.join(self.exp_path, model_name)
        if len(os.listdir(m_path)) == 1:
            m_path = os.path.join(m_path, os.listdir(m_path)[0])

        elif 'best' in os.listdir(m_path):
            # within best folder thre is another folder
            m_path = os.path.join(m_path, 'best')
            assert len(os.listdir(m_path)) == 1
            m_path = os.path.join(m_path, os.listdir(m_path)[0])

        else:
            folders = [path for path in os.listdir(m_path) if
                       os.path.isdir(os.path.join(m_path, path)) and path.startswith('1_')]
            if len(folders) == 1:
                m_path = os.path.join(m_path, folders[0])
            else:
                raise ValueError(f"Cant find best model in {m_path}")
        return m_path



class TransformationExperiments(Experiments):
    """Helper class to conduct experiments with different transformations

        Example:
            >>> from ai4water.datasets import busan_beach
            >>> from ai4water.experiments import TransformationExperiments
            >>> from ai4water.hyperopt import Integer, Categorical, Real
            ... # Define your experiment
            >>> class MyTransformationExperiments(TransformationExperiments):
            ...
            ...    def update_paras(self, **kwargs):
            ...        _layers = {
            ...            "LSTM": {"config": {"units": int(kwargs['lstm_units']}},
            ...            "Dense": {"config": {"units": 1, "activation": kwargs['dense_actfn']}},
            ...            "reshape": {"config": {"target_shape": (1, 1)}}
            ...        }
            ...        return {'model': {'layers': _layers},
            ...                'lookback': int(kwargs['lookback']),
            ...                'batch_size': int(kwargs['batch_size']),
            ...                'lr': float(kwargs['lr']),
            ...                'transformation': kwargs['transformation']}
            >>> data = busan_beach()
            >>> inputs = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm']
            >>> outputs = ['tetx_coppml']
            >>> cases = {'model_minmax': {'transformation': 'minmax'},
            ...         'model_zscore': {'transformation': 'zscore'}}
            >>> search_space = [
            ...            Integer(low=16, high=64, name='lstm_units', num_samples=2),
            ...            Integer(low=3, high=15, name="lookback", num_samples=2),
            ...            Categorical(categories=[4, 8, 12, 16, 24, 32], name='batch_size'),
            ...            Real(low=1e-6, high=1.0e-3, name='lr', prior='log', num_samples=2),
            ...            Categorical(categories=['relu', 'elu'], name='dense_actfn'),
            ...        ]
            >>> x0 = [20, 14, 12, 0.00029613, 'relu']
            >>> experiment = MyTransformationExperiments(cases=cases, input_features=inputs,
            ...                output_features=outputs, exp_name="testing"
            ...                 param_space=search_space, x0=x0)
    """

    @property
    def mode(self):
        return "regression"

    @property
    def category(self):
        return "ML"

    def __init__(self,
                 param_space=None,
                 x0=None,
                 cases: dict = None,
                 exp_name: str = None,
                 num_samples: int = 5,
                 verbosity: int = 1,
                 **model_kws):
        self.param_space = param_space
        self.x0 = x0

        exp_name = exp_name or 'TransformationExperiments' + f'_{dateandtime_now()}'

        super().__init__(
            cases=cases,
            exp_name=exp_name,
            num_samples=num_samples,
            verbosity=verbosity,
            **model_kws
        )

    @property
    def tpot_estimator(self):
        return None

    def update_paras(self, **suggested_paras):
        raise NotImplementedError(f"""
You must write the method `update_paras` which should build the Model with suggested parameters
and return the keyword arguments including `model`. These keyword arguments will then
be used to build ai4water's Model class.
""")

    def _build(self, title=None, **suggested_paras):
        """Builds the ai4water Model class"""

        suggested_paras = jsonize(suggested_paras)

        verbosity = max(self.verbosity - 1, 0)
        if 'verbosity' in self.model_kws:
            verbosity = self.model_kws.pop('verbosity')

        model = Model(
            prefix=title,
            verbosity=verbosity,
            **self.update_paras(**suggested_paras),
            **self.model_kws
        )

        setattr(self, 'model_', model)
        return model

    def process_model_before_fit(self, model):
        """So that the user can perform processing of the model by overwriting this method"""
        return model


def sort_array(array):
    """
    array: [4, 7, 3, 9, 4, 8, 2, 8, 7, 1]
    returns: [4, 4, 3, 3, 3, 3, 2, 2, 2, 1]
    """
    results = np.array(array, dtype=np.float32)
    iters = range(1, len(results) + 1)
    return [np.min(results[:i]) for i in iters]


def consider_exclude(exclude: Union[str, list],
                     models,
                     models_to_filter: Union[list, dict] = None
                     ):
    if isinstance(exclude, str):
        exclude = [exclude]

    if exclude is not None:
        exclude = ['model_' + _model if not _model.startswith('model_') else _model for _model in exclude]
        for elem in exclude:
            assert elem in models, f"""
                {elem} to `exclude` is not available.
                Available models are {models} and you wanted to exclude
                {exclude}"""

            if models_to_filter is not None:
                # maybe the model has already been removed from models_to_filter
                # when we considered include keyword argument.
                if elem in models_to_filter:
                    if isinstance(models_to_filter, list):
                        models_to_filter.remove(elem)
                    else:
                        models_to_filter.pop(elem)
                else:
                    assert elem in models, f'{elem} is not in models'
    return


def load_json_file(fpath):
    with open(fpath, 'r') as fp:
        result = json.load(fp)
    return result


def save_json_file(fpath, obj):
    with open(fpath, 'w') as fp:
        json.dump(jsonize(obj), fp, sort_keys=True, indent=4)


def shred_model_name(model_name):
    key = model_name[6:] if model_name.startswith('model_') else model_name
    key = key[0:-9] if key.endswith("Regressor") else key
    return key


def _combine_training_validation_data(
        x_train,
        y_train,
        validation_data=None,
)->tuple:
    """
    combines x,y pairs of training and validation data.
    """

    if validation_data is None:
        return x_train, y_train
    x_val, y_val = validation_data

    if isinstance(x_train, list):
        x = []
        for val in range(len(x_train)):
            if x_val is not None:
                _val = np.concatenate([x_train[val], x_val[val]])
                x.append(_val)
            else:
                _val = x_train[val]

        y = y_train
        if hasattr(y_val, '__len__') and len(y_val) > 0:
            y = np.concatenate([y_train, y_val])

    elif isinstance(x_train, np.ndarray):
        x, y = x_train, y_train
        # if not validation data is available then use only training data
        if x_val is not None:
            if hasattr(x_val, '__len__') and len(x_val)>0:
                x = np.concatenate([x_train, x_val])
                y = np.concatenate([y_train, y_val])
    else:
        raise NotImplementedError

    return x, y
