import os
import json
import warnings
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ai4water.hyperopt import HyperOpt
from ai4water.postprocessing.SeqMetrics import RegressionMetrics
from ai4water.utils.taylor_diagram import taylor_plot
from ai4water.hyperopt import Real, Categorical, Integer
from ai4water.utils.utils import init_subplots, process_axis, jsonize, ERROR_LABELS
from ai4water.utils.utils import clear_weights, dateandtime_now, dict_to_file
from ai4water.backend import tf
from ai4water.utils.plotting_tools import bar_chart
from ai4water.utils.visualizations import PlotResults
from ai4water.preprocessing import DataHandler

if tf is not None:
    if 230 <= int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 250:
        from ai4water.functional import Model
        print(f"Switching to functional API due to tensorflow version {tf.__version__} for experiments")
    else:
        from ai4water import Model
else:
    from ai4water import Model

SEP = os.sep

# todo plots comparing different models in following youtube videos at 6:30 and 8:00 minutes.
# https://www.youtube.com/watch?v=QrJlj0VCHys
# compare models using statistical tests wuch as Giacomini-White test or Diebold-Mariano test
# paired ttest 5x2cv





class Experiments(object):
    """
    Base class for all the experiments.

    All the expriments must be subclasses of this class.
    The core idea of of `Experiments` is `model`. An experiment consists of one
    or more models. The models differ from each other in their structure/idea/concept.
    When [fit][ai4water.experiments.Experiments.fit] is called, each model is trained.

    Attributes
    ------------
    - simulations
    - trues
    - exp_path
    - _model
    - models

    Methods
    --------
    - fit
    - taylor_plot
    - compare_losses
    - plot_convergence
    - from_config

    """
    def __init__(
            self,
            cases: dict = None,
            exp_name: str = None,
            num_samples: int = 5,
            verbosity: int = 1,
    ):
        """

        Arguments:
            cases:
                python dictionary defining different cases/scenarios
            exp_name:
                name of experiment, used to define path in which results are saved
            num_samples:
                only relevent when you wan to optimize hyperparameters of models
                using `grid` method
            verbosity:
                determines the amount of information
        """
        self.opt_results = None
        self.optimizer = None
        self.exp_name = 'Experiments_' + str(dateandtime_now()) if exp_name is None else exp_name
        self.num_samples = num_samples
        self.verbosity = verbosity

        self.models = [method for method in dir(self) if callable(getattr(self, method)) if method.startswith('model_')]
        if cases is None:
            cases = {}
        self.cases = {'model_'+key if not key.startswith('model_') else key: val for key, val in cases.items()}
        self.models = self.models + list(self.cases.keys())

        self.exp_path = os.path.join(os.getcwd(), "results", self.exp_name)
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)

        self.update_config(models=self.models, exp_path=self.exp_path, exp_name=self.exp_name, cases=self.cases)

    def update_and_save_config(self, **kwargs):
        self.update_config(**kwargs)
        self.save_config()

    def update_config(self, **kwargs):
        if not hasattr(self, 'config'):
            setattr(self, 'config', {})
        self.config.update(kwargs.copy())

    def save_config(self):
        dict_to_file(self.exp_path, config=self.config)

    def build_and_run(self, predict=False, title=None, fit_kws=None, **kwargs):
        setattr(self, '_model', None)
        raise NotImplementedError

    def build_from_config(self, config_path, weights, fit_kws, **kwargs):
        setattr(self, '_model', None)
        raise NotImplementedError

    @property
    def tpot_estimator(self):
        raise NotImplementedError

    @property
    def num_samples(self):
        return self._num_samples

    @num_samples.setter
    def num_samples(self, x):
        self._num_samples = x

    def fit(
            self,
            run_type: str = "dry_run",
            opt_method: str = "bayes",
            num_iterations: int = 12,
            include: Union[None, list] = None,
            exclude: Union[None, list, str] = '',
            cross_validate: bool = False,
            post_optimize: str = 'eval_best',
            fit_kws: dict = None,
            hpo_kws: dict = None
    ):
        """
        Runs the fit loop for the specified models.
        todo, post_optimize not working for 'eval_best' with ML methods.

        Arguments:
            run_type :
                One of `dry_run` or `optimize`. If `dry_run`, the all
                the `models` will be trained only once. if `optimize`, then
                hyperparameters of all the models will be optimized.
            opt_method :
                which optimization method to use. options are `bayes`,
                `random`, `grid`. ONly valid if `run_type` is `optimize`
            num_iterations : number of iterations for optimization. Only valid
                if `run_type` is `optimize`.
            include :
                name of models to included. If None, all the models found
                will be trained and or optimized.
            exclude :
                name of `models` to be excluded
            cross_validate :
                whether to cross validate the model or not. This
                depends upon `cross_validator` agrument to the `Model`.
            post_optimize :
                one of `eval_best` or `train_best`. If eval_best,
                the weights from the best models will be uploaded again and the model
                will be evaluated on train, test and all the data. If `train_best`,
                then a new model will be built and trained using the parameters of
                the best model.
            fit_kws :
                key word arguments that will be passed to ai4water's [fit][ai4water.Model.fit] function
            hpo_kws :
                keyword arguments for [`HyperOpt`][ai4water.hyperopt.HyperOpt.__init__] class.
        """
        assert run_type in ['optimize', 'dry_run']

        assert post_optimize in ['eval_best', 'train_best']

        if exclude == '':
            exclude = []

        predict = False
        if run_type == 'dry_run':
            predict = True

        if hpo_kws is None:
            hpo_kws = {}

        include = self._check_include_arg(include)

        if exclude is None:
            exclude = []
        elif isinstance(exclude, str):
            exclude = [exclude]

        consider_exclude(exclude, self.models)

        self.trues = {'train': {},
                      'test': {}}

        self.simulations = {'train': {},
                            'test': {}}

        self.cv_scores = {}

        self.config['eval_models'] = {}
        self.config['optimized_models'] = {}

        for model_type in include:

            model_name = model_type.split('model_')[1]

            if model_type not in exclude:

                def objective_fn(**suggested_paras):
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

                    return self.build_and_run(predict=predict,
                                              cross_validate=cross_validate,
                                              title=f"{self.exp_name}{SEP}{model_name}",
                                              fit_kws=fit_kws,
                                              **config)

                if run_type == 'dry_run':
                    if self.verbosity > 0: print(f"running  {model_type} model")
                    train_results, test_results = objective_fn()
                    self._populate_results(model_name, train_results, test_results)
                else:
                    # there may be attributes int the model, which needs to be loaded so run the method first.
                    # such as param_space etc.
                    if hasattr(self, model_type):
                        getattr(self, model_type)()

                    opt_dir = os.path.join(os.getcwd(), f"results{SEP}{self.exp_name}{SEP}{model_name}")
                    if self.verbosity > 0: print(f"optimizing  {model_type} using {opt_method} method")
                    self.optimizer = HyperOpt(opt_method,
                                              objective_fn=objective_fn,
                                              param_space=self.param_space,
                                              # use_named_args=True,
                                              opt_path=opt_dir,
                                              num_iterations=num_iterations,  # number of iterations
                                              x0=self.x0,
                                              verbosity=self.verbosity,
                                              **hpo_kws
                                              )

                    self.opt_results = self.optimizer.fit()

                    self.config['optimized_models'][model_type] = self.optimizer.opt_path

                    if post_optimize == 'eval_best':
                        self.eval_best(model_name, opt_dir, fit_kws)
                    elif post_optimize == 'train_best':
                        self.train_best(model_name)

                if not hasattr(self, '_model'):  # todo asking user to define this parameter is not good
                    raise ValueError(f'The `build_and_run` method must set a class level attribute named `_model`.')
                self.config['eval_models'][model_type] = self._model.path

                if cross_validate:
                    cv_scoring = self._model.config['val_metric']
                    self.cv_scores[model_type] = getattr(self._model, f'cross_val_{cv_scoring}')
                    setattr(self, '_cv_scoring', cv_scoring)

        self.save_config()
        return

    def eval_best(self, model_type, opt_dir, fit_kws, **kwargs):
        """Evaluate the best models."""
        best_models = clear_weights(opt_dir, rename=False, write=False)
        # TODO for ML, best_models is empty
        if len(best_models) < 1:
            return self.train_best(model_type, fit_kws)
        for mod, props in best_models.items():
            mod_path = os.path.join(props['path'], "config.json")
            mod_weights = props['weights']

            train_results, test_results = self.build_from_config(mod_path, mod_weights, fit_kws, **kwargs)

            if mod.startswith('1_'):
                self._populate_results(model_type, train_results, test_results)
        return

    def train_best(self, model_type, fit_kws=None):
        """Train the best model."""
        best_paras = self.optimizer.best_paras()
        if best_paras.get('lookback', 1) > 1:
            _model = 'layers'
        else:
            _model = model_type
        train_results, test_results = self.build_and_run(predict=True,
                                                         # view=True,
                                                         fit_kws=fit_kws,
                                                         model={_model: self.optimizer.best_paras()},
                                                         title=f"{self.exp_name}{SEP}{model_type}{SEP}best")

        self._populate_results(model_type, train_results, test_results)
        return

    def _populate_results(self, model_type: str,
                          train_results: Tuple[np.ndarray, np.ndarray],
                          test_results: Tuple[np.ndarray, np.ndarray]
                          ):

        if not model_type.startswith('model_'):  # internally we always use model_ at the start.
            model_type = f'model_{model_type}'

        # it is possible that different models slightly differ in the number of examples
        # in training/test sets for example if `lookback` is different, thus trues and simulations
        # must be saved for each model.
        self.trues['train'][model_type] = train_results[0]
        self.trues['test'][model_type] = test_results[0]

        self.simulations['train'][model_type] = train_results[1]
        self.simulations['test'][model_type] = test_results[1]
        return

    def taylor_plot(
            self,
            include: Union[None, list] = None,
            exclude: Union[None, list] = None,
            figsize: tuple = (9, 7),
            **kwargs
    ):
        """
        Compares the models using [taylor plot][ai4water.utils.taylor_plot].

        Arguments:
            include :
                if not None, must be a list of models which will be included.
                None will result in plotting all the models.
            exclude :
                if not None, must be a list of models which will excluded.
                None will result in no exclusion
            figsize :
            kwargs :  all the keyword arguments for [taylor_plot][ai4water.utils.taylor_plot] function.
        """

        include = self._check_include_arg(include)
        simulations = {'train': {},
                       'test': {}}
        for m in include:
            if m in self.simulations['train']:
                simulations['train'][m.split('model_')[1]] = self.simulations['train'][m]
                simulations['test'][m.split('model_')[1]] = self.simulations['test'][m]

        if exclude is not None:
            consider_exclude(exclude, self.models)

            for m in exclude:
                simulations['train'].pop(m[0])
                simulations['test'].pop(m[0])

        if 'name' in kwargs:
            fname = kwargs.pop('name')
        else:
            fname = 'taylor'
        fname = os.path.join(os.getcwd(), f'results{SEP}{self.exp_name}{SEP}{fname}.png')

        train_lenths = [len(self.trues['train'][obj]) for obj in self.trues['train']]
        if not len(set(train_lenths)) <= 1:
            warnings.warn(f'{train_lenths}')

            trues = {'train': None, 'test': None}
            for k, v in self.trues.items():

                for _k, _v in v.items():

                    trues[k] = {'std': np.std(_v)}

            _simulations = {'train': {}, 'test': {}}
            for scen, v in simulations.items():

                for (_tk, _tv), (_pk, _pv) in zip(self.trues[scen].items(), simulations[scen].items()):

                    _simulations[scen][_pk] = {
                        'std': np.std(_pv),
                        'corr_coeff': RegressionMetrics(_tv, _pv).corr_coeff(),
                        'pbias': RegressionMetrics(_tv, _pv).pbias()
                    }
        else:
            trues = {'train': None, 'test': None}
            for k, v in self.trues.items():

                for idx, (_k, _v) in enumerate(v.items()):
                    trues[k] = _v
            _simulations = simulations

        taylor_plot(
            trues=trues,
            simulations=_simulations,
            figsize=figsize,
            name=fname,
            **kwargs
        )
        return

    def _consider_include(self, include: [str, list], to_filter):

        filtered = {}

        include = self._check_include_arg(include)

        for m in include:
            if m in to_filter:
                filtered[m] = to_filter[m]

        return filtered

    def _check_include_arg(self, include):

        if isinstance(include, str):
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

    def _get_inital_best_results(self, model_name, run_type, matric_name):

        initial_run = os.listdir(os.path.join(self.exp_path, model_name.split('model_')[1]))[0]
        initial_run = os.path.join(self.exp_path, model_name.split('model_')[1],  initial_run)

        if "best" in os.listdir(os.path.join(self.exp_path, model_name.split('model_')[1])):
            best_folder = os.path.join(self.exp_path, model_name.split('model_')[1], "best")
            best_run = os.path.join(best_folder, os.listdir(best_folder)[0])
        else:
            return None, None

        cpath = os.path.join(best_run, 'config.json')
        with open(cpath, 'r') as fp:
            conf = json.load(fp)
        output_features = conf['config']['output_features']
        assert len(output_features) == 1
        output = output_features[0]

        fpath_initial = os.path.join(initial_run, output, f"{run_type}_{output}_0.csv")
        initial = pd.read_csv(fpath_initial, index_col=['index'])

        fpath_best: str = os.path.join(best_run, output, f"{run_type}_{output}_0.csv")
        best = pd.read_csv(fpath_best, index_col=['index'])

        initial_metric: float = getattr(RegressionMetrics(initial.values[:, 0], initial.values[:, 1]), matric_name)()
        best_metric: float = getattr(RegressionMetrics(best.values[:, 0], best.values[:, 1]), matric_name)()

        # -ve values become difficult to plot, moreover they do not reveal anything significant
        # as compared to when a metric is 0, therefore consider -ve values as zero.
        return np.max([initial_metric, 0.0]), np.max([best_metric, 0.0])

    def plot_improvement(
            self,
            matric_name: str,
            save: bool = True,
            run_type: str = 'test',
            orient: str = 'horizontal',
            **kwargs
    ) -> dict:
        """Shows how much improvement was observed after hyperparameter
        optimization. This plot is only available if `run_type` was set to
        `optimize` in [`fit`][ai4water.experiments.Experiments.fit].

        Arguments:
            matric_name : the peformance metric to compare
            save : whether to save the plot or not
            run_type : which run to use, valid values are `training`, `test` and `validation`.
            orient : valid values are `horizontal` or `vertical`
            kwargs : any of the following keyword arguments

                - rotation :
                - name :
                - dpi :
        """
        rotation = kwargs.get('rotation', 0)
        name = kwargs.get('name', '')
        dpi = kwargs.get('dpi', 200)

        assert run_type in ['training', 'test', 'validation']

        colors = {
            'Initial': np.array([0, 56, 104]) / 256,
            'Improvement': np.array([126, 154, 178]) / 256
        }

        exec_models = list(self.trues['train'].keys())

        data = {k: [] for k in colors.keys()}

        for model in exec_models:
            initial, best = self._get_inital_best_results(model, run_type=run_type, matric_name=matric_name)

            data['Initial'].append(initial)
            data['Improvement'].append(best)

        if np.isnan(np.array(data['Improvement'], dtype=np.float32)).sum() == len(data['Improvement']):
            print("No best results not found")
            return data

        plt.close('all')
        fig, axis = plt.subplots()

        if matric_name in ['r2', 'nse', 'kge', 'corr_coeff', 'r2_mod']:
            order = ['Improvement', 'Initial']
        else:
            order = list(colors.keys())

        names = [m.split('model_')[1] for m in exec_models]
        for key in order:
            if orient == "horizontal":
                axis.barh(range(len(exec_models)), data[key], color=colors[key], label=key)
                plt.xlabel("{}".format(ERROR_LABELS.get(matric_name, matric_name)))
                plt.yticks(ticks=range(len(exec_models)), labels=names, rotation=rotation)
            else:
                axis.bar(range(len(exec_models)), data[key], color=colors[key], label=key)
                plt.ylabel("{}".format(ERROR_LABELS.get(matric_name, matric_name)))
                plt.xticks(ticks=range(len(exec_models)), labels=names, rotation=rotation)

        axis.legend()
        plt.title('Improvement after tuning')

        if save:
            fname = os.path.join(os.getcwd(), f'results{SEP}{self.exp_name}{SEP}{name}_improvement_{matric_name}.png')
            plt.savefig(fname, dpi=dpi, bbox_inches=kwargs.get('bbox_inches', 'tight'))
        plt.show()

        return data

    def compare_errors(
            self,
            matric_name: str,
            cutoff_val: float = None,
            cutoff_type: str = None,
            save: bool = True,
            sort_by: str = 'test',
            ignore_nans: bool = True,
            name: str = 'ErrorComparison',
            show: bool = True,
            **kwargs
    ) -> dict:
        """
        Plots a specific performance matric for all the models which were
        run during [fit][ai4water.experiments.Experiments.fit] call.

        Arguments:
            matric_name:
                 performance matric whose value to plot for all the models
            cutoff_val:
                 if provided, only those models will be plotted for whome the matric is greater/smaller
                 than this value. This works in conjuction with `cutoff_type`.
            cutoff_type:
                 one of `greater`, `greater_equal`, `less` or `less_equal`.
                 Criteria to determine cutoff_val. For example if we want to
                 show only those models whose r2 is > 0.5, it will be 'max'.
            save:
                whether to save the plot or not
            sort_by:
                either 'test' or 'train'. How to sort the results for plotting. If 'test', then test
                performance matrics will be sorted otherwise train performance matrics will be sorted.
            ignore_nans:
                default True, if True, then performance matrics with nans are ignored otherwise
                nans/empty bars will be shown to depict which models have resulted in nans for the given
                performance matric.
            name:
                name of the saved file.
            show : whether to show the plot at the end or not?

            kwargs :

                - fig_height :
                - fig_width :
                - title_fs :
                - xlabel_fs :
                - color :

        returns:
            dictionary whose keys are models and values are performance metrics.

        Example:
            >>> from ai4water.experiments import MLRegressionExperiments
            >>> from ai4water.datasets import arg_beach
            >>> data = arg_beach()
            >>> inputs = list(data.columns)[0:-1]
            >>> outputs = list(data.columns)[-1]
            >>> experiment = MLRegressionExperiments(data=data, input_features=inputs, output_features=outputs)
            >>> experiment.fit()
            >>> experiment.compare_errors('mse')
            >>> experiment.compare_errors('r2', 0.2, 'greater')
        """

        models = self.sort_models_by_metric(matric_name, cutoff_val, cutoff_type,
                                            ignore_nans, sort_by)

        names = [i[1] for i in models.values()]
        test_matrics = list(models.keys())
        train_matrics = [i[0] for i in models.values()]

        plt.close('all')
        fig, axis = plt.subplots(1, 2, sharey='all')
        fig.set_figheight(kwargs.get('fig_height', 8))
        fig.set_figwidth(kwargs.get('fig_width', 8))

        bar_chart(axis=axis[0],
                  labels=names[::-1],
                  values=train_matrics[::-1],
                  color=kwargs.get('color', None),
                  title="Train",
                  xlabel=ERROR_LABELS.get(matric_name, matric_name),
                  xlabel_fs=kwargs.get('xlabel_fs', 16),
                  title_fs=kwargs.get('title_fs', 20)
                  )

        bar_chart(axis=axis[1],
                  labels=names[::-1],
                  values=test_matrics[::-1],
                  title="Test",
                  color=kwargs.get('color', None),
                  xlabel=ERROR_LABELS.get(matric_name, matric_name),
                  xlabel_fs=kwargs.get('xlabel_fs', 16),
                  title_fs=kwargs.get('title_fs', 20),
                  show_yaxis=False
                  )

        appendix = f"{cutoff_val or ''}{cutoff_type or ''}{len(models)}"
        if save:
            fname = os.path.join(os.getcwd(), f'results{SEP}{self.exp_name}{SEP}{name}_{matric_name}_{appendix}.png')
            plt.savefig(fname, dpi=100, bbox_inches=kwargs.get('bbox_inches', 'tight'))

        if show:
            plt.show()
        return models

    def plot_losses(
            self,
            loss_name: Union[str, list] = 'loss',
            save: bool = True,
            name: str = 'loss_comparison',
            show: bool = True,
            **kwargs
    ) -> plt.Axes:
        """
        Plots the loss curves of the evaluated models.

        Arguments:
            loss_name:
                the name of loss value, must be recorded during training
            save:
                whether to save the plot or not
            name:
                name of saved file
            show:
                whether to show the plot or now
            kwargs : following keyword arguments can be used

                - width:
                - height:
                - bbox_inches:
        Returns:
            matplotlib axes
        """

        if not isinstance(loss_name, list):
            assert isinstance(loss_name, str)
            loss_name = [loss_name]
        loss_curves = {}
        for _model, _path in self.config['eval_models'].items():
            df = pd.read_csv(os.path.join(_path, 'losses.csv'), usecols=loss_name)
            loss_curves[_model] = df

        _kwargs = {'linestyle': '-',
                   'xlabel': "Epochs",
                   'ylabel': 'Loss'}

        if len(loss_curves) > 5:
            _kwargs['legend_kws'] = {'bbox_to_anchor': (1.1, 0.99)}

        _, axis = init_subplots(kwargs.get('width', None), kwargs.get('height', None))

        for _model, _loss in loss_curves.items():
            if _model.startswith('model_'):
                _model = _model.split('model_')[1]
            axis = process_axis(axis=axis, data=_loss, label=_model, **_kwargs)

        if save:
            fname = os.path.join(self.exp_path, f'{name}_{loss_name}.png')
            plt.savefig(fname, dpi=100, bbox_inches=kwargs.get('bbox_inches', 'tight'))

        if show:
            plt.show()

        return axis

    def plot_convergence(
            self,
            show: bool = True,
            save: bool = False,
            name: str = 'convergence_comparison',
            **kwargs
    ) -> Union[plt.Axes, None]:
        """
        Plots the convergence plots of hyperparameter optimization runs.
        Only valid if `run_type=optimize` during [`fit`][ai4water.experiments.experiments.Experiments.fit]
        call.

        Arguments:
            show:
                whether to show the plot or now
            save : whether to save the plot or not
            name : name of file to save the plot
            kwargs : keyword arguments like:
                bbox_inches :
        Returns:
            if the optimized models are >1 then it returns the maplotlib axes
            on which the figure is drawn otherwise it returns None.
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
            process_axis(axis=axis, data=convergence, label=_model.split('model_')[1],
                         linestyle='--',
                         xlabel='Number of iterations $n$',
                         ylabel=r"$\min f(x)$ after $n$ calls")
        if save:
            fname = os.path.join(self.exp_path, f'{name}.png')
            plt.savefig(fname, dpi=100, bbox_inches=kwargs.get('bbox_inches', 'tight'))

        if show:
            plt.show()
        return axis

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

        cls.config = config
        cls._from_config = True

        trues = {'train': {},
                 'test': {}}

        simulations = {'train': {},
                       'test': {}}

        cv_scores = {}
        scoring = "mse"

        for model_name, model_path in config['eval_models'].items():

            with open(os.path.join(model_path, 'config.json'), 'r') as fp:
                model_config = json.load(fp)

            output_features = model_config['config']['output_features']

            if len(output_features) > 1:
                raise NotImplementedError

            fpath = os.path.join(model_path, output_features[0])
            fname = [f for f in os.listdir(fpath) if f.startswith('training') and f.endswith('.csv')]
            assert len(fname) == 1, f'{model_name} does not have training results'
            fname = fname[0]
            train_df = pd.read_csv(os.path.join(fpath, fname), index_col='index')

            fname = [f for f in os.listdir(fpath) if f.startswith('test') and f.endswith('.csv')]
            assert len(fname) == 1, f'{model_name} does not have training results'
            fname = fname[0]
            test_df = pd.read_csv(os.path.join(fpath, fname), index_col='index')

            trues['train'][model_name] = train_df[f'true_{output_features[0]}']
            trues['test'][model_name] = test_df[f'true_{output_features[0]}']

            simulations['train'][model_name] = train_df[f'pred_{output_features[0]}']
            simulations['test'][model_name] = test_df[f'pred_{output_features[0]}']

            # if cross validation was performed, then read those results.
            cross_validator = model_config['config']['cross_validator']
            if cross_validator is not None:
                cv_name = str(list(cross_validator.keys())[0])
                scoring = model_config['config']['val_metric']
                cv_fname = os.path.join(model_path, f'{cv_name}_{scoring}' + ".json")
                if os.path.exists(cv_fname):
                    with open(cv_fname, 'r') as fp:
                        cv_scores[model_name] = json.load(fp)

        cls.trues = trues
        cls.simulations = simulations
        cls.cv_scores = cv_scores
        cls._cv_scoring = scoring

        return cls(exp_name=config['exp_name'], cases=config['cases'], **kwargs)

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
        [fit()][ai4water.experiments.experiments.Experiments.fit].

        Arguments:
            show : whether to show the plot or not
            name : name of the plot
            include : models to include
            exclude : models to exclude
            kwargs : any of the following keyword arguments

                - notch
                - vert
                - figsize
                - bbox_inches

        Returns:
            matplotlib axes if the figure is drawn otherwise None

        """
        if len(self.cv_scores) == 0:
            return

        scoring = self._cv_scoring
        cv_scores = self.cv_scores

        consider_exclude(exclude, self.models, cv_scores)
        cv_scores = self._consider_include(include, cv_scores)

        model_names = [m.split('model_')[1] for m in list(cv_scores.keys())]
        if len(model_names) < 5:
            rotation = 0
        else:
            rotation = 90

        plt.close()

        _, axis = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))

        d = axis.boxplot(list(cv_scores.values()),
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

        fname = os.path.join(os.getcwd(), f'results{SEP}{self.exp_name}{SEP}{name}_{len(model_names)}.png')
        plt.savefig(fname, dpi=300, bbox_inches=kwargs.get('bbox_inches', 'tight'))

        if show:
            plt.show()

        return axis

    def sort_models_by_metric(
            self,
            matric_name,
            cutoff_val=None,
            cutoff_type=None,
            ignore_nans: bool = True,
            sort_by="test"
    ) -> dict:
        """returns the models sorted according to their performance"""
        def find_matric_array(true, sim):
            errors = RegressionMetrics(true, sim)
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
        models = {}

        for mod in self.models:
            # find the models which have been run
            if mod in self.simulations['test']:  # maybe we have not done some models by using include/exclude
                test_matric = find_matric_array(self.trues['test'][mod], self.simulations['test'][mod])
                if test_matric is not None:
                    test_matrics.append(test_matric)
                    models[mod.split('model_')[1]] = {'test': test_matric}

                    train_matric = find_matric_array(self.trues['train'][mod], self.simulations['train'][mod])
                    if train_matric is None:
                        train_matric = np.nan
                    train_matrics.append(train_matric)
                    models[mod.split('model_')[1]] = {'train': train_matric, 'test': test_matric}

        if len(models) <= 1:
            warnings.warn(f"Comparison can not be plotted because the obtained models are <=1 {models}", UserWarning)
            return models

        if sort_by == 'test':
            d = sort_metric_dicts(ignore_nans, test_matrics, train_matrics, list(models.keys()))

        elif sort_by == 'train':
            d = sort_metric_dicts(ignore_nans, train_matrics, test_matrics, list(models.keys()))
        else:
            raise ValueError(f'sort_by must be either train or test but it is {sort_by}')

        sorted_models = dict(sorted(d.items(), reverse=True))

        return sorted_models

    def fit_with_tpot(
            self,
            models: Union[int, List[str], dict, str] = None,
            selection_criteria: str = 'mse',
            scoring: str = None,
            **tpot_args
    ):
        """
        Fits the [tpot's](http://epistasislab.github.io/tpot/) fit  method which
        finds out the best pipline for the given data.

        Arguments:
            models:
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
                - You can also set it to `all` consider all models available in ai4water's
                    Experiment module.
                - default is None, which means, the `tpot_config` argument will be None
            selection_criteria :
                If `models` is integer, then according to which criteria
                the models will be choosen. By default the models will be selected
                based upon their mse values on test data.
            scoring : the performance metric to use for finding the pipeline.
            tpot_args : any keyword argument for tpot's [Regressor](http://epistasislab.github.io/tpot/api/#regression)
                or [Classifier](http://epistasislab.github.io/tpot/api/#classification) class.
                This can include arguments like `generations`, `population_size` etc.

        Returns:
            the tpot object

        Example:
            >>> from ai4water.experiments import MLRegressionExperiments
            >>> from ai4water.datasets import arg_beach
            >>> exp = MLRegressionExperiments(data=arg_beach(), exp_name=f"tpot_reg_{dateandtime_now()}")
            >>> exp.fit()
            >>> tpot_regr = exp.fit_with_tpot(2, generations=1, population_size=2)
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
            trues = getattr(self, 'trues', {})
            assert len(trues)>1, f"you must first run .fit() method in order to choose top {models} models"

            # sort the models w.r.t their performance
            sorted_models = self.sort_models_by_metric(selection_criteria)

            # get names of models
            models = [v[1] for idx, v in enumerate(sorted_models.values()) if idx < models]

            tpot_config = {}
            for m in models:
                c: dict = param_space[f"model_{m}"]
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
            verbosity=self.verbosity+1,
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

        dh = DataHandler(self.data, **model_kws)
        train_x, train_y = dh.training_data()
        tpot.fit(train_x, train_y.reshape(-1, 1))

        visualizer = PlotResults(path=self.exp_path)

        for idx, data_name in enumerate(['training', 'test']):

            x_data, y_data = getattr(dh, f"{data_name}_data")(key=str(idx))

            pred = tpot.fitted_pipeline_.predict(x_data)
            r2 = RegressionMetrics(y_data, pred).r2()

            # todo, perform inverse transform and deindexification
            visualizer.plot_results(
                pd.DataFrame(y_data.reshape(-1,)),
                pd.DataFrame(pred.reshape(-1,)),
                annotation_key='$R^2$', annotation_val=r2,
                show=self.verbosity,
                where='',  name=data_name
            )
        # save the python code of fitted pipeline
        tpot.export(os.path.join(self.exp_path, "tpot_fitted_pipeline.py"))

        # save each iteration
        fname = os.path.join(self.exp_path, "evaludated_individuals.json")
        with open(fname, 'w') as fp:
            json.dump(tpot.evaluated_individuals_, fp, indent=True)
        return tpot


class TransformationExperiments(Experiments):
    """Helper to conduct experiments with different transformations

        Example:
            >>> from ai4water.datasets import arg_beach
            >>>from ai4water.experiments import TransformationExperiments
            ...# Define your experiment
            >>>class MyTransformationExperiments(TransformationExperiments):
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
            >>>data = arg_beach()
            >>>inputs = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm']
            >>>outputs = ['tetx_coppml']
            >>>cases = {'model_minmax': {'transformation': 'minmax'},
            ...         'model_zscore': {'transformation': 'zscore'}}
            >>>search_space = [
            ...            Integer(low=16, high=64, name='lstm_units', num_samples=2),
            ...            Integer(low=3, high=15, name="lookback", num_samples=2),
            ...            Categorical(categories=[4, 8, 12, 16, 24, 32], name='batch_size'),
            ...            Real(low=1e-6, high=1.0e-3, name='lr', prior='log', num_samples=2),
            ...            Categorical(categories=['relu', 'elu'], name='dense_actfn'),
            ...        ]
            >>>x0 = [20, 14, 12, 0.00029613, 'relu']
            >>>experiment = MyTransformationExperiments(cases=cases, input_features=inputs,
            ...                output_features=outputs, data=data, exp_name="testing"
            ...                 param_space=search_space, x0=x0)
    """

    def __init__(self,
                 data,
                 param_space=None,
                 x0=None,
                 cases: dict = None,
                 exp_name: str = None,
                 num_samples: int = 5,
                 ai4water_model=None,
                 verbosity: int = 1,
                 **model_kws):
        self.data = data
        self.param_space = param_space
        self.x0 = x0
        self.model_kws = model_kws
        self.ai4water_model = Model if ai4water_model is None else ai4water_model

        exp_name = exp_name or 'TransformationExperiments' + f'_{dateandtime_now()}'

        super().__init__(cases=cases,
                         exp_name=exp_name,
                         num_samples=num_samples,
                         verbosity=verbosity)

    @property
    def tpot_estimator(self):
        return None

    def update_paras(self, **suggested_paras):
        raise NotImplementedError(f"""
You must write the method `update_paras` which should build the Model with suggested parameters
and return the keyword arguments including `model`. These keyword arguments will then
be used to build ai4water's Model class.
""")

    def build_and_run(self,
                      predict=False,
                      title=None,
                      fit_kws=None,
                      cross_validate=False,
                      **suggested_paras):

        suggested_paras = jsonize(suggested_paras)

        verbosity = self.verbosity
        if 'verbosity' in self.model_kws:
            verbosity = self.model_kws.pop('verbosity')

        model = self.ai4water_model(
            prefix=title,
            verbosity=verbosity,
            **self.update_paras(**suggested_paras),
            **self.model_kws
        )

        setattr(self, '_model', model)

        model = self.process_model_before_fit(model)

        if cross_validate:
            val_score = model.cross_val_score(data=self.data)
        else:
            model.fit(data=self.data,)
            val_true, val_pred = model.predict(data='validation', return_true=True)
            val_score = getattr(RegressionMetrics(val_true, val_pred), model.config['val_metric'])()

        if predict:
            trt, trp = model.predict(data='training', return_true=True)

            testt, testp = model.predict(return_true=True)

            model.config['allow_nan_labels'] = 2
            model.predict()
            # model.plot_train_data()
            return (trt, trp), (testt, testp)

        return val_score

    def build_from_config(self, config_path, weight_file, fit_kws, **kwargs):

        model = self.ai4water_model.from_config_file(config_path=config_path)
        weight_file = os.path.join(model.w_path, weight_file)
        model.update_weights(weight_file=weight_file)

        model = self.process_model_before_fit(model)

        train_true, train_pred = model.predict(data=self.data, return_true=True)

        test_true, test_pred = model.predict(data='test', return_true=True)

        model.data['allow_nan_labels'] = 1
        model.predict()

        return (train_true, train_pred), (test_true, test_pred)

    def process_model_before_fit(self, model):
        """So that the user can perform procesisng of the model by overwriting this method"""
        return model


def sort_array(array):
    """
    array: [4, 7, 3, 9, 4, 8, 2, 8, 7, 1]
    returns: [4, 4, 3, 3, 3, 3, 2, 2, 2, 1]
    """
    results = np.array(array, dtype=np.float32)
    iters = range(1, len(results) + 1)
    return [np.min(results[:i]) for i in iters]


def consider_exclude(exclude: [str, list], models, models_to_filer: Union[dict] = None):

    if isinstance(exclude, str):
        exclude = [exclude]

    if exclude is not None:
        exclude = ['model_' + _model if not _model.startswith('model_') else _model for _model in exclude]
        for elem in exclude:
            assert elem in models, f"""
                {elem} to `exclude` is not available.
                Available models are {models} and you wanted to exclude
                {exclude}"""

            if models_to_filer is not None:
                assert elem in models_to_filer, f'{elem} is not in models'
                models_to_filer.pop(elem)
    return


def sort_metric_dicts(ignore_nans, first, second, model_names):

    if ignore_nans:
        d = {key: [a, _name] for key, a, _name in zip(first, second, model_names) if
             not np.isnan(key)}
    else:
        d = {key: [a, _name] for key, a, _name in zip(first, second, model_names)}

    return d
