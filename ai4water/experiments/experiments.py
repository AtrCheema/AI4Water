import os
import json
import warnings
from typing import Union, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

from ai4water.hyper_opt import HyperOpt
from ai4water.utils.SeqMetrics import RegressionMetrics
from ai4water.utils.taylor_diagram import taylor_plot
from ai4water.hyper_opt import Real, Categorical, Integer
from ai4water.utils.utils import init_subplots, process_axis
from ai4water.utils.utils import clear_weights, dateandtime_now, save_config_file
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
    When fit() is called, each model is trained.

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
    def __init__(self,
                 cases:dict=None,
                 exp_name:str=None,
                 num_samples:int=5
                 ):
        """
        Arguments:
            cases : python dictionary defining different cases/scenarios
            exp_name : name of experiment, used to define path in which results are saved
            num_samples : only relevent when you wan to optimize hyperparameters of models
                using `grid` method
        """
        self.opt_results = None
        self.optimizer = None
        self.exp_name = 'Experiments_' + str(dateandtime_now()) if exp_name is None else exp_name
        self.num_samples = num_samples

        self.models = [method for method in dir(self) if callable(getattr(self, method)) if method.startswith('model_')]
        if cases is None:
            cases = {}
        self.cases = {'model_'+key if not key.startswith('model_') else key:val for key,val in cases.items()}
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
        save_config_file(self.exp_path, config=self.config)

    def build_and_run(self, predict=False, title=None, fit_kws=None, **kwargs):
        setattr(self, '_model', None)
        raise NotImplementedError

    def build_from_config(self, config_path, weights, fit_kws, **kwargs):
        setattr(self, '_model', None)
        raise NotImplementedError

    @property
    def num_samples(self):
        return self._num_samples

    @num_samples.setter
    def num_samples(self, x):
        self._num_samples = x

    def fit(self,
            run_type:str="dry_run",
            opt_method:str="bayes",
            num_iterations:int=12,
            include: Union[None, list] = None,
            exclude: Union[None, list, str] = '',
            cross_validate:bool = False,
            post_optimize:str='eval_best',
            fit_kws: dict=None,
            hpo_kws: dict = None
            ):
        """
        Runs the fit loop for the specified models.
        todo, post_optimize not working for 'eval_best' with ML methods.

        Arguments:
            run_type : One of `dry_run` or `optimize`. If `dry_run`, the all
                the `models` will be trained only once. if `optimize`, then
                hyperparameters of all the models will be optimized.
            opt_method : which optimization method to use. options are `bayes`,
                `random`, `grid`. ONly valid if `run_type` is `optimize`
            num_iterations : number of iterations for optimization. Only valid
                if `run_type` is `optimize`.
            include : name of models to included. If None, all the models found
                will be trained and or optimized.
            exclude : name of `models` to be excluded
            cross_validate : whether to cross validate the model or not. This
                depends upon `cross_validator` agrument to the `Model`.
            post_optimize : one of `eval_best` or `train_best`. If eval_best,
                the weights from the best models will be uploaded again and the model
                will be evaluated on train, test and all the data. If `train_best`,
                then a new model will be built and trained using the parameters of
                the best model.
            fit_kws :  key word arguments that will be passed to ai4water's model.fit
            hpo_kws : keyword arguments for `HyperOpt` class.
        """

        assert run_type in ['optimize', 'dry_run']
        assert post_optimize in ['eval_best', 'train_best']

        if exclude == '': exclude = []

        predict = False
        if run_type == 'dry_run':
            predict = True

        if hpo_kws is None:
            hpo_kws = {}

        include = self.check_include_arg(include)

        if exclude is None:
            exclude = []
        elif isinstance(exclude, str):
            exclude = [exclude]

        if isinstance(exclude, list):
            exclude = ['model_' + _model if not _model.startswith('model_') else _model for _model in exclude ]
            assert all(elem in self.models for elem in exclude), f"""
One or more models to `exclude` are not available.
Available cases are {self.models} and you wanted to exclude
{exclude}"""

        self.trues = {'train': {},
                      'test': {}}

        self.simulations = {'train': {},
                            'test': {}}

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
                    print(f"running  {model_type} model")
                    train_results, test_results = objective_fn()
                    self._populate_results(model_name, train_results, test_results)
                else:
                    # there may be attributes int the model, which needs to be loaded so run the method first.
                    # such as param_space etc.
                    if hasattr(self, model_type):
                        getattr(self, model_type)()

                    opt_dir = os.path.join(os.getcwd(), f"results{SEP}{self.exp_name}{SEP}{model_name}")
                    print(f"optimizing  {model_type} using {opt_method} method")
                    self.optimizer = HyperOpt(opt_method,
                                              objective_fn=objective_fn,
                                              param_space=self.param_space,
                                              #use_named_args=True,
                                              opt_path=opt_dir,
                                              num_iterations=num_iterations,  # number of iterations
                                              x0=self.x0,
                                              **hpo_kws
                                              )

                    self.opt_results = self.optimizer.fit()

                    self.config['optimized_models'][model_type] = self.optimizer.opt_path

                    if post_optimize == 'eval_best':
                        self.eval_best(model_name, opt_dir, fit_kws)
                    elif post_optimize=='train_best':
                        self.train_best(model_name)

                if not hasattr(self, '_model'):  # todo asking user to define this parameter is not good
                    raise ValueError(f'The `build_and_run` method must set a class level attribute named `_model`.')
                self.config['eval_models'][model_type] = self._model.path

        self.save_config()
        return

    def eval_best(self, model_type, opt_dir, fit_kws, **kwargs):
        """Evaluate the best models."""
        best_models = clear_weights(opt_dir, rename=False, write=False)
        # TODO for ML, best_models is empty
        if len(best_models)<1:
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
        best_paras =  self.optimizer.best_paras()
        if best_paras.get('lookback', 1)>1:
            _model = 'layers'
        else:
            _model = model_type
        train_results, test_results = self.build_and_run(predict=True,
                                                         #view=True,
                                                         fit_kws=fit_kws,
                                                         model={_model: self.optimizer.best_paras()},
                                                         title=f"{self.exp_name}{SEP}{model_type}{SEP}best")

        self._populate_results(model_type, train_results, test_results)
        return

    def _populate_results(self, model_type:str,
                          train_results:Tuple[np.ndarray, np.ndarray],
                          test_results:Tuple[np.ndarray, np.ndarray]
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

    def taylor_plot(self,
                     include: Union[None, list] = None,
                     exclude: Union[None, list] = None,
                     figsize: tuple = (9, 7),
                     **kwargs):
        """
        Arguments:
            include list:
                if not None, must be a list of models which will be included.
                None will result in plotting all the models.
            exclude list:
                if not None, must be a list of models which will excluded.
                None will result in no exclusion
            figsize tuple:
            kwargs dict:  all the keyword arguments from taylor_plot().
        """

        include = self.check_include_arg(include)
        simulations = {'train': {},
                       'test': {}}
        for m in include:
            if m in self.simulations['train']:
                simulations['train'][m.split('model_')[1]] = self.simulations['train'][m]
                simulations['test'][m.split('model_')[1]] = self.simulations['test'][m]

        if exclude is not None:
            exclude = ['model_' + _model if not _model.startswith('model_') else _model for _model in exclude]
            assert all(elem in self.models for elem in exclude), f"""
    One or more models to `exclude` are not available.
    Available cases are {self.models} and you wanted to exclude
    {exclude}
    """
            for m in exclude:
                simulations['train'].pop(m[0])
                simulations['test'].pop(m[0])

        fname = kwargs.get('name', 'taylor.png')
        fname = os.path.join(os.getcwd(),f'results{SEP}{self.exp_name}{SEP}{fname}.png')

        train_lenths = [len(self.trues['train'][obj]) for obj in self.trues['train']]
        if not len(set(train_lenths))<= 1:
            warnings.warn(f'{train_lenths}')

            trues = {'train': None, 'test': None}
            for k,v in self.trues.items():

                for idx, (_k,_v) in enumerate(v.items()):

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
            for k,v in self.trues.items():

                for idx, (_k,_v) in enumerate(v.items()):
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

    def check_include_arg(self, include):
        if include is None:
            include = self.models
        include = ['model_' + _model if not _model.startswith('model_') else _model for _model in include]
        # make sure that include contains same elements which are present in models
        assert all(elem in self.models for elem in include), f"""
One or more models to `include` are not available.
Available cases are {self.models} and you wanted to include
{include}
"""
        return include

    def compare_errors(
            self,
            matric_name: str,
            cutoff_val:float = None,
            cutoff_type:str=None,
            save:bool = True,
            sort_by: str = 'test',
            ignore_nans: bool = True,
            name:str = 'ErrorComparison',
            **kwargs
    )->dict:
        """
        Plots a specific performance matric for all the models which were
        run during `experiment.fit()`.

        Arguments:
            matric_name str:
                 performance matric whose value to plot for all the models
            cutoff_val float:
                 if provided, only those models will be plotted for whome the matric is greater/smaller
                 than this value. This works in conjuction with `cutoff_type`.
            cutoff_type str:
                 one of `greater`, `greater_equal`, `less` or `less_equal`.
                 Criteria to determine cutoff_val. For example if we want to
                 show only those models whose r2 is > 0.5, it will be 'max'.
            save bool:
                whether to save the plot or not
            sort_by str:
                either 'test' or 'train'. How to sort the results for plotting. If 'test', then test
                performance matrics will be sorted otherwise train performance matrics will be sorted.
            ignore_nans bool:
                default True, if True, then performance matrics with nans are ignored otherwise
                nans/empty bars will be shown to depict which models have resulted in nans for the given
                performance matric.
            name str:
                name of the saved file.
            kwargs :
                fig_height:
                fig_width:
                title_fs:
                xlabel_fs:

        returns:
            dictionary whose keys are models and values are performance metrics.

        Example
        -----------
        ```python
        >>>from ai4water.experiments import MLRegressionExperiments
        >>>from ai4water.utils.datasets import arg_beach
        >>>data = arg_beach()
        >>>inputs = list(data.columns)[0:-1]
        >>>outputs = list(data.columns)[-1]
        >>>experiment = MLRegressionExperiments(data=data, input_features=inputs, output_features=outputs)
        >>>experiment.fit()
        >>>experiment.compare_errors('mse')
        >>>experiment.compare_errors('r2', 0.2, 'greater')
        ```
        """

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

        labels = {
            'r2': "$R^{2}$",
            'nse': 'NSE',
            'rmse': 'RMSE',
            'mse': 'MSE',
            'msle': 'MSLE',
            'nrmse': 'Normalized RMSE',
            'mape': 'MAPE'
        }

        if len(models) <=1:
            warnings.warn(f"Comparison can not be plotted because the obtained models are <=1 {models}", UserWarning)
            return models

        if sort_by == 'test':
            if ignore_nans:
                d = {key: [a, _name] for key, a, _name in zip(test_matrics, train_matrics, list(models.keys())) if
                     not np.isnan(key)}
            else:
                d = {key: [a, _name] for key, a, _name in zip(test_matrics, train_matrics, list(models.keys()))}
        elif sort_by == 'train':
            if ignore_nans:
                d = {key: [a, _name] for key, a, _name in zip(train_matrics, test_matrics, list(models.keys())) if
                     not np.isnan(key)}
            else:
                d = {key: [a, _name] for key, a, _name in zip(train_matrics, test_matrics, list(models.keys()))}
        else:
            raise ValueError(f'sort_by must be either train or test but it is {sort_by}')

        models = dict(sorted(d.items(), reverse=True))
        names = [i[1] for i in models.values()]
        test_matrics = list(models.keys())
        train_matrics = [i[0] for i in models.values()]

        plt.close('all')
        fig, axis = plt.subplots(1, 2, sharey='all')
        fig.set_figheight(kwargs.get('fig_height', 8))
        fig.set_figwidth(kwargs.get('fig_width', 8))

        ax = sns.barplot(y=names, x=train_matrics, orient='h', ax=axis[0])
        ax.set_title("Train", fontdict={'fontsize': kwargs.get('title_fs', 20)})
        ax.set_xlabel(labels.get(matric_name, matric_name), fontdict={'fontsize': kwargs.get('xlabel_fs', 16)})

        ax = sns.barplot(y=names, x=test_matrics, orient='h', ax=axis[1])
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel(labels.get(matric_name, matric_name), fontdict={'fontsize': kwargs.get('xlabel_fs', 16)})
        ax.set_title("Test", fontdict={'fontsize': kwargs.get('title_fs', 20)})

        if save:
            fname = os.path.join(os.getcwd(),f'results{SEP}{self.exp_name}{SEP}{name}_{matric_name}.png')
            plt.savefig(fname, dpi=100, bbox_inches=kwargs.get('bbox_inches', 'tight'))
        plt.show()
        return models

    def plot_losses(self,
                    loss_name:Union[str, list]='loss',
                    save:bool=True,
                    name:str='loss_comparison',
                    **kwargs):
        """Plots the loss curves of the evaluated models.
        Arguments:
            loss_name : the name of loss value, must be recorded during training
            save : whether to save the plot or not
            name : name of saved file
            kwargs : following keyword arguments can be used
                width:
                height:
                bbox_inches:
        """

        if not isinstance(loss_name, list):
            assert isinstance(loss_name, str)
            loss_name = [loss_name]
        loss_curves = {}
        for _model, _path in self.config['eval_models'].items():
            df = pd.read_csv(os.path.join(_path, 'losses.csv'), usecols=loss_name)
            loss_curves[_model] = df

        _kwargs = {'linestyle': '-',
                   'x_label':"Epochs",
                   'y_label':'Loss'}

        if len(loss_curves)>5:
            _kwargs['bbox_to_anchor'] = (1.1, 0.99)

        fig, axis = init_subplots(kwargs.get('width', None), kwargs.get('height', None))

        for _model, _loss in loss_curves.items():
            if _model.startswith('model_'):
                _model = _model.split('model_')[1]
            axis = process_axis(axis=axis, data=_loss, label=_model, **_kwargs)

        if save:
            fname = os.path.join(self.exp_path,f'{name}_{loss_name}.png')
            plt.savefig(fname, dpi=100, bbox_inches=kwargs.get('bbox_inches', 'tight'))
        plt.show()
        return

    def plot_convergence(self,
                         save:bool=False,
                         name='convergence_comparison',
                         **kwargs):
        """
        Plots the convergence plots of hyperparameter optimization runs.
        Only valid if `fit` was run with `run_type=optimize`.

        Arguments:
            save : whether to save the plot or not
            name : name of file to save the plot
            kwargs : keyword arguments like:
                bbox_inches :
        """
        if len(self.config['optimized_models']) <1:
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
                         x_label='Number of iterations $n$',
                         y_label=r"$\min f(x)$ after $n$ calls")
        if save:
            fname = os.path.join(self.exp_path,f'{name}.png')
            plt.savefig(fname, dpi=100, bbox_inches=kwargs.get('bbox_inches', 'tight'))
        plt.show()
        return

    @classmethod
    def from_config(cls, config_path:str, **kwargs)->"Experiments":
        """Loads the experiment from the config file.
        Arguments:
            config_path : complete path of experiment
            kwargs : keyword arguments to experiment
        Returns:
            an instance of Experiments class
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

        for model_name, model_path in config['eval_models'].items():

            with open(os.path.join(model_path, 'config.json'), 'r') as fp:
                model_config = json.load(fp)

            output_features = model_config['config']['output_features']

            if len(output_features)>1:
                raise NotImplementedError

            fpath = os.path.join(model_path, output_features[0])
            fname = [f for f in os.listdir(fpath) if f.startswith('training') and f.endswith('.csv')][0]
            train_df = pd.read_csv(os.path.join(fpath, fname), index_col='index')

            fname = [f for f in os.listdir(fpath) if f.startswith('test') and f.endswith('.csv')][0]
            test_df = pd.read_csv(os.path.join(fpath, fname), index_col='index')

            trues['train'][model_name] = train_df[f'true_{output_features[0]}']
            trues['test'][model_name] = test_df[f'true_{output_features[0]}']

            simulations['train'][model_name] = train_df[f'pred_{output_features[0]}']
            simulations['test'][model_name] = test_df[f'pred_{output_features[0]}']

        cls.trues = trues
        cls.simulations = simulations

        return cls(exp_name=config['exp_name'], cases=config['cases'], **kwargs)


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
                 num_samples=10,
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
        >>>from ai4water.utils.datasets import arg_beach
        >>>from ai4water.experiments import MLRegressionExperiments
        >>> # first compare the performance of all available models without optimizing their parameters
        >>>data = arg_beach()  # read data file, in this case load the default data
        >>>inputs = list(data.columns)[0:-1]  # define input and output columns in data
        >>>outputs = list(data.columns)[-1]
        >>>comparisons = MLRegressionExperiments(data=data, input_features=inputs, output_features=outputs,
        ...                                      input_nans={'SimpleImputer': {'strategy':'mean'}} )
        >>>comparisons.fit(run_type="dry_run")
        >>>comparisons.compare_errors('r2')
        >>> # find out the models which resulted in r2> 0.5
        >>>best_models = comparisons.compare_errors('r2', cutoff_type='greater', cutoff_val=0.3)
        >>>best_models = [m[1] for m in best_models.values()]
        >>> # now build a new experiment for best models and otpimize them
        >>>comparisons = MLRegressionExperiments(data=data, inputs_features=inputs, output_features=outputs,
        ...                                   input_nans={'SimpleImputer': {'strategy': 'mean'}}, exp_name="BestMLModels")
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

        """Builds and run one 'model' of the experiment.
        Since and experiment consists of many models, this method
        is also run many times. """

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

        model.fit(**fit_kws)

        if cross_validate:
            to_return = model.cross_val_score(model.config['val_metric'])
        else:
            vt, vp = model.predict('validation')
            to_return =  getattr(RegressionMetrics(vt, vp), model.config['val_metric'])()

        tt, tp = model.predict('test')

        if view:
            model.view_model()

        if predict:
            t, p = model.predict('training')

            return (t,p), (tt, tp)

        return to_return

    def model_ADABoostRegressor(self, **kwargs):
        ## https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
        self.param_space = [
            Integer(low=5, high=100, name='n_estimators', num_samples=self.num_samples),
            Real(low=0.001, high=1.0, name='learning_rate', num_samples=self.num_samples)
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
            Real(low=0.0001, high=0.5, name='learning_rate', num_samples=self.num_samples), # Used for reducing the gradient step.
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
        self.param_space = [
            Integer(low=5, high=50, name='n_estimators', num_samples=self.num_samples),
            Integer(low=3, high=30, name='max_depth', num_samples=self.num_samples),
            Real(low=0.1, high=0.5, name='min_samples_split', num_samples=self.num_samples),
            #Real(low=0.1, high=1.0, name='min_samples_leaf'),
            Real(low=0.0, high=0.5, name='min_weight_fraction_leaf', num_samples=self.num_samples),
            Categorical(categories=['auto', 'sqrt', 'log2'], name='max_features')
        ]
        self.x0 = [10, 5,  0.4, #0.2,
                   0.1, 'auto']
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
            Real(low=0.001, high=1.0, name='learning_rate', num_samples=self.num_samples),   #  shrinks the contribution of each tree
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
            Real(low=0.0001, high=0.9, name='learning_rate', num_samples=self.num_samples),  # Used for reducing the gradient step.
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
            Real(low=0.0001, high=0.1, name='learning_rate', num_samples=self.num_samples),
            Integer(low=20, high=500, name='n_estimators', num_samples=self.num_samples)
        ]
        self.x0 = ['gbdt', 50, 0.001, 20]
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
            Real(low=0.0001, high=0.5, name='learning_rate', num_samples=self.num_samples),     #
            #Categorical(categories=['gbtree', 'gblinear', 'dart'], name='booster'),  # todo solve error
            Real(low=0.1, high=0.9, name='gamma', num_samples=self.num_samples),  # Minimum loss reduction required to make a further partition on a leaf node of the tree.
            Real(low=0.1, high=0.9, name='min_child_weight ', num_samples=self.num_samples),  # Minimum sum of instance weight(hessian) needed in a child.
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
        self.x0 = None #[50, 5, 0.5, 'gbtree', 0.2, 0.2, 0.2,
                   #0.2, 0.2, 0.2, 0.2, 0.2, 0.2
                   #]
        return {'model': {'XGBOOSTREGRESSOR': kwargs}}



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

        t, p = model.predict()

        if view:
            model.view_model()

        if predict:
            tt, tp = model.predict('training')

        return RegressionMetrics(t, t).mse()

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
            Real(low=0.0001, high=0.1, name='learning_rate', num_samples=self.num_samples),
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
            Real(low=0.0001, high=0.5, name='learning_rate', num_samples=self.num_samples),  #
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


class TransformationExperiments(Experiments):
    """Helper to conduct experiments with different transformations
    Examples
    --------
    >>>from ai4water.utils.datasets import load_u1
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
    >>>data = load_u1()
    >>>inputs = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
    >>>outputs = ['target']
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
                 cases:dict=None,
                 exp_name:str=None,
                 num_samples:int=5,
                 ai4water_model=None,
                 **model_kws):
        self.data = data
        self.param_space = param_space
        self.x0 = x0
        self.model_kws = model_kws
        self.ai4water_model = Model if ai4water_model is None else ai4water_model

        exp_name = exp_name or 'TransformationExperiments' + f'_{dateandtime_now()}'

        super().__init__(cases=cases,
                         exp_name=exp_name,
                         num_samples=num_samples)

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

        if fit_kws is None:
            fit_kws = {}

        verbosity = 0
        if 'verbosity' in self.model_kws:
            verbosity = self.model_kws.pop('verbosity')

        model = self.ai4water_model(
            data=self.data,
            prefix=title,
            verbosity=verbosity,
            **self.update_paras(**suggested_paras),
            **self.model_kws
        )

        setattr(self, '_model', model)

        model = self.process_model_before_fit(model)

        history = model.fit(**fit_kws)

        if predict:
            trt, trp = model.predict('training')

            testt, testp = model.predict()

            model.config['allow_nan_labels'] = 2
            model.predict()
            # model.plot_train_data()
            return (trt, trp), (testt, testp)

        target_matric_array = history.history['val_loss']

        if all(np.isnan(target_matric_array)):
            raise ValueError(f"""
Validation loss during all the epochs is NaN. Suggested parameters were
{suggested_paras}
""")
        return np.nanmin(target_matric_array)

    def build_from_config(self, config_path, weight_file, fit_kws, **kwargs):

        if fit_kws is None:
            fit_kws = {}

        model = self.ai4water_model.from_config(config_path=config_path, data=self.data
                                              )
        model.update_weights(weight_file=weight_file)

        model = self.process_model_before_fit(model)

        train_true, train_pred = model.predict('training')

        test_true, test_pred = model.predict('test')

        model.data['allow_nan_labels'] = 1
        model.predict()

        #model.plot_layer_outputs()
        #model.plot_weights()

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
