import warnings
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_roc_curve, plot_confusion_matrix, plot_precision_recall_curve
from sklearn import tree
from dl4seq.backend import xgboost
import matplotlib as mpl
import matplotlib.ticker as ticker

try:
    from dtreeviz import trees
except ModuleNotFoundError:
    trees = None

from dl4seq.utils.transformations import Transformations
from dl4seq.utils.utils import _missing_vals
from dl4seq.utils.TSErrors import FindErrors

try:
    from dl4seq.utils.utils_from_see_rnn import rnn_histogram
    from see_rnn.visuals_gen import features_0D, features_1D, features_2D
except ModuleNotFoundError:
    rnn_histogram = None
    features_0D, features_1D, features_2D = None, None, None

# TODO
# rank histogram, reliability diagram, ROC curve


rnn_info = {"LSTM": {'rnn_type': 'LSTM',
                     'gate_names': ['INPUT', 'FORGET', 'CELL', 'OUTPUT'],
                     'n_gates': 4,
                     'is_bidir': False,
                     'rnn_dim': 64,
                     'uses_bias': True,
                     'direction_names': [[]]}}

class Plots(object):
    # TODO initialte this class with at least path

    def __init__(self, path, problem, category, model, config):
        self.path = path
        self.problem = problem
        self.category=category
        self._model = model
        self.config = config

    @property
    def train_data(self):
        raise AttributeError

    @property
    def val_data(self):
        raise AttributeError

    @property
    def test_data(self):
        raise AttributeError

    @property
    def data(self):
        raise AttributeError

    @property
    def in_cols(self):
        return self.config['inputs']

    @property
    def out_cols(self):
        return self.config['outputs']

    @property
    def ins(self):
        return len(self.config['inputs'])

    @property
    def outs(self):
        return self.config['outputs']

    @property
    def lookback(self):
        return self.config['lookback']

    def feature_imporance(self):
        if self.category.upper() == "ML":

            model_name = list(self.config['model'].keys())[0]
            if model_name.upper() in ["SVC", "SVR"]:
                if self._model.kernel == "linear":
                    # https://stackoverflow.com/questions/41592661/determining-the-most-contributing-features-for-svm-classifier-in-sklearn
                    return self._model.coef_
            elif hasattr(self._model, "feature_importances_"):
                return self._model.feature_importances_

    def plot_data(self, save=True, freq=None, cols=None, max_subplots=10, **kwargs):
        """
        :param save:
        :param max_subplots: int, number of subplots within one plot. Each feature will be shown in a separate subplot.
        :param kwargs: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html
        :param freq: str, one of 'daily', 'weekly', 'monthly', 'yearly', determines interval of plot of data. It is
                     valid for only time-series data.
        :param cols: columns in self.data to plot
        :return:

        ------
        examples:

        >>>model.plot_data(subplots=True, figsize=(12, 14), sharex=True)
        >>>model.plot_data(freq='monthly', subplots=True, figsize=(12, 14), sharex=True)
        """
        if isinstance(self.data, pd.DataFrame):
            self.plot_df(self.data, cols=cols, save=save, freq=freq, max_subplots=max_subplots, **kwargs)

        if isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                self.plot_df(data, cols=cols[idx], save=save, freq=freq, prefix=str(idx), max_subplots=max_subplots, **kwargs)
        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                if isinstance(data, pd.DataFrame):
                    self.plot_df(data, cols=cols, prefix=data_name, save=save, freq=freq, max_subplots=max_subplots,
                                 **kwargs)
        return

    def plot_df(self, df, cols=None, save=True, freq=None, max_subplots=10,
                prefix='',
                leg_kws=None,
                label_kws=None,
                tick_kws=None,
                **kwargs):
        """Plots each columns of dataframe and saves it if `save` is True.
         max_subplots: determines how many sub_plots are to be plotted within one plot. If dataframe contains columns
         greater than max_subplots, a separate plot will be generated for remaining columns.
         """
        assert isinstance(df, pd.DataFrame)
        if leg_kws is None:
            leg_kws = {'fontsize': 14}
        if label_kws is None:
            label_kws = {'fontsize': 14}
        if tick_kws is None:
            tick_kws = {'axis':"both", 'which':'major', 'labelsize':12}

        if cols is None:
            cols = list(df.columns)
        df = df[cols]

        if df.shape[1] <= max_subplots:

            if freq is None:
                kwargs = plot_style(df, **kwargs)
                axis = df.plot(**kwargs)
                if isinstance(axis, np.ndarray):
                    for ax in axis:
                        set_axis_paras(ax, leg_kws, label_kws, tick_kws)
                else:
                    set_axis_paras(axis, leg_kws, label_kws, tick_kws)

                self.save_or_show(save=save, fname=f"input_{prefix}",  where='data')
            else:
                self.plot_df_with_freq(df, freq, save, **kwargs)
        else:
            tot_plots = find_tot_plots(df.shape[1], max_subplots)

            for i in range(len(tot_plots) - 1):
                st, en = tot_plots[i], tot_plots[i + 1]
                sub_df = df.iloc[:, st:en]

                if freq is None:
                    kwargs = plot_style(sub_df, **kwargs)
                    axis = sub_df.plot(**kwargs)
                    for ax in axis:
                        ax.legend(**leg_kws)
                        ax.set_ylabel(ax.get_ylabel(), **label_kws)
                        ax.set_xlabel(ax.get_xlabel(), **label_kws)
                        ax.tick_params(**tick_kws)
                    self.save_or_show(save=save, fname=f'input_{prefix}_{st}_{en}', where='data')
                else:
                    self.plot_df_with_freq(sub_df, freq, save, prefix=f'{prefix}_{st}_{en}', **kwargs)
        return

    def plot_df_with_freq(self, df:pd.DataFrame, freq:str, save:bool=True, prefix:str='', **kwargs):
        """Plots a dataframe which has data as time-series and its index is pd.DatetimeIndex"""

        validate_freq(df, freq)

        st_year = df.index[0].year
        en_year = df.index[-1].year

        for yr in range(st_year, en_year + 1):

            _df = df[df.index.year == yr]

            if freq == 'yearly':
                kwargs = plot_style(_df, **kwargs)
                _df.plot(**kwargs)
                self.save_or_show(save=save, fname=f'input_{prefix}_{str(yr)}', where='data')

            elif freq == 'monthly':
                st_mon = _df.index[0].month
                en_mon = _df.index[-1].month

                for mon in range(st_mon, en_mon+1):

                    __df = _df[_df.index.month == mon]
                    kwargs = plot_style(__df, **kwargs)
                    __df.plot(**kwargs)
                    self.save_or_show(save=save, fname=f'input_{prefix}_{str(yr)} _{str(mon)}', where='data/monthly')

            elif freq == 'weekly':
                st_week = _df.index[0].isocalendar()[1]
                en_week = _df.index[-1].isocalendar()[1]

                for week in range(st_week, en_week+1):
                    __df = _df[_df.index.week == week]
                    kwargs = plot_style(__df, **kwargs)
                    __df.plot(**kwargs)
                    self.save_or_show(save=save, fname=f'input_{prefix}_{str(yr)} _{str(week)}', where='data/weely')
        return

    def _imshow_3d(self, activation,
                   lyr_name,
                   save=True,
                   where='activations'):

        act_2d = []
        for i in range(activation.shape[2]):
            act_2d.append(activation[:, :, i])
        activation_2d = np.concatenate(act_2d, axis=1)
        self._imshow(activation_2d,
                     lyr_name + " Activations (3d of {})".format(activation.shape),
                     save,
                     lyr_name,
                     where=where,
                     xlabel=self.in_cols)
        return

    def _imshow(self, img,
                label: str = '',
                save=True,
                fname=None,
                interpolation:str='none',
                where='activations',
                rnn_args=None,
                **kwargs):

        assert np.ndim(img) == 2, "can not plot {} with shape {} and ndim {}".format(label, img.shape, np.ndim(img))
        plt.close('all')
        plt.rcParams.update(plt.rcParamsDefault)
        plt.imshow(img, aspect='auto', interpolation=interpolation)

        if rnn_args is not None:
            assert isinstance(rnn_args, dict)

            rnn_dim = int(img.shape[1] / rnn_args['n_gates'])
            [plt.axvline(rnn_dim * gate_idx - .5, linewidth=0.5, color='k')
                     for gate_idx in range(1, rnn_args['n_gates'])]

            plt.xlabel(rnn_args['gate_names_str'])
            if "RECURRENT" in label.upper():
                plt.ylabel("Hidden Units")
            else:
                plt.ylabel("Channel Units")
        else:
            plt.ylabel('Examples' if 'weight' not in label.lower() else '')
            plt.xlabel(kwargs.get('xlabel', ''))

        plt.colorbar()
        plt.title(label)
        self.save_or_show(save, fname, where=where)

        return

    def plot1d(self, array, label: str = '', save=True, fname=None, rnn_args=None, where='activations'):
        plt.close('all')
        plt.style.use('ggplot')
        plt.plot(array)
        plt.xlabel("Examples")
        plt.title(label)
        plt.grid(b=True, which='major', color='0.2', linestyle='-')

        if rnn_args is not None:
            assert isinstance(rnn_args, dict)

            rnn_dim = int(array.shape[0] / rnn_args['n_gates'])
            [plt.axvline(rnn_dim * gate_idx - .5, linewidth=0.5, color='k')
                     for gate_idx in range(1, rnn_args['n_gates'])]
            plt.xlabel(rnn_args['gate_names_str'])

        self.save_or_show(save, fname, where=where)

        return

    def save_or_show(self, save: bool = True, fname=None, where='', dpi=300, bbox_inches='tight', close=True):

        if save:
            assert isinstance(fname, str)
            if "/" in fname:
                fname = fname.replace("/", "__")
            if ":" in fname:
                fname = fname.replace(":", "__")

            save_dir = os.path.join(self.path, where)

            if not os.path.exists(save_dir):
                assert os.path.dirname(where) in ['', 'activations', 'weights', 'plots', 'data', 'results'], f"unknown directory: {where}"
                save_dir = os.path.join(self.path, where)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

            fname = os.path.join(save_dir, fname + ".png")

            plt.savefig(fname, dpi=dpi, bbox_inches=bbox_inches)
        else:
            plt.show()

        if close:
            plt.close('all')
        return

    def plot2d_act_for_a_sample(self, activations, sample=0, save:bool = False, name: str = None):
        fig, axis = plt.subplots()
        fig.set_figheight(8)
        # for idx, ax in enumerate(axis):
        im = axis.imshow(activations[sample, :, :].transpose(), aspect='auto')
        axis.set_xlabel('lookback')
        axis.set_ylabel('inputs')
        print(self.in_cols)
        axis.set_title('Activations of all inputs at different lookbacks for sample ' + str(sample))
        fig.colorbar(im)
        self.save_or_show(save=save, fname=name + '_' + str(sample), where='path')

        return

    def plot1d_act_for_a_sample(self, activations, sample=0, save=False, name=None):
        fig, axis = plt.subplots()

        for idx in range(self.lookback-1):
            axis.plot(activations[sample, idx, :].transpose(), label='lookback '+str(idx))
        axis.set_xlabel('inputs')
        axis.set_ylabel('activation weight')
        axis.set_title('Activations at different lookbacks for all inputs for sample ' + str(sample))
        self.save_or_show(save=save, fname=name + '_' + str(sample), where='path')

        return

    def plot_train_data(self, how='3d', save=True,  **kwargs):

        x, _, y = self.train_data(**kwargs)
        self.plot_model_input_data(x, how=how, save=save, which='training')

        return

    def plot_val_data(self, how='3d', save=True,  **kwargs):

        x, _, y = self.val_data(**kwargs)
        self.plot_model_input_data(x, how=how, save=save, which='validation')

        return

    def plot_test_data(self, how='3d', save=True,  **kwargs):

        x, _, y = self.test_data(**kwargs)
        self.plot_model_input_data(x, how=how, save=save, which='test')

        return

    def plot_model_input_data(self, in_data, how, save, which='training'):

        for idx, inputs in enumerate(in_data):
            if np.ndim(inputs) == 3:
                if how.upper() == "3D":
                    self._imshow_3d(inputs, which + '_data_' + str(idx), save=save, where='data')
            elif np.ndim(inputs) == 2:
                if how.upper() == "3D":
                    self._imshow(inputs, save=save, fname= which + '_data_' + str(idx), where='data')
        return

    def features_2d(self, data, name, save=True, slices=32, slice_dim=0, where='activations', **kwargs):
        """Calls the features_2d from see-rnn"""
        st=0
        if features_2D is None:
            warnings.warn("install see-rnn to plot features-2D plot", UserWarning)
        else:
            for en in np.arange(slices, data.shape[slice_dim] + slices, slices):

                if save:
                    fname = name + f"_{st}_{en}"
                    save = os.path.join(os.path.join(self.path, where), fname+".png")
                else:
                    save = None

                if slice_dim == 0:
                    features_2D(data[st:en, :], savepath=save, **kwargs)
                else:
                    # assuming it will always be the last dim if not first
                    features_2D(data[..., st:en], savepath=save, **kwargs)
                st=en
        return

    def features_1d(self, data, save=True, name='', **kwargs):

        if save:
            save = os.path.join(self.act_path, name + ".png")
        else:
            save=None

        if features_1D is None:
            warnings.warn("install see-rnn to plot features-1D plot", UserWarning)
        else:
            features_1D(data, savepath=save, **kwargs)

        return

    def features_0d(self, data, save=True, name='', **kwargs):
        if save:
            save = os.path.join(self.act_path, name + "0D.png")
        else:
            save=None

        if features_0D is None:
            warnings.warn("install see-rnn to plot 0D-features plot")
        else:
            return features_0D(data, savepath=save, **kwargs)

    def rnn_histogram(self, data, save=True, name='', **kwargs):

        if save:
            save = os.path.join(self.act_path, name + "0D.png")
        else:
            save=None

        if rnn_histogram is None:
            warnings.warn("install see-rnn to plot rnn_histogram plot", UserWarning)
        else:
            rnn_histogram(data, rnn_info["LSTM"], bins=400, savepath=save, **kwargs)

        return

    def plot_quantiles2(self, true_outputs, predicted, st=0, en=None, save=True):
        plt.close('all')
        plt.style.use('ggplot')

        if en is None:
            en = true_outputs.shape[0]
        for q in range(len(self.quantiles) - 1):
            st_q = "{:.1f}".format(self.quantiles[q] * 100)
            en_q = "{:.1f}".format(self.quantiles[q + 1] * 100)

            plt.plot(np.arange(st, en), true_outputs[st:en, 0], label="True", color='navy')
            plt.fill_between(np.arange(st, en), predicted[st:en, q], predicted[st:en, q + 1], alpha=0.2,
                             color='g', edgecolor=None, label=st_q + '_' + en_q)
            plt.legend(loc="best")
            self.save_or_show(save, fname='q' + st_q + '_' + en_q + ".png", where='results')
        return

    def plot_quantile(self, true_outputs, predicted, min_q: int, max_q, st=0, en=None, save=False):
        plt.close('all')
        plt.style.use('ggplot')

        if en is None:
            en = true_outputs.shape[0]
        q_name = "{:.1f}_{:.1f}_{}_{}".format(self.quantiles[min_q] * 100, self.quantiles[max_q] * 100, str(st),
                                              str(en))

        plt.plot(np.arange(st, en), true_outputs[st:en, 0], label="True", color='navy')
        plt.fill_between(np.arange(st, en), predicted[st:en, min_q], predicted[st:en, max_q], alpha=0.2,
                         color='g', edgecolor=None, label=q_name + ' %')
        plt.legend(loc="best")
        self.save_or_show(save, fname= "q_" + q_name + ".png", where='results')
        return

    def plot_all_qs(self, true_outputs, predicted, save=False):
        plt.close('all')
        plt.style.use('ggplot')

        st, en = 0, true_outputs.shape[0]

        plt.plot(np.arange(st, en), true_outputs[st:en, 0], label="True", color='navy')

        for idx, q in enumerate(self.quantiles):
            q_name = "{:.1f}".format(self.quantiles[idx] * 100)
            plt.plot(np.arange(st, en), predicted[st:en, idx], label="q {} %".format(q_name))

        plt.legend(loc="best")
        self.save_or_show(save, fname="all_quantiles", where='results')

        return

    def plot_quantiles1(self, true_outputs, predicted, st=0, en=None, save=True):
        plt.close('all')
        plt.style.use('ggplot')

        if en is None:
            en = true_outputs.shape[0]
        for q in range(len(self.quantiles) - 1):
            st_q = "{:.1f}".format(self.quantiles[q] * 100)
            en_q = "{:.1f}".format(self.quantiles[-q] * 100)

            plt.plot(np.arange(st, en), true_outputs[st:en, 0], label="True", color='navy')
            plt.fill_between(np.arange(st, en), predicted[st:en, q], predicted[st:en, -q], alpha=0.2,
                             color='g', edgecolor=None, label=st_q + '_' + en_q)
            plt.legend(loc="best")
            self.save_or_show(save, fname='q' + st_q + '_' + en_q, where='results')
        return

    def plot_feature_importance(self, importance=None, save=True, use_xgb=False, **kwargs):

        if importance is None:
            importance = self.feature_imporance()

        if self.category == "ML":
            model_name = list(self.config['model'].keys())[0]
            if model_name.upper() in ["SVC", "SVR"]:
                if self._model.kernel == "linear":
                    return self.f_importances_svm(importance, self.in_cols, save=save)
                else:
                    warnings.warn(f"for {self._model.kernel} kernels of {model_name}, feature importance can not be plotted.")
                return
            return 

        if isinstance(importance, np.ndarray):
            assert importance.ndim <= 2

        use_prev = self.config['use_predicted_output']
        all_cols = self.config['inputs'] if use_prev else self.config['inputs'] + \
                                                                                self.config['outputs']
        plt.close('all')
        plt.figure()
        plt.title("Feature importance")
        if use_xgb:
            if xgboost is None:
                warnings.warn("install xgboost to plot plot_importance using xgboost", UserWarning)
            else:
                xgboost.plot_importance(self._model, **kwargs)
        else:
            plt.bar(range(self.ins if use_prev else self.ins + self.outs), importance, **kwargs)
            plt.xticks(ticks=range(len(all_cols)), labels=list(all_cols), rotation=90, fontsize=5)
        self.save_or_show(save, fname="feature_importance.png")
        return

    def plot_feature_feature_corr(self, cols=None, remove_targets=True, save=True, **kwargs):

        if cols is None:
            cols = self.in_cols if remove_targets else self.in_cols + self.out_cols

        if isinstance(self.data, pd.DataFrame):
            self._feature_feature_corr(self.data, cols, save=save, **kwargs)

        elif isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                if isinstance(data, pd.DataFrame):
                    self._feature_feature_corr(data, cols[idx], prefix=str(idx), save=save, **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                if isinstance(data, pd.DataFrame):
                    self._feature_feature_corr(data, cols, prefix=data_name, save=save, **kwargs)
        return

    def _feature_feature_corr(self,
                              data,
                              cols=None,
                              prefix='',
                              save=True,
                              split=None,
                              threshold=0,
                              figsize=(20,20),
                              **kwargs):
        """
        split : Optional[str], optional
        Type of split to be performed {None, "pos", "neg", "high", "low"}, by default None
        method : str, optional
                 {"pearson", "spearman", "kendall"}, by default "pearson"

        kwargs
            * vmax: float, default is calculated from the given correlation \
                coefficients.
                Value between -1 or vmin <= vmax <= 1, limits the range of the cbar.
            * vmin: float, default is calculated from the given correlation \
                coefficients.
                Value between -1 <= vmin <= 1 or vmax, limits the range of the cbar.
        To plot positive correlation only:
        _feature_feature_corr(model.data, list(model.data.columns), split="pos")
        To plot negative correlation only
        _feature_feature_corr(model.data, list(model.data.columns), split="neg")
        """
        plt.close('all')
        method = kwargs.get('method', 'pearson')

        if cols is None:
            cols = data.columns

        corr = data[cols].corr(method=method)

        if split == "pos":
            corr = corr.where((corr >= threshold) & (corr > 0))
        elif split == "neg":
            corr = corr.where((corr <= threshold) & (corr < 0))

        mask = np.zeros_like(corr, dtype=np.bool)

        vmax = np.round(np.nanmax(corr.where(~mask)) - 0.05, 2)
        vmin = np.round(np.nanmin(corr.where(~mask)) + 0.05, 2)

        fig, ax = plt.subplots(figsize=figsize)

        _kwargs = dict()
        _kwargs['annot'] = kwargs.get('annot', True if len(cols) <= 20 else False)
        _kwargs['cmap'] = kwargs.get('cmap', "BrBG")
        _kwargs['vmax'] = kwargs.get('vmax', vmax)
        _kwargs['vmin'] = kwargs.get('vmin', vmin)
        _kwargs['linewidths'] = kwargs.get('linewidths', 0.5)
        _kwargs['annot_kws'] = kwargs.get('annot_kws', {"size": 10})
        _kwargs['cbar_kws'] = kwargs.get('cbar_kws', {"shrink": 0.95, "aspect": 30})

        ax = sns.heatmap(corr, center=0, fmt=".2f", ax=ax, **_kwargs)
        ax.set(frame_on=True)
        self.save_or_show(save, fname=f"{split if split else ''}_feature_corr_{prefix}", where="data")
        return

    def roc_curve(self, x, y, save=True):
        assert self.problem.upper().startswith("CLASS")
        plot_roc_curve(self._model, *x, y.reshape(-1, ))
        self.save_or_show(save, fname="roc", where="results")
        return

    def confusion_matrx(self, x, y, save=True):
        assert self.problem.upper().startswith("CLASS")
        plot_confusion_matrix(self._model, *x, y.reshape(-1, ))
        self.save_or_show(save, fname="confusion_matrix", where="results")
        return

    def precision_recall_curve(self, x, y, save=True):
        assert self.problem.upper().startswith("CLASS")
        plot_precision_recall_curve(self._model, *x, y.reshape(-1, ))
        self.save_or_show(save, fname="plot_precision_recall_curve", where="results")
        return

    def decision_tree(self, which="sklearn", save=True, **kwargs):
        """For kwargs see https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.plot_tree"""
        plt.close('')
        if which == "sklearn":
            if hasattr(self._model, "tree_"):
                tree.plot_tree(self._model, **kwargs)
        else:  # xgboost
            if xgboost is None:
                warnings.warn("install xgboost to use plot_tree method")
            else:
                xgboost.plot_tree(self._model, **kwargs)
        self.save_or_show(save, fname="decision_tree", where="results")
        return

    def plot_treeviz_leaves(self, save=True, **kwargs):
        """Plots dtreeviz related plots if dtreeviz is installed"""

        model = list(self.config['model'].keys())[0].upper()
        if model in ["DECISIONTREEREGRESSON", "DECISIONTREECLASSIFIER"] or model.startswith("XGBOOST"):
            if trees is None:
                print("dtreeviz related plots can not be plotted")
            elif model in ['XGBOOSTRFREGRESSOR']:  # dtreeviz doesn't plot this
                pass
            else:
                x, _, y = self.test_data()

                if np.ndim(y) > 2:
                    y = np.squeeze(y, axis=2)

                trees.viz_leaf_samples(self._model, *x, self.in_cols)
                self.save_or_show(save, fname="viz_leaf_samples", where="plots")

                trees.ctreeviz_leaf_samples(self._model, *x, y, self.in_cols)
                self.save_or_show(save, fname="ctreeviz_leaf_samples", where="plots")

    def plot_results(self, true, predicted:pd.DataFrame, save=True, name=None, where=None):
        """
        # kwargs can be any/all of followings
            # fillstyle:
            # marker:
            # linestyle:
            # markersize:
            # color:
        """

        self.regplot_using_searborn(true, predicted, save=save, name=name, where=where)
        plt.close('all')
        mpl.rcParams.update(mpl.rcParamsDefault)

        fig, axis = plt.subplots()
        set_fig_dim(fig, 12, 8)

        # it is quite possible that when data is datetime indexed, then it is not equalidistant and large amount of graph
        # will have not data in that case lines plot will create a lot of useless interpolating lines where no data is present.
        style = '.' if isinstance(true.index, pd.DatetimeIndex) else '-'

        if np.isnan(true.values).sum() > 0:
            style = '.' # For Nan values we should be using this style otherwise nothing is plotted.

        ms = 4 if style == '.' else 2

        axis.plot(predicted, style, color='r', linestyle='-', marker='', label='Prediction')

        axis.plot(true, style, color='b', marker='o', fillstyle='none',  markersize=ms, label='True')

        axis.legend(loc="best", fontsize=22, markerscale=4)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel("Time", fontsize=18)

        self.save_or_show(save=save, fname=name, close=False, where=where)
        return

    def regplot_using_searborn(self, true, pred, save, name, where='plots', **kwargs):
        # https://seaborn.pydata.org/generated/seaborn.regplot.html
        if any([isinstance(true, _type) for _type in [pd.DataFrame, pd.Series]]):
            true = true.values.reshape(-1,)
        if any([isinstance(pred, _type) for _type in [pd.DataFrame, pd.Series]]):
            pred = pred.values.reshape(-1,)

        plt.close('all')

        s = kwargs.get('s', 20)
        cmap = kwargs.get('cmap', 'Spectral')
        figsize = kwargs.get('figsize', (8, 5.5))

        plt.figure(figsize=figsize)
        points = plt.scatter(true, pred, c=pred, s=s, cmap=cmap)  # set style options

        if kwargs.get('annotate', True):
            plt.annotate(f'$R^{2}$: {round(FindErrors(true, pred).r2(), 3)}', xy=(0.50, 0.95), xycoords='axes fraction',
                     horizontalalignment='right', verticalalignment='top', fontsize=16)

        if kwargs.get('colorbar', False):
            plt.colorbar(points)

        sns.regplot(x=true, y=pred, scatter=False, color=".1")
        plt.xlabel('Observed', fontsize=14)
        plt.ylabel('Predicted', fontsize=14)

        self.save_or_show(save=save, fname=name + "_reg", close=False, where=where)
        return

    def f_importances_svm(self, coef, names, save):

        plt.close('all')
        mpl.rcParams.update(mpl.rcParamsDefault)
        classes = coef.shape[0]
        features = coef.shape[1]
        fig, axis = plt.subplots(classes, sharex='all')
        axis = axis if hasattr(axis, "__len__") else [axis]

        for idx, ax in enumerate(axis):
            colors = ['red' if c < 0 else 'blue' for c in self._model.coef_[idx]]
            ax.bar(range(features), self._model.coef_[idx], 0.4)

        plt.xticks(ticks=range(features), labels=self.in_cols, rotation=90, fontsize=12)
        self.save_or_show(save=save, fname=f"{list(self.config['model'].keys())[0]}_feature_importance")
        return

    def plot_loss(self, history: dict, name="loss_curve"):


        plt.clf()
        plt.close('all')
        fig = plt.figure()
        plt.style.use('ggplot')
        i = 1

        legends = {
            'mean_absolute_error': 'Mean Absolute Error',
            'mape': 'Mean Absolute Percentage Error',
            'mean_squared_logarithmic_error': 'Mean Squared Logrithmic Error',
            'pbias': "Percent Bias",
            "nse": "Nash-Sutcliff Efficiency",
            "kge": "Kling-Gupta Efficiency"
        }

        sub_plots = {1: {'axis': (1,1), 'width': 10, 'height': 10},
                     2: (1, 1),
                     3: (1, 2),
                     4: {'axis': (1, 2), 'width': 10, 'height': 10},
                     5: (1, 3),
                     6: {'axis': (1, 3), 'width': 20, 'height': 20},
                     7: {'axis': (3, 2), 'width': 20, 'height': 20},
                     8: {'axis': (4, 2), 'width': 20, 'height': 20},
                     9: {'axis': (5, 2), 'width': 20, 'height': 20},
                     10: {'axis': (5, 2), 'width': 20, 'height': 20},
                     12: {'axis': (4, 3), 'width': 20, 'height': 20},
                     }

        epochs = range(1, len(history['loss']) + 1)
        axis_cache = {}

        for key, val in history.items():

            m_name = key.split('_')[1:] if 'val' in key and '_' in key else key

            if isinstance(m_name, list):
                m_name = '_'.join(m_name)
            if m_name in list(axis_cache.keys()):
                axis = axis_cache[m_name]
                axis.plot(epochs, val, color=[0.96707953, 0.46268314, 0.45772886], label= 'Validation ')
                axis.legend()
            else:
                axis = fig.add_subplot(*sub_plots[len(history)]['axis'], i)
                axis.plot(epochs, val, color=[0.13778617, 0.06228198, 0.33547859], label= 'Training ')
                axis.legend()
                axis.set_xlabel("Epochs")
                axis.set_ylabel(legends.get(key, key))
                axis_cache[key] = axis
                i += 1
            axis.set(frame_on=True)

        fig.set_figheight(sub_plots[len(history)]['height'])
        fig.set_figwidth(sub_plots[len(history)]['width'])
        self.save_or_show(fname=name, save=True if name is not None else False)
        return

    def box_plot(self,
                 inputs=True,
                 outputs=True,
                 save=True,
                 violen=False,
                 normalize=True,
                 cols=None,
                 figsize=(12,8),
                 max_features=8,
                 show_datapoints=False,
                 freq=None,
                 **kwargs):
        """
        Plots box whister or violen plot of data.
        freq: str, one of 'weekly', 'monthly', 'yearly'. If given, box plot will be plotted for these intervals.
        max_features: int, maximum number of features to appear in one plot.
        violen: bool, if True, then violen plot will be plotted else box_whisker plot
        cols: list, the name of columns from data to be plotted.
        kwargs: any args for seaborn.boxplot/seaborn.violenplot or seaborn.swarmplot.
        show_datapoints: if True, sns.swarmplot() will be plotted. Will be time consuming for bigger data."""

        data = self.data
        fname = "violen_plot_" if violen else "box_plot_"

        if cols is None:
            cols = []

            if inputs:
                cols += self.in_cols
                fname += "inputs_"
            if outputs:
                fname += "outptuts_"
        else:
            assert isinstance(cols, list)

        if isinstance(data, list):
            for idx, d in enumerate(data):
                _cols  = cols +  [self.out_cols[idx]] if outputs else cols
                self._box_plot(d, _cols, save, normalize, figsize, max_features, show_datapoints, freq,
                               violen=violen,
                               prefix=str(idx),
                               **kwargs)

        elif isinstance(data, dict):
            for data_name, _data in data.items():
                self._box_plot(_data, list(_data.columns), save, normalize, figsize, max_features, show_datapoints,
                               freq,
                               violen=violen,
                               prefix=data_name,
                               **kwargs)
        else:
            cols = cols + self.out_cols if outputs else cols
            self._box_plot(data, cols, save, normalize, figsize, max_features, show_datapoints, freq,
                           violen=violen,
                           **kwargs)

    def _box_plot(self, data,
                  cols, save, normalize, figsize, max_features, show_datapoints, freq,
                  violen=False,
                  prefix='',
                  **kwargs):
        data = data[cols]

        if data.shape[1] <= max_features:
            self.box_plot_df(data, normalize=normalize, show_datapoints=show_datapoints, save=save,
                             violen=violen,
                             freq=freq,
                             prefix=f"{'violen' if violen else 'box'}_{prefix}",
                              figsize=figsize, **kwargs)
        else:
            tot_plots = find_tot_plots(data.shape[1], max_features)
            for i in range(len(tot_plots) - 1):
                st, en = tot_plots[i], tot_plots[i + 1]
                sub_df = data.iloc[:, st:en]
                self.box_plot_df(sub_df, normalize=normalize, show_datapoints=show_datapoints, save=save,
                                 violen=violen,
                                  figsize=figsize,
                                  freq=freq,
                                  prefix=f"{'violen' if violen else 'box'}_{prefix}_{st}_{en}",
                                  **kwargs)
        return

    def box_plot_df(self, data,
                    normalize=True,
                    show_datapoints=False,
                    violen=False,
                    save=True,
                    figsize=(12,8),
                     prefix="box_plot",
                     freq=None,
                     **kwargs):

        data = data.copy()
        # if data contains duplicated columns, transformation will not work
        data = data.loc[:, ~data.columns.duplicated()]
        if normalize:
            transformer = Transformations(data=data)
            data = transformer.transform()

        if freq is not None:
            return self.box_plot_with_freq(data, freq, show_datapoints, save, figsize,
                                           violen=violen,
                                           prefix=prefix, **kwargs)

        return self._box_plot_df(data=data,
                                 name=prefix,
                                 violen=violen,
                                 save=save,
                                 figsize=figsize,
                                 show_datapoints=show_datapoints,
                                 **kwargs)

    def _box_plot_df(self,
                     data,
                     name,
                     violen=False,
                     save=True,
                     figsize=(12,8),
                     show_datapoints=False,
                     where='data',
                     **kwargs):

        plt.close('all')
        plt.figure(figsize=figsize)

        if violen:
            sns.violinplot(data=data, **kwargs)
        else:
            axis = sns.boxplot(data=data, **kwargs)
            axis.set_xticklabels(list(data.columns), fontdict={'rotation': 70})

        if show_datapoints:
            sns.swarmplot(data=data)

        self.save_or_show(fname=name, save=save, where=where)
        return

    def box_plot_with_freq(self, data, freq,
                           violen=False,
                           show_datapoints=False,
                           save=True,
                           figsize=(12,8),
                           name='bw',
                           prefix='',
                           **kwargs):

        validate_freq(data, freq)

        st_year = data.index[0].year
        en_year = data.index[-1].year

        for yr in range(st_year, en_year + 1):

            _df = data[data.index.year == yr]

            if freq == 'yearly':
                self._box_plot_df(_df,
                                  name=f'{name}_input_{prefix}_{str(yr)}',
                                  figsize=figsize,
                                  violen=violen,
                                  save=save,
                                  show_datapoints=show_datapoints,
                                  **kwargs)

            elif freq == 'monthly':
                st_mon = _df.index[0].month
                en_mon = _df.index[-1].month

                for mon in range(st_mon, en_mon+1):

                    __df = _df[_df.index.month == mon]

                    self._box_plot_df(__df,
                                      name=f'{prefix}_{str(yr)} _{str(mon)}',
                                      where='data/monthly',
                                      figsize=figsize,
                                      violen=violen,
                                      save=save,
                                      show_datapoints=show_datapoints,
                                      **kwargs)

            elif freq == 'weekly':
                st_week = _df.index[0].isocalendar()[1]
                en_week = _df.index[-1].isocalendar()[1]

                for week in range(st_week, en_week+1):
                    __df = _df[_df.index.week == week]

                    self._box_plot_df(__df,
                                      name=f'{prefix}_{str(yr)} _{str(week)}',
                                      where='data/weely',
                                      violen=violen,
                                      figsize=figsize,
                                      save=save,
                                      show_datapoints=show_datapoints,
                                      **kwargs)
        return

    def grouped_scatter(self, inputs=True, outputs=True, cols=None, save=True, max_subplots=8, **kwargs):

        fname = "scatter_plot_"

        if cols is None:
            cols = []

            if inputs:
                cols += self.in_cols
                fname += "inputs_"
            if outputs:
                fname += "outptuts_"
        else:
            assert isinstance(cols, list)

        if isinstance(self.data, pd.DataFrame):
            self.grouped_scatter_plot_df(self.data[cols], max_subplots, save, **kwargs)

        elif isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                if isinstance(data, pd.DataFrame):
                    _cols = cols + [self.out_cols[idx]] if outputs else cols
                    self.grouped_scatter_plot_df(data[_cols], max_subplots, save=save, prefix=str(idx), **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                if isinstance(data, pd.DataFrame):
                    self.grouped_scatter_plot_df(data, max_subplots, save=save, prefix=data_name,
                                                 **kwargs)
        return

    def grouped_scatter_plot_df(self, data:pd.DataFrame, max_subplots:int=10, save=True, prefix='', **kwargs):
        """
        max_subplots: int, it can be set to large number to show all the scatter plots on one axis.
        """
        data = data.copy()
        if data.shape[1] <= max_subplots:
            self._grouped_scatter_plot(data, save=save,name=f'grouped_scatter_{prefix}',  **kwargs)
        else:
            tot_plots = find_tot_plots(data.shape[1], max_subplots)
            for i in range(len(tot_plots) - 1):
                st, en = tot_plots[i], tot_plots[i + 1]
                sub_df = data.iloc[:, st:en]
                self._grouped_scatter_plot(sub_df, name=f'grouped_scatter_{prefix}_{st}_{en}', **kwargs)
        return

    def _grouped_scatter_plot(self, df, save=True, name='grouped_scatter', **kwargs):
        plt.close('all')
        sns.set()
        sns.pairplot(df, size=2.5, **kwargs)
        self.save_or_show(fname=name, save=save, where='data')
        return

    def plot_pcs(self, num_pcs=None, save=True, save_as_csv=False, figsize=(12, 8), **kwargs):
        """Plots principle components.
        kwargs will go to sns.pairplot."""
        if isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                self._plot_pcs(data[self.in_cols], num_pcs, save=save, prefix=str(idx), save_as_csv=save_as_csv,
                               hue=self.out_cols[idx], figsize=figsize, **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                self._plot_pcs(data[self.in_cols], num_pcs, save=save, prefix=data_name, save_as_csv=save_as_csv,
                               hue=self.out_cols,
                               figsize=figsize, **kwargs)
        else:
            self._plot_pcs(self.data[self.in_cols], num_pcs, save=save, save_as_csv=save_as_csv, hue=self.out_cols,
                           figsize=figsize, **kwargs)
        return

    def _plot_pcs(self, data, num_pcs, save=True, prefix='', save_as_csv=False, hue=None, figsize=(12,8), **kwargs):

        if num_pcs is None:
            num_pcs = int(data.shape[1]/2)

        #df_pca = data[self.in_cols]
        #pca = PCA(n_components=num_pcs).fit(df_pca)
        #df_pca = pd.DataFrame(pca.transform(df_pca))

        transformer = Transformations(data=data, method='pca', n_components=num_pcs, replace_nans=True)
        df_pca = transformer.transform()

        pcs = ['pc' + str(i + 1) for i in range(num_pcs)]
        df_pca.columns = pcs

        if hue is not None:
            if isinstance(hue, list):
                hue = hue[0]
            if hue in data:
                df_pca[hue] = data[hue]

                if df_pca[hue].isna().sum() > 0: # output columns contains nans, so don't use it as hue.
                    hue = None
            else: # ignore it
                hue = None

        if save_as_csv:
            df_pca.to_csv(os.path.join(self.path, f"data\\first_{num_pcs}_pcs_{prefix}"))

        plt.close('all')
        plt.figure(figsize=figsize)
        sns.pairplot(data=df_pca, vars=pcs, hue=hue, **kwargs)
        self.save_or_show(fname=f"first_{num_pcs}_pcs_{prefix}", save=save, where='data')
        return

    def plot_missing(self, save=True, cols=None, **kwargs):

        if isinstance(self.data, pd.DataFrame):
            self.plot_missing_df(self.data, cols=cols, save=save, **kwargs)

        elif isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                _cols = cols[idx] if isinstance(cols, list) else None
                self.plot_missing_df(data, cols=None, prefix=str(idx), save=save, **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                if isinstance(data, pd.DataFrame):
                    _cols = cols[data_name] if cols else None
                    self.plot_missing_df(data, cols=_cols, fname=data_name, save=save, **kwargs)
        return

    def plot_missing_df(self,
                        data:pd.DataFrame,
                        cols=None,
                        figsize:tuple=(20,20),
                        fname:str='',
                        save:bool=True,
                        **kwargs):
        if cols is None:
            cols = data.columns
        data = data[cols]
        # Identify missing values
        mv_total, mv_rows, mv_cols, _, mv_cols_ratio = _missing_vals(data).values()

        if mv_total == 0:
            print("No missing values found in the dataset.")
        else:
            # Create figure and axes
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(nrows=1, ncols=1, left=0.1, wspace=0.05)
            ax1 = fig.add_subplot(gs[:1, :5])

            # ax1 - Barplot
            ax1 = sns.barplot(x=list(data.columns), y=np.round((mv_cols_ratio) * 100, 2), ax=ax1, **kwargs)
            ax1.set(frame_on=True, xlim=(-0.5, len(mv_cols) - 0.5))
            ax1.set_ylim(0, np.max(mv_cols_ratio) * 100)
            ax1.grid(linestyle=":", linewidth=1)
            ax1.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
            ax1.set_yticklabels(ax1.get_yticks(),
                                fontsize="18")
            ax1.set_ylabel("Missing Percentage", fontsize="24")
            ax1.set_xticklabels(
                ax1.get_xticklabels(),
                horizontalalignment="center",
                fontweight="light",
                rotation=90,
                fontsize="12",
            )
            ax1.tick_params(axis="y", colors="#111111", length=1)

            # annotate missing values on top of the bars
            for rect, label in zip(ax1.patches, mv_cols):
                height = rect.get_height()
                ax1.text(
                    0.1 + rect.get_x() + rect.get_width() / 2,
                    height + 0.5,
                    label,
                    ha="center",
                    va="bottom",
                    rotation="90",
                    alpha=0.5,
                    fontsize="11",
                )
        self.save_or_show(save=save, fname=fname+'_missing_vals', where='data', dpi=500)
        return

    def plot_histograms(self, save=True, cols=None, **kwargs):
        """Plots distribution of data as histogram.
        kwargs: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.hist.html
        """
        if isinstance(self.data, pd.DataFrame):
            self.plot_his_df(self.data, save=save, cols=cols, **kwargs)

        elif isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                self.plot_his_df(data, prefix=str(idx), cols=cols, save=save, **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                if isinstance(data, pd.DataFrame):
                    self.plot_his_df(data, prefix=data_name, save=save, **kwargs)
        return

    def plot_his_df(self, data:pd.DataFrame, prefix='', cols=None, save=True, bins=100, figsize=(20, 14), **kwargs):
        if cols is None:
            cols = data.columns
        data[cols].hist(bins=bins, figsize=figsize, **kwargs)
        self.save_or_show(fname=f"hist_{prefix}", save=save, where='data')

    def data_heatmap(self, cols=None, **kwargs):
        if isinstance(self.data, pd.DataFrame):
            self._data_heatmap(self.data, cols=cols, **kwargs)

        elif isinstance(self.data, list):
            for idx, data in self.data:
                if isinstance(data, pd.DataFrame):
                    self._data_heatmap(data, cols=cols[idx], fname=f"data_heatmap_{idx}", **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                if isinstance(data, pd.DataFrame):
                    _cols = cols[data_name] if cols is not None else None
                    self._data_heatmap(data, _cols, fname=data_name, **kwargs)
        return

    def _data_heatmap(self,
                      data:pd.DataFrame,
                      cols=None,
                      figsize: tuple = (20, 20),
                      spine_color: str = "#EEEEEE",
                      save=True,
                      fname=""
                     ):

        if cols is None:
            cols = data.columns
        fig, ax2 = plt.subplots(figsize=figsize)
        # ax2 - Heatmap
        sns.heatmap(data[cols].isna(), cbar=False, cmap="binary", ax=ax2)
        ax2.set_yticks(ax2.get_yticks()[0::5].astype('int'))
        ax2.set_yticklabels(ax2.get_yticks(),
                            fontsize="16")
        ax2.set_xticklabels(
            ax2.get_xticklabels(),
            horizontalalignment="center",
            fontweight="light",
            fontsize="12",
        )
        ax2.tick_params(length=1, colors="#111111")
        ax2.set_ylabel("Examples", fontsize="24")
        for _, spine in ax2.spines.items():
            spine.set_visible(True)
            spine.set_color(spine_color)

        self.save_or_show(save=save, fname=fname+'_heat_map', where='data', dpi=500)
        return

    def plot_index(self, save=True, **kwargs):
        if isinstance(self.data, pd.DataFrame):
            self._plot_index(self.data, save=save, **kwargs)

        elif isinstance(self.data, list):
            for data in self.data:
                if isinstance(data, pd.DataFrame):
                    self._plot_index(data, save=save, **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.values():
                if isinstance(data, pd.DataFrame):
                    self._plot_index(data, save=save, **kwargs)
        return

    def _plot_index(self,
                    index,
                    save=True,
                    fname="index",
                    figsize=(10,5),
                    dpi=200,
                    label_fs=18,
                    title_fs=20,
                    leg_fs=14,
                    leg_ms=4,
                    color='r',
                    ):
        """
        Plots the index of a datafram.
        index: can be pandas dataframe or index itself. if dataframe, its index will be used for plotting
        """
        plt.close('all')
        if isinstance(index, pd.DataFrame):
            index=index.index

        idx = pd.DataFrame(np.ones(len(index)), index=index, columns=['Observations'])
        axis = idx.plot(linestyle='', marker='.', color=color, figsize=figsize)
        axis.legend(fontsize=leg_fs, markerscale=leg_ms)
        axis.set_xlabel(axis.get_xlabel(), fontdict={'fontsize': label_fs})
        axis.set_title("Temporal distribution of Observations", fontsize=title_fs)
        axis.get_yaxis().set_visible(False)
        self.save_or_show(save=save, fname=fname, where='data', dpi=dpi)
        return


def plot_style(df:pd.DataFrame, **kwargs):
    if 'style' not in kwargs and df.isna().sum().sum() > 0:
        kwargs['style'] = ['.' for _ in range(df.shape[1])]
    return kwargs


def validate_freq(df, freq):
    assert isinstance(df.index, pd.DatetimeIndex), "index of dataframe must be pandas DatetimeIndex"
    assert freq in ["weekly", "monthly",
                    "yearly"], f"freq must be one of {'weekly', 'monthly', 'yearly'} but it is {freq}"
    return


def find_tot_plots(features, max_subplots):

    tot_plots = np.linspace(0, features, int(features / max_subplots) + 1 if features % max_subplots == 0 else int(
        features / max_subplots) + 2)
    # converting each value to int because linspace can return array containing floats if features is odd
    tot_plots = [int(i) for i in tot_plots]
    return tot_plots


def set_fig_dim(fig, width, height):
    fig.set_figwidth(width)
    fig.set_figheight(height)


def _get_nrows_and_ncols(n_subplots, n_rows=None):
    if n_rows is None:
        n_rows = int(np.sqrt(n_subplots))
    n_cols = max(int(n_subplots / n_rows), 1)  # ensure n_cols != 0
    n_rows = int(n_subplots / n_cols)

    while not ((n_subplots / n_cols).is_integer() and
               (n_subplots / n_rows).is_integer()):
        n_cols -= 1
        n_rows = int(n_subplots / n_cols)
    return n_rows, n_cols


def set_axis_paras(axis, leg_kws, label_kws, tick_kws):
    axis.legend(**leg_kws)
    axis.set_ylabel(axis.get_ylabel(), **label_kws)
    axis.set_xlabel(axis.get_xlabel(), **label_kws)
    axis.tick_params(**tick_kws)
    return