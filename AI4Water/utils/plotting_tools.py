import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve, plot_confusion_matrix, plot_precision_recall_curve

try:
    from dtreeviz import trees
except ModuleNotFoundError:
    trees = None

from AI4Water.backend import xgboost
from AI4Water.utils.utils import find_tot_plots, init_subplots
from AI4Water.utils.transformations import Transformations

try:
    from AI4Water.utils.utils_from_see_rnn import rnn_histogram
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

    def __init__(self, path, problem, category, config, model=None):
        self.path = path
        self.problem = problem
        self.category=category
        self.ml_model = model
        self.config = config

    @property
    def training_data(self):
        raise AttributeError

    @property
    def validation_data(self):
        raise AttributeError

    @property
    def test_data(self):
        raise AttributeError

    @property
    def data(self):
        raise AttributeError

    @property
    def in_cols(self):
        return self.config['input_features']

    @property
    def out_cols(self):
        return self.config['output_features']

    @property
    def ins(self):
        return len(self.config['input_features'])

    @property
    def outs(self):
        return self.config['output_features']

    @property
    def lookback(self):
        return self.config['lookback']

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
            xlabels = kwargs.get('xlabel', None)
            if xlabels is not None:
                if len(xlabels)<30:
                    plt.xlabel(xlabels, rotation=90)
            else:
                plt.xlabel("Inputs")

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
        fig, axis = init_subplots(height=8)
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

        x, _, y = self.training_data(**kwargs)
        self.plot_model_input_data(x, how=how, save=save, which='training')

        return

    def plot_val_data(self, how='3d', save=True,  **kwargs):

        x, y = self.validation_data(**kwargs)
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
            plt.fill_between(np.arange(st, en),
                             predicted[st:en, q].reshape(-1,),
                             predicted[st:en, q + 1].reshape(-1,),
                             alpha=0.2,
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
        plt.fill_between(np.arange(st, en),
                         predicted[st:en, min_q].reshape(-1,),
                         predicted[st:en, max_q].reshape(-1,),
                         alpha=0.2,
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
        assert true_outputs.shape[-2:] == (1,1)
        if en is None:
            en = true_outputs.shape[0]
        for q in range(len(self.quantiles) - 1):
            st_q = "{:.1f}".format(self.quantiles[q] * 100)
            en_q = "{:.1f}".format(self.quantiles[-q] * 100)

            plt.plot(np.arange(st, en), true_outputs[st:en, 0], label="True", color='navy')
            plt.fill_between(np.arange(st, en), predicted[st:en, q].reshape(-1,),
                             predicted[st:en, -q].reshape(-1,), alpha=0.2,
                             color='g', edgecolor=None, label=st_q + '_' + en_q)
            plt.legend(loc="best")
            self.save_or_show(save, fname='q' + st_q + '_' + en_q, where='results')
        return


    def roc_curve(self, x, y, save=True):
        assert self.problem.upper().startswith("CLASS")
        plot_roc_curve(self.ml_model, *x, y.reshape(-1, ))
        self.save_or_show(save, fname="roc", where="results")
        return

    def confusion_matrx(self, x, y, save=True):
        assert self.problem.upper().startswith("CLASS")
        plot_confusion_matrix(self.ml_model, *x, y.reshape(-1, ))
        self.save_or_show(save, fname="confusion_matrix", where="results")
        return

    def precision_recall_curve(self, x, y, save=True):
        assert self.problem.upper().startswith("CLASS")
        plot_precision_recall_curve(self.ml_model, *x, y.reshape(-1, ))
        self.save_or_show(save, fname="plot_precision_recall_curve", where="results")
        return

    def decision_tree(self, which="sklearn", save=True, **kwargs):
        """For kwargs see https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.plot_tree"""
        plt.close('')
        if which == "sklearn":
            if hasattr(self.ml_model, "tree_"):
                tree.plot_tree(self.ml_model, **kwargs)
        else:  # xgboost
            if xgboost is None:
                warnings.warn("install xgboost to use plot_tree method")
            else:
                xgboost.plot_tree(self.ml_model, **kwargs)
        self.save_or_show(save, fname="decision_tree", where="results")
        return

    def plot_treeviz_leaves(self, save=True, **kwargs):
        """Plots dtreeviz related plots if dtreeviz is installed"""

        model = list(self.config['model'].keys())[0].upper()
        if model in ["DECISIONTREEREGRESSON", "DECISIONTREECLASSIFIER"] or model.startswith("XGBOOST"):
            if trees is None:
                print("dtreeviz related plots can not be plotted")
            elif model in ['XGBOOSTRFREGRESSOR', 'XGBOOSTREGRESSOR']:  # dtreeviz doesn't plot this
                pass
            else:
                x, _, y = self.test_data()

                if np.ndim(y) > 2:
                    y = np.squeeze(y, axis=2)

                trees.viz_leaf_samples(self.ml_model, *x, self.in_cols)
                self.save_or_show(save, fname="viz_leaf_samples", where="plots")

                trees.ctreeviz_leaf_samples(self.ml_model, *x, y, self.in_cols)
                self.save_or_show(save, fname="ctreeviz_leaf_samples", where="plots")

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
                if isinstance(self.in_cols, dict):
                    cols_ = [item for sublist in list(self.in_cols.values()) for item in sublist]
                    _cols = []
                    for c in cols_:
                        if c in d:
                            cols_.append(c)
                else:
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



def validate_freq(df, freq):
    assert isinstance(df.index, pd.DatetimeIndex), "index of dataframe must be pandas DatetimeIndex"
    assert freq in ["weekly", "monthly",
                    "yearly"], f"freq must be one of {'weekly', 'monthly', 'yearly'} but it is {freq}"
    return


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
