import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import plot_roc_curve, plot_confusion_matrix, plot_precision_recall_curve
from sklearn import tree
from dl4seq.backend import xgboost
import matplotlib as mpl
from sklearn.decomposition import PCA
from dl4seq.utils.transformations import Transformations

try:
    from dl4seq.utils.utils_from_see_rnn import rnn_histogram
    from see_rnn.visuals_gen import features_0D, features_1D, features_2D
except ModuleNotFoundError:
    rnn_histogram = None
    features_0D, features_1D, features_2D = None, None, None

try:
    from dtreeviz import trees
except ModuleNotFoundError:
    trees = None

rnn_info = {"LSTM": {'rnn_type': 'LSTM',
                     'gate_names': ['INPUT', 'FORGET', 'CELL', 'OUTPUT'],
                     'n_gates': 4,
                     'is_bidir': False,
                     'rnn_dim': 64,
                     'uses_bias': True,
                     'direction_names': [[]]}}

class Plots(object):
    # TODO initialte this class with at least path

    def __init__(self, path, problem, category, model, data_config, model_config):
        self.path = path
        self.problem = problem
        self.category=category
        self._model = model
        self.data_config = data_config
        self.model_cofig = model_config
        self.act_path = os.path.join(path, "activations")

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
        return self.data_config['inputs']

    @property
    def out_cols(self):
        return self.data_config['outputs']

    @property
    def ins(self):
        return len(self.data_config['inputs'])

    @property
    def outs(self):
        return self.data_config['outputs']

    @property
    def lookback(self):
        return self.data_config['lookback']

    def feature_imporance(self):
        if self.category.upper() == "ML":

            if self.model_cofig["ml_model"].upper() in ["SVC", "SVR"]:
                if self._model.kernel == "linear":
                    # https://stackoverflow.com/questions/41592661/determining-the-most-contributing-features-for-svm-classifier-in-sklearn
                    return self._model.coef_
            elif hasattr(self._model, "feature_importances_"):
                return self._model.feature_importances_

    def plot_data(self, save=True, freq=None, **kwargs):
        """
        :param save:
        :param kwargs: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html
        :param freq: str, one of 'daily', 'weekly', 'monthly', 'yearly', determines interval of plot of data. It is
                     valid for only time-series data.
        :return:

        ------
        examples:

        >>model.plot_input_data(subplots=True, figsize=(12, 14), sharex=True)
        >>model.plot_data(freq='monthly', subplots=True, figsize=(12, 14), sharex=True)
        """
        data = self.data
        if isinstance(data, list):
            for idx,df in enumerate(data):
                self.plot_df(df, save=save, freq=freq, prefix=str(idx), **kwargs)
        else:
            self.plot_df(data, save=save, freq=freq, **kwargs)
        return

    def plot_df(self, df, save=True, freq=None, max_subplots=10, prefix='', **kwargs):
        """Plots each columns of dataframe and saves it if `save` is True.
         max_subplots: determines how many sub_plots are to be plotted within one plot. If dataframe contains columns
         greater than max_subplots, a separate plot will be generated for remaining columns."""
        assert isinstance(df, pd.DataFrame)

        if df.shape[1] <= max_subplots:

            if freq is None:
                df.plot(**kwargs)
                self.save_or_show(save=save, fname=f"input_{prefix}",  where='data')
            else:
                self.plot_df_with_freq(df, freq, save, **kwargs)
        else:
            tot_plots = find_tot_plots(df.shape[1], max_subplots)

            for i in range(len(tot_plots) - 1):
                st, en = tot_plots[i], tot_plots[i + 1]
                sub_df = df.iloc[:, st:en]

                if freq is None:
                    sub_df.plot(**kwargs)
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
                _df.plot(**kwargs)
                self.save_or_show(save=save, fname=f'input_{prefix}_{str(yr)}', where='data')

            elif freq == 'monthly':
                st_mon = _df.index[0].month
                en_mon = _df.index[-1].month

                for mon in range(st_mon, en_mon+1):

                    __df = _df[_df.index.month == mon]

                    __df.plot(**kwargs)
                    self.save_or_show(save=save, fname=f'input_{prefix}_{str(yr)} _{str(mon)}', where='data/monthly')

            elif freq == 'weekly':
                st_week = _df.index[0].isocalendar()[1]
                en_week = _df.index[-1].isocalendar()[1]

                for week in range(st_week, en_week+1):
                    __df = _df[_df.index.week == week]

                    __df.plot(**kwargs)
                    self.save_or_show(save=save, fname=f'input_{prefix}_{str(yr)} _{str(week)}', where='data/weely')
        return

    def _imshow_3d(self, activation,
                   lyr_name,
                   save=True,
                   where='activations'):

        act_2d = []
        for i in range(activation.shape[0]):
            act_2d.append(activation[i, :])
        activation_2d = np.vstack(act_2d)
        self._imshow(activation_2d,
                     lyr_name + " Activations (3d of {})".format(activation.shape),
                     save,
                     lyr_name,
                     where=where)
        return

    def _imshow(self, img,
                label: str = '',
                save=True,
                fname=None,
                interpolation:str='none',
                where='activations',
                rnn_args=None):

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

        plt.colorbar()
        plt.title(label)
        self.save_or_show(save, fname, where=where)

        return

    def plot1d(self, array, label: str = '', save=True, fname=None, rnn_args=None):
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

        self.save_or_show(save, fname, where='activations')

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

        x,y = self.train_data(**kwargs)
        self.plot_model_input_data(x, how=how, save=save, which='training')

        return

    def plot_val_data(self, how='3d', save=True,  **kwargs):

        x,y = self.val_data(**kwargs)
        self.plot_model_input_data(x, how=how, save=save, which='validation')

        return

    def plot_test_data(self, how='3d', save=True,  **kwargs):

        x,y = self.test_data(**kwargs)
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

    def features_2d(self, data, name, save=True, slices=32, slice_dim=0, **kwargs):
        """Calls the features_2d from see-rnn"""
        st=0
        if features_2D is None:
            warnings.warn("install see-rnn to plot features-2D plot", UserWarning)
        else:
            for en in np.arange(slices, data.shape[slice_dim] + slices, slices):

                if save:
                    name = name + f"_{st}_{en}"
                    save = os.path.join(self.act_path, name+".png")
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
            save = os.path.join(self.act_path, name + "0D.png", UserWarning)
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

    def plot_feature_importance(self, importance=None, save=True, use_xgb=False):

        if importance is None:
            importance = self.feature_imporance()

        if self.category == "ML" and self.model_cofig["ml_model"] is not None:
            if self.model_cofig["ml_model"].upper() in ["SVC", "SVR"]:
                if self._model.kernel == "linear":
                    return self.f_importances_svm(importance, self.in_cols, save=save)
                else:
                    warnings.warn(f"for {self._model.kernel} kernels of {self.model_cofig['ml_model']}, feature importance can not be plotted.")
                return
            return 

        if isinstance(importance, np.ndarray):
            assert importance.ndim <= 2

        use_prev = self.data_config['use_predicted_output']
        all_cols = self.data_config['inputs'] if use_prev else self.data_config['inputs'] + \
                                                                                self.data_config['outputs']
        plt.close('all')
        plt.figure()
        plt.title("Feature importance")
        if use_xgb:
            if xgboost is None:
                warnings.warn("install xgboost to plot plot_importance using xgboost", UserWarning)
            else:
                xgboost.plot_importance(self._model)
        else:
            plt.bar(range(self.ins if use_prev else self.ins + self.outs), importance)
            plt.xticks(ticks=range(len(all_cols)), labels=list(all_cols), rotation=90, fontsize=5)
        self.save_or_show(save, fname="feature_importance.png")
        return

    def plot_feature_feature_corr(self, remove_targets=True, save=True, **kwargs):
        plt.close('')
        cols = self.in_cols if remove_targets else self.in_cols + self.out_cols
        if isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                self._feature_feature_corr(data, cols, prefix=str(idx), save=save, **kwargs)
        else:
            self._feature_feature_corr(self.data, cols, save=save, **kwargs)

    def _feature_feature_corr(self, data, cols, prefix='', save=True, **kwargs):
        corr = data[cols].corr()
        sns.heatmap(corr, **kwargs)
        self.save_or_show(save, fname=f"feature_feature_corr_{prefix}", where="data")
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

        model = self.model_cofig['ml_model'].upper()
        if model in ["DECISIONTREEREGRESSON", "DECISIONTREECLASSIFIER"] or model.startswith("XGBOOST"):
            if trees is None:
                print("dtreeviz related plots can not be plotted")
            else:
                x,y = self.test_data()

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

        mpl.rcParams.update(mpl.rcParamsDefault)

        fig, axis = plt.subplots()
        set_fig_dim(fig, 12, 8)

        # it is quite possible that when data is datetime indexed, then it is not equalidistant and large amount of graph
        # will have not data in that case lines plot will create a lot of useless interpolating lines where no data is present.
        style = '.' if isinstance(true.index, pd.DatetimeIndex) else '-'

        if np.isnan(true.values).sum() > 0:
            style = '.' # For Nan values we should be using this style otherwise nothing is plotted.

        ms = 4 if style == '.' else 2
        axis.plot(true, style, color='b', markersize=ms, label='True')

        axis.plot(predicted, style, color='r', label='Prediction')

        axis.legend(loc="best", fontsize=22, markerscale=4)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel("Time", fontsize=18)

        self.save_or_show(save=save, fname=name, close=False, where=where)
        return

    def regplot_using_searborn(self, true, pred, save, name, where='plots'):
        # https://seaborn.pydata.org/generated/seaborn.regplot.html
        plt.close('all')
        sns.regplot(x=true, y=pred, color="g")
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
        self.save_or_show(save=save, fname=f"{self.model_cofig['ml_model']}_feature_importance")
        return

    def plot_loss(self, history: dict, name="loss_curve"):


        plt.clf()
        plt.close('all')
        fig = plt.figure()
        plt.style.use('ggplot')
        i = 1

        sub_plots = {1: (1,1),
                     2: (1, 1),
                     3: (1, 2),
                     4: (1, 2),
                     5: (1, 3),
                     6: (1, 3),
                     7: (2, 2),
                     8: (2, 2),
                     9: (3, 2),
                     10: (3, 2)
                     }

        epochs = range(1, len(history['loss']) + 1)
        axis_cache = {}

        for key, val in history.items():

            m_name = key.split('_')[1] if '_' in key else key

            if m_name in list(axis_cache.keys()):
                axis = axis_cache[m_name]
                axis.plot(epochs, val, color=[0.96707953, 0.46268314, 0.45772886], label= 'Validation ' + m_name)
                axis.legend()
            else:
                axis = fig.add_subplot(*sub_plots[len(history)], i)
                axis.plot(epochs, val, color=[0.13778617, 0.06228198, 0.33547859], label= 'Training ' + key)
                axis.legend()
                axis_cache[key] = axis
                i += 1

        self.save_or_show(fname=name, save=True if name is not None else False)
        return

    def box_plot(self, inputs=True,
                 outputs=True,
                 save=True,
                 normalize=True,
                 cols=None,
                 figsize=(12,8),
                 max_features=8,
                 show_datapoints=False,
                 freq=None,
                 **kwargs):
        """
        Plots box whister plot of data.
        freq: str, one of 'weekly', 'monthly', 'yearly'. If given, box plot will be plotted for these intervals.
        max_features: int, maximum number of features to appear in one plot.
        cols: list, the name of columns from data to be plotted.
        kwargs: any args for seaborn.boxplot or seaborn.swarmplot.
        show_datapoints: if True, sns.swarmplot() will be plotted. Will be time consuming for bigger data."""

        data = self.data
        fname = "box_plot_"

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
                self._box_plot(d, _cols, save, normalize, figsize, max_features, show_datapoints, freq, prefix=str(idx),
                               **kwargs)
        else:
            cols = cols + self.out_cols if outputs else cols
            self._box_plot(data, cols, save, normalize, figsize, max_features, show_datapoints, freq, **kwargs)


    def _box_plot(self, data, cols, save, normalize, figsize, max_features, show_datapoints, freq,
                  prefix='',
                  **kwargs):
        data = data[cols]

        if data.shape[1] <= max_features:
            self.box_plot_df(data, normalize=normalize, show_datapoints=show_datapoints, save=save,
                              freq=freq,
                             prefix=f"box_plot_{prefix}",
                              figsize=figsize, **kwargs)
        else:
            tot_plots = find_tot_plots(data.shape[1], max_features)
            for i in range(len(tot_plots) - 1):
                st, en = tot_plots[i], tot_plots[i + 1]
                sub_df = data.iloc[:, st:en]
                self.box_plot_df(sub_df, normalize=normalize, show_datapoints=show_datapoints, save=save,
                                  figsize=figsize,
                                  freq=freq,
                                  prefix=f"box_plot_{prefix}_{st}_{en}",
                                  **kwargs)
        return

    def box_plot_df(self, data, normalize=True, show_datapoints=False, save=True, figsize=(12,8),
                     prefix="box_plot",
                     freq=None,
                     **kwargs):

        if normalize:
            transformer = Transformations(data=data)
            data = transformer.transform()

        if freq is not None:
            return self.box_plot_with_freq(data, freq, show_datapoints, save, figsize, prefix=prefix, **kwargs)

        return self._box_plot_df(data, prefix, save, figsize, show_datapoints, **kwargs)

    def _box_plot_df(self,
                     data,
                     name,
                     save=True,
                     figsize=(12,8),
                     show_datapoints=False,
                     where='data',
                     **kwargs):

        plt.close('all')
        plt.figure(figsize=figsize)

        ax = sns.boxplot(data=data, **kwargs)

        if show_datapoints:
            sns.swarmplot(data=data)

        self.save_or_show(fname=name, save=save, where=where)
        return

    def box_plot_with_freq(self, data, freq, show_datapoints=False, save=True, figsize=(12,8), name='bw', prefix='',
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
                                      figsize=figsize,
                                      save=save,
                                      show_datapoints=show_datapoints,
                                      **kwargs)
        return

    def grouped_scatter(self, inputs=True, outputs=True, cols=None, save=True, max_subplots=8, **kwargs):

        data = self.data
        fname = "box_plot_"

        if cols is None:
            cols = []

            if inputs:
                cols += self.in_cols
                fname += "inputs_"
            if outputs:
                cols += self.out_cols
                fname += "outptuts_"
        else:
            assert isinstance(cols, list)

        if isinstance(data, pd.DataFrame):
            data = data[cols]
            if data.shape[1] <= max_subplots:
                self._grouped_scatter_plot(data, save=save, **kwargs)
            else:
                tot_plots = find_tot_plots(data.shape[1], max_subplots)
                for i in range(len(tot_plots) - 1):
                    st, en = tot_plots[i], tot_plots[i + 1]
                    sub_df = data.iloc[:, st:en]
                    self._grouped_scatter_plot(sub_df, name=f'grouped_scatter {st}_{en}', **kwargs)
        return

    def _grouped_scatter_plot(self, df, save=True, name='grouped_scatter', **kwargs):
        plt.close('all')
        sns.set()
        sns.pairplot(df, size=2.5, **kwargs)
        self.save_or_show(fname=name, save=save)
        return

    def plot_pcs(self, num_pcs=None, save=True, save_as_csv=False, figsize=(12, 8), **kwargs):
        """Plots principle components.
        kwargs will go to sns.pairplot."""
        if isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                self._plot_pcs(data, num_pcs, save=save, prefix=str(idx), save_as_csv=save_as_csv,
                               hue=self.out_cols[idx], figsize=figsize, **kwargs)
        else:
            self._plot_pcs(self.data, num_pcs, save=save, save_as_csv=save_as_csv, hue=self.out_cols,
                           figsize=figsize, **kwargs)
        return

    def _plot_pcs(self, data, num_pcs, save=True, prefix='', save_as_csv=False, hue=None, figsize=(12,8), **kwargs):

        if num_pcs is None:
            num_pcs = int(data.shape[1]/2)

        df_pca = data[self.in_cols]
        pca = PCA(n_components=num_pcs).fit(df_pca)
        df_pca = pd.DataFrame(pca.transform(df_pca))
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