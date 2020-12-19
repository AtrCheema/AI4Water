import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import plot_roc_curve, plot_confusion_matrix, plot_precision_recall_curve
from sklearn import tree
from xgboost import plot_importance, plot_tree
import matplotlib as mpl

try:
    from dl4seq.utils.utils_from_see_rnn import rnn_histogram
    from see_rnn.visuals_gen import features_0D, features_1D, features_2D
except ModuleNotFoundError:
    rnn_histogram = None

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

    def plot_input_data(self, save=True, **kwargs):
        """
        :param save:
        :param kwargs: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html
        :return:
        model.plot_input_data(subplots=True, figsize=(12, 14), sharex=True)
        """
        data = self.data
        if isinstance(data, list):
            for df in data:
                self.plot_df(df, save=save, **kwargs)
        else:
            self.plot_df(data, save=save, **kwargs)
        return

    def plot_df(self, df, save=True, **kwargs):
        assert isinstance(df, pd.DataFrame)

        if df.shape[1] <= 10:

            df.plot(**kwargs)
            self.save_or_show(save=save, fname='input_',  where='data')
        else:
            tot_plots = np.arange(0, df.shape[1], 10)

            for i in range(len(tot_plots) - 1):
                st, en = tot_plots[i], tot_plots[i + 1]
                sub_df = df.iloc[:, st:en]


                sub_df.plot(**kwargs)
                self.save_or_show(save=save, fname='input'+str(st) + str(en), where='data')
        return

    def _imshow_3d(self, activation,
                   lyr_name,
                   save=True,
                   where='act'):

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
                where='act',
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

        self.save_or_show(save, fname)

        return

    def save_or_show(self, save: bool = True, fname=None, where='', dpi=300, bbox_inches='tight', close=True):

        if save:
            assert isinstance(fname, str)
            if "/" in fname:
                fname = fname.replace("/", "__")
            if ":" in fname:
                fname = fname.replace(":", "__")

            if not os.path.exists(where):
                assert where in ['', 'act', 'activation', 'weights', 'plots', 'data', 'results']
                save_dir = os.path.join(self.path, where)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
            else:
                save_dir = where

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

        features_1D(data, savepath=save, **kwargs)

        return

    def features_0d(self, data, save=True, name='', **kwargs):
        if save:
            save = os.path.join(self.act_path, name + "0D.png")
        else:
            save=None

        return features_0D(data, savepath=save, **kwargs)

    def rnn_histogram(self, data, save=True, name='', **kwargs):

        if save:
            save = os.path.join(self.act_path, name + "0D.png")
        else:
            save=None
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
            plot_importance(self._model)
        else:
            plt.bar(range(self.ins if use_prev else self.ins + self.outs), importance)
            plt.xticks(ticks=range(len(all_cols)), labels=list(all_cols), rotation=90, fontsize=5)
        self.save_or_show(save, fname="feature_importance.png")
        return

    def plot_feature_feature_corr(self, remove_targets=True, save=True, **kwargs):
        plt.close('')
        cols = self.in_cols if remove_targets else self.in_cols + self.out_cols
        corr = self.data[cols].corr()
        sns.heatmap(corr, **kwargs)
        self.save_or_show(save, fname="feature_feature_corr", where="data")
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
            plot_tree(self._model, **kwargs)
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