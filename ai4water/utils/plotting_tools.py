import os
import random
import warnings
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve, plot_confusion_matrix, plot_precision_recall_curve

from ai4water.utils.utils import init_subplots


BAR_CMAPS = ['Blues', 'BuGn', 'gist_earth_r',
             'GnBu', 'PuBu', 'PuBuGn', 'summer_r']


# TODO
# rank histogram, reliability diagram, ROC curve



class Plots(object):
    # TODO initialte this class with at least path

    def __init__(self, path, problem, category, config, model=None):
        self.path = path
        self.problem = problem
        self.category = category
        self.ml_model = model
        self.config = config

    @property
    def training_data(self, *args, **kwargs):
        raise AttributeError

    @property
    def validation_data(self, *args, **kwargs):
        raise AttributeError

    @property
    def test_data(self, *args, **kwargs):
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
                   xticklabels=None,
                   save=True,
                   where='activations',
                   xlabel=None):

        act_2d = []
        for i in range(activation.shape[2]):
            act_2d.append(activation[:, :, i])
        activation_2d = np.concatenate(act_2d, axis=1)
        self._imshow(activation_2d,
                     lyr_name + " Activations (3d of {})".format(activation.shape),
                     save,
                     lyr_name,
                     where=where,
                     xticklabels=xticklabels,
                     xlabel=xlabel)
        return

    def _imshow(self,
                img,
                label: str = '',
                save=True,
                fname=None,
                interpolation: str = 'none',
                where='',
                rnn_args=None,
                cmap = None,
                show=False,
                **kwargs):

        assert np.ndim(img) == 2, "can not plot {} with shape {} and ndim {}".format(label, img.shape, np.ndim(img))

        _, axis = plt.subplots()
        im = axis.imshow(img, aspect='auto', interpolation=interpolation, cmap=cmap)

        if rnn_args is not None:
            assert isinstance(rnn_args, dict)

            rnn_dim = int(img.shape[1] / rnn_args['n_gates'])
            [plt.axvline(rnn_dim * gate_idx - .5, linewidth=0.8, color='k')
                     for gate_idx in range(1, rnn_args['n_gates'])]

            kwargs['xlabel'] = rnn_args['gate_names_str']
            if "RECURRENT" in label.upper():
                plt.ylabel("Hidden Units")
            else:
                plt.ylabel("Channel Units")
        else:
            axis.set_ylabel('Examples' if 'weight' not in label.lower() else '')
            xlabels = kwargs.get('xticklabels', None)
            if xlabels is not None:
                if len(xlabels) < 30:
                    axis.set_xticklabels(xlabels, rotation=90)

        plt.xlabel(kwargs.get('xlabel', 'inputs'))

        plt.colorbar(im)
        plt.title(label)
        self.save_or_show(save, fname, where=where, show=show)

        return

    def plot1d(self, array, label: str = '', show=False, fname=None, rnn_args=None, where=''):
        plt.close('all')
        plt.plot(array)
        plt.xlabel("Examples")
        plt.title(label)

        if rnn_args is not None:
            assert isinstance(rnn_args, dict)

            rnn_dim = int(array.shape[0] / rnn_args['n_gates'])
            [plt.axvline(rnn_dim * gate_idx - .5, linewidth=0.5, color='k')
                     for gate_idx in range(1, rnn_args['n_gates'])]
            plt.xlabel(rnn_args['gate_names_str'])

        self.save_or_show(save=True, fname=fname, where=where, show=show)

        return

    def save_or_show(self, *args, **kwargs):

        return save_or_show(self.path, *args, **kwargs)

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
        _, axis = plt.subplots()

        for idx in range(self.lookback-1):
            axis.plot(activations[sample, idx, :].transpose(), label='lookback '+str(idx))
        axis.set_xlabel('inputs')
        axis.set_ylabel('activation weight')
        axis.set_title('Activations at different lookbacks for all inputs for sample ' + str(sample))
        self.save_or_show(save=save, fname=name + '_' + str(sample), where='path')

        return

    def plot_train_data(self, how='3d', save=True,  **kwargs):

        x,  _ = self.training_data(**kwargs)
        self.plot_model_input_data(x, how=how, save=save, which='training')

        return

    def plot_val_data(self, how='3d', save=True,  **kwargs):

        x, y = self.validation_data(**kwargs)
        self.plot_model_input_data(x, how=how, save=save, which='validation')

        return

    def plot_test_data(self, how='3d', save=True,  **kwargs):

        x, _ = self.test_data(**kwargs)
        self.plot_model_input_data(x, how=how, save=save, which='test')

        return

    def plot_model_input_data(self,
                              in_data: Union[list, np.ndarray],
                              how: str,
                              save: bool,
                              which: str = 'training'
                              ) -> None:

        assert how in ['3d', 'hist']

        if not isinstance(in_data, list):
            if isinstance(in_data, dict):
                in_data = list(in_data.values())
            else:
                assert isinstance(in_data, np.ndarray)
                in_data = [in_data]

        for idx, inputs in enumerate(in_data):
            if np.ndim(inputs) == 3:
                if how.upper() == "3D":
                    self._imshow_3d(inputs, which + '_data_' + str(idx), save=save, where='data')
            elif np.ndim(inputs) == 2:
                if how.upper() == "3D":
                    self._imshow(inputs, save=save, fname=which + '_data_' + str(idx), where='data')
                else:
                    self.plot_histogram(inputs,
                                        save=save,
                                        fname=which+'_data_' + str(idx),
                                        features=self.in_cols,
                                        where='data')
            else:
                print(f'skipping shape is {inputs.shape}')
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
        self.save_or_show(save, fname="q_" + q_name + ".png", where='results')
        return

    def plot_all_qs(self, true_outputs, predicted, save=False):
        plt.close('all')
        plt.style.use('ggplot')

        st, en = 0, true_outputs.shape[0]

        plt.plot(np.arange(st, en), true_outputs[st:en, 0], label="True", color='navy')

        for idx, q in enumerate(self.quantiles):
            q_name = "{:.1f}".format(q * 100)
            plt.plot(np.arange(st, en), predicted[st:en, idx], label="q {} %".format(q_name))

        plt.legend(loc="best")
        self.save_or_show(save, fname="all_quantiles", where='results')

        return

    def plot_quantiles1(self, true_outputs, predicted, st=0, en=None, save=True):
        plt.close('all')
        plt.style.use('ggplot')
        assert true_outputs.shape[-2:] == (1, 1)
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
        plot_roc_curve(self._model, x, y.reshape(-1, ))
        self.save_or_show(save, fname="roc", where="results")
        return

    def confusion_matrx(self, x, y, save=True):
        assert self.problem.upper().startswith("CLASS")
        plot_confusion_matrix(self._model, x, y.reshape(-1, ))
        self.save_or_show(save, fname="confusion_matrix", where="results")
        return

    def precision_recall_curve(self, x, y, save=True):
        assert self.problem.upper().startswith("CLASS")
        plot_precision_recall_curve(self._model, x, y.reshape(-1, ))
        self.save_or_show(save, fname="plot_precision_recall_curve", where="results")
        return

    def plot_histogram(self,
                       data: np.ndarray,
                       save: bool = True,
                       fname='hist',
                       features=None,
                       where='data'
                       ):

        assert data.ndim == 2

        data = pd.DataFrame(data, columns=features)
        data.hist(figsize=(12, 12))

        self.save_or_show(save=save, fname=fname+'hist', where=where)
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


def bar_chart(labels,
              values,
              axis=None,
              orient='h',
              sort=False,
              color=None,
              xlabel=None,
              xlabel_fs=None,
              title=None,
              title_fs=None,
              show_yaxis=True,
              rotation=0
              ):

    cm = get_cmap(random.choice(BAR_CMAPS), len(values), 0.2)
    color = color if color is not None else cm

    if not axis:
        _, axis = plt.subplots()

    if sort:
        sort_idx = np.argsort(values)
        values = values[sort_idx]
        labels = np.array(labels)[sort_idx]

    if orient=='h':
        axis.barh(np.arange(len(values)), values, color=color)
        axis.set_yticks(np.arange(len(values)))
        axis.set_yticklabels(labels, rotation=rotation)

    else:
        axis.bar(np.arange(len(values)), values, color=color)
        axis.set_xticks(np.arange(len(values)))
        axis.set_xticklabels(labels, rotation=rotation)

    if not show_yaxis:
        axis.get_yaxis().set_visible(False)

    if xlabel:
        axis.set_xlabel(xlabel, fontdict={'fontsize': xlabel_fs})

    if title:
        axis.set_title(title, fontdict={'fontsize': title_fs})

    return axis


def save_or_show(path, save: bool = True, fname=None, where='', dpi=300, bbox_inches='tight', close=True,
                 show=False):

    if save:
        assert isinstance(fname, str)
        if "/" in fname:
            fname = fname.replace("/", "__")
        if ":" in fname:
            fname = fname.replace(":", "__")

        save_dir = os.path.join(path, where)

        if not os.path.exists(save_dir):
            assert os.path.dirname(where) in ['',
                                              'activations',
                                              'weights',
                                              'plots', 'data', 'results'], f"unknown directory: {where}"
            save_dir = os.path.join(path, where)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        fname = os.path.join(save_dir, fname + ".png")

        plt.savefig(fname, dpi=dpi, bbox_inches=bbox_inches)
    if show:
        plt.show()

    if close:
        plt.close('all')
    return


def get_cmap(cm:str, num_cols:int, low=0.0, high=1.0):

    cols = getattr(plt.cm, cm)(np.linspace(low, high, num_cols))
    return cols
