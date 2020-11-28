import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import plot_roc_curve, plot_confusion_matrix, plot_precision_recall_curve
from xgboost import plot_importance

class Plots(object):
    # TODO initialte this class with at least path

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

    def save_or_show(self, save: bool = True, fname=None, where='act'):
        if save:
            assert isinstance(fname, str)
            if "/" in fname:
                fname = fname.replace("/", "__")
            if ":" in fname:
                fname = fname.replace(":", "__")

            if where.upper() == 'DATA':
                fpath = os.path.join(self.path, 'data')
                if not os.path.exists(fpath):
                    os.makedirs(fpath)
                fname = os.path.join(fpath, fname)
            elif where.upper() == 'ACT':
                fname = os.path.join(self.act_path, fname + ".png")
            else:
                fname = os.path.join(self.path, fname + ".png")

            plt.savefig(fname)
        else:
            plt.show()

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

    def features_2d(self, data, lyr_name,
                    reflect_half=False,
                    timesteps_xaxis=False, max_timesteps=None,
                    **kwargs):
        """Plots 2D heatmaps in a standalone graph or subplot grid.

        iter == list/tuple (both work)

        Arguments:
            data: np.ndarray, 2D/3D. Data to plot.
                  2D -> standalone graph; 3D -> subplot grid.
                  3D: (samples, timesteps, channels)
                  2D: (timesteps, channels)

            reflect_half: bool. If True, second half of channels dim will be
                  flipped about the timesteps dim.
            timesteps_xaxis: bool. If True, the timesteps dim (`data` dim 1)
                  if plotted along the x-axis.
            max_timesteps:  int/None. Max number of timesteps to show per plot.
                  If None, keeps original.

        kwargs:
            w: float. Scale width  of resulting plot by a factor.
            h: float. Scale height of resulting plot by a factor.
            show_borders:  bool.  If True, shows boxes around plot(s).
            show_xy_ticks: int/bool iter. Slot 0 -> x, Slot 1 -> y.
                  Ex: [1, 1] -> show both x- and y-ticks (and their labels).
                      [0, 0] -> hide both.
            show_colorbar: bool. If True, shows one colorbar next to plot(s).
            title: bool/str. If True, shows generic supertitle.
                  If str in {'grads', 'outputs', 'generic'}, shows supertitle
                  tailored to `data` dim (2D/3D). If other str, shows `title`
                  as supertitle. If False, no title is shown.
            tight: bool. If True, plots compactly by removing subplot padding.
            channel_axis: int, 0 or -1. `data` axis holding channels/features.
                  -1 --> (samples,  timesteps, channels)
                  0  --> (channels, timesteps, samples)
            borderwidth: float / None. Width of subplot borders.
            bordercolor: str / None. Color of subplot borders. Default black.
            save: bool, .

        Returns:
            (figs, axes) of generated plots.
        """

        w, h          = kwargs.get('w', 1), kwargs.get('h', 1)
        show_borders  = kwargs.get('show_borders', True)
        show_xy_ticks = kwargs.get('show_xy_ticks', (1, 1))
        show_colorbar = kwargs.get('show_colorbar', False)
        tight         = kwargs.get('tight', False)
        channel_axis  = kwargs.get('channel_axis', -1)
        borderwidth   = kwargs.get('borderwidth', None)
        bordercolor   = kwargs.get('bordercolor', None)
        save      = kwargs.get('savepath', True)


        def _process_data(data, max_timesteps, reflect_half,
                          timesteps_xaxis, channel_axis):
            if data.ndim not in (2, 3):
                raise Exception("`data` must be 2D or 3D (got ndim=%s)" % data.ndim)

            if max_timesteps is not None:
                data = data[..., :max_timesteps, :]

            if reflect_half:
                data = data.copy()  # prevent passed array from changing
                half_chs = data.shape[-1] // 2
                data[..., half_chs:] = np.flip(data[..., half_chs:], axis=0)

            if data.ndim != 3:
                # (1, width, height) -> one image
                data = np.expand_dims(data, 0)
            if timesteps_xaxis:
                data = np.transpose(data, (0, 2, 1))
            return data

        def _style_axis(ax, show_borders, show_xy_ticks):
            ax.axis('tight')
            if not show_xy_ticks[0]:
                ax.set_xticks([])
            if not show_xy_ticks[1]:
                ax.set_yticks([])
            if not show_borders:
                ax.set_frame_on(False)

        if isinstance(show_xy_ticks, (int, bool)):
            show_xy_ticks = (show_xy_ticks, show_xy_ticks)
        data = _process_data(data, max_timesteps, reflect_half,
                             timesteps_xaxis, channel_axis)
        n_rows, n_cols = _get_nrows_and_ncols(n_subplots=len(data))

        fig, axes = plt.subplots(n_rows, n_cols, dpi=76, figsize=(10, 10), sharex='all', sharey='all')
        axes = np.asarray(axes)

        fig.suptitle(lyr_name, weight='bold', fontsize=14, y=.93 + .12 * tight)

        for ax_idx, ax in enumerate(axes.flat):
            img = ax.imshow(data[ax_idx], cmap='bwr', vmin=-1, vmax=1)
            _style_axis(ax, show_borders, show_xy_ticks)

        if show_colorbar:
            fig.colorbar(img, ax=axes.ravel().tolist())
        if tight:
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        if borderwidth is not None or bordercolor is not None:
            for ax in axes.flat:
                for s in ax.spines.values():
                    if borderwidth is not None:
                        s.set_linewidth(borderwidth)
                    if bordercolor is not None:
                        s.set_color(bordercolor)

        self.save_or_show(save, lyr_name)

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
        self.save_or_show(save, fname="feature_importance", where='results')
        return

    def plot_feature_feature_corr(self, remove_targets=True, save=True, **kwargs):
        plt.close('')
        cols = self.in_cols if remove_targets else self.in_cols + self.out_cols
        corr = self.data[cols].corr()
        sns.heatmap(corr, **kwargs)
        self.save_or_show(save, fname="feature_feature_corr", where="data")
        return

    def plot_roc_curve(self, x, y, save=True):
        assert self.problem.upper().startswith("CLASS")
        plot_roc_curve(self._model, *x, y.reshape(-1, ))
        self.save_or_show(save, fname="roc", where="results")
        return

    def plot_confusion_matrx(self, x, y, save=True):
        assert self.problem.upper().startswith("CLASS")
        plot_confusion_matrix(self._model, *x, y.reshape(-1, ))
        self.save_or_show(save, fname="confusion_matrix", where="results")
        return

    def plot_precision_recall_curve(self, x, y, save=True):
        assert self.problem.upper().startswith("CLASS")
        plot_precision_recall_curve(self._model, *x, y.reshape(-1, ))
        self.save_or_show(save, fname="plot_precision_recall_curve", where="results")
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