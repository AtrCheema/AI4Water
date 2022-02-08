
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from easy_mpl import regplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import plot_roc_curve, plot_confusion_matrix, plot_precision_recall_curve

from ai4water.utils.visualizations import Plot, init_subplots
from ai4water.utils.utils import dateandtime_now, ts_features, dict_to_file

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

# TODO add Murphy's plot as shown in MLAir
# prediction_distribution aka actual_plot of PDPbox
# average_target_value aka target_plot of PDPbox
# competitive skill score plot/ bootstrap skill score plot as in MLAir
# rank histogram and reliability diagram for probabilitic forecasting model.
# show availability plot of data
# residual plot as in yellow brick
# classification report as in yellow brick
# class prediction error as in yellow brick
# discremination threshold as in yellow brick

class ProcessResults(Plot):
    """post processing of results after training"""
    def __init__(self,
                 forecast_len=None,
                 output_features=None,
                 is_multiclass=None,
                 config: dict = None,
                 path=None,
                 dpi=300,
                 in_cols=None,
                 out_cols=None,
                 verbosity=1,
                 backend: str = 'plotly'
                 ):

        self.forecast_len = forecast_len
        self.output_features = output_features
        self.is_multiclass = is_multiclass
        self.config = config
        self.dpi = dpi
        self.in_cols = in_cols
        self.out_cols = out_cols
        self.quantiles = None
        self.verbosity = verbosity

        super().__init__(path, backend=backend)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, x):
        self._config = x

    @property
    def quantiles(self):
        return self._quantiles

    @quantiles.setter
    def quantiles(self, x):
        self._quantiles = x

    def horizon_plots(self, errors: dict, fname='', save=True):
        plt.close('')
        _, axis = plt.subplots(len(errors), sharex='all')

        legends = {'r2': "$R^2$", 'rmse': "RMSE", 'nse': "NSE"}
        idx = 0
        for metric_name, val in errors.items():
            ax = axis[idx]
            ax.plot(val, '--o', label=legends.get(metric_name, metric_name))
            ax.legend(fontsize=14)
            if idx >= len(errors)-1:
                ax.set_xlabel("Horizons", fontsize=14)
            ax.set_ylabel(legends.get(metric_name, metric_name), fontsize=14)
            idx += 1
        self.save_or_show(save=save, fname=fname)
        return

    def plot_results(self, true, predicted: pd.DataFrame, save=True, name=None, where=None,
                     annotation_key=None, annotation_val=None, show=False):
        """
        # kwargs can be any/all of followings
            # fillstyle:
            # marker:
            # linestyle:
            # markersize:
            # color:
        """

        regplot(true,
                predicted,
                title=name,
                annotation_key=annotation_key,
                annotation_val=annotation_val,
                show=False
                )

        self.save_or_show(save=save, fname=f"{name}_reg", close=False,
                          where=where, show=show)

        mpl.rcParams.update(mpl.rcParamsDefault)

        _, axis = init_subplots(width=12, height=8)

        # it is quite possible that when data is datetime indexed, then it is not
        # equalidistant and large amount of graph
        # will have not data in that case lines plot will create a lot of useless
        # interpolating lines where no data is present.
        datetime_axis = False
        if isinstance(true.index, pd.DatetimeIndex) and pd.infer_freq(true.index) is not None:
            style = '.'
            true = true
            predicted = predicted
            datetime_axis = True
        else:
            if np.isnan(true.values).sum() > 0:
                style = '.'  # For Nan values we should be using this style otherwise nothing is plotted.
            else:
                style = '-'
                true = true.values
                predicted = predicted.values

        ms = 4 if style == '.' else 2

        if len(true) > 1000:  # because the data is very large, so better to use small marker size
            ms = 2

        axis.plot(predicted, style, color='r',  label='Prediction')

        axis.plot(true, style, color='b', marker='o', fillstyle='none',  markersize=ms, label='True')

        axis.legend(loc="best", fontsize=22, markerscale=4)

        if datetime_axis:
            loc = mdates.AutoDateLocator(minticks=4, maxticks=6)
            axis.xaxis.set_major_locator(loc)
            fmt = mdates.AutoDateFormatter(loc)
            axis.xaxis.set_major_formatter(fmt)

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel("Time", fontsize=18)

        self.save_or_show(save=save, fname=name, close=False, where=where)
        return

    def plot_loss(self, history: dict, name="loss_curve", show=False):
        """Considering history is a dictionary of different arrays, possible
        training and validation loss arrays, this method plots those arrays."""

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
            "kge": "Kling-Gupta Efficiency",
            "tf_r2": "$R^{2}$",
            "r2": "$R^{2}$"
        }

        sub_plots = {1: {'axis': (1, 1), 'width': 9, 'height': 6},
                     2: {'axis': (1, 2), 'width': 9, 'height': 6},
                     3: {'axis': (1, 3), 'width': 9, 'height': 6},
                     4: {'axis': (2, 2), 'width': 9, 'height': 6},
                     5: {'axis': (5, 1), 'width': 8, 'height': 12},
                     6: {'axis': (3, 2), 'width': 8, 'height': 12},
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
                axis.plot(epochs, val, color=[0.96707953, 0.46268314, 0.45772886], label='Validation ')
                axis.legend()
            else:
                axis = fig.add_subplot(*sub_plots[len(history)]['axis'], i)
                axis.plot(epochs, val, color=[0.13778617, 0.06228198, 0.33547859], label='Training ')
                axis.legend()
                axis.set_xlabel("Epochs")
                axis.set_ylabel(legends.get(key, key))
                axis_cache[key] = axis
                i += 1
            axis.set(frame_on=True)

        fig.set_figheight(sub_plots[len(history)]['height'])
        fig.set_figwidth(sub_plots[len(history)]['width'])
        self.save_or_show(fname=name, save=True if name is not None else False, show=show)
        mpl.rcParams.update(mpl.rcParamsDefault)
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

    def roc_curve(self, estimator, x, y, save=True):

        plot_roc_curve(estimator, x, y.reshape(-1, ))
        self.save_or_show(save, fname="roc", where="results")
        return

    def confusion_matrx(self,estimator, x, y, save=True):

        plot_confusion_matrix(estimator, x, y.reshape(-1, ))
        self.save_or_show(save, fname="confusion_matrix", where="results")
        return

    def precision_recall_curve(self, estimator, x, y, save=True):

        plot_precision_recall_curve(estimator, x, y.reshape(-1, ))
        self.save_or_show(save, fname="plot_precision_recall_curve", where="results")
        return

    def process_regres_results(
            self,
            true: np.ndarray,
            predicted: np.ndarray,
            metrics="minimal",
            prefix=None,
            index=None,
            remove_nans=True,
            user_defined_data: bool = False,
            annotate_with="r2",
    ):
        """
        predicted, true are arrays of shape (examples, outs, forecast_len).
        annotate_with : which value to write on regression plot
        """
        from ai4water.postprocessing.SeqMetrics import RegressionMetrics

        metric_names = {'r2': "$R^2$"}

        if user_defined_data:
            # when data is user_defined, we don't know what out_cols, and forecast_len are
            if predicted.ndim == 1:
                out_cols = ['output']
            else:
                out_cols = [f'output_{i}' for i in range(predicted.shape[-1])]
            forecast_len = 1
            true, predicted = self.maybe_not_3d_data(true, predicted)
        else:
            # for cases if they are 2D/1D, add the third dimension.
            true, predicted = self.maybe_not_3d_data(true, predicted)

            forecast_len = self.forecast_len
            if isinstance(forecast_len, dict):
                forecast_len = np.unique(list(forecast_len.values())).item()

            out_cols = self.output_features
            if isinstance(out_cols, dict):
                _out_cols = []
                for cols in out_cols.values():
                    _out_cols = _out_cols + cols
                out_cols = _out_cols
                if len(out_cols)>1 and not isinstance(predicted, np.ndarray):
                    raise NotImplementedError("can not process results with more than 1 output arrays")

        for idx, out in enumerate(out_cols):

            horizon_errors = {metric_name: [] for metric_name in ['nse', 'rmse']}
            for h in range(forecast_len):

                errs = dict()

                fpath = os.path.join(self.path, out)
                if not os.path.exists(fpath):
                    os.makedirs(fpath)

                t = pd.DataFrame(true[:, idx, h], index=index, columns=['true_' + out])
                p = pd.DataFrame(predicted[:, idx, h], index=index, columns=['pred_' + out])

                if wandb is not None and self.config['wandb_config'] is not None:
                    _wandb_scatter(t.values, p.values, out)

                df = pd.concat([t, p], axis=1)
                df = df.sort_index()
                fname = prefix + out + '_' + str(h) + dateandtime_now() + ".csv"
                df.to_csv(os.path.join(fpath, fname), index_label='index')

                annotation_val = getattr(RegressionMetrics(t, p), annotate_with)()
                self.plot_results(t,
                                  p,
                                  name=prefix + out + '_' + str(h),
                                  where=out,
                                  annotation_key=metric_names.get(annotate_with, annotate_with),
                                  annotation_val=annotation_val,
                                  show=bool(self.verbosity))

                if remove_nans:
                    nan_idx = np.isnan(t)
                    t = t.values[~nan_idx]
                    p = p.values[~nan_idx]

                errors = RegressionMetrics(t, p)
                errs[out + '_errors_' + str(h)] = getattr(errors, f'calculate_{metrics}')()
                errs[out + 'true_stats_' + str(h)] = ts_features(t)
                errs[out + 'predicted_stats_' + str(h)] = ts_features(p)

                dict_to_file(fpath, errors=errs, name=prefix)

                for p in horizon_errors.keys():
                    horizon_errors[p].append(getattr(errors, p)())

            if forecast_len > 1:
                self.horizon_plots(horizon_errors, f'{prefix}_{out}_horizons.png')
        return

    def process_class_results(
            self,
            true: np.ndarray,
            predicted: np.ndarray,
            metrics="minimal",
            prefix=None,
            index=None,
            user_defined_data: bool = False
    ):
        """post-processes classification results."""
        if user_defined_data:
            return

        from ai4water.postprocessing.SeqMetrics import ClassificationMetrics

        if self.is_multiclass:
            pred_labels = [f"pred_{i}" for i in range(predicted.shape[1])]
            true_labels = [f"true_{i}" for i in range(true.shape[1])]
            fname = os.path.join(self.path, f"{prefix}_prediction.csv")
            pd.DataFrame(np.concatenate([true, predicted], axis=1),
                         columns=true_labels + pred_labels, index=index).to_csv(fname)
            class_metrics = ClassificationMetrics(true, predicted, multiclass=True)

            dict_to_file(self.path,
                         errors=class_metrics.calculate_all(),
                         name=f"{prefix}_{dateandtime_now()}.json")
        else:
            if predicted.ndim == 1:
                predicted = predicted.reshape(-1, 1)
            for idx, _class in enumerate(self.output_features):
                _true = true[:, idx]
                _pred = predicted[:, idx]

                fpath = os.path.join(self.path, _class)
                if not os.path.exists(fpath):
                    os.makedirs(fpath)

                class_metrics = ClassificationMetrics(_true, _pred, multiclass=False)
                dict_to_file(fpath,
                             errors=getattr(class_metrics, f"calculate_{metrics}")(),
                             name=f"{prefix}_{_class}_{dateandtime_now()}.json"
                             )

                fname = os.path.join(fpath, f"{prefix}_{_class}.csv")
                array = np.concatenate([_true.reshape(-1, 1), _pred.reshape(-1, 1)], axis=1)
                pd.DataFrame(array, columns=['true', 'predicted'],  index=index).to_csv(fname)

        return

    def maybe_not_3d_data(self, true, predicted):

        forecast_len = self.forecast_len

        if true.ndim < 3:
            if isinstance(forecast_len, dict):
                forecast_len = set(list(forecast_len.values()))
                assert len(forecast_len) == 1
                forecast_len = forecast_len.pop()

            assert forecast_len == 1, f'{forecast_len}'
            axis = 2 if true.ndim == 2 else (1, 2)
            true = np.expand_dims(true, axis=axis)

        if predicted.ndim < 3:
            assert forecast_len == 1
            axis = 2 if predicted.ndim == 2 else (1, 2)
            predicted = np.expand_dims(predicted, axis=axis)

        return true, predicted


def choose_examples(x, examples_to_use, y=None):
    """Chooses exampels from x and y"""
    if isinstance(examples_to_use, int):
        if len(x) == examples_to_use:
            # since using all the examples, don't randomize them
            x, index = x, np.arange(examples_to_use)
        else:
            # randomly chose x values from test_x
            x, index = choose_n_imp_exs(x, examples_to_use, y)
    elif isinstance(examples_to_use, float):
        assert examples_to_use < 1.0
        # randomly choose x fraction from test_x
        x, index = choose_n_imp_exs(x, int(examples_to_use * len(x)), y)

    elif hasattr(examples_to_use, '__len__'):
        index = np.array(examples_to_use)
        x = x[index]
    else:
        raise ValueError(f"unrecognized value of examples_to_use: {examples_to_use}")

    return x, index


def choose_n_imp_exs(x: np.ndarray, n: int, y=None):
    """Chooses the n important examples from x and y"""

    n = min(len(x), n)

    st = n // 2
    en = n - st

    if y is None:
        idx = np.random.randint(0, len(x), n)
    else:
        st = np.argsort(y, axis=0)[0:st].reshape(-1,)
        en = np.argsort(y, axis=0)[-en:].reshape(-1,)
        idx = np.hstack([st, en])

    x = x[idx]

    return x, idx


def _wandb_scatter(true: np.ndarray, predicted: np.ndarray, name: str) -> None:
    """Adds a scatter plot on wandb."""
    data = [[x, y] for (x, y) in zip(true.reshape(-1,), predicted.reshape(-1,))]
    table = wandb.Table(data=data, columns=["true", "predicted"])
    wandb.log({
        "scatter_plot": wandb.plot.scatter(table, "true", "predicted",
                                           title=name)
               })
    return
