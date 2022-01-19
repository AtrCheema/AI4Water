import os
from typing import Union, Callable

import numpy as np
import pandas as pd
import matplotlib as mpl
from easy_mpl import regplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import plot_roc_curve, plot_confusion_matrix, plot_precision_recall_curve

try:
    import plotly
except ModuleNotFoundError:
    plotly = None

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

from .plotting_tools import save_or_show, to_1d_array
from .utils import dateandtime_now, ts_features, dict_to_file


# TODO add Murphy's plot as shown in MLAir
# prediction_distribution aka actual_plot of PDPbox
# average_target_value aka target_plot of PDPbox
# competitive skill score plot/ bootstrap skill score plot as in MLAir
# rank histogram and reliability diagram for probabilitic forecasting model.
# show availability plot of data

#

class Plot(object):

    def __init__(self, path=None, backend='plotly', save=True, dpi=300):
        self.path = path
        self.backend = backend
        self.save = save
        self.dpi = dpi

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, x):

        _backend = x
        assert x in ['plotly', 'matplotlib'], f"unknown backend {x}. Allowed values are `plotly` and `matplotlib`"

        if x == 'plotly':
            if plotly is None:
                _backend = 'matplotlib'

        self._backend = _backend

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, x):
        if x is None:
            x = os.getcwd()
        self._path = x

    def save_or_show(self, save: bool = None, fname=None, where='', dpi=None, bbox_inches='tight',
                     close=True, show=False):

        if save is None:
            save = self.save

        if dpi is None:
            dpi = self.dpi

        return save_or_show(self.path, save, fname, where, dpi, bbox_inches, close, show=show)


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
                if len(out_cols)>1:
                    raise NotImplementedError("can not process results with more than 1 output arrays")
                else:
                    out_cols = list(out_cols.values())[0]

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
                                  show=self.verbosity)

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


def linear_model(
        model_name: str,
        inputs,
        target
):
    import sklearn
    from ai4water.backend import get_attributes

    models = get_attributes(sklearn, "linear_model", case_sensitive=True)
    if model_name not in models:
        raise ValueError(f"Can not find {model_name} in sklearn.linear_model")
    model = models[model_name]
    reg = model().fit(inputs, target)

    return reg.predict(inputs)


def murphy_diagram(
        observed: Union[list, np.ndarray, pd.Series, pd.DataFrame],
        predicted: Union[list, np.ndarray, pd.Series, pd.DataFrame],
        reference: Union[list, np.ndarray, pd.Series, pd.DataFrame] = None,
        reference_model: Union[str, Callable] = None,
        inputs=None,
        plot_type: str = "scores",
        xaxis: str = "theta",
        ax: plt.Axes = None,
        line_colors: tuple = None,
        fill_color: str = "lightgray",
        show: bool = True
) -> plt.Axes:
    """Murphy diagram as introducted by [Ehm et al., 2015](https://arxiv.org/pdf/1503.08195.pdf)
     and illustrated by [Rob Hyndman](https://robjhyndman.com/hyndsight/murphy-diagrams/)

    Arguments:
        observed:
            observed or true values
        predicted:
             model's prediction
        reference:
             reference prediction
        reference_model:
             The model for reference prediction. Only relevent if `reference` is
             None and `plot_type` is `diff`. It can be callable or a string. If it is a
             string, then it can be any model name from
             [sklearn.linear_model](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)
        inputs:
             inputs for reference model. Only relevent if `reference_model` is not
             None and `plot_type` is `diff`
        plot_type:
             either of `scores` or `diff`
        xaxis:
             either of `theta` or `time`
        ax:
             the axis to use for plotting
        line_colors:
             colors of line
        fill_color:
             color to fill confidence interval
        show:
             whether to show the plot or not

    Returns:
         matplotlib axes

    Example:
        >>> import numpy as np
        >>> from ai4water.utils.visualizations import murphy_diagram
        >>> yy = np.random.randint(1, 1000, 100)
        >>> ff1 = np.random.randint(1, 1000, 100)
        >>> ff2 = np.random.randint(1, 1000, 100)
        >>> murphy_diagram(yy, ff1, ff2)
        ...
        >>> murphy_diagram(yy, ff1, ff2, plot_type="diff")
    """
    assert plot_type in ("scores", "diff")
    assert xaxis in ("theta", "time")

    y = to_1d_array(observed)
    f1 = to_1d_array(predicted)

    if reference is None:
        if plot_type == "diff":
            assert reference_model is not None
            if callable(reference_model):
                reference = reference_model(inputs)
            else:
                assert inputs is not None, f"You must specify the inputs for {reference_model}"
                reference = linear_model(reference_model, inputs, predicted)
            f2 = to_1d_array(reference)
        else:
            f2 = None
    else:
        f2 = to_1d_array(reference)

    line_colors = line_colors or ["dimgrey", "tab:orange"]

    n = len(y)
    _min, _max = np.nanmin(np.hstack([y, f1, f2])), np.nanmax(np.hstack([y, f1, f2]))
    tmp = _min-0.2*(_max-_min), _max+0.2*(_max-_min)

    theta = np.linspace(tmp[0], tmp[1], 501)

    s1 = np.full((501, n), np.nan)
    s2 = np.full((501, n), np.nan)

    max1 = np.maximum(f1, y)
    max2 = np.maximum(f2, y)

    min1 = np.minimum(f1, y)
    min2 = np.minimum(f2, y)

    for j in range(n):

        s1[:, j] = abs(y[j]-theta) * (max1[j] > theta) * (min1[j] <= theta)
        s2[:, j] = abs(y[j] - theta) * (max2[j] > theta) * (min2[j] <= theta)

    # grab the axes
    if ax is None:
        ax = plt.gca()

    if xaxis == "theta":
        s1ave, s2ave = _data_for_theta(s1, s2)
    else:
        raise NotImplementedError

    if plot_type == "scores":
        _plot_scores(theta, s1ave, s2ave, ax, line_colors)
        ax.set_ylabel("Empirical Scores", fontsize=16)
    else:
        _plot_diff(theta, s1, s2, n, ax, line_colors[0], fill_color)
        ax.set_ylabel("Difference in scores", fontsize=16)

    ax.set_xlabel(xaxis, fontsize=16)

    if show:
        plt.show()

    return ax


def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def _plot_diff(theta, s1, s2, n, ax, line_color="black", fill_color="lightgray"):

    se = np.std(s1-s2)/np.sqrt(n)

    diff = np.mean(s1-s2, axis=1)

    upper = diff + 1.96 * se
    lower = diff - 1.96 * se

    ax.plot(theta, diff, color=line_color)

    # first_nonzero occurence
    st = (diff != 0).argmax(axis=0)
    en = last_nonzero(diff, axis=0).item()

    ax.fill_between(theta[st:en], upper[st:en], lower[st:en],  # alpha=0.2,
                    color=fill_color)

    return ax


def fdc_plot(
        sim: Union[list, np.ndarray, pd.Series, pd.DataFrame],
        obs: Union[list, np.ndarray, pd.Series, pd.DataFrame],
        ax: plt.Axes = None,
        legend: bool = True,
        xlabel: str = "Exceedence [%]",
        ylabel: str = "Flow",
        show: bool = True
) -> plt.Axes:
    """Plots flow duration curve

    Arguments:
        sim:
            simulated flow
        obs:
            observed flow
        ax:
            axis on which to plot
        legend:
            whether to apply legend or not
        xlabel:
            label to set on x-axis. set to None for no x-label
        ylabel:
            label to set on y-axis
        show:
            whether to show the plot or not

    Returns:
        matplotlib axes

    Example:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from ai4water.utils.visualizations import fdc_plot
        >>> simulated = np.random.random(100)
        >>> observed = np.random.random(100)
        >>> fdc_plot(simulated, observed)
        >>> plt.show()
    """

    sim = to_1d_array(sim)
    obs = to_1d_array(obs)

    sort_obs = np.sort(sim)[::-1]
    exceedence_obs = np.arange(1., len(sort_obs)+1) / len(sort_obs)
    sort_sim = np.sort(obs)[::-1]
    exceedence_sim = np.arange(1., len(sort_sim)+1) / len(sort_sim)

    if ax is None:
        ax = plt.gca()

    ax.plot(exceedence_obs * 100, sort_obs, color='b', label="Observed")
    ax.plot(exceedence_sim * 100, sort_sim, color='r', label="Simulated")

    if legend:
        ax.legend()

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if show:
        plt.show()

    return ax


def _plot_scores(theta, s1ave, s2ave, ax, line_colors):

    ax.plot(theta, s1ave, color=line_colors[0])
    ax.plot(theta, s2ave, color=line_colors[1])

    return ax


def _data_for_time(s1, s2):
    s1ave, s2ave = np.mean(s1, axis=0), np.mean(s2, axis=0)

    return s1ave, s2ave


def _data_for_theta(s1, s2):
    return np.mean(s1, axis=1), np.mean(s2, axis=1)


def _wandb_scatter(true: np.ndarray, predicted: np.ndarray, name: str) -> None:
    """Adds a scatter plot on wandb."""
    data = [[x, y] for (x, y) in zip(true.reshape(-1,), predicted.reshape(-1,))]
    table = wandb.Table(data=data, columns=["true", "predicted"])
    wandb.log({
        "scatter_plot": wandb.plot.scatter(table, "true", "predicted",
                                           title=name)
               })
    return


def init_subplots(width=None, height=None, nrows=1, ncols=1, **kwargs):
    """Initializes the fig for subplots"""
    plt.close('all')
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, **kwargs)
    if width is not None:
        fig.set_figwidth(width)
    if height is not None:
        fig.set_figheight(height)
    return fig, ax