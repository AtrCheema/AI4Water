import os
import random
from typing import Union

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    import plotly
except ModuleNotFoundError:
    plotly = None

from .plotting_tools import save_or_show
from ai4water.utils.utils import init_subplots

# TODO add Murphy's plot as shown in MLAir
# https://robjhyndman.com/hyndsight/murphy-diagrams/
# competitive skill score plot/ bootstrap skill score plot as in MLAir
# rank histogram and reliability diagram for probabilitic forecasting model.
# show availability plot of data


regplot_combs = [
    ['cadetblue', 'slateblue', 'darkslateblue'],
    ['cadetblue', 'mediumblue', 'mediumblue'],
    ['cornflowerblue', 'dodgerblue', 'darkblue'],
    ['cornflowerblue', 'dodgerblue', 'steelblue'],
    ['cornflowerblue', 'mediumblue', 'dodgerblue'],
    ['cornflowerblue', 'steelblue', 'mediumblue'],
    ['darkslateblue', 'aliceblue', 'mediumblue'],
    ['darkslateblue', 'blue', 'royalblue'],
    ['darkslateblue', 'blueviolet', 'royalblue'],
    ['darkslateblue', 'darkblue', 'midnightblue'],
    ['darkslateblue', 'mediumblue', 'darkslateblue'],
    ['darkslateblue', 'midnightblue', 'mediumblue'],
    ['seagreen', 'darkslateblue', 'cadetblue'],
    ['cadetblue', 'darkblue', 'midnightblue'],
    ['cadetblue', 'deepskyblue', 'cadetblue']
]

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


class PlotResults(Plot):

    def __init__(self,
                 data=None,
                 config: dict = None,
                 path=None,
                 dpi=300,
                 in_cols=None,
                 out_cols=None,
                 backend: str = 'plotly'
                 ):
        self.config = config
        self.data = data
        self.dpi = dpi
        self.in_cols = in_cols
        self.out_cols = out_cols

        super().__init__(path, backend=backend)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, x):
        self._config = x

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, x):
        self._data = x

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

        regplot(true, predicted, title=name,
                annotation_key=annotation_key,
                annotation_val=annotation_val)

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

        if len(true)>1000: # because the data is very large, so better to use small marker size
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


def regplot(
        x:Union[np.ndarray, pd.DataFrame, pd.Series, list],
        y:Union[np.ndarray, pd.DataFrame, pd.Series, list],
        title:str = None,
        show:bool = False,
        annotation_key:str=None,
        annotation_val=None,
        line_color = None,
        marker_color = None,
        fill_color = None,
        marker_size:int = 20,
        ci:Union[int, None] = 95,
        figsize:tuple = None,
        xlabel:str = 'Observed',
        ylabel:str = 'Predicted'
):
    """
    Regpression plot with regression line and confidence interval

    Arguments:
        x : array like, the 'x' value.
        y : array like
        ci : confidence interval. Set to None if not required.
        show : whether to show the plot or not
        annotation_key : The name of the value to annotate with.
        annotation_val : The value to annotate with.
        marker_size :
        line_color :
        marker_color:
        fill_color : only relevent if ci is not None.
        figsize : tuple
        title : name to be used for title
        xlabel :
        ylabel :

    Example
    --------
    ```python
    >>>from ai4water.datasets import arg_beach
    >>>from ai4water.utils.visualizations import regplot
    >>>data = arg_beach()
    >>>regplot(data['pcp3_mm'], data['pcp6_mm'], show=True)
    ```
    """
    x = to_1d_array(x)
    y = to_1d_array(y)

    mc, lc, fc = random.choice(regplot_combs)
    _metric_names = {'r2': '$R^2$'}
    plt.close('all')

    _, axis = plt.subplots(figsize=figsize or (6, 5))

    axis.scatter(x, y, c=marker_color or mc,
                         s=marker_size)  # set style options

    if annotation_key is not None:
        assert annotation_val is not None

        plt.annotate(f'{annotation_key}: {round(annotation_val, 3)}',
                     xy=(0.3, 0.95),
                     xycoords='axes fraction',
                     horizontalalignment='right', verticalalignment='top', fontsize=16)
    _regplot(x,
             y,
             ax=axis,
             ci=ci,
             line_color=line_color or lc,
             fill_color=fill_color or fc)

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=26)

    if show:
        plt.show()

    return axis


def _ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return np.nanpercentile(a, p, axis)


def reg_func(_x, _y):
    return np.linalg.pinv(_x).dot(_y)


def bootdist(f, args, n_boot=1000, **func_kwargs):

    n = len(args[0])
    integers = np.random.randint
    boot_dist = []
    for i in range(int(n_boot)):
        resampler = integers(0, n, n, dtype=np.intp)  # intp is indexing dtype
        sample = [a.take(resampler, axis=0) for a in args]
        boot_dist.append(f(*sample, **func_kwargs))

    return np.array(boot_dist)


def _regplot(x, y, ax, ci=None, line_color=None, fill_color=None):

    grid = np.linspace(np.min(x), np.max(x), 100)
    X = np.c_[np.ones(len(x)), x]
    grid = np.c_[np.ones(len(grid)), grid]
    yhat = grid.dot(reg_func(X, y))

    ax.plot(grid[:, 1], yhat, color=line_color)

    if ci:
        boots = bootdist(reg_func, args=[X, y], n_boot=1000).T

        yhat_boots = grid.dot(boots).T

        err_bands = _ci(yhat_boots, ci, axis=0)

        ax.fill_between(grid[:, 1], *err_bands,
                        facecolor=fill_color,
                        alpha=.15)
    return ax


def to_1d_array(array_like)->np.ndarray:

    if array_like.__class__.__name__ in ['list', 'tuple', 'Series']:
        return np.array(array_like)

    elif array_like.__class__.__name__ == 'ndarray':
        if array_like.ndim == 1:
            return array_like
        else:
            assert array_like.size == len(array_like), f'cannot convert multidim ' \
                                                       f'array of shape {array_like.shape} to 1d'
            return array_like.reshape(-1, )

    elif array_like.__class__.__name__ == 'DataFrame' and array_like.ndim == 2:
        return array_like.values.reshape(-1,)
    else:
        raise ValueError(f'cannot convert object array {array_like.__class__.__name__}  to 1d ')


def linear_model(
        model_name:str,
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
        observed,
        predicted,
        reference=None,
        reference_model=None,
        inputs=None,
        plot_type="scores",
        xaxis = "theta",
        ax=None,
        line_colors=None,
        fill_color="lightgray",
        show=True
)->plt.axis:
    """Murphy diagram as introducted by [Ehm et al., 2015](https://arxiv.org/pdf/1503.08195.pdf)
     and illustrated by [Rob Hyndman](https://robjhyndman.com/hyndsight/murphy-diagrams/)

     Arguments:
         observed : array like
            observed or true values
         predicted : array like
            model's prediction
         reference : array like
            reference prediction
        reference_model :
            The model for reference prediction. Only relevent if `reference` is
            None and `plot_type` is `diff`. It can be callable or a string. If it is a
            string, then it can be any model name from [sklearn.linear_model](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)
        inputs :
            inputs for reference model. Only relevent if `reference_model` is not
            None and `plot_type` is `diff`
        plot_type :
            either of `scores` or `diff`
        xaxis :
            either of `theta` or `time`
        ax :
            the axis to use for plotting
        line_colors : tuple
            colors of line
        fill_color :
            color to fill confidence interval
        show :
            whether to show the plot or not

    Returns:
        matplotlib axis

    Example
    -------
    ```python
    >>> import numpy as np
    >>> from ai4water.utils.visualizations import murphy_diagram
    >>> yy = np.random.randint(1, 1000, 100)
    >>> ff1 = np.random.randint(1, 1000, 100)
    >>> ff2 = np.random.randint(1, 1000, 100)
    >>> murphy_diagram(yy, ff1, ff2)
    ...
    >>> murphy_diagram(yy, ff1, ff2, plot_type="diff")
    ```
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
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

def _plot_diff(theta, s1, s2, n, ax, line_color="black", fill_color="lightgray"):

    se = np.std(s1-s2)/np.sqrt(n)

    diff = np.mean(s1-s2, axis=1)

    upper = diff + 1.96 * se
    lower = diff - 1.96 * se

    ax.plot(theta, diff, color=line_color)

    # first_nonzero occurence
    st = (diff!=0).argmax(axis=0)
    en = last_nonzero(diff, axis=0).item()

    ax.fill_between(theta[st:en], upper[st:en], lower[st:en], #alpha=0.2,
                    color=fill_color)

    return ax



def fdc_plot(
        sim:Union[list, np.ndarray, pd.Series, pd.DataFrame],
        obs:Union[list, np.ndarray, pd.Series, pd.DataFrame],
        ax=None,
        legend=True,
        xlabel:str = "Exceedence [%]",
        ylabel:str = "Flow",
        show:bool = True
):
    """Plots flow duration curve
    Arguments:
        sim :
            simulated
        obs :
            observed
        ax :
            axis on which to plot
        legend :
            whether to apply legend or not
        xlabel :
            label to set on x-axis. set to None for no x-label
        ylabel :
            label to set on y-axis
        show :
            whether to show the plot or not

    Example
    -------
    ```python
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from ai4water.utils.visualizations import fdc_plot
    >>> simulated = np.random.random(100)
    >>> observed = np.random.random(100)
    >>> fdc_plot(simulated, observed)
    >>> plt.show()
    ```
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