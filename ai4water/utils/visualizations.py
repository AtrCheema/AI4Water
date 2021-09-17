import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

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

    def save_or_show(self, save: bool = None, fname=None, where='', dpi=None, bbox_inches='tight', close=True):

        if save is None:
            save = self.save

        if dpi is None:
            dpi = self.dpi

        return save_or_show(self.path, save, fname, where, dpi, bbox_inches, close)


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
                     annotation_key=None, annotation_val=None):
        """
        # kwargs can be any/all of followings
            # fillstyle:
            # marker:
            # linestyle:
            # markersize:
            # color:
        """

        regplot(true, predicted, save=save, title=name, annotation_key=annotation_key, annotation_val=annotation_val)
        self.save_or_show(save=save, fname=f"{name}_reg", close=False, where=where)

        mpl.rcParams.update(mpl.rcParamsDefault)

        _, axis = init_subplots(width=12, height=8)

        # it is quite possible that when data is datetime indexed, then it is not
        # equalidistant and large amount of graph
        # will have not data in that case lines plot will create a lot of useless
        # interpolating lines where no data is present.
        if isinstance(true.index, pd.DatetimeIndex) and pd.infer_freq(true.index) is not None:
            style = '.'
            true = true
            predicted = predicted
        else:
            if np.isnan(true.values).sum() > 0:
                style = '.'  # For Nan values we should be using this style otherwise nothing is plotted.
            else:
                style = '-'
                true = true.values
                predicted = predicted.values

        ms = 4 if style == '.' else 2

        axis.plot(predicted, style, color='r',  label='Prediction')

        axis.plot(true, style, color='b', marker='o', fillstyle='none',  markersize=ms, label='True')

        axis.legend(loc="best", fontsize=22, markerscale=4)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel("Time", fontsize=18)

        self.save_or_show(save=save, fname=name, close=False, where=where)
        return

    def plot_loss(self, history: dict, name="loss_curve"):
        """Considering history is a dictionary of different arrays, possible training and validation loss arrays,
        this method plots those arrays."""

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
        self.save_or_show(fname=name, save=True if name is not None else False)
        return


def regplot(true, pred, title=None, show=False,
            annotation_key=None, annotation_val=None,
            **kwargs):
    """
    :param true: array like
    :param pred, array like
    :param title, name to be used for title
    :param show : whether to show the plot or not
    :param annotation_key :
    :param annotation_val :
    Following kwargs are allowed:
        figsize: tuple
        colorbar: for plt.colorbar
        cmap: for plt.scatter
        s: for plt.scatter

    :Note, This function will neither show nor saves the plot. The user has to manually
           do it after calling this function as shown below.

    >>>from ai4water.utils.visualizations import regplot
    >>>import numpy as np
    >>>t = np.random.random(100)
    >>>p = np.random.random(100)
    >>>regplot(t, p)
    >>>plt.show()
    """
    # https://seaborn.pydata.org/generated/seaborn.regplot.html
    if any([isinstance(true, _type) for _type in [pd.DataFrame, pd.Series]]):
        true = true.values.reshape(-1,)
    if any([isinstance(pred, _type) for _type in [pd.DataFrame, pd.Series]]):
        pred = pred.values.reshape(-1,)

    plt.close('all')

    s = kwargs.get('s', 20)
    cmap = kwargs.get('cmap', 'winter')  # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    figsize = kwargs.get('figsize', (8, 5.5))

    plt.figure(figsize=figsize)
    points = plt.scatter(true, pred, c=pred, s=s, cmap=cmap)  # set style options

    if annotation_key is not None:
        assert annotation_val is not None

        plt.annotate(f'{annotation_key}: {round(annotation_val, 3)}', xy=(0.3, 0.95),
                     xycoords='axes fraction',
                     horizontalalignment='right', verticalalignment='top', fontsize=16)

    if kwargs.get('colorbar', False):
        plt.colorbar(points)

    sns.regplot(x=true, y=pred, scatter=False, color=".1")
    plt.xlabel('Observed', fontsize=14)
    plt.ylabel('Predicted', fontsize=14)
    plt.title(title, fontsize=26)

    if show:
        plt.show()

    return
