
"""
easy_mpl stands for easy maplotlib. The purpose of this is to ease the use of
matplotlib while keeping the flexibility intact.
"""

__all__ = [
    "plot",
    "bar_chart",
    "regplot",
    "imshow",
    "hist",
    "process_axis",
    "init_subplots",
    "pie"
]

import random
from typing import Union

import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .plotting_tools import to_1d_array


BAR_CMAPS = ['Blues', 'BuGn', 'gist_earth_r',
             'GnBu', 'PuBu', 'PuBuGn', 'summer_r']

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


def get_cmap(cm: str, num_cols: int, low=0.0, high=1.0):

    cols = getattr(plt.cm, cm)(np.linspace(low, high, num_cols))
    return cols


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
              rotation=0,
              show=True,
              ):

    cm = get_cmap(random.choice(BAR_CMAPS), len(values), 0.2)
    color = color if color is not None else cm

    if not axis:
        _, axis = plt.subplots()

    if sort:
        sort_idx = np.argsort(values)
        values = values[sort_idx]
        labels = np.array(labels)[sort_idx]

    if orient == 'h':
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

    if show:
        plt.show()

    return axis


def plot(*args, show=True, **kwargs):
    """
    One liner plot function. It should not be more complex than axis.plot() or
    plt.plot() yet it must accomplish all in one line what requires multiple
    lines in matplotlib. args and kwargs can be anything which goes into plt.plot()
    or axis.plot(). They can also be anything which goes into `process_axis`.
    """
    _, axis = init_subplots()
    axis = process_axis(axis, *args, **kwargs)
    if kwargs.get('save', False):
        plt.savefig(f"{kwargs.get('name', 'fig.png')}")
    if show:
        plt.show()
    return axis


def regplot(
        x: Union[np.ndarray, pd.DataFrame, pd.Series, list],
        y: Union[np.ndarray, pd.DataFrame, pd.Series, list],
        title: str = None,
        show: bool = False,
        annotation_key: str = None,
        annotation_val: float = None,
        line_color=None,
        marker_color=None,
        fill_color=None,
        marker_size: int = 20,
        ci: Union[int, None] = 95,
        figsize: tuple = None,
        xlabel: str = 'Observed',
        ylabel: str = 'Predicted'
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

    Example:
        >>> from ai4water.datasets import busan_beach
        >>> from ai4water.utils.easy_mpl import regplot
        >>> data = busan_beach()
        >>> regplot(data['pcp3_mm'], data['pcp6_mm'], show=True)
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
                     horizontalalignment='right', verticalalignment='top',
                     fontsize=16)
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


def _regplot(x, y, ax, ci=None, line_color=None, fill_color=None):

    grid = np.linspace(np.min(x), np.max(x), 100)
    x = np.c_[np.ones(len(x)), x]
    grid = np.c_[np.ones(len(grid)), grid]
    yhat = grid.dot(reg_func(x, y))

    ax.plot(grid[:, 1], yhat, color=line_color)

    if ci:
        boots = bootdist(reg_func, args=[x, y], n_boot=1000).T

        yhat_boots = grid.dot(boots).T

        err_bands = _ci(yhat_boots, ci, axis=0)

        ax.fill_between(grid[:, 1], *err_bands,
                        facecolor=fill_color,
                        alpha=.15)
    return ax


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


def init_subplots(width=None, height=None, nrows=1, ncols=1, **kwargs):
    """Initializes the fig for subplots"""
    plt.close('all')
    fig, axis = plt.subplots(nrows=nrows, ncols=ncols, **kwargs)
    if width is not None:
        fig.set_figwidth(width)
    if height is not None:
        fig.set_figheight(height)
    return fig, axis


def process_axis(axis,
                 data: Union[list, np.ndarray, pd.Series, pd.DataFrame],
                 x: Union[list, np.ndarray] = None,  # array to plot as x
                 marker='',
                 fillstyle=None,
                 linestyle='-',
                 color=None,
                 ms=6.0,  # markersize
                 label=None,  # legend
                 log=False,
                 log_nz=False,
                 **kwargs
                 ):

    """Purpose to act as a middle man between axis.plot/plt.plot.
    Returns:
        axis
        """
    # TODO
    # default fontsizes should be same as used by matplotlib
    # should not complicate plt.plot or axis.plto
    # allow multiple plots on same axis

    if log and log_nz:
        raise ValueError

    use_third = False
    if x is not None:
        if isinstance(x, str):  # the user has not specified x so x is currently plot style.
            style = x
            x = None
            if marker == '.':
                use_third = True

    if log_nz:
        data = deepcopy(data)
        _data = data.values
        d_nz_idx = np.where(_data > 0.0)
        data_nz = _data[d_nz_idx]
        d_nz_log = np.log(data_nz)
        _data[d_nz_idx] = d_nz_log
        _data = np.where(_data < 0.0, 0.0, _data)
        data = pd.Series(_data, index=data.index)

    if log:
        data = deepcopy(data)
        _data = np.where(data.values < 0.0, 0.0, data.values)
        print(len(_data[np.where(_data < 0.0)]))
        data = pd.Series(_data, index=data.index)

    if x is not None:
        axis.plot(x, data, fillstyle=fillstyle, color=color, marker=marker,
                  linestyle=linestyle, ms=ms, label=label)
    elif use_third:
        axis.plot(data, style, color=color, ms=ms, label=label)
    else:
        axis.plot(data, fillstyle=fillstyle, color=color, marker=marker,
                  linestyle=linestyle, ms=ms, label=label)

    return _process_axis(axis=axis, label=label, log=log, **kwargs)


def _process_axis(
        axis: plt.Axes=None,
        label=None,  # legend
        log=False,
        legend_kws:dict = None, # kwargs for axis.legend such as loc, fontsize, bbox_to_anchor, markerscale
        xlabel=None,
        xlabel_kws:dict=None,
        xtick_kws:dict = None, # for axis.tick_params such as which, labelsize, colors etc
        ylim:tuple=None,  # limit for y axis
        ylabel=None,
        ylabel_kws:dict=None,  # ylabel kwargs
        ytick_kws:dict = None, # for axis.tick_params(  such as which, labelsize, colors etc
        show_xaxis=True,
        top_spine=True,
        bottom_spine=True,
        right_spine=True,
        left_spine=True,
        invert_yaxis=False,
        max_xticks=None,
        min_xticks=None,
        title=None,
        title_kws:dict=None,  # title kwargs
        grid=None,
        grid_kws:dict = None,  # keyword arguments for axes.grid
)-> plt.Axes:
    """
    processing of matplotlib Axes
    Returns:
        plt.Axes
    """
    if axis is None:
        axis = plt.gca()

    if label:
        if label != "__nolabel__":
            legend_kws = legend_kws or {}
            axis.legend(**legend_kws)

    if ylabel:
        ylabel_kws = ylabel_kws or {}
        axis.set_ylabel(ylabel, **ylabel_kws)

    if log:
        axis.set_yscale('log')

    if invert_yaxis:
        axis.set_ylim(axis.get_ylim()[::-1])

    if ylim:
        axis.set_ylim(ylim)

    if xlabel:  # better not change these paras if user has not defined any x_label
        xtick_kws = xtick_kws or {}
        axis.tick_params(axis="x", **xtick_kws)

    if ylabel:
        ytick_kws = ytick_kws or {}
        axis.tick_params(axis="y", **ytick_kws)

    axis.get_xaxis().set_visible(show_xaxis)

    if xlabel:
        xlabel_kws = xlabel_kws or {}
        axis.set_xlabel(xlabel, **xlabel_kws)

    if top_spine:
        axis.spines['top'].set_visible(top_spine)
    if bottom_spine:
        axis.spines['bottom'].set_visible(bottom_spine)
    if right_spine:
        axis.spines['right'].set_visible(right_spine)
    if left_spine:
        axis.spines['left'].set_visible(left_spine)

    if max_xticks is not None:
        min_xticks = min_xticks or max_xticks-1
        assert isinstance(min_xticks, int)
        assert isinstance(max_xticks, int)
        loc = mdates.AutoDateLocator(minticks=min_xticks, maxticks=max_xticks)
        axis.xaxis.set_major_locator(loc)
        fmt = mdates.AutoDateFormatter(loc)
        axis.xaxis.set_major_formatter(fmt)

    if title:
        title_kws = title_kws or {}
        axis.set_title(title, **title_kws)

    if grid:
        grid_kws = grid_kws or {}
        axis.grid(grid, **grid_kws)

    return axis


def imshow(values,
           axis=None,
           vmin=None,
           vmax=None,
           xlabel=None,
           aspect=None,
           interpolation=None,
           title=None,
           cmap=None,
           ylabel=None,
           yticklabels=None,
           xticklabels=None,
           show=True,
           annotate=False,
           annotate_kws=None,
           colorbar:bool=False,
           colorbar_orientation:str = 'vertical'
           ):
    """One stop shop for imshow
    Example:
        >>> import numpy as np
        >>> from ai4water.utils.easy_mpl import imshow
        >>> x = np.random.random((10, 5))
        >>> imshow(x, annotate=True)
    """

    if axis is None:
        axis = plt.gca()

    im = axis.imshow(values, aspect=aspect, vmin=vmin, vmax=vmax,
                     cmap=cmap, interpolation=interpolation)

    if annotate:
        annotate_kws = annotate_kws or {"color": "w", "ha":"center", "va":"center"}
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                _ = axis.text(j, i, round(values[i, j], 2),
                            **annotate_kws)

    if yticklabels is not None:
        axis.set_yticks(np.arange(len(yticklabels)))
        axis.set_yticklabels(yticklabels)

    if xticklabels is not None:
        axis.set_xticks(np.arange(len(xticklabels)))
        axis.set_xticklabels(xticklabels)

    _process_axis(axis, xlabel=xlabel, ylabel=ylabel, title=title)

    if colorbar:
        fig: plt.Figure = plt.gcf()
        fig.colorbar(im, orientation=colorbar_orientation, pad=0.2)

    if show:
        plt.show()

    return axis, im


def hist(
        x:Union[list, np.ndarray, pd.Series, pd.DataFrame],
        hist_kws: dict = None,
        grid:bool = True,
        ax: plt.Axes = None,
        show: bool = True,
        **kwargs
)->plt.Axes:
    """
    one stop shop for histogram
    Arguments:
        x: array like, must not be greader than 1d
        grid: whether to show the grid or not
        show: whether to show the plot or not
        ax: axes on which to draw the plot
        hist_kws: any keyword arguments for [axes.hist](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html)
        kwargs: any keyword arguments for axes manipulation such as title, xlable, ylable etc
    Returns:
        matplotlib Axes
    """
    if not ax:
        ax = plt.gca()
    hist_kws = hist_kws or {}
    n, bins, patches = ax.hist(x, **hist_kws)

    _process_axis(ax, grid=grid, **kwargs)

    if show:
        plt.show()

    return ax


def pie(
        vals: Union[list, np.ndarray, pd.Series] = None,
        fractions: Union[list, np.ndarray, pd.Series] = None,
        labels: list = None,
        ax: plt.Axes = None,
        title: str = None,
        name: str = None,
        save: bool = True,
        show: bool = True,
        **kwargs
)->plt.Axes:
    """
    Arguments:
        vals: array like, unique values and their counts will be inferred from this array
        fractions: if given, vals must not be given
        labels: labels for unique values in vals, if given, must be equal to unique vals
             in vals. Otherwise "unique_value (counts)" will be used for labeling.
        ax: the axes on which to draw, if not given current active axes will be used
        title: if given, will be used for title
        name:
        save:
        show:
        kwargs: any keyword argument will go to axes.pie
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pie.html

    Returns:
        a matplotlib axes. This can be used for further processing by making show=False.

    Example:
        >>> pie(np.random.randint(0, 3, 100))
        or by directly providing fractions
        >>> pie([0.2, 0.3, 0.1, 0.4])
    """

    if ax is None:
        ax = plt.gca()

    if fractions is None:
        fractions = pd.Series(vals).value_counts(normalize=True).values
        vals = pd.Series(vals).value_counts().to_dict()
        if labels is None:
            labels = [f"{value} ({count}) " for value, count in vals.items()]
    else:
        assert vals is None
        if labels is None:
            labels = [f"f{i}" for i in range(len(fractions))]

    if 'autopct' not in kwargs:
        kwargs['autopct'] = '%1.1f%%'

    ax.pie(fractions,
           labels=labels,
           **kwargs)

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    if title:
        plt.title(title, fontsize=20)

    if save:
        name = name or "pie.png"
        plt.savefig(name, dpi=300)

    if show:
        plt.show()

    return ax