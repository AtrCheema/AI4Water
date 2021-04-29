import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def take(st, en, d):
    keys = list(d.keys())[st:en]
    values = list(d.values())[st:en]

    return {k:v for k,v in zip(keys, values)}


def plot_errors(errors, ranges=((0.0, 1.0), (1.0, 10), (10, 1000)), save=True, save_path=None, **kwargs):
    """Plots the given errors as dictionary as radial plot between ranges.
    :param errors: dict, dictionary whose keys are names are erros and values are error values.
    :param ranges: tuple of tuples defining range of errors to plot in one plot
    :param save: bool, if True, the figure will be saved.
    :param save_path: string/pathlike, if given, the figure will the saved at this location.
    """
    for _range in ranges:
        plot_errors_between(errors, *_range, save=save, save_path=save_path, **kwargs)
    return

def plot_errors_between(errors: dict, lower:int, upper:int, save=True, save_path=None, **kwargs):
    zero_to_one = {}
    for k, v in errors.items():
        if v is not None:
            if lower < v < upper:
                zero_to_one[k] = v

    st = 0
    n = len(zero_to_one)
    for i in np.array(np.linspace(0, n, int(n/15)+1), dtype=np.int):
        if i == 0:
            pass
        else:
            en = i
            d = take(st,en, zero_to_one)
            plot_radial(d, lower, upper, save=save, save_path=save_path, **kwargs)
            st = i

    return


def plot_radial(errors:dict, low:int, up:int, save=True, save_path=None, **kwargs):
    """Plots all the errors in errors dictionary. low and up are used to draw the limits of radial plot."""

    fill = kwargs.get('fill', None)
    fillcolor = kwargs.get('fillcolor', None)
    line = kwargs.get('line', None)
    marker = kwargs.get('marker', None)

    OrderedDict(sorted(errors.items(), key=lambda kv: kv[1]))

    lower = round(np.min(list(errors.values())), 4)
    upper = round(np.max(list(errors.values())), 4)

    fig = go.Figure()
    categories = list(errors.keys())

    fig.add_trace(go.Scatterpolar(
        r=list(errors.values()),
        theta=categories,  # angular coordinates
        fill=fill,
        fillcolor=fillcolor,
        line=line,
        marker=marker,
        name='errors'
    ))

    fig.update_layout(
        title_text=f"Errors from {lower} to {upper}",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[low, up]
            )),
        showlegend=False
    )

    fig.show()
    if save:
        fname = f"errors_from_{lower}_to_{upper}.png"
        if save_path is not None:
            fname = os.path.join(save_path, fname)
        fig.write_image(fname)
    return


def plot1d(true, predicted, save=True, name="plot", show=False):
    fig, axis = plt.subplots()

    axis.plot(np.arange(len(true)), true, label="True")
    axis.plot(np.arange(len(predicted)), predicted, label="Predicted")
    axis.legend(loc="best")

    if save:
        plt.savefig(name, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    plt.close('all')
    return