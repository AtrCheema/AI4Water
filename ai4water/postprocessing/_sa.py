"""sensitivity analysis"""

import importlib
import os
from typing import Callable

import matplotlib.pyplot as plt
from easy_mpl import bar_chart
import numpy as np
from SALib.plotting.hdmr import plot
from SALib.plotting.bar import plot as barplot
from SALib.plotting.morris import horizontal_bar_plot, covariance_plot


def sensitivity_analysis(
        sampler,
        analyzer,
        func:Callable,
        bounds: list,
        sampler_kwds: dict = None,
        analyzer_kwds: dict = None,
        names: list = None
):
    sampler = importlib.import_module(f"SALib.sample.{sampler}")
    _analyzer = importlib.import_module(f"SALib.analyze.{analyzer}")

    if names is None:
        names = [f"Feat{i}" for i in range(len(bounds))]

    # Define the model inputs
    problem = {
        'num_vars': len(bounds),
        'names': names,
        'bounds': bounds
    }

    sampler_kwds = sampler_kwds or {'N': 100}

    param_values = sampler.sample(problem=problem, **sampler_kwds)
    print("total samples:", len(param_values))

    y = func(x=param_values)

    y = np.array(y)

    assert np.size(y) == len(y) , f"output must be 1 dimensional"
    y = y.reshape(-1, )

    analyzer_kwds = analyzer_kwds or {}

    if analyzer in ["hdmr",
                    "morris",
                    "dgsm",
        "ff",
                    "pawn"] and 'X' not in analyzer_kwds:
        analyzer_kwds['X'] = param_values

    Si = _analyzer.analyze(problem=problem, Y=y, **analyzer_kwds)

    return Si


def _plots(analyzer, si, path):

    if analyzer == "morris":
        morris_plots(si, path=path)
        _bar_plot(si.to_df(), path=os.path.join(path, "morris_bar_plot"))

    elif analyzer in ["sobol"]:
        total, first, second = si.to_df()
        _bar_plot(total, path=os.path.join(path, "total"))
        _bar_plot(first, path=os.path.join(path, "first_order"))
        _bar_plot(second, path=os.path.join(path, "second_order"))

    elif analyzer == "hdmr":

        plt.close('all')
        plot(si)
        plt.savefig(os.path.join(path, "hdmr"), bbox_inches="tight")

    elif analyzer in ["pawn"]:
        ax = bar_chart(si['mean'], si['names'], orient='v', sort=True,
                  figsize=(8, 6), title="Mean values", show=False)
        means = si["mean"]
        mins = si["minimum"]
        maxes = si["maximum"]
        # ax.errorbar(np.arange(len(means)), means, [means - mins, maxes - means],
        #              fmt='ok', ecolor='gray', lw=1)
        plt.show()
    return


def morris_plots(si, show=False, path=None):

    fig, ax = plt.subplots()
    horizontal_bar_plot(ax=ax, Si=si)
    if show:
        plt.show()

    if path:
        plt.savefig(os.path.join(path, "morris_bar_plot"), bbox_inches="tight")

    fig, ax = plt.subplots()
    covariance_plot(ax, si)
    if show:
        plt.show()

    if path:
        plt.savefig(os.path.join(path, "covariance_plot"), bbox_inches="tight")

    return


def _bar_plot(si_df, path, show=False):

    barplot(si_df)
    plt.show()

    if show:
        plt.show()

    plt.savefig(path, bbox_inches="tight")

    return
