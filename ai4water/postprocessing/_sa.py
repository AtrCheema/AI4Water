"""sensitivity analysis"""

import importlib
from typing import Callable, Union

from SALib.plotting.hdmr import plot
from SALib.plotting.bar import plot as barplot
from SALib.plotting.morris import covariance_plot

from ai4water.backend import easy_mpl as ep
from ai4water.backend import np, pd, plt, os


def sensitivity_analysis(
        sampler:str,
        analyzer:Union[str, list],
        func:Callable,
        bounds: list,
        sampler_kwds: dict = None,
        analyzer_kwds: dict = None,
        names: list = None,
        **kwargs
)->dict:
    """
    Parameters
    ----------
    sampler :
    analyzer :
    func :
    bounds :
    sampler_kwds :
    analyzer_kwds :
    names :
    **kwargs :
    """
    sampler = importlib.import_module(f"SALib.sample.{sampler}")

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

    y = func(x=param_values, **kwargs)

    y = np.array(y)

    assert np.size(y) == len(y) , f"output must be 1 dimensional"
    y = y.reshape(-1, )

    results = {}
    if isinstance(analyzer, list):
        for _analyzer in analyzer:
            print(f"Analyzing with {_analyzer}")
            results[_analyzer] = analyze(_analyzer, param_values, y, problem, analyzer_kwds)
    else:
        assert isinstance(analyzer, str)
        results[analyzer] = analyze(analyzer, param_values, y, problem, analyzer_kwds)

    return results


def analyze(analyzer, param_values, y, problem, analyzer_kwds):
    _analyzer = importlib.import_module(f"SALib.analyze.{analyzer}")
    analyzer_kwds = analyzer_kwds or {}

    if analyzer in ["hdmr",
                    "morris",
                    "dgsm",
                    "ff",
                    "pawn",
        "rbd_fast", "delta",
                    ] and 'X' not in analyzer_kwds:
        analyzer_kwds['X'] = param_values

    Si = _analyzer.analyze(problem=problem, Y=y, **analyzer_kwds)

    if 'X' in analyzer_kwds:
        analyzer_kwds.pop('X')

    return Si


def sensitivity_plots(analyzer, si, path=None, show=False):

    if analyzer == "morris":
        morris_plots(si, path=path, show=show)

    elif analyzer in ["sobol"]:
        sobol_plots(si, show, path)

    elif analyzer == "hdmr":

        plt.close('all')
        plot(si)
        if path:
            plt.savefig(os.path.join(path, "hdmr"), bbox_inches="tight")

    elif analyzer in ["pawn"]:
        plt.close('all')
        si_df = si.to_df()
        bar_plot(si_df[["CV", "median"]], conf_col="median")
        if path:
            plt.savefig(os.path.join(path, "pawn_cv"), bbox_inches="tight")
        if show:
            plt.show()

    elif analyzer == "fast":
        plt.close('all')
        si_df = si.to_df()
        bar_plot(si_df[["S1", "S1_conf"]])
        if path:
            plt.savefig(os.path.join(path, "fast_s1"), bbox_inches="tight")
        if show:
            plt.show()

        plt.close('all')
        bar_plot(si_df[["ST", "ST_conf"]])
        if path:
            plt.savefig(os.path.join(path, "fast_s1"), bbox_inches="tight")
        if show:
            plt.show()


    elif analyzer == "delta":
        plt.close('all')
        si_df = si.to_df()
        bar_plot(si_df[["delta", "delta_conf"]])
        if path:
            plt.savefig(os.path.join(path, "fast_s1"), bbox_inches="tight")
        if show:
            plt.show()

        plt.close('all')
        bar_plot(si_df[["S1", "S1_conf"]])
        if path:
            plt.savefig(os.path.join(path, "fast_s1"), bbox_inches="tight")
        if show:
            plt.show()

    elif analyzer == "rbd_fast":
        plt.close('all')
        si_df = si.to_df()
        bar_plot(si_df[["S1", "S1_conf"]])
        if path:
            plt.savefig(os.path.join(path, "rbd_fast_s1"), bbox_inches="tight")
        if show:
            plt.show()
    return


def sobol_plots(si, show=False, path:str=None):
    total, first, second = si.to_df()

    plt.close('all')
    bar_plot(total)
    if path:
        plt.savefig(os.path.join(path, "total"), bbox_inches="tight")
    if show:
        plt.show()

    plt.close('all')
    bar_plot(first)
    if path:
        plt.savefig(os.path.join(path, "first_order"), bbox_inches="tight")
    if show:
        plt.show()

    fig, ax = plt.subplots(figsize=(16, 6))
    bar_plot(second, ax=ax)
    if path:
        plt.savefig(os.path.join(path, "first_order"), bbox_inches="tight")
    if show:
        plt.show()

    return

def morris_plots(si, show:bool=False, path:str=None, annotate=True):

    plt.close('all')
    si_df = si.to_df()
    bar_plot(si_df[["mu_star", "mu_star_conf"]])
    if show:
        plt.tight_layout()
        plt.show()
    if path:
        plt.savefig(os.path.join(path, "morris_bar_plot"), bbox_inches="tight")

    fig, ax = plt.subplots()
    covariance_plot(ax, si)

    if si['sigma'] is not None and annotate:
        y = si['sigma']
        z = si['mu_star']
        for i, txt in enumerate(si['names']):
            ax.annotate(txt, (z[i], y[i]))
    if show:
        plt.tight_layout()
        plt.show()
    if path:
        plt.savefig(os.path.join(path, "covariance_plot"), bbox_inches="tight")

    plt.close('all')
    barplot(si_df, ax=ax)

    if path:
        plt.savefig(os.path.join(path, "morris_bar_plot_all"), bbox_inches="tight")
    if show:
        plt.tight_layout()
        plt.show()

    return


def bar_plot(sis_df:pd.DataFrame, sort=True, conf_col = "_conf", **kwargs):

    conf_cols = sis_df.columns.str.contains(conf_col)

    sis = sis_df.loc[:, ~conf_cols].values
    confs = sis_df.loc[:, conf_cols].values
    names = sis_df.index

    if isinstance(names[0], tuple):
        names = np.array([str(i) for i in names])

    if len(sis) == sis.size:
        confs = confs.reshape(-1, )
        sis = sis.reshape(-1,)
    else:
        raise ValueError

    if sort:
        sort_idx = np.argsort(sis)
        confs = confs[sort_idx]
        sis = sis[sort_idx]
        names = names[sort_idx]

    label = sis_df.columns[~conf_cols][0]

    ax = ep.bar_chart(sis, names, orient="v", sort=sort, rotation=90, show=False,
                   label=label, **kwargs)
    if sort:
        ax.legend(loc="upper left")
    else:
        ax.legend(loc="best")

    ax.errorbar(np.arange(len(sis)), sis, yerr=confs, fmt=".", color="black")
    return ax

def _show_save(path=None, show=True):
    if path:
        plt.savefig(path, bbox_inches="tight")

    if show:
        plt.show()
    return


def _make_predict_func(model, **kwargs):

    from ai4water.preprocessing import DataSet

    lookback = model.config["ts_args"]['lookback']

    def func(x):
        x = pd.DataFrame(x, columns=model.input_features)

        ds = DataSet(data=x,
                     ts_args=model.config["ts_args"],
                     input_features=model.input_features,
                     train_fraction=1.0,
                     val_fraction=0.0,
                     verbosity=0)

        x, _ = ds.training_data()
        p = model.predict(x=x, **kwargs)

        return np.concatenate([p, np.zeros((lookback-1, 1))])

    return func
