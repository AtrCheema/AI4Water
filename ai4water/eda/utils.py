
from typing import Any, Dict

from scipy import linalg

from ai4water.backend import np, pd, plt


def auto_corr(x, nlags, demean=True):
    """
    autocorrelation like statsmodels
    https://stackoverflow.com/a/51168178
    """

    var = np.var(x)

    if demean:
        x -= np.mean(x)

    corr = np.full(nlags+1, np.nan, np.float64)
    corr[0] = 1.

    for lag in range(1, nlags+1):
        corr[lag] = np.sum(x[lag:]*x[:-lag])/len(x)/var

    return corr


def pac_yw(x, nlags):
    """partial autocorrelation according to ywunbiased method"""

    pac = np.full(nlags+1, fill_value=np.nan, dtype=np.float64)
    pac[0] = 1.

    for lag in range(1, nlags+1):
        pac[lag] = ar_yw(x, lag)[-1]

    return pac


def ar_yw(x, order=1, adj_needed=True, demean=True):
    """Performs autoregressor using Yule-Walker method.
    Returns:
        rho : np array
        coefficients of AR
    """
    x = np.array(x, dtype=np.float64)

    if demean:
        x -= x.mean()

    n = len(x)
    r = np.zeros(order+1, np.float64)
    r[0] = (x ** 2).sum() / n
    for k in range(1, order+1):
        r[k] = (x[0:-k] * x[k:]).sum() / (n - k * adj_needed)
    R = linalg.toeplitz(r[:-1])

    rho = np.linalg.solve(R, r[1:])
    return rho


def plot_autocorr(
        x,
        axis=None,
        plot_marker=True,
        show=True,
        legend=None,
        title=None,
        xlabel=None,
        vlines_colors=None,
        hline_color=None,
        marker_color=None,
        legend_fs=None
):

    if not axis:
        _, axis = plt.subplots()

    if plot_marker:
        axis.plot(x, 'o', color=marker_color, label=legend)
        if legend:
            axis.legend(fontsize=legend_fs)
    axis.vlines(range(len(x)), [0], x, colors=vlines_colors)
    axis.axhline(color=hline_color)

    if title:
        axis.set_title(title)
    if xlabel:
        axis.set_xlabel("Lags")

    if show:
        plt.show()

    return axis


def ccovf_np(x, y, unbiased=True, demean=True):
    n = len(x)
    if demean:
        xo = x - x.mean()
        yo = y - y.mean()
    else:
        xo = x
        yo = y
    if unbiased:
        xi = np.ones(n)
        d = np.correlate(xi, xi, 'full')
    else:
        d = n
    return (np.correlate(xo, yo, 'full') / d)[n - 1:]


def ccf_np(x, y, unbiased=True):
    """cross correlation between two time series
    # https://stackoverflow.com/a/24617594
    """
    cvf = ccovf_np(x, y, unbiased=unbiased, demean=True)
    return cvf / (np.std(x) * np.std(y))


def _missing_vals(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Modified after https://github.com/akanz1/klib/blob/main/klib/utils.py#L197
     Gives metrics of missing values in the dataset.
    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame
    Returns
    -------
    Dict[str, float]
        mv_total: float, number of missing values in the entire dataset
        mv_rows: float, number of missing values in each row
        mv_cols: float, number of missing values in each column
        mv_rows_ratio: float, ratio of missing values for each row
        mv_cols_ratio: float, ratio of missing values for each column
    """

    data = pd.DataFrame(data).copy()
    mv_rows = data.isna().sum(axis=1)
    mv_cols = data.isna().sum(axis=0)
    mv_total = data.isna().sum().sum()
    mv_rows_ratio = mv_rows / data.shape[1]
    mv_cols_ratio = mv_cols / data.shape[0]

    return {
        "mv_total": mv_total,
        "mv_rows": mv_rows,
        "mv_cols": mv_cols,
        "mv_rows_ratio": mv_rows_ratio,
        "mv_cols_ratio": mv_cols_ratio,
    }