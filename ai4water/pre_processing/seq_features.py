# extracts features of 1d array like data.

import numpy as np
import scipy
from scipy.stats import norm, rankdata

class Features(object):

    def __init__(self, x):
        self.x = x

class Trends(Features):

    """
    Arguments:
    x array/list/series: 1d array or array like whose features are to be calculated.
    """

    def sen_slope(self, alpha=None):
        # https://github.com/USGS-python/trend/blob/master/trend/__init__.py
        """A nonparametric estimate of trend.
            Parameters
        ----------
        x : array_like
            Observations taken at a fixed frequency.
        Notes
        -----
        This method works with missing or censored data, as long as less <20% of
        observations are censored.
        References
        ----------
        .. [1] Helsel and Hirsch, R.M. 2002. Statistical Methods in Water Resources.
        .. [2] https://vsp.pnnl.gov/help/vsample/nonparametric_estimate_of_trend.htm
        """
        s = sen_diff(self.x)
        s.sort()

        if alpha:
            N = len(s)
            # calculate confidence limits
            C_alpha = norm.ppf(1 - alpha / 2) * np.sqrt(np.nanvar(self.x))
            U = int(np.round(1 + (N + C_alpha) / 2))
            L = int(np.round((N - C_alpha) / 2))
            return np.nanmedian(s), s[L], s[U]

        else:
            return np.nanmedian(s)

    def seasonal_sen_slope(self, period=12, alpha=None):
        """A nonparametric estimate of trend for seasonal time series.
        Paramters
        ---------
        x : array_like
            Observations taken at a fixed frequency.
        period : int
            Number of observations in a cycle. The number of seasons.
        """
        s = 0

        for season in np.arange(0, period):
            x_season = self.x[season::period]
            s = np.append(s, sen_diff(x_season))

        s.sort()

        if alpha:
            #  XXX This code needs to be verified
            N = len(s)
            # calculate confidence limits
            C_alpha = norm.ppf(1-alpha/2)*np.sqrt(np.nanvar(self.x))
            U = int(np.round(1 + (N + C_alpha)/2))
            L = int(np.round((N - C_alpha)/2))
            return np.nanmedian(s), s[L], s[U]

        else:
            return np.nanmedian(s)

    def pettitt(self, alpha=0.05):
        """Pettitt's change-point test
        A nonparameteric test for detecting change points in a time series.
        Parameters
        ----------
        x : array_like
            Observations taken at a fixed frequency.
        alpha : float
            Significance level
        Return
        ------
        The index of the change point of the series, provided that it is
        statistically significant.
        """
        U_t = np.zeros_like(self.x)
        n = len(self.x)

        r = rankdata(self.x)
        for i in np.arange(n):
            U_t[i] = 2 * np.sum(r[:i+1]) - (i+1)*(n-1)

        t = np.argmax(np.abs(U_t))
        K_t = U_t[t]

        p = 2.0 * np.exp((-6.0 * K_t**2)/(n**3 + n**2))

        if p > alpha:
            return t
        else:
            return np.nan

    def mann_kendall(self, alpha=0.05):
        """Mann-Kendall (MK) is a nonparametric test for monotonic trend.
        Parameters
        ----------
        x : array
            Observations taken at a fixed frequency.
        Returns
        -------
        z : float
            Normalized MK test statistic.
        Examples
        --------
        >>> x = np.random.rand(100) + np.linspace(0,.5,100)
        >>> z,p = kendall(x)
        Attribution
        -----------
        Modified from code by Michael Schramn available at
        https://github.com/mps9506/Mann-Kendall-Trend/blob/master/mk_test.py
        """
        # n = len(self.x)

        s = mk_score(self.x)
        var_s = mk_score_variance(self.x)

        z = mk_z(s, var_s)
        # calculate the p_value
        p_value = 2*(1-norm.cdf(abs(z)))  # two tail test

        return p_value


    def seasonal_mann_kendall(self, period=12):
        """ Seasonal nonparametric test for detecting a monotonic trend.
        Parameters
        ----------
        x : array
            A sequence of chronologically ordered observations with fixed
            frequency.
        period : int
            The number of observations that define period. This is the number of seasons.
        """
        # Compute the SK statistic, S, for each season
        s = 0
        var_s = 0

        for season in np.arange(period):
            x_season = self.x[season::period]
            s += mk_score(x_season)
            var_s += mk_score_variance(x_season)

        # Compute the SK test statistic, Z, for each season.
        z = mk_z(s, var_s)

        # calculate the p_value
        p_value = 2*(1-norm.cdf(abs(z)))  # two tail test

        return p_value


def mk_z(s, var_s):
    """Computes the MK test statistic, Z.
    Parameters
    ----------
    s : float
        The MK trend statistic, S.
    var_s : float
        Variance of S.
    Returns
    -------
    MK test statistic, Z.
    """
    # calculate the MK test statistic
    if s > 0:
        z = (s - 1)/np.sqrt(var_s)
    elif s < 0:
        z = (s + 1)/np.sqrt(var_s)
    else:
        z = 0

    return z

def mk_score_variance(x):
    """Computes corrected variance of S statistic used in Mann-Kendall tests.
    Equation 8.4 from Helsel and Hirsch (2002).
    Parameters
    ----------
    x : array_like
    Returns
    -------
    Variance of S statistic
    Note that this might be equivalent to:
        See page 728 of Hirsch and Slack
    References
    ----------
    .. [1] Helsel and Hirsch, R.M. 2002. Statistical Methods in Water Resources.
    """
    x = x[~np.isnan(x)]
    n = len(x)
    # calculate the unique data
    unique_x = np.unique(x)
    # calculate the number of tied groups
    g = len(unique_x)

    # calculate the var(s)
    if n == g:  # there is no tie
        var_s = (n * (n - 1) * (2 * n + 5)) / 18

    else:  # there are some ties in data
        tp = np.zeros_like(unique_x)
        for i in range(len(unique_x)):
            tp[i] = sum(x == unique_x[i])
        var_s = (n * (n - 1) * (2 * n + 5) - np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18

    return var_s


class Stats(Features):

    def auc(self):
        return

    def auto_corr(self):
        return

    def centroid(self):
        return

    def slope(self):
        return

    def zero_crossing_rate(self):
        return

    def sum_abs_diff(self):
        return np.sum(np.abs(np.diff(self.x)))

    def min_max_diff(self):
        return np.abs(np.max(self.x) - np.min(self.x))

    def mean_abs_dev(self):
        return np.mean(np.abs(self.x - np.mean(self.x, axis=0)), axis=0)

    def median_abs_dev(self):
        """Median absolute deviation"""
        return scipy.stats.median_absolute_deviation(self.x, scale=1)

    def rms(self):
        """Root mean square"""
        return np.sqrt(np.sum(np.array(self.x) ** 2) / len(self.x))

def mk_score(x):
    """Computes S statistic used in Mann-Kendall tests.
    Parameters
    ----------
    x : array_like
        Chronologically ordered array of observations.
    Returns
    -------
    MK trend statistic (S).
    """
    x = x[~np.isnan(x)]
    n = len(x)
    s = 0

    for j in np.arange(1, n):
        s += np.sum(np.sign(x[j] - x[0:j]))

    return s

def sen_diff(x):
    """Sen's difference operator.
    Paramaters
    ----------
    x : array_like
        Observations taken at a fixed frequency.
    Returns
    -------
    Sen difference
    """
    #x = x[~np.isnan(x)]
    n = len(x)
    N = int(n*(n-1)/2)  # number of slope estimates
    s = np.zeros(N)
    i = 0
    for j in np.arange(1, n):
        #s[i:j+i] = (x[j] - x[0:j])/np.arange(1, j+1)
        s[i:j+i] = (x[j] - x[0:j])/np.arange(j, 0, -1)
        i += j

    return s


if __name__ == "__main__":
    f = Features(np.random.random(10))
