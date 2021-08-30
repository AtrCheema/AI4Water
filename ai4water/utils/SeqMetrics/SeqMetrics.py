import json
import warnings
import numpy as np
from math import sqrt
from typing import Union

from sklearn import preprocessing
from sklearn.metrics import hinge_loss
from scipy.stats import gmean, kendalltau
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

from ai4water.utils.utils import ts_features
from ai4water.utils.SeqMetrics.utils import _geometric_mean, _mean_tweedie_deviance, _foo, list_subclass_methods

# TODO remove repeated calculation of mse, std, mean etc
# TODO make weights, class attribute
# TODO write tests
# TODO standardized residual sum of squares
# http://documentation.sas.com/?cdcId=fscdc&cdcVersion=15.1&docsetId=fsug&docsetTarget=n1sm8nk3229ttun187529xtkbtpu.htm&locale=en
# https://arxiv.org/ftp/arxiv/papers/1809/1809.03006.pdf
# https://www.researchgate.net/profile/Mark-Tschopp/publication/322147437_Quantifying_Similarity_and_Distance_Measures_for_Vector-Based_Datasets_Histograms_Signals_and_Probability_Distribution_Functions/links/5a48089ca6fdcce1971c8142/Quantifying-Similarity-and-Distance-Measures-for-Vector-Based-Datasets-Histograms-Signals-and-Probability-Distribution-Functions.pdf
# maximum absolute error
# relative mean absolute error
# relative rmse
# Log mse
# Jeffreys Divergence
# kullback-Leibler divergence
# Peak flow ratio https://hess.copernicus.org/articles/24/869/2020/
# Legates׳s coefficient of efficiency
# outliear percentage : pysteps
# mean squared error skill score, mean absolute error skill score, https://doi.org/10.1016/j.ijforecast.2018.11.010
# root mean quartic error, Kolmogorov–Smirnov test integral, OVERPer, Rényi entropy, 95th percentile: https://doi.org/10.1016/j.solener.2014.10.016
# Friedman test: https://doi.org/10.1016/j.solener.2014.10.016

EPS = 1e-10  # epsilon

# TODO classification metrics
# cross entropy

# TODO probability losses
# log normal loss
# skill score

# TODO multi horizon metrics


class Metrics(object):
    """
    This class does some pre-processign and handles metadata regaring true and
    predicted arrays.

    The arguments other than `true` and `predicted` are dynamic i.e. they can be
    changed from outside the class. This means the user can change their value after
    creating the class. This will be useful if we want to calculate an error once by
    ignoring NaN and then by not ignoring the NaNs. However, the user has to run
    the method `treat_arrays` in order to have the changed values impact on true and
    predicted arrays.

    Literature:
        https://www-miklip.dkrz.de/about/murcss/


    """

    def __init__(self,
                 true: Union[np.ndarray, list],
                 predicted: Union[np.ndarray, list],
                 replace_nan: Union[int, float, None] = None,
                 replace_inf: Union[int, float, None] = None,
                 remove_zero: bool = False,
                 remove_neg: bool = False,
                 metric_type: str = 'regression'
                 ):

        """
        Arguments:
            true : array like, ture/observed/actual/target values
            predicted : array like, simulated values
            replace_nan : default None. if not None, then NaNs in true
                and predicted will be replaced by this value.
            replace_inf : default None, if not None, then inf vlaues in true and
                predicted will be replaced by this value.
            remove_zero : default False, if True, the zero values in true
                or predicted arrays will be removed. If a zero is found in one
                array, the corresponding value in the other array will also be
                removed.
            remove_neg : default False, if True, the negative values in true
                or predicted arrays will be removed.
            metric_type : type of metric.

        """
        self.metric_type = metric_type
        self.true, self.predicted = self._pre_process(true, predicted)
        self.replace_nan = replace_nan
        self.replace_inf = replace_inf
        self.remove_zero = remove_zero
        self.remove_neg = remove_neg

    @property
    def replace_nan(self):
        return self._replace_nan

    @replace_nan.setter
    def replace_nan(self, x):
        self._replace_nan = x

    @property
    def replace_inf(self):
        return self._replace_inf

    @replace_inf.setter
    def replace_inf(self, x):
        self._replace_inf = x

    @property
    def remove_zero(self):
        return self._remove_zero

    @remove_zero.setter
    def remove_zero(self, x):
        self._remove_zero = x

    @property
    def remove_neg(self):
        return self._remove_neg

    @remove_neg.setter
    def remove_neg(self, x):
        self._remove_neg = x

    @property
    def assert_greater_than_one(self):
        # assert that both true and predicted arrays are greater than one.
        if len(self.true) <= 1 or len(self.predicted) <= 1:
            raise ValueError(f"""Expect length of true and predicted arrays to be larger than 1 but they are
                            {len(self.true)} and {len(self.predicted)}""")
        return

    def _pre_process(self, true, predicted):

        predicted = self._assert_1darray(predicted)
        true = self._assert_1darray(true)
        assert len(predicted) == len(true), "lengths of provided arrays mismatch, predicted array: {}, true array: {}" \
            .format(len(predicted), len(true))

        return true, predicted

    def _assert_1darray(self, array_like) -> np.ndarray:
        """Makes sure that the provided `array_like` is 1D numpy array"""
        if not isinstance(array_like, np.ndarray):
            if not isinstance(array_like, list):
                # it can be pandas series or datafrmae
                if array_like.__class__.__name__ in ['Series', 'DataFrame']:
                    if len(array_like.shape) > 1:  # 1d series has shape (x,) while 1d dataframe has shape (x,1)
                        if array_like.shape[1] > 1:  # it is a 2d datafrmae
                            raise TypeError("only 1d pandas Series or dataframe are allowed")
                    np_array = np.array(array_like).reshape(-1, )
                else:
                    raise TypeError(f"all inputs must be numpy array or list but one is of type {type(array_like)}")
            else:
                np_array = np.array(array_like).reshape(-1, )
        else:
            if np.ndim(array_like)>1:
                sec_dim = array_like.shape[1]
                if self.metric_type != 'classification' and sec_dim>1:
                    raise ValueError(f"Array must not be 2d but it has shape {array_like.shape}")
                np_array = np.array(array_like).reshape(-1, ) if self.metric_type != 'classification' else array_like
            else:
                # maybe the dimension is >1 so make sure it is more
                np_array = array_like.reshape(-1, ) if self.metric_type != 'classification' else array_like

        if self.metric_type != 'classification':
            assert len(np_array.shape) == 1
        return np_array

    def calculate_all(self, statistics=False, verbose=False, write=False, name=None):
        """ calculates errors using all available methods except brier_score..
        write: bool, if True, will write the calculated errors in file.
        name: str, if not None, then must be path of the file in which to write."""
        errors = {}
        for m in self.all_methods:
            if m not in ["brier_score"]:
                try:
                    error = float(getattr(self, m)())
                # some errors might not have been computed and returned a non float-convertible value e.g. None
                except TypeError:
                    error = getattr(self, m)()
                errors[m] = error
                if verbose:
                    if error is None:
                        print('{0:25} :  {1}'.format(m, error))
                    else:
                        print('{0:25} :  {1:<12.3f}'.format(m, error))

        if statistics:
            errors['stats'] = self.stats(verbose=verbose)

        if write:
            if name is not None:
                assert isinstance(name, str)
                fname = name
            else:
                fname = 'errors'

            with open(fname + ".json", 'w') as fp:
                json.dump(errors, fp, sort_keys=True, indent=4)

        return errors

    def _error(self, true=None, predicted=None):
        """ simple difference """
        if true is None:
            true = self.true
        if predicted is None:
            predicted = self.predicted
        return true - predicted

    def _percentage_error(self):
        """
        Percentage error
        """
        return self._error() / (self.true + EPS) * 100

    def _naive_prognose(self, seasonality: int = 1):
        """ Naive forecasting method which just repeats previous samples """
        return self.true[:-seasonality]

    def _relative_error(self, benchmark: np.ndarray = None):
        """ Relative Error """
        if benchmark is None or isinstance(benchmark, int):
            # If no benchmark prediction provided - use naive forecasting
            if not isinstance(benchmark, int):
                seasonality = 1
            else:
                seasonality = benchmark
            return self._error(self.true[seasonality:], self.predicted[seasonality:]) / \
                   (self._error(self.true[seasonality:], self._naive_prognose(seasonality)) + EPS)

        return self._error() / (self._error(self.true, benchmark) + EPS)

    def _bounded_relative_error(self, benchmark: np.ndarray = None):
        """ Bounded Relative Error """
        if benchmark is None or isinstance(benchmark, int):
            # If no benchmark prediction provided - use naive forecasting
            if not isinstance(benchmark, int):
                seasonality = 1
            else:
                seasonality = benchmark

            abs_err = np.abs(self._error(self.true[seasonality:], self.predicted[seasonality:]))
            abs_err_bench = np.abs(self._error(self.true[seasonality:], self._naive_prognose(seasonality)))
        else:
            abs_err = np.abs(self._error())
            abs_err_bench = np.abs(self._error())

        return abs_err / (abs_err + abs_err_bench + EPS)

    def _ae(self):
        """Absolute error """
        return np.abs(self.true - self.predicted)

    def scale_free_metrics(self):
            pass

    def scale_dependent_metrics(self):
            pass

    def stats(self, verbose: bool = False) -> dict:
        """ returs some important stats about true and predicted values."""
        _stats = dict()
        _stats['true'] = ts_features(self.true)
        _stats['predicted'] = ts_features(self.predicted)

        if verbose:
            print("\nName            True         Predicted  ")
            print("----------------------------------------")
            for k in _stats['true'].keys():
                print("{:<25},  {:<10},  {:<10}"
                      .format(k, round(_stats['true'][k], 4), round(_stats['predicted'][k])))

        return _stats

    def percentage_metrics(self):
            pass

    def relative_metrics(self):
            pass

    def composite_metrics(self):
            pass

    def treat_values(self):
        """
        This function is applied by default at the start/at the time of initiating
        the class. However, it can used any time after that. This can be handy
        if we want to calculate error first by ignoring nan and then by no ignoring
        nan.
        Adopting from https://github.com/BYU-Hydroinformatics/HydroErr/blob/master/HydroErr/HydroErr.py#L6210
        Removes the nan, negative, and inf values in two numpy arrays
        """
        sim_copy = np.copy(self.predicted)
        obs_copy = np.copy(self.true)

        # Treat missing data in observed_array and simulated_array, rows in simulated_array or
        # observed_array that contain nan values
        all_treatment_array = np.ones(obs_copy.size, dtype=bool)

        if np.any(np.isnan(obs_copy)) or np.any(np.isnan(sim_copy)):
            if self.replace_nan is not None:
                # Finding the NaNs
                sim_nan = np.isnan(sim_copy)
                obs_nan = np.isnan(obs_copy)
                # Replacing the NaNs with the input
                sim_copy[sim_nan] = self.replace_nan
                obs_copy[obs_nan] = self.replace_nan

                warnings.warn("Elements(s) {} contained NaN values in the simulated array and "
                              "elements(s) {} contained NaN values in the observed array and have been "
                              "replaced (Elements are zero indexed).".format(np.where(sim_nan)[0],
                                                                             np.where(obs_nan)[0]),
                              UserWarning)
            else:
                # Getting the indices of the nan values, combining them, and informing user.
                nan_indices_fcst = ~np.isnan(sim_copy)
                nan_indices_obs = ~np.isnan(obs_copy)
                all_nan_indices = np.logical_and(nan_indices_fcst, nan_indices_obs)
                all_treatment_array = np.logical_and(all_treatment_array, all_nan_indices)

                warnings.warn("Row(s) {} contained NaN values and the row(s) have been "
                              "removed (Rows are zero indexed).".format(np.where(~all_nan_indices)[0]),
                              UserWarning)

        if np.any(np.isinf(obs_copy)) or np.any(np.isinf(sim_copy)):
            if self.replace_nan is not None:
                # Finding the NaNs
                sim_inf = np.isinf(sim_copy)
                obs_inf = np.isinf(obs_copy)
                # Replacing the NaNs with the input
                sim_copy[sim_inf] = self.replace_inf
                obs_copy[obs_inf] = self.replace_inf

                warnings.warn("Elements(s) {} contained Inf values in the simulated array and "
                              "elements(s) {} contained Inf values in the observed array and have been "
                              "replaced (Elements are zero indexed).".format(np.where(sim_inf)[0],
                                                                             np.where(obs_inf)[0]),
                              UserWarning)
            else:
                inf_indices_fcst = ~(np.isinf(sim_copy))
                inf_indices_obs = ~np.isinf(obs_copy)
                all_inf_indices = np.logical_and(inf_indices_fcst, inf_indices_obs)
                all_treatment_array = np.logical_and(all_treatment_array, all_inf_indices)

                warnings.warn(
                    "Row(s) {} contained Inf or -Inf values and the row(s) have been removed (Rows "
                    "are zero indexed).".format(np.where(~all_inf_indices)[0]),
                    UserWarning
                )

        # Treat zero data in observed_array and simulated_array, rows in simulated_array or
        # observed_array that contain zero values
        if self.remove_zero:
            if (obs_copy == 0).any() or (sim_copy == 0).any():
                zero_indices_fcst = ~(sim_copy == 0)
                zero_indices_obs = ~(obs_copy == 0)
                all_zero_indices = np.logical_and(zero_indices_fcst, zero_indices_obs)
                all_treatment_array = np.logical_and(all_treatment_array, all_zero_indices)

                warnings.warn(
                    "Row(s) {} contained zero values and the row(s) have been removed (Rows are "
                    "zero indexed).".format(np.where(~all_zero_indices)[0]),
                    UserWarning
                )

        # Treat negative data in observed_array and simulated_array, rows in simulated_array or
        # observed_array that contain negative values

        # Ignore runtime warnings from comparing
        if self.remove_neg:
            with np.errstate(invalid='ignore'):
                obs_copy_bool = obs_copy < 0
                sim_copy_bool = sim_copy < 0

            if obs_copy_bool.any() or sim_copy_bool.any():
                neg_indices_fcst = ~sim_copy_bool
                neg_indices_obs = ~obs_copy_bool
                all_neg_indices = np.logical_and(neg_indices_fcst, neg_indices_obs)
                all_treatment_array = np.logical_and(all_treatment_array, all_neg_indices)

                warnings.warn("Row(s) {} contained negative values and the row(s) have been "
                              "removed (Rows are zero indexed).".format(np.where(~all_neg_indices)[0]),
                              UserWarning)

        self.true = obs_copy[all_treatment_array]
        self.predicted = sim_copy[all_treatment_array]

        return


class RegressionMetrics(Metrics):
    """
    Calculates more than 100 regression performance metrics related to sequence data.

    Example
    ---------
    ```python
    import numpy as np
    from ai4water.utils.SeqMetrics import RegressionMetrics
    t = np.random.random(10)
    p = np.random.random(10)
    errors = RegressionMetrics(t,p)
    all_errors = errors.calculate_all()
    ```
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.all_methods = list_subclass_methods(RegressionMetrics, True)

        # if arrays contain negative values, following three errors can not be computed
        for array in [self.true, self.predicted]:

            assert len(array) > 0, "Input arrays should not be empty"

            if len(array[array < 0.0]) > 0:
                self.all_methods = [m for m in self.all_methods if m not in ('mean_gamma_deviance',
                                                                             'mean_poisson_deviance',
                                                                             'mean_square_log_error')]
            if (array <= 0).any():  # mean tweedie error is not computable
                self.all_methods = [m for m in self.all_methods if m not in ('mean_gamma_deviance',
                                                                             'mean_poisson_deviance')]
        """
        Initializes `Metrics`.
        """
    def abs_pbias(self) -> float:
        """ Absolute Percent bias"""
        _apb = 100.0 * sum(abs(self.predicted - self.true)) / sum(self.true)  # Absolute percent bias
        return float(_apb)

    def acc(self) -> float:
        """Anomaly correction coefficient.
        Reference:
            [Langland et al., 2012](https://doi.org/10.3402/tellusa.v64i0.17531).
            Miyakoda et al., 1972. Murphy et al., 1989."""
        a = self.predicted - np.mean(self.predicted)
        b = self.true - np.mean(self.true)
        c = np.std(self.true, ddof=1) * np.std(self.predicted, ddof=1) * self.predicted.size
        return float(np.dot(a, b / c))

    def adjusted_r2(self) -> float:
        """
        Adjusted R squared
        """
        k = 1
        n = len(self.predicted)
        adj_r = 1 - ((1 - self.r2()) * (n - 1)) / (n - k - 1)
        return float(adj_r)

    def agreement_index(self) -> float:
        """
        Agreement Index (d) developed by [Willmott, 1981](https://doi.org/10.1080/02723646.1981.10642213).
        It detects additive and pro-portional differences in the observed and
        simulated means and vari-ances [Moriasi et al., 2015](https://doi.org/10.13031/trans.58.10715).
        It is overly sensitive to extreme values due to the squared
        differences [2]. It can also be used as a substitute for R2 to identify the degree to which model predic-tions
        are error-free [2].
            .. math::
            d = 1 - \\frac{\\sum_{i=1}^{N}(e_{i} - s_{i})^2}{\\sum_{i=1}^{N}(\\left | s_{i} - \\bar{e}
             \\right | + \\left | e_{i} - \\bar{e} \\right |)^2}

        [2] Legates and McCabe, 199
        """
        agreement_index = 1 - (np.sum((self.true - self.predicted) ** 2)) / (np.sum(
            (np.abs(self.predicted - np.mean(self.true)) + np.abs(self.true - np.mean(self.true))) ** 2))
        return float(agreement_index)

    def aic(self, p=1) -> float:
        """
        [Akaike’s Information Criterion](https://doi.org/10.1007/978-1-4612-1694-0_15)
        Modifying from https://github.com/UBC-MDS/RegscorePy/blob/master/RegscorePy/aic.py
        """
        assert p > 0
        self.assert_greater_than_one  # noac

        n = len(self.true)
        resid = np.subtract(self.predicted, self.true)
        rss = np.sum(np.power(resid, 2))
        return float(n * np.log(rss / n) + 2 * p)

    def aitchison(self, center='mean') -> float:
        """Aitchison distance. used in [Zhang et al., 2020](https://doi.org/10.5194/hess-24-2505-2020)"""
        lx = np.log(self.true)
        ly = np.log(self.predicted)
        if center.upper() == 'MEAN':
            m = np.mean
        elif center.upper() == 'MEDIAN':
            m = np.median
        else:
            raise ValueError

        clr_x = lx - m(lx)
        clr_y = ly - m(ly)
        d = (sum((clr_x - clr_y) ** 2)) ** 0.5
        return float(d)

    def amemiya_adj_r2(self) -> float:
        """Amemiya’s Adjusted R-squared"""
        k = 1
        n = len(self.predicted)
        adj_r = 1 - ((1 - self.r2()) * (n + k)) / (n - k - 1)
        return float(adj_r)

    def amemiya_pred_criterion(self) -> float:
        """Amemiya’s Prediction Criterion"""
        k = 1
        n = len(self.predicted)
        return float(((n + k) / (n - k)) * (1/n) * self.sse())

    def bias(self) -> float:
        """
        Bias as shown in  https://doi.org/10.1029/97WR03495 and given by [Gupta et al., 1998](https://doi.org/10.1080/02626667.2018.1552002
            .. math::
            Bias=\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})
        """
        bias = np.nansum(self.true - self.predicted) / len(self.true)
        return float(bias)

    def bic(self, p=1) -> float:
        """
        Bayesian Information Criterion
        Minimising the BIC is intended to give the best model. The
        model chosen by the BIC is either the same as that chosen by the AIC, or one
        with fewer terms. This is because the BIC penalises the number of parameters
        more heavily than the AIC [1].
        Modified after https://github.com/UBC-MDS/RegscorePy/blob/master/RegscorePy/bic.py
        [1]: https://otexts.com/fpp2/selecting-predictors.html#schwarzs-bayesian-information-criterion
        """
        assert p >= 0

        n = len(self.true)
        return float(n * np.log(self.sse() / n) + p * np.log(n))

    def brier_score(self) -> float:
        """
        Adopted from https://github.com/PeterRochford/SkillMetrics/blob/master/skill_metrics/brier_score.py
        Calculates the Brier score (BS), a measure of the mean-square error of
        probability forecasts for a dichotomous (two-category) event, such as
        the occurrence/non-occurrence of precipitation. The score is calculated
        using the formula:
        BS = sum_(n=1)^N (f_n - o_n)^2/N

        where f is the forecast probabilities, o is the observed probabilities
        (0 or 1), and N is the total number of values in f & o. Note that f & o
        must have the same number of values, and those values must be in the
        range [0,1].
        https://data.library.virginia.edu/a-brief-on-brier-scores/

        Output:
        BS : Brier score

        Reference:
        Glenn W. Brier, 1950: Verification of forecasts expressed in terms
        of probabilities. Mon. We. Rev., 78, 1-23.
        D. S. Wilks, 1995: Statistical Methods in the Atmospheric Sciences.
        Cambridge Press. 547 pp.

        """
        # Check for valid values
        index = np.where(np.logical_or(self.predicted < 0, self.predicted > 1))
        if np.sum(index) > 0:
            msg = 'Forecast has values outside interval [0,1].'
            raise ValueError(msg)

        index = np.where(np.logical_and(self.true != 0, self.true != 1))
        if np.sum(index) > 0:
            msg = 'Observed has values not equal to 0 or 1.'
            raise ValueError(msg)

        # Calculate score
        bs = np.sum(np.square(self.predicted - self.true)) / len(self.predicted)

        return bs

    def corr_coeff(self) -> float:
        """
        Correlation Coefficient
            .. math::
            r = \\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}{\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2}
             \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}}
        """
        correlation_coefficient = np.corrcoef(self.true, self.predicted)[0, 1]
        return float(correlation_coefficient)

    def covariance(self) -> float:
        """
        Covariance
            .. math::
            Covariance = \\frac{1}{N} \\sum_{i=1}^{N}((e_{i} - \\bar{e}) * (s_{i} - \\bar{s}))
        """
        obs_mean = np.mean(self.true)
        sim_mean = np.mean(self.predicted)
        covariance = np.mean((self.true - obs_mean) * (self.predicted - sim_mean))
        return float(covariance)

    def cronbach_alpha(self) -> float:
        """
        It is a measure of internal consitency of data
        https://stats.idre.ucla.edu/spss/faq/what-does-cronbachs-alpha-mean/
        https://stackoverflow.com/a/20799687/5982232
        """
        itemscores = np.stack([self.true, self.predicted])
        itemvars = itemscores.var(axis=1, ddof=1)
        tscores = itemscores.sum(axis=0)
        nitems = len(itemscores)
        return float(nitems / (nitems - 1.) * (1 - itemvars.sum() / tscores.var(ddof=1)))

    def centered_rms_dev(self) -> float:
        """
        Modified after https://github.com/PeterRochford/SkillMetrics/blob/master/skill_metrics/centered_rms_dev.py
        Calculates the centered root-mean-square (RMS) difference between true and predicted
        using the formula:
        (E')^2 = sum_(n=1)^N [(p_n - mean(p))(r_n - mean(r))]^2/N
        where p is the predicted values, r is the true values, and
        N is the total number of values in p & r.

        Output:
        CRMSDIFF : centered root-mean-square (RMS) difference (E')^2
        """
        # Calculate means
        pmean = np.mean(self.predicted)
        rmean = np.mean(self.true)

        # Calculate (E')^2
        crmsd = np.square((self.predicted - pmean) - (self.true - rmean))
        crmsd = np.sum(crmsd) / self.predicted.size
        crmsd = np.sqrt(crmsd)

        return float(crmsd)

    def cosine_similarity(self)->float:
        """[cosine similary](https://en.wikipedia.org/wiki/Cosine_similarity)
        It is a judgment of orientation and not magnitude: two vectors with
        the same orientation have a cosine similarity of 1, two vectors oriented
        at 90° relative to each other have a similarity of 0, and two vectors diametrically
        opposed have a similarity of -1, independent of their magnitude.
        """
        return float(np.dot(self.true.reshape(-1,), self.predicted.reshape(-1,))/(np.linalg.norm(self.true)*np.linalg.norm(self.predicted)))

    def decomposed_mse(self) -> float:
        """
        Decomposed MSE developed by Kobayashi and Salam (2000)
            .. math ::
            dMSE = (\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i}))^2 + SDSD + LCS
            SDSD = (\\sigma(e) - \\sigma(s))^2
            LCS = 2 \\sigma(e) \\sigma(s) * (1 - \\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}
            {\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2} \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}})
        """
        e_std = np.std(self.true)
        s_std = np.std(self.predicted)

        bias_squared = self.bias() ** 2
        sdsd = (e_std - s_std) ** 2
        lcs = 2 * e_std * s_std * (1 - self.corr_coeff())

        decomposed_mse = bias_squared + sdsd + lcs

        return float(decomposed_mse)

    def euclid_distance(self) -> float:
        """Euclidian distance

        Referneces: Kennard et al., 2010
        """
        return float(np.linalg.norm(self.true - self.predicted))

    def exp_var_score(self, weights=None) -> Union[float, None]:
        """
        Explained variance score
        https://stackoverflow.com/questions/24378176/python-sci-kit-learn-metrics-difference-between-r2-score-and-explained-varian
        best value is 1, lower values are less accurate.
        """
        y_diff_avg = np.average(self.true - self.predicted, weights=weights, axis=0)
        numerator = np.average((self.true - self.predicted - y_diff_avg) ** 2,
                               weights=weights, axis=0)

        y_true_avg = np.average(self.true, weights=weights, axis=0)
        denominator = np.average((self.true - y_true_avg) ** 2,
                                 weights=weights, axis=0)

        if numerator == 0.0:
            return None
        output_scores = _foo(denominator, numerator)

        return float(np.average(output_scores, weights=weights))

    def expanded_uncertainty(self, cov_fact=1.96) -> float:
        """By default it calculates uncertainty with 95% confidence interval. 1.96 is the coverage factor
         corresponding 95% confidence level [2]. This indicator is used in order to show more information about the
         model deviation [2].
        Using formula from by [1] and [2].
        [1] https://doi.org/10.1016/j.enconman.2015.03.067
        [2] https://doi.org/10.1016/j.rser.2014.07.117
        """
        sd = np.std(self._error(self.true, self.predicted))
        return float(cov_fact * np.sqrt(sd ** 2 + self.rmse() ** 2))

    def fdc_fhv(self, h: float = 0.02) -> float:
        """
        modified after: https://github.com/kratzert/ealstm_regional_modeling/blob/64a446e9012ecd601e0a9680246d3bbf3f002f6d/papercode/metrics.py#L190
        Peak flow bias of the flow duration curve (Yilmaz 2008).
        used in kratzert et al., 2018
        Returns
        -------
        float
            Bias of the peak flows

        Raises
        ------

        RuntimeError
            If `h` is not in range(0,1)
        """
        if (h <= 0) or (h >= 1):
            raise RuntimeError("h has to be in the range (0,1)")

        # sort both in descending order
        obs = -np.sort(-self.true)
        sim = -np.sort(-self.predicted)

        # subset data to only top h flow values
        obs = obs[:np.round(h * len(obs)).astype(int)]
        sim = sim[:np.round(h * len(sim)).astype(int)]

        fhv = np.sum(sim - obs) / (np.sum(obs) + 1e-6)

        return float(fhv * 100)

    def fdc_flv(self, low_flow: float = 0.3) -> float:
        """
        bias of the bottom 30 % low flows
        modified after: https://github.com/kratzert/ealstm_regional_modeling/blob/64a446e9012ecd601e0a9680246d3bbf3f002f6d/papercode/metrics.py#L237
        used in kratzert et al., 2018
        Parameters
        ----------
        low_flow : float, optional
            Upper limit of the flow duration curve. E.g. 0.3 means the bottom 30% of the flows are
            considered as low flows, by default 0.3

        Returns
        -------
        float
            Bias of the low flows.

        Raises
        ------
        RuntimeError
            If `low_flow` is not in the range(0,1)
        """

        low_flow = 1.0 - low_flow
        # make sure that metric is calculated over the same dimension
        obs = self.true.flatten()
        sim = self.predicted.flatten()

        if (low_flow <= 0) or (low_flow >= 1):
            raise RuntimeError("l has to be in the range (0,1)")

        # for numerical reasons change 0s to 1e-6
        sim[sim == 0] = 1e-6
        obs[obs == 0] = 1e-6

        # sort both in descending order
        obs = -np.sort(-obs)
        sim = -np.sort(-sim)

        # subset data to only top h flow values
        obs = obs[np.round(low_flow * len(obs)).astype(int):]
        sim = sim[np.round(low_flow * len(sim)).astype(int):]

        # transform values to log scale
        obs = np.log(obs + 1e-6)
        sim = np.log(sim + 1e-6)

        # calculate flv part by part
        qsl = np.sum(sim - sim.min())
        qol = np.sum(obs - obs.min())

        flv = -1 * (qsl - qol) / (qol + 1e-6)

        return float(flv * 100)

    def gmae(self) -> float:
        """ Geometric Mean Absolute Error """
        return _geometric_mean(np.abs(self._error()))

    def gmean_diff(self) -> float:
        """Geometric mean difference. First geometric mean is calculated for each of two samples and their difference
        is calculated."""
        sim_log = np.log1p(self.predicted)
        obs_log = np.log1p(self.true)
        return float(np.exp(gmean(sim_log) - gmean(obs_log)))

    def gmrae(self, benchmark: np.ndarray = None) -> float:
        """ Geometric Mean Relative Absolute Error """
        return _geometric_mean(np.abs(self._relative_error(benchmark)))

    def inrse(self) -> float:
        """ Integral Normalized Root Squared Error """
        return float(np.sqrt(np.sum(np.square(self._error())) / np.sum(np.square(self.true - np.mean(self.true)))))

    def irmse(self) -> float:
        """Inertial RMSE. RMSE divided by standard deviation of the gradient of true."""
        # Getting the gradient of the observed data
        obs_len = self.true.size
        obs_grad = self.true[1:obs_len] - self.true[0:obs_len - 1]

        # Standard deviation of the gradient
        obs_grad_std = np.std(obs_grad, ddof=1)

        # Divide RMSE by the standard deviation of the gradient of the observed data
        return float(self.rmse() / obs_grad_std)

    def JS(self) -> float:
        """Jensen-shannon divergence"""
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        d1 = self.true * np.log2(2 * self.true / (self.true + self.predicted))
        d2 = self.predicted * np.log2(2 * self.predicted / (self.true + self.predicted))
        d1[np.isnan(d1)] = 0
        d2[np.isnan(d2)] = 0
        d = 0.5 * sum(d1 + d2)
        return float(d)

    def kendaull_tau(self, return_p=False)->Union[float, tuple]:
        """Kendall's tau
        https://machinelearningmastery.com/how-to-calculate-nonparametric-rank-correlation-in-python/
        used in https://www.jmlr.org/papers/volume20/18-444/18-444.pdf
        """
        coef, p = kendalltau(self.true, self.predicted)
        if return_p:
            return coef, p
        return float(p)

    def kge(self, return_all=False):
        """
        Kling-Gupta Efficiency
        Gupta, Kling, Yilmaz, Martinez, 2009, Decomposition of the mean squared error and NSE performance
         criteria: Implications for improving hydrological modelling
        output:
            kge: Kling-Gupta Efficiency
            cc: correlation
            alpha: ratio of the standard deviation
            beta: ratio of the mean
        """
        cc = np.corrcoef(self.true, self.predicted)[0, 1]
        alpha = np.std(self.predicted) / np.std(self.true)
        beta = np.sum(self.predicted) / np.sum(self.true)
        kge = float(1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
        if return_all:
            return np.vstack((kge, cc, alpha, beta))
        else:
            return kge

    def kge_bound(self) -> float:
        """
        Bounded Version of the Original Kling-Gupta Efficiency
        https://iahs.info/uploads/dms/13614.21--211-219-41-MATHEVET.pdf
        """
        kge_ = self.kge(return_all=True)[0, :]
        kge_c2m_ = kge_ / (2 - kge_)

        return float(kge_c2m_)

    def kge_mod(self, return_all=False):
        """
        Modified Kling-Gupta Efficiency (Kling et al. 2012 - https://doi.org/10.1016/j.jhydrol.2012.01.011)
        """
        # calculate error in timing and dynamics r (Pearson's correlation coefficient)
        sim_mean = np.mean(self.predicted, axis=0, dtype=np.float64)
        obs_mean = np.mean(self.true, dtype=np.float64)
        r = np.sum((self.predicted - sim_mean) * (self.true - obs_mean), axis=0, dtype=np.float64) / \
            np.sqrt(np.sum((self.predicted - sim_mean) ** 2, axis=0, dtype=np.float64) *
                    np.sum((self.true - obs_mean) ** 2, dtype=np.float64))
        # calculate error in spread of flow gamma (avoiding cross correlation with bias by dividing by the mean)
        gamma = (np.std(self.predicted, axis=0, dtype=np.float64) / sim_mean) / \
                (np.std(self.true, dtype=np.float64) / obs_mean)
        # calculate error in volume beta (bias of mean discharge)
        beta = np.mean(self.predicted, axis=0, dtype=np.float64) / np.mean(self.true, axis=0, dtype=np.float64)
        # calculate the modified Kling-Gupta Efficiency KGE'
        kgeprime_ = float(1 - np.sqrt((r - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2))

        if return_all:
            return np.vstack((kgeprime_, r, gamma, beta))
        else:
            return kgeprime_

    def kge_np(self, return_all=False):
        """
        Non parametric Kling-Gupta Efficiency
        Corresponding paper:
        Pool, Vis, and Seibert, 2018 Evaluating model performance: towards a non-parametric variant of the
         Kling-Gupta efficiency, Hydrological Sciences Journal.
        https://doi.org/10.1080/02626667.2018.1552002
        output:
            kge: Kling-Gupta Efficiency
            cc: correlation
            alpha: ratio of the standard deviation
            beta: ratio of the mean
        """
        # # self-made formula
        cc = self.spearmann_corr()

        fdc_sim = np.sort(self.predicted / (np.nanmean(self.predicted) * len(self.predicted)))
        fdc_obs = np.sort(self.true / (np.nanmean(self.true) * len(self.true)))
        alpha = 1 - 0.5 * np.nanmean(np.abs(fdc_sim - fdc_obs))

        beta = np.mean(self.predicted) / np.mean(self.true)
        kge = float(1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
        if return_all:
            return np.vstack((kge, cc, alpha, beta))
        else:
            return kge

    def kgeprime_c2m(self) -> float:
        """
        https://iahs.info/uploads/dms/13614.21--211-219-41-MATHEVET.pdf
         Bounded Version of the Modified Kling-Gupta Efficiency
        """
        kgeprime_ = self.kge_mod(return_all=True)[0, :]
        kgeprime_c2m_ = kgeprime_ / (2 - kgeprime_)

        return float(kgeprime_c2m_)

    def kgenp_bound(self):
        """
        Bounded Version of the Non-Parametric Kling-Gupta Efficiency
        """
        kgenp_ = self.kge_np(return_all=True)[0, :]
        kgenp_c2m_ = kgenp_ / (2 - kgenp_)

        return float(kgenp_c2m_)

    def KLsym(self) -> Union[float, None]:
        """Symmetric kullback-leibler divergence"""
        if not all((self.true == 0) == (self.predicted == 0)):
            return None  # ('KL divergence not defined when only one distribution is 0.')
        x, y = self.true, self.predicted
        # set values where both distributions are 0 to the same (positive) value.
        # This will not contribute to the final distance.
        x[x == 0] = 1
        y[y == 0] = 1
        d = 0.5 * np.sum((x - y) * (np.log2(x) - np.log2(y)))
        return float(d)

    def lm_index(self, obs_bar_p=None) -> float:
        """Legate-McCabe Efficiency Index.
        Less sensitive to outliers in the data.
        obs_bar_p: float, Seasonal or other selected average. If None, the mean of the observed array will be used.
        """
        mean_obs = np.mean(self.true)
        a = np.abs(self.predicted - self.true)

        if obs_bar_p is not None:

            b = np.abs(self.true - obs_bar_p)
        else:
            b = np.abs(self.true - mean_obs)

        return float(1 - (np.sum(a) / np.sum(b)))

    def maape(self) -> float:
        """
        Mean Arctangent Absolute Percentage Error
        Note: result is NOT multiplied by 100
        """
        return float(np.mean(np.arctan(np.abs((self.true - self.predicted) / (self.true + EPS)))))

    def mae(self, true=None, predicted=None) -> float:
        """ Mean Absolute Error """
        if true is None:
            true = self.true
        if predicted is None:
            predicted = self.predicted
        return float(np.mean(np.abs(true - predicted)))

    def mape(self) -> float:
        """ Mean Absolute Percentage Error.
        The MAPE is often used when the quantity to predict is known to remain way above zero [1]. It is useful when
        the size or size of a prediction variable is significant in evaluating the accuracy of a prediction [2]. It has
        advantages of scale-independency and interpretability [3]. However, it has the significant disadvantage that it
        produces infinite or undefined values for zero or close-to-zero actual values [3].

        [1] https://doi.org/10.1016/j.neucom.2015.12.114
        [2] https://doi.org/10.1088/1742-6596/930/1/012002
        [3] https://doi.org/10.1016/j.ijforecast.2015.12.003
        """
        return float(np.mean(np.abs((self.true - self.predicted) / self.true)) * 100)

    def mbe(self) -> float:
        """Mean bias error. This indicator expresses a tendency of model to underestimate (negative value)
        or overestimate (positive value) global radiation, while the MBE values closest to zero are desirable.
        The drawback of this test is that it does not show the correct performance when the model presents
        overestimated and underestimated values at the same time, since overestimation and underestimation
        values cancel each other. [1]

        [1] https://doi.org/10.1016/j.rser.2015.08.035
        """
        return float(np.mean(self._error(self.true, self.predicted)))

    def mbrae(self, benchmark: np.ndarray = None) -> float:
        """ Mean Bounded Relative Absolute Error """
        return float(np.mean(self._bounded_relative_error(benchmark)))

    def mapd(self) -> float:
        """Mean absolute percentage deviation."""
        a = np.sum(np.abs(self.predicted - self.true))
        b = np.sum(np.abs(self.true))
        return float(a / b)

    def mase(self, seasonality: int = 1):
        """
        Mean Absolute Scaled Error
        Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
        modified after https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
        Hyndman, R. J. (2006). Another look at forecast-accuracy metrics for intermittent demand.
        Foresight: The International Journal of Applied Forecasting, 4(4), 43-46.
        """
        return self.mae() / self.mae(self.true[seasonality:], self._naive_prognose(seasonality))

    def mare(self) -> float:
        """ Mean Absolute Relative Error. When expressed in %age, it is also known as mape. [1]
        https://doi.org/10.1016/j.rser.2015.08.035
        """
        return float(np.mean(np.abs(self._error(self.true, self.predicted) / self.true)))

    def max_error(self) -> float:
        """
        maximum error
        """
        return float(np.max(self._ae()))

    def mb_r(self) -> float:
        """Mielke-Berry R value.
        Berry and Mielke, 1988.
        Mielke, P. W., & Berry, K. J. (2007). Permutation methods: a distance function approach.
         Springer Science & Business Media.
        """
        # Calculate metric
        n = self.predicted.size
        tot = 0.0
        for i in range(n):
            tot = tot + np.sum(np.abs(self.predicted - self.true[i]))
        mae_val = np.sum(np.abs(self.predicted - self.true)) / n
        mb = 1 - ((n ** 2) * mae_val / tot)

        return float(mb)

    def mda(self) -> float:
        """ Mean Directional Accuracy
         modified after https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
         """
        dict_acc = np.sign(self.true[1:] - self.true[:-1]) == np.sign(self.predicted[1:] - self.predicted[:-1])
        return float(np.mean(dict_acc))

    def mde(self) -> float:
        """Median Error"""
        return float(np.median(self.predicted - self.true))

    def mdape(self) -> float:
        """
        Median Absolute Percentage Error
        """
        return float(np.median(np.abs(self._percentage_error())) * 100)

    def mdrae(self, benchmark: np.ndarray = None) -> float:
        """ Median Relative Absolute Error """
        return float(np.median(np.abs(self._relative_error(benchmark))))

    def me(self):
        """Mean error """
        return float(np.mean(self._error()))

    def mean_bias_error(self) -> float:
        """
        Mean Bias Error
        It represents overall bias error or systematic error. It shows average interpolation bias; i.e. average over-
        or underestimation. [1][2].This indicator expresses a tendency of model to underestimate (negative value)
        or overestimate (positive value) global radiation, while the MBE values closest to zero are desirable.
        The drawback of this test is that it does not show the correct performance when the model presents
        overestimated and underestimated values at the same time, since overestimation and underestimation
        values cancel each other.

    [2] Willmott, C. J., & Matsuura, K. (2006). On the use of dimensioned measures of error to evaluate the performance
        of spatial interpolators. International Journal of Geographical Information Science, 20(1), 89-102.
         https://doi.org/10.1080/1365881050028697
    [1] Valipour, M. (2015). Retracted: Comparative Evaluation of Radiation-Based Methods for Estimation of Potential
        Evapotranspiration. Journal of Hydrologic Engineering, 20(5), 04014068.
         http://dx.doi.org/10.1061/(ASCE)HE.1943-5584.0001066
    [3]  https://doi.org/10.1016/j.rser.2015.08.035
         """
        return float(np.sum(self.true - self.predicted) / len(self.true))

    def mean_var(self) -> float:
        """Mean variance"""
        return float(np.var(np.log1p(self.true) - np.log1p(self.predicted)))

    def mean_poisson_deviance(self, weights=None) -> float:
        """
        mean poisson deviance
        """
        return _mean_tweedie_deviance(self.true, self.predicted, weights=weights, power=1)

    def mean_gamma_deviance(self, weights=None) -> float:
        """
        mean gamma deviance
        """
        return _mean_tweedie_deviance(self.true, self.predicted, weights=weights, power=2)

    def median_abs_error(self) -> float:
        """
        median absolute error
        """
        return float(np.median(np.abs(self.predicted - self.true), axis=0))

    def med_seq_error(self) -> float:
        """Median Squared Error
        Same as mse but it takes median which reduces the impact of outliers.
        """
        return float(np.median((self.predicted - self.true) ** 2))

    def mle(self) -> float:
        """Mean log error"""
        return float(np.mean(np.log1p(self.predicted) - np.log1p(self.true)))

    def mod_agreement_index(self, j=1) -> float:
        """Modified agreement of index.
        j: int, when j==1, this is same as agreement_index. Higher j means more impact of outliers."""
        a = (np.abs(self.predicted - self.true)) ** j
        b = np.abs(self.predicted - np.mean(self.true))
        c = np.abs(self.true - np.mean(self.true))
        e = (b + c) ** j
        return float(1 - (np.sum(a) / np.sum(e)))

    def mpe(self) -> float:
        """ Mean Percentage Error """
        return float(np.mean(self._percentage_error()))

    def mrae(self, benchmark: np.ndarray = None):
        """ Mean Relative Absolute Error """
        return float(np.mean(np.abs(self._relative_error(benchmark))))

    def mse(self, weights=None) -> float:
        """ mean square error """
        return float(np.average((self.true - self.predicted) ** 2, axis=0, weights=weights))

    def msle(self, weights=None) -> float:
        """
        mean square logrithmic error
        """
        return float(np.average((np.log1p(self.true) - np.log1p(self.predicted)) ** 2, axis=0, weights=weights))

    def norm_euclid_distance(self) -> float:
        """Normalized Euclidian distance"""

        a = self.true / np.mean(self.true)
        b = self.predicted / np.mean(self.predicted)
        return float(np.linalg.norm(a - b))

    def nrmse_range(self) -> float:
        """Range Normalized Root Mean Squared Error.
        RMSE normalized by true values. This allows comparison between data sets with different scales. It is more
        sensitive to outliers.

        Reference: Pontius et al., 2008
        """

        return float(self.rmse() / (np.max(self.true) - np.min(self.true)))

    def nrmse_ipercentile(self, q1=25, q2=75) -> float:
        """
        RMSE normalized by inter percentile range of true. This is least sensitive to outliers.
        q1: any interger between 1 and 99
        q2: any integer between 2 and 100. Should be greater than q1.
        Reference: Pontius et al., 2008.
        """

        q1 = np.percentile(self.true, q1)
        q3 = np.percentile(self.true, q2)
        iqr = q3 - q1

        return float(self.rmse() / iqr)

    def nrmse_mean(self) -> float:
        """Mean Normalized RMSE
        RMSE normalized by mean of true values.This allows comparison between datasets with different scales.

        Reference: Pontius et al., 2008
        """
        return float(self.rmse() / np.mean(self.true))

    def norm_ae(self) -> float:
        """ Normalized Absolute Error """
        return float(np.sqrt(np.sum(np.square(self._error() - self.mae())) / (len(self.true) - 1)))

    def norm_ape(self) -> float:
        """ Normalized Absolute Percentage Error """
        return float(np.sqrt(np.sum(np.square(self._percentage_error() - self.mape())) / (len(self.true) - 1)))

    def nrmse(self) -> float:
        """ Normalized Root Mean Squared Error """
        return float(self.rmse() / (np.max(self.true) - np.min(self.true)))

    def nse(self) -> float:
        """Nash-Sutcliff Efficiency.
        It determine how well the model simulates trends for the output response of concern. But cannot help identify
        model bias and cannot be used to identify differences in timing and magnitude of peak flows and shape of
        recession curves; in other words, it cannot be used for single-event simulations. It is sensitive to extreme
        values due to the squared differ-ences [1]. To make it less sensitive to outliers, [2] proposed
        log and relative nse.
        [1] Moriasi, D. N., Gitau, M. W., Pai, N., & Daggupati, P. (2015). Hydrologic and water quality models:
            Performance measures and evaluation criteria. Transactions of the ASABE, 58(6), 1763-1785.
        [2] Krause, P., Boyle, D., & Bäse, F. (2005). Comparison of different efficiency criteria for hydrological
            model assessment. Adv. Geosci., 5, 89-97. http://dx.doi.org/10.5194/adgeo-5-89-2005.
        """
        _nse = 1 - sum((self.predicted - self.true) ** 2) / sum((self.true - np.mean(self.true)) ** 2)
        return float(_nse)

    def nse_alpha(self) -> float:
        """
        Alpha decomposition of the NSE, see [Gupta et al. 2009](https://doi.org/10.1029/97WR03495)
        used in kratzert et al., 2018
        Returns
        -------
        float
            Alpha decomposition of the NSE

        """
        return float(np.std(self.predicted) / np.std(self.true))

    def nse_beta(self) -> float:
        """
        Beta decomposition of NSE. See [Gupta et. al 2009](https://doi.org/10.1016/j.jhydrol.2009.08.003)
        used in kratzert et al., 2018
        Returns
        -------
        float
            Beta decomposition of the NSE
        """
        return float((np.mean(self.predicted) - np.mean(self.true)) / np.std(self.true))

    def nse_mod(self, j=1) -> float:
        """
        Gives less weightage of outliers if j=1 and if j>1, gives more weightage to outliers.
        Reference: Krause et al., 2005
        """
        a = (np.abs(self.predicted - self.true)) ** j
        b = (np.abs(self.true - np.mean(self.true))) ** j
        return float(1 - (np.sum(a) / np.sum(b)))

    def nse_rel(self) -> float:
        """
        Relative NSE.
        """

        a = (np.abs((self.predicted - self.true) / self.true)) ** 2
        b = (np.abs((self.true - np.mean(self.true)) / np.mean(self.true))) ** 2
        return float(1 - (np.sum(a) / np.sum(b)))

    def nse_bound(self) -> float:
        """
        Bounded Version of the Nash-Sutcliffe Efficiency
        https://iahs.info/uploads/dms/13614.21--211-219-41-MATHEVET.pdf
        """
        nse_ = self.nse()
        nse_c2m_ = nse_ / (2 - nse_)

        return nse_c2m_

    def log_nse(self, epsilon=0.0) -> float:
        """
        log Nash-Sutcliffe model efficiency
            .. math::
            NSE = 1-\\frac{\\sum_{i=1}^{N}(log(e_{i})-log(s_{i}))^2}{\\sum_{i=1}^{N}(log(e_{i})-log(\\bar{e})^2}-1)*-1
        """
        s, o = self.predicted + epsilon, self.true + epsilon
        return float(1 - sum((np.log(o) - np.log(o)) ** 2) / sum((np.log(o) - np.mean(np.log(o))) ** 2))

    def log_prob(self) -> float:
        """
        Logarithmic probability distribution
        """
        scale = np.mean(self.true) / 10
        if scale < .01:
            scale = .01
        y = (self.true - self.predicted) / scale
        normpdf = -y ** 2 / 2 - np.log(np.sqrt(2 * np.pi))
        return float(np.mean(normpdf))

    def pbias(self) -> float:
        """
        Percent Bias.
        It determine how well the model simulates the average magnitudes for the output response of interest. It can
        also determine over and under-prediction. It cannot be used (1) for single-event simula-tions to identify
        differences in timing and magnitude of peak flows and the shape of recession curves nor (2) to determine how
        well the model simulates residual variations and/or trends for the output response of interest. It can  give a
        deceiving rating of model performance if the model overpredicts as much as it underpredicts, in which case
        PBIAS will be close to zero even though the model simulation is poor. [1]
        [1] Moriasi et al., 2015
        """
        return float(100.0 * sum(self.predicted - self.true) / sum(self.true))

    def pearson_r(self) -> float:
        """
        Pearson correlation coefficient.
        It measures linear correlatin between true and predicted arrays.
        It is sensitive to outliers.
        Reference: Pearson, K 1895.
        """
        sim_mean = np.mean(self.predicted)
        obs_mean = np.mean(self.true)

        top = np.sum((self.true - obs_mean) * (self.predicted - sim_mean))
        bot1 = np.sqrt(np.sum((self.true - obs_mean) ** 2))
        bot2 = np.sqrt(np.sum((self.predicted - sim_mean) ** 2))

        return float(top / (bot1 * bot2))

    def rmsle(self) -> float:
        """Root mean square log error. Compared to RMSE, RMSLE only considers the relative error between predicted and
         actual values, and the scale of the error is nullified by the log-transformation. Furthermore, RMSLE penalizes
         underestimation more than overestimation. This is especially useful in our studies where the underestimation
         of the target variable is not acceptable but overestimation can be tolerated. [1]

         [1] https://doi.org/10.1016/j.scitotenv.2020.137894
         """
        return float(np.sqrt(np.mean(np.power(np.log1p(self.predicted) - np.log1p(self.true), 2))))

    def rmdspe(self) -> float:
        """
        Root Median Squared Percentage Error
        """
        return float(np.sqrt(np.median(np.square(self._percentage_error()))) * 100.0)

    def rse(self) -> float:
        """Relative Squared Error"""
        return float(np.sum(np.square(self.true - self.predicted)) / np.sum(np.square(self.true - np.mean(self.true))))

    def rrse(self) -> float:
        """ Root Relative Squared Error """
        return float(np.sqrt(self.rse()))

    def rae(self) -> float:
        """ Relative Absolute Error (aka Approximation Error) """
        return float(np.sum(self._ae()) / (np.sum(np.abs(self.true - np.mean(self.true))) + EPS))

    def ref_agreement_index(self) -> float:
        """Refined Index of Agreement. From -1 to 1. Larger the better.
        Refrence: Willmott et al., 2012"""
        a = np.sum(np.abs(self.predicted - self.true))
        b = 2 * np.sum(np.abs(self.true - self.true.mean()))
        if a <= b:
            return float(1 - (a / b))
        else:
            return float((b / a) - 1)

    def rel_agreement_index(self) -> float:
        """Relative index of agreement. from 0 to 1. larger the better."""
        a = ((self.predicted - self.true) / self.true) ** 2
        b = np.abs(self.predicted - np.mean(self.true))
        c = np.abs(self.true - np.mean(self.true))
        e = ((b + c) / np.mean(self.true)) ** 2
        return float(1 - (np.sum(a) / np.sum(e)))

    def rmse(self, weights=None) -> float:
        """ root mean square error"""
        return sqrt(np.average((self.true - self.predicted) ** 2, axis=0, weights=weights))

    def r2(self) -> float:
        """
        Quantifies the percent of variation in the response that the 'model'
        explains. The 'model' here is anything from which we obtained predicted
        array. It is also called coefficient of determination or square of pearson
        correlation coefficient. More heavily affected by outliers than pearson correlatin r.
        https://data.library.virginia.edu/is-r-squared-useless/
        """
        zx = (self.true - np.mean(self.true)) / np.std(self.true, ddof=1)
        zy = (self.predicted - np.mean(self.predicted)) / np.std(self.predicted, ddof=1)
        r = np.sum(zx * zy) / (len(self.true) - 1)
        return float(r ** 2)

    def r2_mod(self, weights=None):
        """
        This is not a symmetric function.
        Unlike most other scores, R^2 score may be negative (it need not actually
        be the square of a quantity R).
        This metric is not well-defined for single samples and will return a NaN
        value if n_samples is less than two.
        """

        if len(self.predicted) < 2:
            msg = "R^2 score is not well-defined with less than two samples."
            warnings.warn(msg)
            return None

        if weights is None:
            weight = 1.
        else:
            weight = weights[:, np.newaxis]

        numerator = (weight * (self.true - self.predicted) ** 2).sum(axis=0,
                                                                     dtype=np.float64)
        denominator = (weight * (self.true - np.average(
            self.true, axis=0, weights=weights)) ** 2).sum(axis=0, dtype=np.float64)

        if numerator == 0.0:
            return None
        output_scores = _foo(denominator, numerator)

        return float(np.average(output_scores, weights=weights))

    def relative_rmse(self) -> float:
        """
        Relative Root Mean Squared Error
            .. math::
            RRMSE=\\frac{\\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2}}{\\bar{e}}
        """
        rrmse = self.rmse() / np.mean(self.true)
        return float(rrmse)

    def rmspe(self) -> float:
        """
        Root Mean Square Percentage Error
        https://stackoverflow.com/a/53166790/5982232
        """
        return float(np.sqrt(np.mean(np.square(((self.true - self.predicted) / self.true)), axis=0)))

    def rsr(self) -> float:
        """
        Moriasi et al., 2007.
        It incorporates the benefits of error index statistics andincludes a scaling/normalization factor,
        so that the resulting statistic and reported values can apply to various constitu-ents."""
        return float(self.rmse() / np.std(self.true))

    def rmsse(self, seasonality: int = 1) -> float:
        """ Root Mean Squared Scaled Error """
        q = np.abs(self._error()) / self.mae(self.true[seasonality:], self._naive_prognose(seasonality))
        return float(np.sqrt(np.mean(np.square(q))))

    def sa(self) -> float:
        """Spectral angle. From -pi/2 to pi/2. Closer to 0 is better.
        It measures angle between two vectors in hyperspace indicating how well the shape of two arrays match instead
        of their magnitude.
        Reference: Robila and Gershman, 2005."""
        a = np.dot(self.predicted, self.true)
        b = np.linalg.norm(self.predicted) * np.linalg.norm(self.true)
        return float(np.arccos(a / b))

    def sc(self) -> float:
        """Spectral correlation.
         From -pi/2 to pi/2. Closer to 0 is better.
        """
        a = np.dot(self.true - np.mean(self.true), self.predicted - np.mean(self.predicted))
        b = np.linalg.norm(self.true - np.mean(self.true))
        c = np.linalg.norm(self.predicted - np.mean(self.predicted))
        e = b * c
        return float(np.arccos(a / e))

    def sga(self) -> float:
        """Spectral gradient angle.
        From -pi/2 to pi/2. Closer to 0 is better.
        """
        sgx = self.true[1:] - self.true[:self.true.size - 1]
        sgy = self.predicted[1:] - self.predicted[:self.predicted.size - 1]
        a = np.dot(sgx, sgy)
        b = np.linalg.norm(sgx) * np.linalg.norm(sgy)
        return float(np.arccos(a / b))

    def smape(self) -> float:
        """
         Symmetric Mean Absolute Percentage Error
         https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
         https://stackoverflow.com/a/51440114/5982232
        """
        _temp = np.sum(2 * np.abs(self.predicted - self.true) / (np.abs(self.true) + np.abs(self.predicted)))
        return float(100 / len(self.true) * _temp)

    def smdape(self) -> float:
        """
        Symmetric Median Absolute Percentage Error
        Note: result is NOT multiplied by 100
        """
        return float(np.median(2.0 * self._ae() / ((np.abs(self.true) + np.abs(self.predicted)) + EPS)))

    def sid(self) -> float:
        """Spectral Information Divergence.
        From -pi/2 to pi/2. Closer to 0 is better. """
        first = (self.true / np.mean(self.true)) - (
                self.predicted / np.mean(self.predicted))
        second1 = np.log10(self.true) - np.log10(np.mean(self.true))
        second2 = np.log10(self.predicted) - np.log10(np.mean(self.predicted))
        return float(np.dot(first, second1 - second2))

    def skill_score_murphy(self) -> float:
        """
        Adopted from https://github.com/PeterRochford/SkillMetrics/blob/278b2f58c7d73566f25f10c9c16a15dc204f5869/skill_metrics/skill_score_murphy.py
        Calculate non-dimensional skill score (SS) between two variables using
        definition of Murphy (1988) using the formula:

        SS = 1 - RMSE^2/SDEV^2

        SDEV is the standard deviation of the true values

        SDEV^2 = sum_(n=1)^N [r_n - mean(r)]^2/(N-1)

        where p is the predicted values, r is the reference values, and N is the total number of values in p & r.
        Note that p & r must have the same number of values. A positive skill score can be interpreted as the percentage
        of improvement of the new model forecast in comparison to the reference. On the other hand, a negative skill
        score denotes that the forecast of interest is worse than the referencing forecast. Consequently, a value of
        zero denotes that both forecasts perform equally [MLAir, 2020].

        Output:
        SS : skill score
        Reference:
        Allan H. Murphy, 1988: Skill Scores Based on the Mean Square Error
        and Their Relationships to the Correlation Coefficient. Mon. Wea.
        Rev., 116, 2417-2424.
        doi: http//dx.doi.org/10.1175/1520-0493(1988)<2417:SSBOTM>2.0.CO;2
        """
        # Calculate RMSE
        rmse2 = self.rmse() ** 2

        # Calculate standard deviation
        sdev2 = np.std(self.true, ddof=1) ** 2

        # Calculate skill score
        ss = 1 - rmse2 / sdev2

        return float(ss)

    def spearmann_corr(self) -> float:
        """Separmann correlation coefficient.
        This is a nonparametric metric and assesses how well the relationship
        between the true and predicted data can be described using a monotonic
        function.
        https://hess.copernicus.org/articles/24/2505/2020/hess-24-2505-2020.pdf
        """
        col = [list(a) for a in zip(self.true, self.predicted)]
        xy = sorted(col, key=lambda _x: _x[0], reverse=False)
        # rang of x-value
        for i, row in enumerate(xy):
            row.append(i + 1)

        a = sorted(xy, key=lambda _x: _x[1], reverse=False)
        # rang of y-value
        for i, row in enumerate(a):
            row.append(i + 1)

        mw_rank_x = np.nanmean(np.array(a)[:, 2])
        mw_rank_y = np.nanmean(np.array(a)[:, 3])

        numerator = np.nansum([float((a[j][2] - mw_rank_x) * (a[j][3] - mw_rank_y)) for j in range(len(a))])
        denominator1 = np.sqrt(np.nansum([(a[j][2] - mw_rank_x) ** 2. for j in range(len(a))]))
        denominator2 = np.sqrt(np.nansum([(a[j][3] - mw_rank_x) ** 2. for j in range(len(a))]))
        return float(numerator / (denominator1 * denominator2))

    def sse(self) -> float:
        """Sum of squared errors (model vs actual).
        measure of how far off our model’s predictions are from the observed values. A value of 0 indicates that all
         predications are spot on. A non-zero value indicates errors.
        https://dziganto.github.io/data%20science/linear%20regression/machine%20learning/python/Linear-Regression-101-Metrics/
        This is also called residual sum of squares (RSS) or sum of squared residuals as per
        https://www.tutorialspoint.com/statistics/residual_sum_of_squares.htm
        """
        squared_errors = (self.true - self.predicted) ** 2
        return float(np.sum(squared_errors))

    def std_ratio(self, **kwargs)->float:
        """ratio of standard deviations of predictions and trues.
        Also known as standard ratio, it varies from 0.0 to infinity while
        1.0 being the perfect value.
        """
        return float(np.std(self.predicted, **kwargs) / np.std(self.true, **kwargs))

    def umbrae(self, benchmark: np.ndarray = None):
        """ Unscaled Mean Bounded Relative Absolute Error """
        return self.mbrae(benchmark) / (1 - self.mbrae(benchmark))

    def ve(self) -> float:
        """
        Volumetric efficiency. from 0 to 1. Smaller the better.
        Reference: Criss and Winston 2008.
        """
        a = np.sum(np.abs(self.predicted - self.true))
        b = np.sum(self.true)
        return float(1 - (a / b))

    def volume_error(self) -> float:
        """
        Returns the Volume Error (Ve).
        It is an indicator of the agreement between the averages of the simulated
        and observed runoff (i.e. long-term water balance).
        used in this paper:
        Reynolds, J.E., S. Halldin, C.Y. Xu, J. Seibert, and A. Kauffeldt. 2017.
        "Sub-Daily Runoff Predictions Using Parameters Calibrated on the Basis of Data with a
        Daily Temporal Resolution."  Journal of Hydrology 550 (July):399?411.
        https://doi.org/10.1016/j.jhydrol.2017.05.012.
            .. math::
            Sum(self.predicted- true)/sum(self.predicted)
        """
        # TODO written formula and executed formula are different.
        ve = np.sum(self.predicted - self.true) / np.sum(self.true)
        return float(ve)

    def wape(self) -> float:
        """
        weighted absolute percentage error
        https://mattdyor.wordpress.com/2018/05/23/calculating-wape/
        """
        return float(np.sum(self._ae() / np.sum(self.true)))

    def watt_m(self) -> float:
        """Watterson's M.
        Refrence: Watterson., 1996"""
        a = 2 / np.pi
        c = np.std(self.true, ddof=1) ** 2 + np.std(self.predicted, ddof=1) ** 2
        e = (np.mean(self.predicted) - np.mean(self.true)) ** 2
        f = c + e
        return float(a * np.arcsin(1 - (self.mse() / f)))

    def wmape(self) -> float:
        """
         Weighted Mean Absolute Percent Error
         https://stackoverflow.com/a/54833202/5982232
        """
        # Take a series (actual) and a dataframe (forecast) and calculate wmape
        # for each forecast. Output shape is (1, num_forecasts)

        # Make an array of mape (same shape as forecast)
        se_mape = abs(self.true - self.predicted) / self.true

        # Calculate sum of actual values
        ft_actual_sum = self.true.sum(axis=0)

        # Multiply the actual values by the mape
        se_actual_prod_mape = self.true * se_mape

        # Take the sum of the product of actual values and mape
        # Make sure to sum down the rows (1 for each column)
        ft_actual_prod_mape_sum = se_actual_prod_mape.sum(axis=0)

        # Calculate the wmape for each forecast and return as a dictionary
        ft_wmape_forecast = ft_actual_prod_mape_sum / ft_actual_sum
        return float(ft_wmape_forecast)


class ClassificationMetrics(Metrics):
    """Calculates classification metrics."""

    def __init__(self, *args, categorical=False, **kwargs):
        self.categorical = categorical
        super().__init__(*args, metric_type='classification', **kwargs)
        self.true_labels = self._true_labels()
        self.true_logits = self._true_logits()
        self.pred_labels = self._pred_labels()
        self.pred_logits = self._pred_logits()

        all_methods = list_subclass_methods(ClassificationMetrics, True)
        self.all_methods=  [m for m in all_methods if not m.startswith('_')]

    def _num_classes(self):
        return len(self._classes())

    def _classes(self):
        array = self.true_labels
        return np.unique(array[~np.isnan(array)])

    def _true_labels(self):
        """retuned array is 1d"""
        if self.categorical:
            return np.argmax(self.true, axis=1)
        assert self.true.ndim == 1
        return self.true

    def _true_logits(self):
        """returned array is 2d"""
        if self.categorical:
            return self.true
        lb = preprocessing.LabelBinarizer()
        return lb.fit_transform(self.true)

    def _pred_labels(self):
        """returns 1d"""
        if self.categorical:
            return np.argmax(self.predicted, axis=1)
        lb = preprocessing.LabelBinarizer()
        lb.fit(self.true_labels)
        return lb.inverse_transform(self.predicted)

    def _pred_logits(self):
        """returned array is 2d"""
        if self.categorical:
            return self.true
        # we can't do it
        return None

    def cross_entropy(self, epsilon=1e-12):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions.
        Input: predictions (N, k) ndarray
               targets (N, k) ndarray
        Returns: scalar
        """
        predictions = np.clip(self.predicted, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = -np.sum(self.true * np.log(predictions + 1e-9)) / N
        return ce

    # def hinge_loss(self):
    #     """hinge loss using sklearn"""
    #     if self.pred_logits is not None:
    #         return hinge_loss(self.true_labels, self.pred_logits)
    #     return None

    def balanced_accuracy_score(self):
        return balanced_accuracy_score(self.true_labels, self.pred_labels)

    def accuracy(self):
        return accuracy_score(self.true_labels, self.pred_labels)

