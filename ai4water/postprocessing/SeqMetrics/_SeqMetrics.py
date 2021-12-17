import json
import warnings
import numpy as np
from typing import Union

from ai4water.utils.utils import ts_features


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
# root mean quartic error, Kolmogorov–Smirnov test integral, OVERPer, Rényi entropy,
# 95th percentile: https://doi.org/10.1016/j.solener.2014.10.016
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

    @staticmethod
    def _minimal() -> list:
        raise NotImplementedError

    @staticmethod
    def _scale_independent_metrics() -> list:
        raise NotImplementedError

    @staticmethod
    def _scale_dependent_metrics() -> list:
        raise NotImplementedError

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
            if np.ndim(array_like) > 1:
                sec_dim = array_like.shape[1]
                if self.metric_type != 'classification' and sec_dim > 1:
                    raise ValueError(f"Array must not be 2d but it has shape {array_like.shape}")
                np_array = np.array(array_like).reshape(-1, ) if self.metric_type != 'classification' else array_like
            else:
                # maybe the dimension is >1 so make sure it is more
                np_array = array_like.reshape(-1, ) if self.metric_type != 'classification' else array_like

        if self.metric_type != 'classification':
            assert len(np_array.shape) == 1
        return np_array

    def calculate_all(self, statistics=False, verbose=False, write=False, name=None) -> dict:
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

    def calculate_minimal(self) -> dict:
        """
        Calculates some basic metrics.

        Returns
        -------
        dict
            Dictionary with all metrics
        """
        metrics = {}

        for metric in self._minimal():
            metrics[metric] = getattr(self, metric)()

        return metrics

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

    def calculate_scale_independent_metrics(self) -> dict:
        """
        Calculates scale independent metrics

        Returns
        -------
        dict
            Dictionary with all metrics
        """
        metrics = {}

        for metric in self._scale_independent_metrics():
            metrics[metric] = getattr(self, metric)()

        return metrics

    def calculate_scale_dependent_metrics(self) -> dict:
        """
        Calculates scale dependent metrics

        Returns
        -------
        dict
            Dictionary with all metrics
        """
        metrics = {}

        for metric in self._scale_dependent_metrics():
            metrics[metric] = getattr(self, metric)()

        return metrics

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
