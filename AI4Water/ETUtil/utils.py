import re
import math
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from dl4seq.ETUtil.converter import Temp, Speed, Pressure
from dl4seq.ETUtil.global_variables import ALLOWED_COLUMNS, SOLAR_CONSTANT, LAMBDA, Colors
from dl4seq.ETUtil.global_variables import default_constants, SB_CONS


class AttributeChecker:
    def __init__(self, input_df):
        self.input = self.check_in_df(input_df)
        self.output = {}
        self.allowed_columns = ALLOWED_COLUMNS
        self.no_of_hours = None

    def check_in_df(self, data_frame) -> pd.DataFrame:
        if not isinstance(data_frame, pd.DataFrame):
            raise TypeError("input must be a pandas dataframe")

        for col in data_frame.columns:
            if col not in ALLOWED_COLUMNS:
                raise ValueError("""col {} given in input dataframe is not allowed. Allowed columns names are {}"""
                                 .format(col, ALLOWED_COLUMNS))

        if not isinstance(data_frame.index, pd.DatetimeIndex):
            index = pd.to_datetime(data_frame.index)
            if not isinstance(index, pd.DatetimeIndex):
                raise TypeError("index of input dataframe must be convertible to pd.DatetimeIndex")

        if data_frame.shape[0] > 1:
            data_frame.index.freq = pd.infer_freq(data_frame.index)
        else:
            setattr(self, 'single_vale', True)

        setattr(self, 'in_freq', data_frame.index.freqstr)

        return data_frame


class PlotData(AttributeChecker):
    """
    Methods:
          plot_inputs
          plot_outputs
    """
    def __init__(self, input_df, units):

        super(PlotData, self).__init__(input_df)
        self.units = units

    def plot_inputs(self, _name=False):

        no_of_plots = len(self.input.columns)

        plt.close('all')
        fig, axis = plt.subplots(no_of_plots, sharex='all')
        fig.set_figheight(no_of_plots+2)
        fig.set_figwidth(10.48)

        idx = 0
        for ax, col in zip(axis, self.input.columns):

            show_xaxis = False

            if idx > no_of_plots-2:
                show_xaxis = True

            if col in self.units:
                yl = self.units[col]
            else:
                yl = ' '

            data = self.input[col]
            process_axis(ax, data, label=col, show_xaxis=show_xaxis, show_leg=True, y_label=yl, leg_ms=8, max_xdates=4)

            idx += 1

        plt.subplots_adjust(wspace=0.001, hspace=0.001)
        if _name:
            plt.savefig(_name, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_outputs(self, name='', _name=False):

        def marker_scale(_col):
            if 'Monthly' in _col:
                return 4
            elif 'Yearly' in _col:
                return 10
            else:
                return 0.5

        to_plot = []

        for key in self.output.keys():
            if name in key:
                to_plot.append(key)

        no_of_plots = len(to_plot)

        plt.close('all')
        fig, axis = plt.subplots(no_of_plots, sharex='all')

        if no_of_plots==1:
            axis = [axis]

        fig.set_figheight(no_of_plots+4)
        fig.set_figwidth(10.48)

        idx = 0
        for ax, col in zip(axis, self.output.keys()):

            show_xaxis = False
            if idx > no_of_plots-2:
                show_xaxis = True

            data = self.output[col]
            process_axis(ax, data, ms=marker_scale(col), label=col, show_xaxis=show_xaxis, show_leg=True, y_label='mm',
                         leg_ms=8, max_xdates=4)
            idx += 1

        plt.subplots_adjust(wspace=0.001, hspace=0.001)
        if _name:
            plt.savefig(_name, dpi=300, bbox_inches='tight')
        plt.show()


class PreProcessing(PlotData):
    """
    Attributes
        freq_str: str
        daily_index: pd.DatetimeIndex
        freq_in_mins: int
    """
    def __init__(self, input_df, units, constants, calculate_at='same', verbosity=1):

        super(PreProcessing, self).__init__(input_df, units)

        self.units = units
        self.default_cons = default_constants
        self.cons = constants
        self.freq_in_mins = calculate_at
        self.sb_cons = self.freq_in_mins
        self.lat_rad = self.cons
        self._check_compatability()
        self.verbosity = verbosity

    @property
    def seconds(self):
        """finds number of seconds between two steps of input data"""
        if len(self.input) > 1:
            return (self.input.index[1]-self.input.index[0])/np.timedelta64(1, 's')

    @property
    def sb_cons(self):
        return self._sb_cons

    @sb_cons.setter
    def sb_cons(self, freq_in_mins):
        self._sb_cons = freq_in_mins * SB_CONS

    @property
    def lat_rad(self):
        return self._lat_rad

    @lat_rad.setter
    def lat_rad(self, constants):
        if 'lat_rad' in constants:
            self._lat_rad = constants['lat_rad']
        elif 'lat_dec_deg' in constants:
            self._lat_rad = constants['lat_dec_deg'] * 0.0174533  # # degree to radians
        else:
            raise ConnectionResetError("Provide latitude information in as lat_rat or as lat_dec_deg in constants")

    @property
    def freq_in_mins(self):
        return self._freq_in_mins

    @freq_in_mins.setter
    def freq_in_mins(self, calculate_at):

        if calculate_at is not None and calculate_at != 'same':
            if isinstance(calculate_at, str):
                in_minutes = freq_in_mins_from_string(calculate_at)
            else:
                raise TypeError("invalid type of frequency demanded", calculate_at)

        else:
            in_minutes = freq_in_mins_from_string(self.input.index.freqstr)

        self._freq_in_mins = in_minutes

    @property
    def freq_str(self) -> str:

        minutes = self.freq_in_mins
        freq_str = min_to_str(minutes)

        return freq_str

    def daily_index(self) -> pd.DatetimeIndex:
        start_year = justify_len(str(self.input.index[0].year))
        end_year = justify_len(str(self.input.index[-1].year))
        start_month = justify_len(str(self.input.index[0].month))
        end_month = justify_len(str(self.input.index[0].month))
        start_day = justify_len(str(self.input.index[0].day))
        end_day = justify_len(str(self.input.index[0].day))

        st = start_year + start_month + start_day
        en = end_year + end_month + end_day

        return pd.date_range(st, en, freq='D')

    def _check_compatability(self):

        self._preprocess_temp()

        self._preprocess_rh()

        self._check_wind_units()

        self._cehck_pressure_units()

        self._check_rad_units()

        # getting julian day
        self.input['jday'] = self.input.index.dayofyear

        if self.freq_in_mins == 60:
            a = self.input.index.hour
            ma = np.convolve(a, np.ones((2,)) / 2, mode='same')
            ma[0] = ma[1] - (ma[2] - ma[1])
            self.input['half_hr'] = ma
            freq = self.input.index.freqstr
            if len(freq) > 1:
                setattr(self, 'no_of_hours', int(freq[0]))
            else:
                setattr(self, 'no_of_hours', 1)

            self.input['t1'] = np.zeros(len(self.input)) + self.no_of_hours

        elif self.freq_in_mins < 60:
            a = self.input.index.hour
            b = (self.input.index.minute + self.freq_in_mins / 2.0) / 60.0
            self.input['half_hr'] = a + b

            self.input['t1'] = np.zeros(len(self.input)) + self.freq_in_mins / 60.0

        for val in ['sol_rad', 'rn']:
            if val in self.input:
                if self.freq_in_mins <= 60:
                    self.input['is_day'] = np.where(self.input[val].values > 0.1, 1, 0)

        return

    def _preprocess_rh(self):
        # make sure that we mean relative humidity calculated if possible
        if 'rel_hum' in self.input.columns:
            rel_hum = self.input['rel_hum']
            rel_hum = np.where(rel_hum < 0.0, 0.0, rel_hum)
            rel_hum = np.where(rel_hum >= 100.0, 100.0, rel_hum)
            self.input['rh_mean'] = rel_hum
            self.input['rel_hum'] = rel_hum
        else:
            if 'rh_min' in self.input.columns:
                self.input['rh_mean'] = np.mean(np.array([self.input['rh_min'].values, self.input['rh_max'].values]),
                                                axis=0)
        return

    def _preprocess_temp(self):
        """ converts temperature related input to units of Centigrade if required. """
        # converting temperature units to celsius
        for val in ['tmin', 'tmax', 'temp', 'tdew']:
            if val in self.input:
                t = Temp(self.input[val].values, self.units[val])
                temp = t.Centigrade
                self.input[val] = np.where(temp < -30, -30, temp)

        # if 'temp' is given, it is assumed to be mean otherwise calculate mean and put it as `temp` in input dataframe.
        if 'temp' not in self.input.columns:
            if 'tmin' in self.input.columns and 'tmax' in self.input.columns:
                self.input['temp'] = np.mean(np.array([self.input['tmin'].values, self.input['tmax'].values]), axis=0)
        return

    def _check_wind_units(self):
        # check units of wind speed and convert if needed
        if 'wind_speed' in self.input:
            wind = self.input['wind_speed'].values
            wind = np.where(wind < 0.0, 0.0, wind)
            w = Speed(wind, self.units['wind_speed'])
            self.input['wind_speed'] = w.MeterPerSecond
        return

    def _cehck_pressure_units(self):
        """ converts pressure related input to units of KiloPascal if required. """
        for pres in ['ea', 'es', 'vp_def']:
            if pres in self.input:
                p = Pressure(self.input[pres].values, self.units[pres])
                self.input[pres] = p.KiloPascal

    def _check_rad_units(self):
        """
        Currently it does not converts radiation units, only makes sure that they are > 0.0.
        """
        for val in ['rn', 'sol_rad']:
            if val in self.input:
                rad = self.input[val].values
                rad = np.where(rad < 0.0, 0.0, rad)
                self.input[val] = rad


class TransFormData(PreProcessing):
    """
     transforms input or output data to different frequencies.
    """
    def __init__(self, input_df, units, constants, calculate_at='same', verbosity=1):
        self.verbosity = verbosity
        input_df = self.freq_check(input_df, calculate_at)
        input_df = self.transform_data(input_df, calculate_at)
        super(TransFormData, self).__init__(input_df, units, constants, calculate_at, verbosity)

    def freq_check(self, input_df, freq: str):
        """
        Makes sure that the input dataframe.index as frequency. It frequency is not there, it means it contains
        missing data. In this case this method fills missing values. In such case, the argument freq must not be `same`.
        """
        if input_df.shape[0] > 1:
            input_df.index.freq = pd.infer_freq(input_df.index)

        if input_df.index.freq is None:
            if freq == 'same' or freq is None:
                raise ValueError("input data does not have uniform time-step. Provide a value for argument"
                                 " `calculate_at` ")
            else:
                new_freq = freq_in_mins_from_string(freq)
                try:
                    input_df.index.freq = freq
                except ValueError:
                    input_df = self.fill_missing_data(input_df, str(new_freq) + 'min')
        return input_df

    def fill_missing_data(self, df: pd.DataFrame, new_freq: str):
        if self.verbosity > 0:
            print("input contains missing values or time-steps")

        df = force_freq(df.copy(), new_freq, 'input', 'nearest')
        assert df.index.freqstr is not None
        return df

    def transform_data(self, input_df, calculate_at):
        if calculate_at == 'same' or calculate_at is None:
            df = input_df
        else:
            new_freq_mins = freq_in_mins_from_string(calculate_at)
            old_freq_mins = freq_in_mins_from_string(input_df.index.freqstr)
            if new_freq_mins == old_freq_mins:
                df = input_df
            elif new_freq_mins > old_freq_mins:
                # we want to calculate at higher/larger time-step
                print('downsampling input data from {} to {}'.format(old_freq_mins, new_freq_mins))
                df = self.downsample_input(input_df, new_freq_mins)
            else:
                print('upsampling input data from {} to {}'.format(old_freq_mins, new_freq_mins))
                # we want to calculate at smaller time-step
                df = self.upsample_input(input_df, new_freq_mins)

        return df

    def upsample_input(self, df,  out_freq):
        # from larger timestep to smaller timestep, such as from daily to hourly
        for col in df.columns:
            df[col] = self.upsample_df(pd.DataFrame(df[col]), col, out_freq)
        return df

    def downsample_input(self, df,  out_freq):
        # from low timestep to high timestep i.e from 1 hour to 24 hour
        # from hourly to daily
        for col in df.columns:
            df[col] = self.downsample_df(pd.DataFrame(df[col]), col, out_freq)
        return df

    def transform_etp(self, name):
        freq_to_trans = self.get_freq()
        down_sample = freq_to_trans['up_sample']
        up_sample = freq_to_trans['down_sample']

        for freq in up_sample:
            in_col_name = 'et_' + name + '_' + self.freq_str
            freq_str = min_to_str(freq)
            out_col_name = 'et_' + name + '_' + freq_str
            self.output[out_col_name] = self.upsample_df(pd.DataFrame(self.output[in_col_name]), 'et', freq)

        for freq in down_sample:
            in_col_name = 'et_' + name + '_' + self.freq_str
            freq_str = min_to_str(freq)
            out_col_name = 'et_' + name + '_' + freq_str
            self.output[out_col_name] = self.downsample_df(pd.DataFrame(self.output[in_col_name]), 'et', freq)

    def downsample_df(self, data_frame: pd.DataFrame, data_name: str, out_freq: int):
        # from low timestep to high timestep i.e from 1 hour to 24 hour
        # from hourly to daily
        col_name = data_frame.columns[0]

        data_frame = data_frame.copy()
        old_freq = data_frame.index.freq
        if self.verbosity > 1:
            print('downsampling {} data from {} to {}'.format(col_name, old_freq, min_to_str(out_freq)))
        out_freq = str(out_freq) + 'min'
        # e.g. from hourly to daily
        if data_name in ['temp', 'rel_hum', 'rh_min', 'rh_max', 'uz', 'u2', 'wind_speed_kph', 'q_lps']:
            return data_frame.resample(out_freq).mean()
        elif data_name in ['rain_mm', 'ss_gpl', 'sol_rad', 'etp', 'et']:
            return data_frame.resample(out_freq).sum()

    def upsample_df(self, data_frame, data_name, out_freq_int):
        # from larger timestep to smaller timestep, such as from daily to hourly
        out_freq = str(out_freq_int) + 'min'
        col_name = data_frame.columns[0]

        old_freq = data_frame.index.freqstr
        nan_idx = data_frame.isna()  # preserving indices with nan values

        nan_idx_r = nan_idx.resample(out_freq).ffill()
        nan_idx_r = nan_idx_r.fillna(False)  # the first value was being filled with NaN, idk y?
        data_frame = data_frame.copy()

        if self.verbosity > 1:
            print('upsampling {} data from {} to {}'.format(data_name, old_freq, min_to_str(out_freq_int)))
        # e.g from monthly to daily or from hourly to sub_hourly
        if data_name in ['temp', 'rel_hum', 'rh_min', 'rh_max', 'uz', 'u2', 'q_lps']:
            data_frame = data_frame.resample(out_freq).interpolate(method='linear')
            # filling those interpolated values with NaNs which were NaN before interpolation
            data_frame[nan_idx_r] = np.nan

        elif data_name in ['rain_mm', 'ss_gpl', 'sol_rad', 'pet', 'pet_hr', 'et', 'etp']:
            # distribute rainfall equally to smaller time steps. like hourly 17.4 will be 1.74 at 6 min resolution
            idx = data_frame.index[-1] + get_offset(data_frame.index.freqstr)
            data_frame = data_frame.append(data_frame.iloc[[-1]].rename({data_frame.index[-1]: idx}))
            data_frame = add_freq(data_frame)
            df1 = data_frame.resample(out_freq).ffill().iloc[:-1]
            df1[col_name] /= df1.resample(data_frame.index.freqstr)[col_name].transform('size')
            data_frame = df1.copy()
            # filling those interpolated values with NaNs which were NaN before interpolation
            data_frame[nan_idx_r] = np.nan

        return data_frame

    def get_freq(self) -> dict:
        """ decides which frequencies to """
        all_freqs = {'Sub_hourly': {'down_sample': [1], 'up_sample': [60, 1440, 43200, 525600]},
                     'Hourly': {'down_sample': [1], 'up_sample': [1440, 43200, 525600]},
                     'Sub_daily': {'down_sample': [1, 60], 'up_sample': [1440, 43200, 525600]},
                     'Daily': {'down_sample': [1, 60], 'up_sample': [43200, 525600]},
                     'Sub_monthly': {'down_sample': [1, 60, 1440], 'up_sample': [43200, 525600]},
                     'Monthly': {'down_sample': [1, 60, 1440], 'up_sample': [525600]},
                     'Annualy': {'down_sample': [1, 60, 1440, 43200], 'up_sample': []}
                     }
        return all_freqs[self.freq_str]


class Utils(TransFormData):
    """
    Contains functions methods for calculation of ETP with various methods.
    Methods:
        net_rad
        atm_pressure
        _wind_2m
    """

    def __init__(self, input_df, units, constants, calculate_at=None, verbosity=1):

        super(Utils, self).__init__(input_df, units, constants, calculate_at=calculate_at, verbosity=verbosity)

    @property
    def seasonal_correction(self):
        """Seasonal correction for solar time (Eqs. 57 & 58)

        uses
        ----------
        doy : scalar or array_like of shape(M, )
            Day of year.

        Returns
        ------
        ndarray
            Seasonal correction [hour]

        """
        doy = self.input['jday']
        b = 2 * math.pi * (doy - 81.) / 364.
        return 0.1645 * np.sin(2 * b) - 0.1255 * np.cos(b) - 0.0250 * np.sin(b)

    def net_rad(self, ea, rs=None):
        """
        Calculate daily net radiation at the crop surface, assuming a grass reference crop.

        Net radiation is the difference between the incoming net shortwave (or solar) radiation and the outgoing net
        longwave radiation. Output can be converted to equivalent evaporation [mm day-1] using ``energy2evap()``.

        Based on equation 40 in Allen et al (1998).

        :uses rns: Net incoming shortwave radiation [MJ m-2 day-1]. Can be
                   estimated using ``net_in_sol_rad()``.
              rnl: Net outgoing longwave radiation [MJ m-2 day-1]. Can be
                   estimated using ``net_out_lw_rad()``.
        :return: net radiation [MJ m-2 timestep-1].
        :rtype: float
        """
        if 'rn' not in self.input:
            if rs is None:
                rs = self.rs()
            if 'rns' not in self.input:
                rns = self.net_in_sol_rad(rs)
            else:
                rns = self.input['rns']
            rnl = self.net_out_lw_rad(rs=rs, ea=ea)
            rn = np.subtract(rns, rnl)
            self.input['rn'] = rn  # for future use
        else:
            rn = self.input['rn']
        return rn

    def rs(self):
        """
        calculate solar radiation either from temperature (as second preference, as it is les accurate) or from daily
        _sunshine hours as second preference). Sunshine hours is given second preference because sunshine hours will
        remain same for all years if sunshine hours data is not provided (which is difficult to obtain), but temperature
        data  which is easy to obtain and thus will be different for different years"""

        if 'sol_rad' not in self.input.columns:
            if 'sunshine_hrs' in self.input.columns:
                rs = self.sol_rad_from_sun_hours()
                if self.verbosity > 0:
                    print("Sunshine hour data is used for calculating incoming solar radiation")
            elif 'tmin' in self.input.columns and 'tmax' in self.input.columns:
                rs = self._sol_rad_from_t()
                if self.verbosity > 0:
                    print("solar radiation is calculated from temperature")
            else:
                raise ValueError("""Unable to calculate solar radiation. Provide either of following inputs:
                                 sol_rad, sunshine_hrs or tmin and tmax""")
        else:
            rs = self.input['sol_rad']

        self.input['sol_rad'] = rs
        return rs

    def net_in_sol_rad(self, rs):
        """
        Calculate net incoming solar (or shortwave) radiation (*Rns*) from gross incoming solar radiation, assuming a
         grass reference crop.

        Net incoming solar radiation is the net shortwave radiation resulting from the balance between incoming and
         reflected solar radiation. The output can be converted to equivalent evaporation [mm day-1] using
        ``energy2evap()``.

        Based on FAO equation 38 in Allen et al (1998).
        Rns = (1-a)Rs

        uses Gross incoming solar radiation [MJ m-2 day-1]. If necessary this can be estimated using functions whose
            name begins with 'solar_rad_from'.
        :param rs: solar radiation
        albedo: Albedo of the crop as the proportion of gross incoming solar
            radiation that is reflected by the surface. Default value is 0.23,
            which is the value used by the FAO for a short grass reference crop.
            Albedo can be as high as 0.95 for freshly fallen snow and as low as
            0.05 for wet bare soil. A green vegetation over has an albedo of
            about 0.20-0.25 (Allen et al, 1998).
        :return: Net incoming solar (or shortwave) radiation [MJ m-2 day-1].
        :rtype: float
        """
        return np.multiply((1 - self.cons['albedo']), rs)

    def net_out_lw_rad(self, rs, ea):
        """
        Estimate net outgoing longwave radiation.

        This is the net longwave energy (net energy flux) leaving the earth's surface. It is proportional to the
        absolute temperature of the surface raised to the fourth power according to the Stefan-Boltzmann law. However,
        water vapour, clouds, carbon dioxide and dust are absorbers and emitters of longwave radiation. This function
        corrects the Stefan- Boltzmann law for humidity (using actual vapor pressure) and cloudiness (using solar
        radiation and clear sky radiation). The concentrations of all other absorbers are assumed to be constant.

        The output can be converted to equivalent evaporation [mm timestep-1] using  ``energy2evap()``.

        Based on FAO equation 39 in Allen et al (1998).

        uses: Absolute daily minimum temperature [degrees Kelvin]
              Absolute daily maximum temperature [degrees Kelvin]
              Solar radiation [MJ m-2 day-1]. If necessary this can be estimated using ``sol+rad()``.
              Clear sky radiation [MJ m-2 day-1]. Can be estimated using  ``cs_rad()``.
              Actual vapour pressure [kPa]. Can be estimated using functions with names beginning with 'avp_from'.
        :param ea: actual vapour pressure, can be calculated using method avp_from
        :param rs: solar radiation
        :return: Net outgoing longwave radiation [MJ m-2 timestep-1]
        :rtype: float
        """
        if 'tmin' in self.input.columns and 'tmax' in self.input.columns:
            added = np.add(np.power(self.input['tmax'].values+273.16, 4), np.power(self.input['tmin'].values+273.16, 4))
            divided = np.divide(added, 2.0)
        else:
            divided = np.power(self.input['temp'].values+273.16, 4.0)

        tmp1 = np.multiply(self.sb_cons, divided)
        tmp2 = np.subtract(0.34, np.multiply(0.14, np.sqrt(ea)))
        tmp3 = np.subtract(np.multiply(1.35, np.divide(rs, self._cs_rad())), 0.35)
        return np.multiply(tmp1, np.multiply(tmp2, tmp3))  # eq 39

    def sol_rad_from_sun_hours(self):
        """
        Calculate incoming solar (or shortwave) radiation, *Rs* (radiation hitting a horizontal plane after
        scattering by the atmosphere) from relative sunshine duration.

        If measured radiation data are not available this method is preferable to calculating solar radiation from
        temperature. If a monthly mean is required then divide the monthly number of sunshine hours by number of
        days in the month and ensure that *et_rad* and *daylight_hours* was calculated using the day of the year
        that corresponds to the middle of the month.

        Based on equations 34 and 35 in Allen et al (1998).

        uses: Number of daylight hours [hours]. Can be calculated  using ``daylight_hours()``.
              Sunshine duration [hours]. Can be calculated  using ``sunshine_hours()``.
              Extraterrestrial radiation [MJ m-2 day-1]. Can be estimated using ``et_rad()``.
        :return: Incoming solar (or shortwave) radiation [MJ m-2 day-1]
        :rtype: float
        """

        # 0.5 and 0.25 are default values of regression constants (Angstrom values)
        # recommended by FAO when calibrated values are unavailable.
        ss_hrs = self.input['sunshine_hrs']  # sunshine_hours
        dl_hrs = self.daylight_fao56()       # daylight_hours
        return np.multiply(np.add(self.cons['a_s'], np.multiply(np.divide(ss_hrs, dl_hrs), self.cons['b_s'])),
                           self._et_rad())

    def _sol_rad_from_t(self, coastal=False):
        """
        Estimate incoming solar (or shortwave) radiation  [Mj m-2 day-1] , *Rs*, (radiation hitting  a horizontal
           plane after scattering by the atmosphere) from min and max temperature together with an empirical adjustment
           coefficient for 'interior' and 'coastal' regions.

        The formula is based on equation 50 in Allen et al (1998) which is the Hargreaves radiation formula (Hargreaves
        and Samani, 1982, 1985). This method should be used only when solar radiation or sunshine hours data are not
        available. It is only recommended for locations where it is not possible to use radiation data from a regional
        station (either because climate conditions are heterogeneous or data are lacking).

        **NOTE**: this method is not suitable for island locations due to the
        moderating effects of the surrounding water. """

        # Determine value of adjustment coefficient [deg C-0.5] for
        # coastal/interior locations
        if coastal:     # for 'coastal' locations, situated on or adjacent to the coast of a large l
            adj = 0.19  # and mass and where air masses are influenced by a nearby water body,
        else:           # for 'interior' locations, where land mass dominates and air
            adj = 0.16  # masses are not strongly influenced by a large water body

        et_rad = None
        cs_rad = None
        if 'et_rad' not in self.input:
            et_rad = self._et_rad()
            self.input['et_rad'] = et_rad
        if 'cs_rad' not in self.input:
            cs_rad = self._cs_rad()
            self.input['cs_rad'] = cs_rad
        sol_rad = np.multiply(adj, np.multiply(np.sqrt(np.subtract(self.input['tmax'].values,
                                                                   self.input['tmin'].values)), et_rad))

        # The solar radiation value is constrained by the clear sky radiation
        return np.min(np.array([sol_rad, cs_rad]), axis=0)

    def _cs_rad(self, method='asce'):
        """
        Estimate clear sky radiation from altitude and extraterrestrial radiation.

        Based on equation 37 in Allen et al (1998) which is recommended when calibrated Angstrom values are not
        available. et_rad is Extraterrestrial radiation [MJ m-2 day-1]. Can be estimated using ``et_rad()``.

        :return: Clear sky radiation [MJ m-2 day-1]
        :rtype: float
        """
        if method.upper() == 'ASCE':
            return (0.00002 * self.cons['altitude'] + 0.75) * self._et_rad()
        elif method.upper() == 'REFET':
            sc = self.seasonal_correction()
            _omega = omega(solar_time_rad(self.cons['long_dec_deg'], self.input['half_hour'], sc))
        else:
            raise ValueError

    def daylight_fao56(self):
        """get number of maximum hours of sunlight for a given latitude using equation 34 in Fao56.
        Annual variation of sunlight hours on earth are plotted in figre 14 in ref 1.

        dr = pd.date_range('20110903 00:00', '20110903 23:59', freq='H')
        sol_rad = np.array([0.45 for _ in range(len(dr))])
        df = pd.DataFrame(np.stack([sol_rad],axis=1), columns=['sol_rad'], index=dr)
        constants = {'lat' : -20}
        units={'solar_rad': 'MegaJoulePerMeterSquarePerHour'}
        eto = ReferenceET(df,units,constants=constants)
        N = np.unique(eto.daylight_fao56())
          array([11.66])

        1) http://www.fao.org/3/X0490E/x0490e07.htm"""
        ws = self.sunset_angle()
        hrs = (24/3.14) * ws
        # if self.input_freq == 'Monthly':
        #     df = pd.DataFrame(hrs, index=self.daily_index)
        #     hrs = df.resample('M').mean().values.reshape(-1,)
        return hrs

    def _et_rad(self):
        """
        Estimate extraterrestrial radiation (*Ra*, 'top of the atmosphere radiation').

        For daily, it is based on equation 21 in Allen et al (1998). If monthly mean radiation is required make
         sure *sol_dec*. *sha* and *irl* have been calculated using the day of the year that corresponds to the middle
         of the month.

        **Note**: From Allen et al (1998): "For the winter months in latitudes greater than 55 degrees (N or S),
          the equations have limited validity. Reference should be made to the Smithsonian Tables to assess possible
          deviations."

        :return: extraterrestrial radiation [MJ m-2 timestep-1]
        :rtype: float

        dr = pd.date_range('20110903 00:00', '20110903 23:59', freq='D')
        sol_rad = np.array([0.45 ])
        df = pd.DataFrame(np.stack([sol_rad],axis=1), columns=['sol_rad'], index=dr)
        constants = {'lat' : -20}
        units={'sol_rad': 'MegaJoulePerMeterSquarePerHour'}
        eto = ReferenceET(df,units,constants=constants)
        ra = eto._et_rad()
        [32.27]
        """
        if self.freq_in_mins < 1440:  # TODO should sub_hourly be different from Hourly?
            j = (3.14/180) * self.cons['lat_dec_deg']  # eq 22  phi
            dr = self.inv_rel_dist_earth_sun()  # eq 23
            sol_dec = self.dec_angle()  # eq 24    # gamma
            w1, w2 = self.solar_time_angle()
            t1 = (12*60)/math.pi
            t2 = np.multiply(t1, np.multiply(SOLAR_CONSTANT, dr))
            t3 = np.multiply(np.subtract(w2, w1), np.multiply(np.sin(j), np.sin(sol_dec)))
            t4 = np.subtract(np.sin(w2), np.sin(w1))
            t5 = np.multiply(np.multiply(np.cos(j), np.cos(sol_dec)), t4)
            t6 = np.add(t5, t3)
            ra = np.multiply(t2, t6)   # eq 28

        elif self.freq_in_mins == 1440:  # daily frequency
            sol_dec = self.dec_angle()  # based on julian day
            sha = self.sunset_angle()   # sunset hour angle[radians], based on latitude
            ird = self.inv_rel_dist_earth_sun()
            tmp1 = (24.0 * 60.0) / math.pi
            tmp2 = np.multiply(sha, np.multiply(math.sin(self.lat_rad), np.sin(sol_dec)))
            tmp3 = np.multiply(math.cos(self.lat_rad), np.multiply(np.cos(sol_dec), np.sin(sha)))
            ra = np.multiply(tmp1, np.multiply(SOLAR_CONSTANT, np.multiply(ird, np.add(tmp2, tmp3))))  # eq 21
        else:
            raise NotImplementedError
        self.input['ra'] = ra
        return ra

    def sunset_angle(self):
        """
        calculates sunset hour angle in radians given by Equation 25  in Fao56 (1)

        1): http://www.fao.org/3/X0490E/x0490e07.htm"""
        if 'sha' not in self.input:
            j = (3.14/180.0) * self.cons['lat_dec_deg']           # eq 22
            d = self.dec_angle()       # eq 24, declination angle
            angle = np.arccos(-np.tan(j)*np.tan(d))      # eq 25
            self.input['sha'] = angle
        else:
            angle = self.input['sha'].values
        return angle

    def inv_rel_dist_earth_sun(self):
        """
        Calculate the inverse relative distance between earth and sun from day of the year.
        Based on FAO equation 23 in Allen et al (1998).
        ird = 1.0 + 0.033 * cos( [2pi/365] * j )

        :return: Inverse relative distance between earth and the sun
        :rtype: np array
        """
        if 'ird' not in self.input:
            inv1 = np.multiply(2*math.pi/365.0,  self.input['jday'].values)
            inv2 = np.cos(inv1)
            inv3 = np.multiply(0.033, inv2)
            ird = np.add(1.0, inv3)
            self.input['ird'] = ird
        else:
            ird = self.input['ird']
        return ird

    def dec_angle(self):
        """
        finds solar declination angle
        """
        if 'sol_dec' not in self.input:
            if self.freq_str == 'monthly':
                solar_dec = np.array(0.409 * np.sin(2*3.14 * self.daily_index().dayofyear/365 - 1.39))
            else:
                solar_dec = 0.409 * np.sin(2*3.14 * self.input['jday'].values/365 - 1.39)     # eq 24, declination angle
            self.input['solar_dec'] = solar_dec
        else:
            solar_dec = self.input['solar_dec']
        return solar_dec

    def solar_time_angle(self):
        """
        returns solar time angle at start, mid and end of period using equation 29, 31 and 30 respectively in Fao
        w = pi/12 [(t + 0.06667 ( lz-lm) + Sc) -12]
        t =standard clock time at the midpoint of the period [hour]. For example for a period between 14.00 and 15.00
           hours, t = 14.5
        lm = longitude of the measurement site [degrees west of Greenwich]
        lz = longitude of the centre of the local time zone [degrees west of Greenwich]

        w1 = w - pi*t1/24
        w2 = w + pi*t1/24
        where:
          w = solar time angle at midpoint of hourly or shorter period [rad]
          t1 = length of the calculation period [hour]: i.e., 1 for hourly period or 0.5 for a 30-minute period

        www.fao.org/3/X0490E/x0490e07.htm
        """

        # TODO find out how to calculate lz
        # https://github.com/djlampert/PyHSPF/blob/c3c123acf7dba62ed42336f43962a5e4db922422/src/pyhspf/preprocessing/etcalculator.py#L610
        lz = np.abs(15 * round(self.cons['long_dec_deg'] / 15.0))
        lm = np.abs(self.cons['long_dec_deg'])
        t1 = 0.0667*(lz-lm)
        t2 = self.input['half_hr'].values + t1 + self.solar_time_cor()
        t3 = np.subtract(t2, 12)
        w = np.multiply((math.pi/12.0), t3)     # eq 31, in rad

        w1 = np.subtract(w, np.divide(np.multiply(math.pi, self.input['t1']).values, 24.0))  # eq 29
        w2 = np.add(w, np.divide(np.multiply(math.pi, self.input['t1']).values, 24.0))   # eq 30
        return w1, w2

    def solar_time_cor(self):
        """seasonal correction for solar time by implementation of eqation 32 in hour, `Sc`"""
        upar = np.multiply((2*math.pi), np.subtract(self.input['jday'].values, 81))
        b = np.divide(upar, 364)   # eq 33
        t1 = np.multiply(0.1645, np.sin(np.multiply(2, b)))
        t2 = np.multiply(0.1255, np.cos(b))
        t3 = np.multiply(0.025, np.sin(b))
        return t1-t2-t3   # eq 32

    def avp_from_rel_hum(self):
        """
        Estimate actual vapour pressure (*ea*) from saturation vapour pressure and relative humidity.

        Based on FAO equation 17 in Allen et al (1998).
        ea = [ e_not(tmin)RHmax/100 + e_not(tmax)RHmin/100 ] / 2

        uses  Saturation vapour pressure at daily minimum temperature [kPa].
              Saturation vapour pressure at daily maximum temperature [kPa].
              Minimum relative humidity [%]
              Maximum relative humidity [%]
        :return: Actual vapour pressure [kPa]
        :rtype: float
        http://www.fao.org/3/X0490E/x0490e07.htm#TopOfPage
        """
        if 'ea' in self.input:
            avp = self.input['ea']
        else:
            avp = 0.0
            # TODO `shub_hourly` calculation should be different from `Hourly`
            # use equation 54 in http://www.fao.org/3/X0490E/x0490e08.htm#TopOfPage
            if self.freq_in_mins <= 60:  # for hourly or sub_hourly
                avp = np.multiply(self.sat_vp_fao56(self.input['temp'].values),
                                  np.divide(self.input['rel_hum'].values, 100.0))

            elif self.freq_in_mins == 1440:
                if 'rh_min' in self.input.columns and 'rh_max' in self.input.columns:
                    tmp1 = np.multiply(self.sat_vp_fao56(self.input['tmin'].values),
                                       np.divide(self.input['rh_max'].values, 100.0))
                    tmp2 = np.multiply(self.sat_vp_fao56(self.input['tmax'].values),
                                       np.divide(self.input['rh_min'].values, 100.0))
                    avp = np.divide(np.add(tmp1, tmp2), 2.0)
                elif 'rel_hum' in self.input.columns:
                    # calculation actual vapor pressure from mean humidity
                    # equation 19
                    t1 = np.divide(self.input['rel_hum'].values, 100)
                    t2 = np.divide(np.add(self.sat_vp_fao56(self.input['tmax'].values),
                                          self.sat_vp_fao56(self.input['tmin'].values)), 2.0)
                    avp = np.multiply(t1, t2)
            else:
                raise NotImplementedError(" for frequency of {} minutes, actual vapour pressure can not be calculated"
                                          .format(self.freq_in_mins))

            self.input['ea'] = avp
        return avp

    def sat_vp_fao56(self, temp):
        """calculates saturation vapor pressure (*e_not*) as given in eq 11 of FAO 56 at a given temp which must be in
         units of centigrade.
        using Tetens equation
        es = 0.6108 * exp((17.26*temp)/(temp+273.3))
        where es is in KiloPascal units.

        Murray, F. W., On the computation of saturation vapor pressure, J. Appl. Meteorol., 6, 203-204, 1967.
        """
        #  e_not_t = multiply(0.6108, np.exp( multiply(17.26939, temp) / add(temp , 237.3)))
        e_not_t = np.multiply(0.6108, np.exp(np.multiply(17.27, np.divide(temp, np.add(temp, 237.3)))))
        return e_not_t

    def soil_heat_flux(self, rn=None):
        if self.freq_in_mins == 1440:
            return 0.0
        elif self.freq_in_mins <= 60:
            gd = np.multiply(0.1, rn)
            gn = np.multiply(0.5, rn)
            return np.where(self.input['is_day'] == 1, gd, gn)
        elif self.freq_in_mins > 1440:
            raise NotImplementedError

    def mean_sat_vp_fao56(self):
        """ calculates mean saturation vapor pressure (*es*) for a day, weak or month according to eq 12 of FAO 56 using
        tmin and tmax which must be in centigrade units
        """
        es = None
        # for case when tmax and tmin are not given and only `temp` is given
        if 'tmax' not in self.input:
            if 'temp' in self.input:
                es = self.sat_vp_fao56(self.input['temp'])

        # for case when `tmax` and `tmin` are provided
        elif 'tmax' in self.input:
            es_tmax = self.sat_vp_fao56(self.input['tmax'].values)
            es_tmin = self.sat_vp_fao56(self.input['tmin'].values)
            es = np.mean(np.array([es_tmin, es_tmax]), axis=0)
        else:
            raise NotImplementedError
        return es

    def psy_const(self) -> float:
        """
        Calculate the psychrometric constant.

        This method assumes that the air is saturated with water vapour at the minimum daily temperature. This
        assumption may not hold in arid areas.

        Based on equation 8, page 95 in Allen et al (1998).

        uses Atmospheric pressure [kPa].
        :return: Psychrometric constant [kPa degC-1].
        :rtype: array
        """
        return np.multiply(0.000665, self.atm_pressure())

    def slope_sat_vp(self, t):
        """
        slope of the relationship between saturation vapour pressure and temperature for a given temperature
        according to equation 13 in Fao56[1].

        delta = 4098 [0.6108 exp(17.27T/T+237.3)] / (T+237.3)^2

        :param t: Air temperature [deg C]. Use mean air temperature for use in Penman-Monteith.
        :return: Saturation vapour pressure [kPa degC-1]

        [1]: http://www.fao.org/3/X0490E/x0490e07.htm#TopOfPage
        """
        to_exp = np.divide(np.multiply(17.27, t), np.add(t, 237.3))
        tmp = np.multiply(4098, np.multiply(0.6108, np.exp(to_exp)))
        return np.divide(tmp, np.power(np.add(t, 237.3), 2))

    def _wind_2m(self, method='fao56', z_o=0.001):
        """
        converts wind speed (m/s) measured at height z to 2m using either FAO 56 equation 47 or McMohan eq S4.4.
         u2 = uz [ 4.87/ln(67.8z-5.42) ]         eq 47 in [1], eq S5.20 in [2].
         u2 = uz [ln(2/z_o) / ln(z/z_o)]         eq S4.4 in [2]

        :param `method` string, either of `fao56` or `mcmohan2013`. if `mcmohan2013` is chosen then `z_o` is used
        :param `z_o` float, roughness height. Default value is from [2]

        :return: Wind speed at 2 m above the surface [m s-1]

        [1] http://www.fao.org/3/X0490E/x0490e07.htm
        [2] McMahon, T., Peel, M., Lowe, L., Srikanthan, R. & McVicar, T. 2012. Estimating actual, potential,
            reference crop and pan evaporation using standard meteorological data: a pragmatic synthesis. Hydrology and
            Earth System Sciences Discussions, 9, 11829-11910.
            https://www.hydrol-earth-syst-sci.net/17/1331/2013/hess-17-1331-2013-supplement.pdf
        """
        # if value of height at which wind is measured is not given, then don't convert
        if 'wind_z' in self.cons:
            wind_z = self.cons['wind_z']
        else:
            wind_z = None

        if wind_z is None:
            if self.verbosity > 0:
                print("""WARNING: givn wind data is not at 2 meter and `wind_z` is also not given. So assuming wind
                 given as measured at 2m height""")
            return self.input['wind_speed'].values
        else:
            if method == 'fao56':
                return np.multiply(self.input['wind_speed'], (4.87 / math.log((67.8 * wind_z) - 5.42)))
            else:
                return np.multiply(self.input['wind_speed'].values, math.log(2/z_o) / math.log(wind_z/z_o))

    def atm_pressure(self) -> float:
        """
        Estimate atmospheric pressure from altitude.

        Calculated using a simplification of the ideal gas law, assuming 20 degrees Celsius for a standard atmosphere.
         Based on equation 7, page 62 in Allen et al (1998).

        :return: atmospheric pressure [kPa]
        :rtype: float
        """
        tmp = (293.0 - (0.0065 * self.cons['altitude'])) / 293.0
        return math.pow(tmp, 5.26) * 101.3

    def tdew_from_t_rel_hum(self):
        """
        Calculates the dew point temperature given temperature and relative humidity.
        Following formulation given at https://goodcalculators.com/dew-point-calculator/
        The formula is
          Tdew = (237.3 × [ln(RH/100) + ( (17.27×T) / (237.3+T) )]) / (17.27 - [ln(RH/100) + ( (17.27×T) / (237.3+T) )])
        Where:

        Tdew = dew point temperature in degrees Celsius (°C),
        T = air temperature in degrees Celsius (°C),
        RH = relative humidity (%),
        ln = natural logarithm.
        The formula also holds true as calculations shown at http://www.decatur.de/javascript/dew/index.html
        """
        temp = self.input['temp']
        neum = (237.3 * (np.log(self.input['rel_hum'] / 100.0) + ((17.27 * temp) / (237.3 + temp))))
        denom = (17.27 - (np.log(self.input['rel_hum'] / 100.0) + ((17.27 * temp) / (237.3 + temp))))
        td = neum / denom
        self.input['tdew'] = td
        return

    def evap_pan(self):
        """
        pan evaporation which is used in almost all penman related methods
        """
        ap = self.cons['pen_ap']

        lat = self.cons['lat_dec_deg']
        rs = self.rs()
        delta = self.slope_sat_vp(self.input['temp'].values)
        gamma = self.psy_const()
        vabar = self.avp_from_rel_hum()  # Vapour pressure
        vas = self.mean_sat_vp_fao56()
        u2 = self._wind_2m()
        r_nl = self.net_out_lw_rad(rs=rs, ea=vabar)   # net outgoing longwave radiation
        ra = self._et_rad()

        # eq 34 in Thom et al., 1981
        f_pan_u = np.add(1.201, np.multiply(1.621, u2))

        # eq 4 and 5 in Rotstayn et al., 2006
        p_rad = np.add(1.32, np.add(np.multiply(4e-4, lat), np.multiply(8e-5, lat**2)))
        f_dir = np.add(-0.11, np.multiply(1.31, np.divide(rs, ra)))
        rs_pan = np.multiply(np.add(np.add(np.multiply(f_dir, p_rad), np.multiply(1.42,
                                                                                  np.subtract(1, f_dir))),
                                    np.multiply(0.42, self.cons['albedo'])), rs)
        rn_pan = np.subtract(np.multiply(1-self.cons['alphaA'], rs_pan), r_nl)

        # S6.1 in McMohan et al 2013
        tmp1 = np.multiply(np.divide(delta, np.add(delta, np.multiply(ap, gamma))), np.divide(rn_pan, LAMBDA))
        tmp2 = np.divide(np.multiply(ap, gamma), np.add(delta, np.multiply(ap, gamma)))
        tmp3 = np.multiply(f_pan_u, np.subtract(vas, vabar))
        tmp4 = np.multiply(tmp2, tmp3)
        epan = np.add(tmp1, tmp4)

        return epan

    def rad_to_evap(self):
        """
         converts solar radiation to equivalent inches of water evaporation

        SRadIn[in/day] = SolRad[Ley/day] / ((597.3-0.57) * temp[centigrade]) * 2.54)    [1]
        or using equation 20 of FAO chapter 3

        from TABLE 3 in FAO chap 3.
        SRadIn[mm/day] = 0.408 * Radiation[MJ m-2 day-1]
        SRadIn[mm/day] = 0.035 * Radiation[Wm-2]
        SRadIn[mm/day] = Radiation[MJ m-2 day-1] / 2.45
        SRadIn[mm/day] = Radiation[J cm-2 day-1] / 245
        SRadIn[mm/day] = Radiation[Wm-2] / 28.4

    [1] https://github.com/respec/BASINS/blob/4356aa9481eb7217cb2cbc5131a0b80a932907bf/atcMetCmp/modMetCompute.vb#L1251
    https://github.com/DanluGuo/Evapotranspiration/blob/8efa0a2268a3c9fedac56594b28ac4b5197ea3fe/R/Evapotranspiration.R
    http://www.fao.org/3/X0490E/x0490e07.htm

        """
        # TODO following equation assumes radiations in langleys/day ando output in Inches
        tmp1 = np.multiply(np.subtract(597.3, np.multiply(0.57, self.input['temp'].values)), 2.54)
        rad_in = np.divide(self.input['sol_rad'].values, tmp1)

        return rad_in

    def equil_temp(self, et_daily):
        # equilibrium temperature T_e
        t_e = self.input['temp'].copy()
        ta = self.input['temp']
        vabar = self.avp_from_rel_hum()
        r_n = self.net_rad(vabar)  # net radiation
        gamma = self.psy_const()
        for i in range(9999):
            v_e = 0.6108 * np.exp(17.27 * t_e/(t_e + 237.3))  # saturated vapour pressure at T_e (S2.5)
            t_e_new = ta - 1 / gamma * (1 - r_n / (LAMBDA * et_daily)) * (v_e - vabar)  # rearranged from S8.8
            delta_t_e = t_e_new - t_e
            maxdelta_t_e = np.abs(np.max(delta_t_e))
            t_e = t_e_new
            if maxdelta_t_e < 0.01:
                break
        return t_e


def freq_in_mins_from_string(input_string: str) -> int:

    if has_numbers(input_string):
        in_minutes = split_freq(input_string)
    elif input_string.upper() in ['D', 'H', 'M', 'DAILY', 'HOURLY', 'MONTHLY', 'YEARLY', 'MIN', 'MINUTE']:
        in_minutes = str_to_mins(input_string.upper())
    else:
        raise TypeError("invalid input string", input_string)

    return int(in_minutes)


def str_to_mins(input_string: str) -> int:
    d = {'MIN': 1,
         'MINUTE': 1,
         'DAILY': 1440,
         'D': 1440,
         'HOURLY': 60,
         'HOUR': 60,
         'H': 60,
         'MONTHLY': 43200,
         'M': 43200,
         'YEARLY': 525600
         }

    return d[input_string]


def split_freq(freq_str: str) -> int:
    match = re.match(r"([0-9]+)([a-z]+)", freq_str, re.I)
    if match:
        minutes, freq = match.groups()
        if freq.upper() in ['H', 'HOURLY', 'HOURS', 'HOUR']:
            minutes = int(minutes) * 60
        elif freq.upper() in ['D', 'DAILY', 'DAY', 'DAYS']:
            minutes = int(minutes) * 1440
        return int(minutes)
    else:
        raise NotImplementedError


def has_numbers(input_string: str) -> bool:
    return bool(re.search(r'\d', input_string))


def justify_len(string: str, length: int = 2, pad: str = '0') -> str:

    if len(string) < length:
        zeros_to_pad = pad * int(len(string) - length)
        new_string = zeros_to_pad + string
    else:
        new_string = string

    return new_string


def add_freq(dataframe,  name=None, _force_freq=None, method=None):
    """Add a frequency attribute to idx, through inference or directly.
    Returns a copy.  If `freq` is None, it is inferred.
    """
    idx = dataframe.index
    idx = idx.copy()
    # if freq is None:
    if idx.freq is None:
        freq = pd.infer_freq(idx)
        idx.freq = freq

        if idx.freq is None:
            if _force_freq is not None:
                dataframe = force_freq(dataframe, _force_freq, name, method=method)
            else:

                raise AttributeError('no discernible frequency found in {} for {}.  Specify'
                                     ' a frequency string with `freq`.'.format(name, name))
        else:
            print('frequency {} is assigned to {}'.format(idx.freq, name))
            dataframe.index = idx

    return dataframe


def force_freq(data_frame, freq_to_force, name, method=None):

    old_nan_counts = data_frame.isna().sum()
    old_shape = data_frame.shape
    dr = pd.date_range(data_frame.index[0], data_frame.index[-1], freq=freq_to_force)

    df_unique = data_frame[~data_frame.index.duplicated(keep='first')]  # first remove duplicate indices if present
    if method:
        df_idx_sorted = df_unique.sort_index()
        df_reindexed = df_idx_sorted.reindex(dr, method='nearest')
    else:
        df_reindexed = df_unique.reindex(dr, fill_value=np.nan)

    df_reindexed.index.freq = pd.infer_freq(df_reindexed.index)
    new_nan_counts = df_reindexed.isna().sum()
    print('Frequency {} is forced to {} dataframe, NaN counts changed from {} to {}, shape changed from {} to {}'
          .format(df_reindexed.index.freq, name, old_nan_counts.values, new_nan_counts.values,
                  old_shape, df_reindexed.shape))
    return df_reindexed


def min_to_str(minutes: int) -> str:
    if minutes == 1:
        freq_str = 'Minute'
    elif 60 > minutes > 1:
        freq_str = 'Sub_hourly'
    elif minutes == 60:
        freq_str = 'Hourly'
    elif 1440 > minutes > 60:
        freq_str = 'Sub-daily'
    elif minutes == 1440:
        freq_str = 'Daily'
    elif 43200 > minutes > 1440:
        freq_str = 'Sub-monthly'
    elif minutes == 43200:
        freq_str = 'Monthly'
    elif 525600 > minutes > 43200:
        freq_str = 'Sub-yearly'
    elif minutes == 525600:
        freq_str = 'Yearly'
    else:
        raise ValueError("Can not calculate frequency string from given frequency in minutes ", minutes)

    return freq_str


time_step = {'D': 'Day', 'H': 'Hour', 'M': 'MonthEnd'}


def get_offset(freqstr: str) -> str:
    offset_step = 1
    if freqstr in time_step:
        freqstr = time_step[freqstr]
    elif has_numbers(freqstr):
        in_minutes = split_freq(freqstr)
        freqstr = 'Minute'
        offset_step = int(in_minutes)

    offset = getattr(pd.offsets, freqstr)(offset_step)

    return offset


def _wrap(x, x_min, x_max):
    """Wrap floating point values into range

    Parameters
    ----------
    x : ndarray
        Values to wrap.
    x_min : float
        Minimum value in output range.
    x_max : float
        Maximum value in output range.

    Returns
    -------
    ndarray

    """
    return np.mod((x - x_min), (x_max - x_min)) + x_min


def omega(solar_time):
    """Solar hour angle (Eq. 55)

    Parameters
    ----------
    solar_time : scalar or array_like of shape(M, )
        Solar time (i.e. noon is 0) [hours].

    Returns
    -------
    omega : ndarray
        Hour angle [radians].

    """
    _omega = (2 * math.pi / 24.0) * solar_time

    # Need to adjust omega so that the values go from -pi to pi
    # Values outside this range are wrapped (i.e. -3*pi/2 -> pi/2)
    _omega = _wrap(_omega, -math.pi, math.pi)
    return _omega


def solar_time_rad(lon, time_mid, sc):
    """Solar time (i.e. noon is 0) (Eq. 55)

    Parameters
    ----------
    lon : scalar or array_like of shape(M, )
        Longitude [radians].
    time_mid : scalar or array_like of shape(M, )
        UTC time at midpoint of period [hours].
    sc : scalar or array_like of shape(M, )
        Seasonal correction [hours].

    Returns
    -------
    ndarray
        Solar time [hours].

    Notes
    -----
    This function could be integrated into the _omega() function since they are
    always called together (i.e. _omega(_solar_time_rad()).  It was built
    independently from _omega to eventually support having a separate
    solar_time functions for longitude in degrees.

    """
    return time_mid + (lon * 24 / (2 * math.pi)) + sc - 12
