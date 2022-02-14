import numpy as np
import pandas as pd

from .utils import Utils
from .global_variables import LAMBDA

# TODO, classify methods which require wind_speed, or which require solar_rad.

class ETBase(Utils):
    """
    This is the base class for evapotranspiration calculation. It calculates
    etp according to Jensen and Haise_ method. Any new ETP calculation must
    inherit from it and must implement
    the ``__call__`` method.

    .. _Haise:
        https://doi.org/10.1061/JRCEA4.0000287
    """
    def __init__(self,
                 input_df: pd.DataFrame,
                 units:dict,
                 constants:dict,
                 **kwargs
                 ):
        """
        Parameters
        ---------
            input_df :
            units :
            constants :
            kwargs :
        """
        self.name = self.__class__.__name__

        super(ETBase, self).__init__(input_df.copy(),
                                     units.copy(),
                                     constants.copy(),
                                     **kwargs)

    def requirements(self, **kwargs):

        if 'constants' in kwargs:
            constants = kwargs['constants']
        else:
            constants = ['lat_dec_deg', 'altitude', 'ct', 'tx']

        if 'ts' in kwargs:
            ts = kwargs['ts']
        else:
            ts = ['temp']

        for cons in constants:
            if cons not in self.cons:
                if cons in self.default_cons:
                    val = self.default_cons[cons]['def_val']
                    desc = self.default_cons[cons]['desc']
                    if val is not None:
                        print("Warning: default value {} of parameter {} which is {} is being used".format(val,
                                                                                                           cons,
                                                                                                           desc))
                        self.cons[cons] = val
                else:
                    raise ValueError("Value of constant {} must be provided to calculate ETP using {}"
                                     .format(cons, self.name))
        for _ts in ts:
            if _ts not in self.input.columns:
                raise ValueError("Timeseries {} is required for calculation of ETP using {}"
                                 .format(_ts, self.name))

    def __call__(self, *args,
                 transform: bool=False,
                 **kwargs):
        """
        as given (eq 9) in Xu and Singh, 2000 and implemented here_

        uses:  a_s, b_s, ct=0.025, tx=-3
        Arguments:
            transform : whether to transform the calculated etp to frequecies
                other than at which it is calculated.

        .. _here:
            https://github.com/DanluGuo/Evapotranspiration/blob/8efa0a2268a3c9fedac56594b28ac4b5197ea3fe/R/Evapotranspiration.R#L2734
        """
        self.requirements(constants=['lat_dec_deg', 'altitude', 'ct', 'tx'],
                          ts=['temp'])
        rs = self.rs()
        tmp1 = np.multiply(np.multiply(self.cons['ct'], np.add(self.input['temp'], self.cons['tx'])), rs)
        et = np.divide(tmp1, LAMBDA)

        self.post_process(et, transform=transform)
        return et

    def post_process(self, et, transform=False):
        if isinstance(et, np.ndarray):
            et = pd.Series(et, index=self.input.index)
        self.output['et_' + self.name + '_' + self.freq_str] = et
        if transform:
            self.transform_etp(self.name)

    def summary(self):

        methods_evaluated = []
        for m in self.output.keys():
            if 'Hourly' in m:
                methods_evaluated.append(m)

        for m in methods_evaluated:
            ts = self.output[m]
            yrs = np.unique(ts.index.year)
            print('For {} \n'.format(m.split('_')[1], end=','))
            for yr in yrs:
                st, en = str(yr) + '0101', str(yr) + '1231'
                yr_ts = ts[st:en]
                yr_sum = yr_ts.sum().values[0]
                yr_mean = yr_ts.mean().values[0]
                print('for year {}:, sum: {:<10.1f} mean: {:<10.1f}'.format(yr, yr_sum, yr_mean))


class Abtew(ETBase):
    """
    daily etp using equation 3 in Abtew_ 1996. `k` is a dimentionless coefficient.

     uses: , k=0.52, a_s=0.23, b_s=0.5
    :param `k` coefficient, default value taken from [1]
    :param `a_s` fraction of extraterrestrial radiation reaching earth on sunless days
    :param `b_s` difference between fracion of extraterrestrial radiation reaching full-sun days
             and that on sunless days.

    .. _Abtew:
        https://doi.org/10.1111/j.1752-1688.1996.tb04044.x

     """
    def __call__(self, *args, **kwargs):

        self.requirements(constants=['lat_dec_deg', 'altitude', 'abtew_k'])

        rs = self.rs()
        et = np.multiply(self.cons['abtew_k'], np.divide(rs, LAMBDA))

        self.post_process(et, kwargs.get('transform', False))
        return et


class Albrecht(ETBase):
    """
     Developed in Germany by Albrecht, 1950. Djaman et al., 2016 Wrote the formula as
      eto = (0.1005 + 0.297 * u2) * (es - ea)
    """
    def __call__(self, *args, **kwargs):

        # Mean saturation vapour pressure
        if 'es' not in self.input:
            if self.freq_str == 'Daily':
                es = self.mean_sat_vp_fao56()
            elif self.freq_str == 'Hourly':
                es = self.sat_vp_fao56(self.input['temp'].values)
            elif self.freq_str == 'sub_hourly':   # TODO should sub-hourly be same as hourly?
                es = self.sat_vp_fao56(self.input['temp'].values)
            else:
                raise NotImplementedError
        else:
            es = self.input['es']

        # actual vapour pressure
        ea = self.avp_from_rel_hum()

        u2 = self._wind_2m()
        eto = (0.1005 + 0.297 * u2) * (es - ea)

        self.post_process(eto, kwargs.get('transform', False))
        return eto


class BlaneyCriddle(ETBase):
    """
    using formulation of Blaney-Criddle for daily reference crop ETP using monthly mean tmin and tmax.
    Inaccurate under extreme climates. underestimates in windy, dry and sunny conditions and overestimates under
    calm, humid and clouded conditions.
    
    Doorenbos, J., & Pruitt, W. O. (1977). Crop water requirements, FAO Irrigation and Drainage.
    Paper 24, 2a ed., Roma, Italy.
    """
    def __call__(self, *args, **kwargs):
        # TODO include modified BlaneyCriddle as introduced  in [3]

        self.requirements(constants=['e0', 'e1', 'e2', 'e3', 'e4'])  # check that all constants are present

        N = self.daylight_fao56()  # mean daily percentage of annual daytime hours
        u2 = self._wind_2m()
        rh_min = self.input['rh_min']
        n = self.input['sunshine_hrs']
        ta = self.input['temp'].values
        # undefined working variable (Allena and Pruitt, 1986; Shuttleworth, 1992) (S9.8)
        a1 = self.cons['e0'] + self.cons['e1'] * rh_min + self.cons['e2'] * n / N
        a2 = self.cons['e3'] * u2
        a3 = self.cons['e4'] * rh_min * n / N + self.cons['e5'] * rh_min * u2
        bvar = a1 + a2 + a3
        # calculate yearly sum of daylight hours and assign that value to each point in array `N`
        n_annual = assign_yearly(N, self.input.index)

        # percentage of actual daytime hours for the day comparing to the annual sum of maximum sunshine hours
        p_y = 100 * n / n_annual['N'].values

        # reference crop evapotranspiration
        et = (0.0043 * rh_min - n / N - 1.41) + bvar * p_y * (0.46 * ta + 8.13)
        self.post_process(et, kwargs.get('transform', False))
        return et


class BrutsaertStrickler(ETBase):
    """
    using formulation given by BrutsaertStrickler

    :param `alpha_pt` Priestley-Taylor coefficient = 1.26 for Priestley-Taylor_ model (Priestley and Taylor, 1972)
    :param `a_s` fraction of extraterrestrial radiation reaching earth on sunless days
    :param `b_s` difference between fracion of extraterrestrial radiation reaching full-sun days
             and that on sunless days.
    :param `albedo`  Any numeric value between 0 and 1 (dimensionless), albedo of the evaporative surface
            representing the portion of the incident radiation that is reflected back at the surface.
            Default is 0.23 for surface covered with short reference crop.
    :return: et

    .. _Priestley-Taylor:
        https://doi.org/10.1029/WR015i002p00443
    """
    def __call__(self, *args, **kwargs):
        self.requirements(constants=['alphaPT'])  # check that all constants are present

        delta = self.slope_sat_vp(self.input['temp'].values)
        gamma = self.psy_const()
        vabar = self.avp_from_rel_hum()  # Vapour pressure, *ea*
        vas = self.mean_sat_vp_fao56()
        u2 = self._wind_2m()
        f_u2 = np.add(2.626, np.multiply(1.381, u2))
        r_ng = self.net_rad(vabar)
        alpha_pt = self.cons['alphaPT']

        et = np.subtract(np.multiply(np.multiply((2*alpha_pt-1),
                                                 np.divide(delta, np.add(delta, gamma))),
                                     np.divide(r_ng, LAMBDA)),
                         np.multiply(np.multiply(np.divide(gamma, np.add(delta, gamma)), f_u2),
                                     np.subtract(vas, vabar)))
        self.post_process(et, kwargs.get('transform', False))
        return et


class Camargo(ETBase):
    """
    Originally presented by Camargo, 1971. Following formula is presented in Fernandes et al., 2012 quoting
    Sedyiama et al., 1997.
         eto = f * Tmean * ra * nd

    Gurski et al., 2018 has not written nd in formula. He expressed formula to convert extra-terresterial radiation
    into equivalent mm/day as
        ra[mm/day] = ra[MegaJoulePerMeterSquare PerDay] / 2.45
        where 2.45 is constant.

     eto: reference etp in mm/day.
     f: an empircal factor taken as 0.01
     ra: extraterrestrial radiation expressed as mm/day
     nd: length of time interval
    """
    def __call__(self, *args, **kwargs):
        self.requirements(constants=['f_camargo'])  # check that all constants are present

        ra = self._et_rad()
        if self.freq_str == 'Daily':
            ra = ra/2.45
        else:
            raise NotImplementedError
        et = self.cons['f_camargo'] * self.input['temp'] * ra
        self.post_process(et, kwargs.get('transform', False))
        return et


class Caprio(ETBase):
    """
    Developed by Caprio (1974). Pandey et al 2016 wrote the equation as
        eto = (0.01092708*t + 0.0060706) * rs
    """
    def __call__(self, *args, **kwargs):

        rs = self.rs()
        eto = (0.01092708 * self.input['temp'] + 0.0060706) * rs

        self.post_process(eto, kwargs.get('transform', False))
        return eto


class ChapmanAustralia(ETBase):
    """using formulation of Chapman_, 2001,

    uses: a_s=0.23, b_s=0.5, ap=2.4, alphaA=0.14, albedo=0.23

    .. _Chapman:
        https://116.90.59.164/MODSIM03/Volume_01/A03/04_Chapman.pdf
    """
    def __call__(self, *args, **kwargs):
        self.requirements(constants=['lat_dec_deg', 'altitude', 'alphaA', 'pan_ap', 'albedo'],
                          ts=['temp'])

        lat = self.cons['lat_dec_deg']
        a_p = 0.17 + 0.011 * abs(lat)
        b_p = np.power(10, (0.66 - 0.211 * abs(lat)))  # constants (S13.3)

        epan = self.evap_pan()

        et = np.add(np.multiply(a_p, epan), b_p)

        self.post_process(et, kwargs.get('transform', False))
        return et


class Copais(ETBase):
    """
    Developed for central Greece by Alexandris et al 2006 and used in Alexandris et al 2008.
    """
    def __call__(self, *args, **kwargs):
        et = None
        self.post_process(et, kwargs.get('transform', False))
        return et


class Dalton(ETBase):
    """
    using Dalton formulation as mentioned here_ in mm/dday

    uses:
      es: mean saturation vapour pressure
      ea: actual vapour pressure
      u2: wind speed


    .. _here:
        https://water-for-africa.org/en/dalton.html
    """
    def __call__(self, *args, **kwargs):

        u2 = self._wind_2m()
        fau = 0.13 + 0.14 * u2

        # Mean saturation vapour pressure
        if 'es' not in self.input:
            if self.freq_str == 'Daily':
                es = self.mean_sat_vp_fao56()
            elif self.freq_str == 'Hourly':
                es = self.sat_vp_fao56(self.input['temp'].values)
            elif self.freq_str == 'sub_hourly':   # TODO should sub-hourly be same as hourly?
                es = self.sat_vp_fao56(self.input['temp'].values)
            else:
                raise NotImplementedError
        else:
            es = self.input['es']

        # actual vapour pressure
        ea = self.avp_from_rel_hum()
        if 'vp_def' not in self.input:
            vp_d = es - ea   # vapor pressure deficit
        else:
            vp_d = self.input['vp_def']

        etp = fau * vp_d
        self.post_process(etp, kwargs.get('transform', False))
        return etp


class DeBruinKeijman(ETBase):
    """
     Calculates daily Pot ETP, developed by deBruin and Jeijman 1979 and used in Rosenberry et al 2004.
    """


class DoorenbosPruitt(ETBase):
    """
    Developed by Doorenbos and Pruitt (1777), Poyen et al wrote following equation
      et = a(delta/(delta+gamma) * rs) + b
      b = -0.3
      a = 1.066 - 0.13 x10^{-2} * rh + 0.045*ud - 0.2x10^{-3}*rh * ud - 0.315x10^{-4}*rh**2 - 0.11x10{-2}*ud**2
      used in Xu HP 2000.
    """


class GrangerGray(ETBase):
    """
    using formulation of Granger & Gray 1989 which is for non-saturated lands and modified form of penman 1948.

     uses: , wind_f`='pen48', a_s=0.23, b_s=0.5, albedo=0.23
    :param `wind_f` str, if 'pen48 is used then formulation of [1] is used otherwise formulation of [3] requires
             wind_f to be 2.626.
    :param `a_s fraction of extraterrestrial radiation reaching earth on sunless days
    :param `b_s` difference between fracion of extraterrestrial radiation reaching full-sun days
             and that on sunless days.
    :param `albedo`  Any numeric value between 0 and 1 (dimensionless), albedo of the evaporative surface
            representing the portion of the incident radiation that is reflected back at the surface.
            Default is 0.23 for surface covered with short reference crop.
    :return:

    https://doi.org/10.1016/0022-1694(89)90249-7

    """
    def __call__(self, *args, **kwargs):

        self.requirements(constants=['wind_f'])  # check that all constants are present

        if self.cons['wind_f'] not in ['pen48', 'pen56']:
            raise ValueError('value of given wind_f is not allowed.')

        if self.cons['wind_f'] == 'pen48':
            _a = 2.626
            _b = 0.09
        else:
            _a = 1.313
            _b = 0.06

        # rs = self.rs()
        delta = self.slope_sat_vp(self.input['temp'].values)
        gamma = self.psy_const()

        vabar = self.avp_from_rel_hum()  # Vapour pressure
        r_n = self.net_rad(vabar)   # net radiation
        vas = self.mean_sat_vp_fao56()

        u2 = self._wind_2m()
        fau = _a + 1.381 * u2
        ea = np.multiply(fau, np.subtract(vas, vabar))

        # dimensionless relative drying power  eq 7 in Granger, 1998
        dry_pow = np.divide(ea, np.add(ea, np.divide(np.subtract(r_n, self.soil_heat_flux()), LAMBDA)))
        # eq 6 in Granger, 1998
        g_g = 1 / (0.793 + 0.20 * np.exp(4.902 * dry_pow)) + 0.006 * dry_pow

        tmp1 = np.divide(np.multiply(delta, g_g), np.add(np.multiply(delta, g_g), gamma))
        tmp2 = np.divide(np.subtract(r_n, self.soil_heat_flux()), LAMBDA)
        tmp3 = np.multiply(np.divide(np.multiply(gamma, g_g), np.add(np.multiply(delta, g_g), gamma)), ea)
        et = np.add(np.multiply(tmp1, tmp2), tmp3)

        self.post_process(et, kwargs.get('transform', False))
        return et


class Hamon(ETBase):
    """calculates evapotranspiration in mm using Hamon 1963 method as given in Lu et al 2005. It uses daily mean
     temperature which can also be calculated
    from daily max and min temperatures. It also requires `daylight_hrs` which is hours of day light, which if not
    provided as input, will be calculated from latitutde. This means if `daylight_hrs` timeseries is not provided as
    input, then argument `lat` must be provided.

    pet = cts * n * n * vdsat
    vdsat = (216.7 * vpsat) / (tavc + 273.3)
    vpsat = 6.108 * exp((17.26939 * tavc)/(tavc + 237.3))

    :uses cts: float, or array of 12 values for each month of year or a time series of equal length as input data.
                 if it is float, then that value will be considered for whole year. Default value of 0.0055 was used
                 by Hamon 1961, although he later used different value but I am using same value as it is used by
                 WDMUtil. It should be also noted that 0.0055 is to be used when pet is in inches. So I am dividing
                 the whole pet by 24.5 in order to convert from inches to mm while still using 0.0055.

    References
    ----------
    Hamon, W. R. (1963). Computation of direct runoff amounts from storm rainfall. International Association of
    Scientific Hydrology Publication, 63, 52-62.

    Lu et al. (2005).  A comparison of six potential evaportranspiration methods for regional use in the
        southeastern United States.  Journal of the American Water Resources Association, 41, 621-633.
     """

    def __call__(self, *args, **kwargs):
        self.requirements(constants=['lat_dec_deg', 'altitude', 'albedo', 'cts'],
                          ts=['temp'])
        # allow cts to be provided as input while calling method, e.g we may want to use array
        if 'cts' in kwargs:
            cts = kwargs['cts']
        else:
            cts = self.cons['cts']

        if 'sunshine_hrs' not in self.input.columns:
            if 'daylight_hrs' not in self.input.columns:
                daylight_hrs = self.daylight_fao56()
            else:
                daylight_hrs = self.input['daylight_hrus']
            sunshine_hrs = daylight_hrs
            print('Warning, sunshine hours are consiered equal to daylight hours')
        else:
            sunshine_hrs = self.input['sunshine_hrs']

        sunshine_hrs = np.divide(sunshine_hrs, 12.0)

        # preference should be given to tmin and tmax if provided and if tmin, tmax is not provided then use temp which
        # is mean temperature. This is because in original equations, vd_sat is calculated as average of max vapour
        # pressure and minimum vapour pressue.
        if 'tmax' not in self.input.columns:
            if 'temp' not in self.input.columns:
                raise ValueError('Either tmax and tmin or mean temperature should be provided as input')
            else:
                vd_sat = self.sat_vp_fao56(self.input['temp'])
        else:
            vd_sat = self.mean_sat_vp_fao56()

        # in some literature, the equation is divided by 100 by then the cts value is 0.55 instead of 0.0055
        et = cts * 25.4 * np.power(sunshine_hrs, 2) * (216.7 * vd_sat * 10 / (np.add(self.input['temp'], 273.3)))

        self.post_process(et, kwargs.get('transform', False))
        return et


class HargreavesSamani(ETBase):
    """
    estimates daily ETo using Hargreaves method Hargreaves and Samani_, 1985.
    :uses
      temp
      tmin
      tmax
    :param
    method: str, if `1985`, then the method of 1985 (Hargreaves and Samani, 1985) is followed as calculated by and
    mentioned by Hargreaves and Allen, 2003.
    if `2003`, then as formula is used as mentioned in [1]
    Note: Current test passes for 1985 method.
    There is a variation of Hargreaves introduced by Trajkovic 2007 as mentioned in Alexandris 2008.

    .. _Samani:
        https://rdrr.io/cran/Evapotranspiration/man/ET.HargreavesSamani.html
    """

    def __call__(self, method='1985', **kwargs):
        self.requirements(constants=['lat_dec_deg', 'altitude', 'albedo'],
                          ts=['temp'])

        if method == '2003':
            tmp1 = np.multiply(0.0023, np.add(self.input['temp'], 17.8))
            tmp2 = np.power(np.subtract(self.input['tmax'].values, self.input['tmin'].values), 0.5)
            tmp3 = np.multiply(0.408, self._et_rad())
            et = np.multiply(np.multiply(tmp1, tmp2), tmp3)

        else:
            ra_my = self._et_rad()
            tmin = self.input['tmin'].values
            tmax = self.input['tmax'].values
            ta = self.input['temp'].values
            # empirical coefficient by Hargreaves and Samani (1985) (S9.13)
            c_hs = 0.00185 * np.power((np.subtract(tmax, tmin)), 2) - 0.0433 * (np.subtract(tmax, tmin)) + 0.4023
            et = 0.0135 * c_hs * ra_my / LAMBDA * np.power((np.subtract(tmax, tmin)), 0.5) * (np.add(ta, 17.8))

        self.post_process(et, kwargs.get('transform', False))
        return et


class Haude(ETBase):
    """
    only requires air temp and relative humidity at 2:00 pm. Good for moderate
    zones despite being simple

    References
    ----------
    [1]. Haude, W. (1954). Zur praktischen Bestimmung der aktuellen und potentiellen Evaporation und Evapotranspiration.
     Schweinfurter Dr. und Verlag-Ges..
    """
    def __call__(self, *args, **kwargs):

        etp = None  # f_mon * (6.11 × 10(7.48 × T / (237+T)) - rf × es)
        self.post_process(etp, kwargs.get('transform', False))
        return etp


class JensenHaiseBasins(ETBase):
    """
    This method generates daily pan evaporation (inches) using a coefficient for the month `cts`, , the daily
    average air temperature (F), a coefficient `ctx`, and solar radiation (langleys/day) as givn in
    BASINS program[2].
    The computations are
    based on the Jensen and Haise (1963) formula.
              PET = CTS * (TAVF - CTX) * RIN

        where
              PET = daily potential evapotranspiration (in)
              CTS = monthly variable coefficient
             TAVF = mean daily air temperature (F), computed from max-min
              CTX = coefficient
              RIN = daily solar radiation expressed in inches of evaporation

              RIN = SWRD/(597.3 - (.57 * TAVC)) * 2.54

        where
             SWRD = daily solar radiation (langleys)
             TAVC = mean daily air temperature (C)
    :uses cts float or array like. Value of monthly coefficient `cts` to be used. If float, then same value is
            assumed for all months. If array like then it must be of length 12.
    :uses ctx `float` constant coefficient value of `ctx` to be used in Jensen and Haise formulation.
    """
    def __call__(self, *args, **kwargs):
        if 'cts_jh' in kwargs:
            cts = kwargs['cts_jh']
        else:
            cts = self.cons['cts_jh']

        if 'cts_jh' in kwargs:
            ctx = kwargs['ctx_jh']
        else:
            ctx = self.cons['ctx_jh']

        if not isinstance(cts, float):
            if not isinstance(np.array(ctx), np.ndarray):
                raise ValueError('cts must be array like')
            else:  # if cts is array like it must be given for 12 months of year, not more not less
                if len(np.array(cts)) > 12:
                    raise ValueError('cts must be of length 12')
        else:  # if only one value is given for all moths distribute it as monthly value
            cts = np.array([cts for _ in range(12)])

        if not isinstance(ctx, float):
            raise ValueError('ctx must be float')

        # distributing cts values for all dates of input data
        self.input['cts'] = np.nan
        for m, i in zip(self.input.index.month, self.input.index):
            for _m in range(m):
                self.input.at[i, 'cts'] = cts[_m]

        cts = self.input['cts']
        taf = self.input['temp'].values

        rad_in = self.rad_to_evap()
        pan_evp = np.multiply(np.multiply(cts, np.subtract(taf, ctx)), rad_in)
        et = np.where(pan_evp < 0.0, 0.0, pan_evp)

        self.post_process(et, kwargs.get('transform', False))
        return et


class Kharrufa(ETBase):
    """
    For monthly potential evapotranspiration estimation, originally presented by Kharrufa, 1885. Xu and Singh, 2001
    presented following formula:
        et = 0.34 * p * Tmean**1.3
    et: pot. evapotranspiration in mm/month.
    Tmean: Average temperature in Degree Centigrade
    p: percentage of total daytime hours for the period used (daily or monthly) outof total daytime hours of the
       year (365 * 12)
    Kharrufa, N. S. (1985). Simplified equation for evapotranspiration in arid regions. Beitrage zur
	Hydrologie, 5(1), 39-47.
    """
    def __call__(self, *args, **kwargs):
        ta = self.input['temp']

        N = self.daylight_fao56()  # mean daily percentage of annual daytime hours
        n_annual = assign_yearly(N, self.input.index)

        et = 0.34 * n_annual['N'].values * ta**1.3

        self.post_process(et, kwargs.get('transform', False))
        return et


class Linacre(ETBase):
    """
     using formulation of Linacre 1977 who simplified Penman method.
     :uses
       temp
       tdew/rel_hum
       https://doi.org/10.1016/0002-1571(77)90007-3
     """
    def __call__(self, *args, **kwargs):

        if 'tdew' not in self.input:
            if 'rel_hum' in self.input:
                self.tdew_from_t_rel_hum()

        tm = np.add(self.input['temp'].values, np.multiply(0.006, self.cons['altitude']))
        tmp1 = np.multiply(500, np.divide(tm, 100 - self.cons['lat_dec_deg']))
        tmp2 = np.multiply(15, np.subtract(self.input['temp'].values, self.input['tdew'].values))
        upar = np.add(tmp1, tmp2)

        et = np.divide(upar, np.subtract(80, self.input['temp'].values))

        self.post_process(et, kwargs.get('transform', False))
        return et


class Makkink(ETBase):
    """
    :uses
      a_s, b_s
      temp
      solar_rad

    using formulation of Makkink
    """
    def __call__(self, *args, **kwargs):

        rs = self.rs()

        delta = self.slope_sat_vp(self.input['temp'].values)
        gamma = self.psy_const()

        et = np.subtract(np.multiply(np.multiply(0.61, np.divide(delta, np.add(delta, gamma))),
                                     np.divide(rs, 2.45)), 0.12)

        self.post_process(et, kwargs.get('transform', False))
        return et


class Irmak(ETBase):
    """
    Pandey et al 2016, presented 3 formulas for Irmak.
    1    eto = -0.611 + 0.149 * rs + 0.079 * t
    2    eto = -0.642 + 0.174 * rs + 0.0353 * t
    3    eto = -0.478 + 0.156 * rs - 0.0112 * tmax + 0.0733 * tmin

    References:
        Irmak 2003
        Tabari et al 2011
        Pandey et al 2016
    """


class Mahringer(ETBase):
    """
     Developed by Mahringer in Germany. [1] Wrote formula as
           eto = 0.15072 * sqrt(3.6) * (es - ea)
    """


class Mather(ETBase):
    """
    Developed by Mather 1978 and used in Rosenberry et al 2004. Calculates daily Pot ETP.
    pet = [1.6 (10T_a/I) ** 6.75e-7 * I**3 - 7.71e-7 * I**2 + 1.79e-2 * I + 0.49] (10/d)
    I = annual heat index, sum(Ta/5)1.514
    d = number of days in month
    """


class MattShuttleworth(ETBase):
    """
    using formulation of Matt-Shuttleworth and Wallace, 2009. This is designed for semi-arid and windy areas as an
    alternative to FAO-56 Reference Crop method
    10.13031/2013.29217
    https://elibrary.asabe.org/abstract.asp?aid=29217
    """
    def __call__(self, *args, **kwargs):

        self.requirements(constants=['CH', 'Roua', 'Ca', 'surf_res'])

        ch = self.cons['CH']  # crop height
        ro_a = self.cons['Roua']
        ca = self.cons['Ca']  # specific heat of the air
        # surface resistance (s m-1) of a well-watered crop equivalent to the FAO crop coefficient
        r_s = self.cons['surf_res']

        vabar = self.avp_from_rel_hum()  # Vapour pressure
        vas = self.mean_sat_vp_fao56()
        r_n = self.net_rad(vabar)  # net radiation
        u2 = self._wind_2m()    # Wind speed
        delta = self.slope_sat_vp(self.input['temp'].values)   # slope of vapour pressure curve
        gam = self.psy_const()    # psychrometric constant

        tmp1 = self.seconds * ro_a * ca
        # clinmatological resistance (s*m^-1) (S5.34)
        r_clim = np.multiply(tmp1, np.divide(np.subtract(vas, vabar), np.multiply(delta, r_n)))
        r_clim = np.where(r_clim == 0, 0.1, r_clim)   # correction for r_clim = 0
        u2 = np.where(u2 == 0, 0.1, u2)               # correction for u2 = 0

        #  ratio of vapour pressure deficits at 50m to vapour pressure deficits at 2m heights, eq S5.35
        a1 = (302 * (delta + gam) + 70 * gam * u2)
        a2 = (208 * (delta + gam) + 70 * gam * u2)
        a3 = 1/r_clim * ((302 * (delta + gam) + 70 * gam * u2) / (208 * (delta + gam) + 70 * gam * u2) * (208 / u2) - (302 / u2))
        vpd50_to_vpd2 = a1/a2 + a3

        # aerodynamic coefficient for crop height (s*m^-1) (eq S5.36 in McMohan et al 2013)
        a1 = 1 / (0.41**2)
        a2 = np.log((50 - 0.67 * ch) / (0.123 * ch))
        a3 = np.log((50 - 0.67 * ch) / (0.0123 * ch))
        a4 = np.log((2 - 0.08) / 0.0148) / np.log((50 - 0.08) / 0.0148)
        rc_50 = a1 * a2 * a3 * a4

        a1 = 1/LAMBDA
        a2 = (delta * r_n + (ro_a * ca * u2 * (vas - vabar)) / rc_50 * vpd50_to_vpd2)
        a3 = (delta + gam * (1 + r_s * u2 / rc_50))
        et = a1 * a2/a3
        self.post_process(et, kwargs.get('transform', False))
        return et


class McGuinnessBordne(ETBase):
    """
    calculates evapotranspiration [mm/day] using Mcguinnes Bordne formulation McGuinnes and Bordne, 1972.

    """
    def __call__(self, *args, **kwargs):

        ra = self._et_rad()
        # latent heat of vaporisation, MJ/Kg
        _lambda = LAMBDA  # multiply((2.501 - 2.361e-3), self.input['temp'].values)
        tmp1 = np.multiply((1/_lambda), ra)
        tmp2 = np.divide(np.add(self.input['temp'].values, 5), 68)
        et = np.multiply(tmp1, tmp2)

        self.post_process(et, kwargs.get('transform', False))
        return et


class Penman(ETBase):
    """
    calculates pan evaporation from open water using formulation of Penman, 1948, as mentioned (as eq 12) in
    McMahon et al., 2012. If wind data is missing then equation 33 from Valiantzas, 2006 is used which does not require
    wind data.

    uses:  wind_f='pen48', a_s=0.23, b_s=0.5, albedo=0.23
           uz
           temp
           rs
           reh_hum

    :param `wind_f` str, if 'pen48 is used then formulation of [1] is used otherwise formulation of [3] requires
             wind_f to be 2.626.
    """
    # todo, gives -ve values sometimes

    def __call__(self, **kwargs):
        self.requirements(constants=['lat_dec_deg', 'altitude', 'wind_f', 'albedo'],
                          ts=['temp', 'rh_mean'])

        if self.cons['wind_f'] not in ['pen48', 'pen56']:
            raise ValueError('value of given wind_f is not allowed.')

        wind_method = 'macmohan'
        if 'wind_method' in kwargs:
            wind_method = kwargs['wind_method']

        if self.cons['wind_f'] == 'pen48':
            _a = 2.626
            _b = 0.09
        else:
            _a = 1.313
            _b = 0.06

        delta = self.slope_sat_vp(self.input['temp'].values)
        gamma = self.psy_const()

        rs = self.rs()

        vabar = self.avp_from_rel_hum()  # Vapour pressure  *ea*
        r_n = self.net_rad(vabar, rs)  # net radiation
        vas = self.mean_sat_vp_fao56()

        if 'wind_speed' in self.input.columns:
            if self.verbosity > 1:
                print("Wind data have been used for calculating the Penman evaporation.")
            u2 = self._wind_2m(method=wind_method)
            fau = _a + 1.381 * u2
            ea = np.multiply(fau, np.subtract(vas, vabar))

            tmp1 = np.divide(delta, np.add(delta, gamma))
            tmp2 = np.divide(r_n, LAMBDA)
            tmp3 = np.multiply(np.divide(gamma, np.add(delta, gamma)), ea)
            evap = np.add(np.multiply(tmp1, tmp2), tmp3)
        # if wind data is not available
        else:
            if self.verbosity > 1:
                print("Alternative calculation for Penman evaporation without wind data has been performed")

            ra = self._et_rad()
            tmp1 = np.multiply(np.multiply(0.047, rs), np.sqrt(np.add(self.input['temp'].values, 9.5)))
            tmp2 = np.multiply(np.power(np.divide(rs, ra), 2.0), 2.4)
            tmp3 = np.multiply(_b, np.add(self.input['temp'].values, 20))
            tmp4 = np.subtract(1, np.divide(self.input['rh_mean'].values, 100))
            tmp5 = np.multiply(tmp3, tmp4)
            evap = np.add(np.subtract(tmp1, tmp2), tmp5)

        self.post_process(evap, kwargs.get('transform', False))
        return evap


class PenPan(ETBase):
    """
    implementing the PenPan formulation for Class-A pan evaporation as given in Rotstayn et al., 2006

    """
    def __call__(self, **kwargs):

        self.requirements(constants=['lat_dec_deg', 'altitude', 'pen_ap', 'albedo', 'alphaA', 'pan_over_est',
                                     'pan_est'],
                          ts=['temp', 'wind_speed'])

        epan = self.evap_pan()

        et = epan

        if self.cons['pan_over_est']:
            if self.cons['pan_est'] == 'pot_et':
                et = np.multiply(np.divide(et, 1.078), self.cons['pan_coef'])
            else:
                et = np.divide(et, 1.078)

        self.post_process(et, kwargs.get('transform', False))
        return et


class PenmanMonteith(ETBase):
    """
    calculates reference evapotrnaspiration according to Penman-Monteith (Allen et al 1998) equation which is
    also recommended by FAO. The etp is calculated at the time step determined by the step size of input data.
    For hourly or sub-hourly calculation, equation 53 is used while for daily time step equation 6 is used.

    # Requirements
    Following timeseries data is used
      relative humidity
      temperature

    Following constants are used
     lm=None, a_s=0.25, b_s=0.5, albedo=0.23

    http://www.fao.org/3/X0490E/x0490e08.htm#chapter%204%20%20%20determination%20of%20eto
    """
    def __call__(self, *args, **kwargs):
        self.requirements(constants=['lat_dec_deg', 'altitude', 'albedo', 'a_s', 'b_s'],
                          ts=['temp', 'wind_speed', 'jday'])

        wind_2m = self._wind_2m()

        d = self.slope_sat_vp(self.input['temp'].values)
        g = self.psy_const()

        # Mean saturation vapour pressure
        if 'es' not in self.input:
            if self.freq_in_mins == 1440:
                es = self.mean_sat_vp_fao56()
            elif self.freq_in_mins == 60:
                es = self.sat_vp_fao56(self.input['temp'].values)
            elif self.freq_in_mins < 60:   # TODO should sub-hourly be same as hourly?
                es = self.sat_vp_fao56(self.input['temp'].values)
            else:
                raise NotImplementedError
        else:
            es = self.input['es']

        # actual vapour pressure
        ea = self.avp_from_rel_hum()
        if 'vp_def' not in self.input:
            vp_d = es - ea   # vapor pressure deficit
        else:
            vp_d = self.input['vp_def']

        rn = self.net_rad(ea)              # eq 40 in Fao
        _g = self.soil_heat_flux(rn)

        t1 = 0.408 * (d*(rn - _g))
        nechay = d + g*(1 + 0.34 * wind_2m)

        if self.freq_in_mins == 1440:
            t5 = t1 / nechay
            t6 = 900/(self.input['temp']+273) * wind_2m * vp_d * g / nechay
            pet = np.add(t5, t6)

        elif self.freq_in_mins < 1440:  # TODO should sub-hourly be same as hourly?
            t3 = np.multiply(np.divide(37, self.input['temp']+273.0), g)
            t4 = np.multiply(t3, np.multiply(wind_2m, vp_d))
            upar = t1 + t4
            pet = upar / nechay
        else:
            raise NotImplementedError("For frequency of {} minutes, {} method can not be implemented"
                                      .format(self.freq_in_mins, self.name))
        self.post_process(pet, kwargs.get('transform', False))
        return pet


class PriestleyTaylor(ETBase):
    """
    following formulation of Priestley & Taylor, 1972.
    uses: , a_s=0.23, b_s=0.5, alpha_pt=1.26, albedo=0.23
    :param `alpha_pt` Priestley-Taylor coefficient = 1.26 for Priestley-Taylor model (Priestley and Taylor, 1972)
    https://doi.org/10.1175/1520-0493(1972)100<0081:OTAOSH>2.3.CO;2
     """
    def __call__(self, *args, **kwargs):
        self.requirements(constants=['lat_dec_deg', 'altitude', 'alpha_pt', 'albedo'])

        delta = self.slope_sat_vp(self.input['temp'].values)
        gamma = self.psy_const()
        vabar = self.avp_from_rel_hum()    # *ea*
        r_n = self.net_rad(vabar)   # net radiation
        # vas = self.mean_sat_vp_fao56()

        tmp1 = np.divide(delta, np.add(delta, gamma))
        tmp2 = np.multiply(tmp1, np.divide(r_n, LAMBDA))
        tmp3 = np.subtract(tmp2, np.divide(self.soil_heat_flux(), LAMBDA))
        et = np.multiply(self.cons['alpha_pt'], tmp3)

        self.post_process(et, kwargs.get('transform', False))
        return et


class Romanenko(ETBase):

    def __call__(self, *args, **kwargs):
        self.requirements(constants=['lat_dec_deg', 'altitude', 'albedo'],
                          ts=['temp'])
        """
        using formulation of Romanenko
        uses:
          temp
          rel_hum
        There are two variants of it in Song et al 2017.
        https://www.scirp.org/(S(czeh2tfqyw2orz553k1w0r45))/reference/ReferencesPapers.aspx?ReferenceID=2151471
        """
        t = self.input['temp'].values
        vas = self.mean_sat_vp_fao56()
        vabar = self.avp_from_rel_hum()  # Vapour pressure  *ea*

        tmp1 = np.power(np.add(1, np.divide(t, 25)), 2)
        tmp2 = np.subtract(1, np.divide(vabar, vas))
        et = np.multiply(np.multiply(4.5, tmp1), tmp2)

        self.post_process(et, kwargs.get('transform', False))
        return et


class SzilagyiJozsa(ETBase):
    """
    using formulation of Azilagyi, 2007.
     https://doi.org/10.1029/2006GL028708
    """
    def __call__(self, *args, **kwargs):

        self.requirements(constants=['wind_f', 'alphaPT'])

        if self.cons['wind_f'] == 'pen48':
            _a = 2.626
            _b = 0.09
        else:
            _a = 1.313
            _b = 0.06
        alpha_pt = self.cons['alphaPT']  # Priestley Taylor constant

        delta = self.slope_sat_vp(self.input['temp'].values)
        gamma = self.psy_const()

        rs = self.rs()
        vabar = self.avp_from_rel_hum()  # Vapour pressure  *ea*
        r_n = self.net_rad(vabar)   # net radiation
        vas = self.mean_sat_vp_fao56()

        if 'uz' in self.input.columns:
            if self.verbosity > 1:
                print("Wind data have been used for calculating the Penman evaporation.")
            u2 = self._wind_2m()
            fau = _a + 1.381 * u2
            ea = np.multiply(fau, np.subtract(vas, vabar))

            tmp1 = np.divide(delta, np.add(delta, gamma))
            tmp2 = np.divide(r_n, LAMBDA)
            tmp3 = np.multiply(np.divide(gamma, np.add(delta, gamma)), ea)
            et_penman = np.add(np.multiply(tmp1, tmp2), tmp3)
        # if wind data is not available
        else:
            if self.verbosity > 1:
                print("Alternative calculation for Penman evaporation without wind data have been performed")

            ra = self._et_rad()
            tmp1 = np.multiply(np.multiply(0.047, rs), np.sqrt(np.add(self.input['temp'].values, 9.5)))
            tmp2 = np.multiply(np.power(np.divide(rs, ra), 2.0), 2.4)
            tmp3 = np.multiply(_b, np.add(self.input['temp'].values, 20))
            tmp4 = np.subtract(1, np.divide(self.input['rh_mean'].values, 100))
            tmp5 = np.multiply(tmp3, tmp4)
            et_penman = np.add(np.subtract(tmp1, tmp2), tmp5)

        # find equilibrium temperature T_e
        t_e = self.equil_temp(et_penman)

        delta_te = self.slope_sat_vp(t_e)   # slope of vapour pressure curve at T_e
        # Priestley-Taylor evapotranspiration at T_e
        et_pt_te = np.multiply(alpha_pt, np.multiply(np.divide(delta_te, np.add(delta_te, gamma)), np.divide(r_n, LAMBDA)))
        et = np.subtract(np.multiply(2, et_pt_te), et_penman)

        self.post_process(et, kwargs.get('transform', False))
        return et


class Thornthwait(ETBase):
    """calculates reference evapotrnaspiration according to empirical temperature based Thornthwaite
    (Thornthwaite 1948) method. The method actualy calculates both ETP and evaporation. It requires only temperature
    and day length as input. Suitable for monthly values.
    """
    def __call__(self, *args, **kwargs):

        if 'daylight_hrs' not in self.input.columns:
            day_hrs = self.daylight_fao56()
        else:
            day_hrs = self.input['daylight_hrs']

        self.input['adj_t'] = np.where(self.input['temp'].values < 0.0, 0.0, self.input['temp'].values)
        I = self.input['adj_t'].resample('A').apply(custom_resampler)  # heat index (I)
        a = (6.75e-07 * I ** 3) - (7.71e-05 * I ** 2) + (1.792e-02 * I) + 0.49239
        self.input['a'] = a
        a_mon = self.input['a']    # monthly values filled with NaN
        a_mon = pd.DataFrame(a_mon)
        a_ann = pd.DataFrame(a)
        a_monthly = a_mon.merge(a_ann, left_index=True, right_index=True, how='left').fillna(method='bfill')
        self.input['I'] = I
        i_mon = self.input['I']  # monthly values filled with NaN
        i_mon = pd.DataFrame(i_mon)
        i_ann = pd.DataFrame(I)
        i_monthly = i_mon.merge(i_ann, left_index=True, right_index=True, how='left').fillna(method='bfill')

        tmp1 = np.multiply(1.6, np.divide(day_hrs, 12.0))
        tmp2 = np.divide(self.input.index.daysinmonth, 30.0)
        tmp3 = np.multiply(np.power(np.multiply(10.0, np.divide(self.input['temp'].values, i_monthly['I'].values)),
                                    a_monthly['a'].values), 10.0)
        pet = np.multiply(tmp1, np.multiply(tmp2, tmp3))

        # self.input['Thornthwait_daily'] = np.divide(self.input['Thornthwait_Monthly'].values, self.input.index.days_in_month)
        self.post_process(pet, kwargs.get('transform', False))
        return pet


class MortonCRAE(ETBase):
    """
    for monthly pot. ET and wet-environment areal ET and actual ET by Morton 1983.
    :return:

    """


class Papadakis(ETBase):
    """
    Calculates monthly values based on saturation vapor pressure and temperature. Following equation is given by
        eto = 0.5625 * (ea_tmax - ed)
        ea: water pressure corresponding to avg max temperature [KiloPascal].
        ed: saturation water pressure corresponding to the dew point temperature [KiloPascal].
    Rosenberry et al., 2004 presented following equation quoting McGuinnes and Bordne, 1972
        pet = 0.5625 * [es_max - (es_min - 2)] (10/d)
        d = number of days in month
        es = saturated vapour pressure at temperature of air in millibars
    """


class Ritchie(ETBase):
    """
    Given by Jones and Ritchie 1990 and quoted by Valipour, 2005 and Pandey et al., 2016
      et = rs * alpha [0.002322 * tmax + 0.001548*tmin + 0.11223]
    """
    def __call__(self, *args, **kwargs):

        self.requirements(constants=['ritchie_a', 'ritchie_b', 'ritchie_b', 'ritchie_alpha'],
                          ts=['tmin', 'tmax'])
        ritchie_a = self.cons['ritchie_a']
        ritchie_b = self.cons['ritchie_b']
        ritchie_c = self.cons['ritchie_c']
        alpha = self.cons['ritchie_alpha']
        rs = self.rs()
        eto = rs * alpha * [ritchie_a * self.input['tmax'] + ritchie_b * self.input['tmin'] + ritchie_c]

        self.post_process(eto, kwargs.get('transform', False))
        return eto


class Turc(ETBase):
    """
    The original formulation is from Turc, 1961 which was developed for southern France and Africa.
    Pandey et al 2016 mentioned a modified version of Turc quoting Xu et al., 2008, Singh, 2008 and Chen and Chen, 2008.
                         eto = alpha_t * 0.013 T/(T+15) ( (23.8856Rs + 50)/gamma)

    A shorter version of this formula is quoted by Valipour, 2015 quoting Xu et al., 2008
                         eto = (0.3107 * Rs + 0.65) [T alpha_t / (T + 15)]

    Here it is implemented as given (as eq 5) in Alexandris, et al., 2008 which is;
                for rh > 50 %:
                   eto = 0.0133 * [T_mean / (T_mean + 15)] ( Rs + 50)
                for rh < 50 %:
                   eto = 0.0133 * [T_mean / (T_mean + 15)] ( Rs + 50) [1 + (50 - Rh) / 70]

    uses
    :param `k` float or array like, monthly crop coefficient. A single value means same crop coefficient for
          whole year
    :param `a_s` fraction of extraterrestrial radiation reaching earth on sunless days
    :param `b_s` difference between fracion of extraterrestrial radiation reaching full-sun days
             and that on sunless days.
    Turc, L. (1961). Estimation of irrigation water requirements, potential evapotranspiration: a simple climatic
    formula evolved up to date. Ann. Agron, 12(1), 13-49.
    """
    def __call__(self, *args, **kwargs):

        self.requirements(constants=['lat_dec_deg', 'altitude', 'turc_k'],
                          ts=['temp'])

        use_rh = False  # because while testing daily, rhmin and rhmax are given and rhmean is calculated by default
        if 'use_rh' in kwargs:
            use_rh = kwargs['use_rh']

        rs = self.rs()
        ta = self.input['temp'].values
        et = np.multiply(np.multiply(self.cons['turc_k'], (np.add(np.multiply(23.88, rs), 50))),
                         np.divide(ta, (np.add(ta, 15))))

        if use_rh:
            if 'rh_mean' in self.input.columns:
                rh_mean = self.input['rh_mean'].values
                eq1 = np.multiply(np.multiply(np.multiply(self.cons['turc_k'], (np.add(np.multiply(23.88, rs), 50))),
                                              np.divide(ta, (np.add(ta, 15)))),
                                  (np.add(1, np.divide((np.subtract(50, rh_mean)), 70))))
                eq2 = np.multiply(np.multiply(self.cons['turc_k'], (np.add(np.multiply(23.88, rs), 50))),
                                  np.divide(ta, (np.add(ta, 15))))
                et = np.where(rh_mean < 50, eq1, eq2)

        self.post_process(et, kwargs.get('transform', False))
        return et


class Valiantzas(ETBase):
    """
    Djaman 2016 mentioned 2 methods from him, however Valipour 2015 tested 5 variants of his formulations in Iran.
    Ahmad et al 2019 used 6 variants of this method however, Djaman et al., 2017 used 9 of its variants.
    These 9 methods are given below:
    method_1:
      This is equation equation 19 in Valiantzas, 2012. This also does not require wind data.
      eto = 0.0393 * Rs* sqrt(T_avg + 9.5) - (0.19 * Rs**0.6 * lat_rad**0.15)
           + 0.0061(T_avg + 20)(1.12*_avg - T_min - 2)**0.7

    method_2:
      This is equation 14 in Valiantzas, 2012. This does not require wind data. The recommended value of alpha is 0.23.
      eto = 0.0393 * Rs * sqrt(T_avg + 9.5) - (0.19 * Rs**0.6 * lat_rad**0.15) + 0.078(T_avg + 20)(1 - rh/100)

    method_3
      eto = 0.0393 * Rs * sqrt(T_avg + 9.5) - (Rs/Ra)**2 - [(T_avg + 20) * (1-rh/100) * ( 0.024 - 0.1 * Waero)]

    method_4:
      This is equation 35 in Valiantzas 2013c paper and was reffered as Fo-PENM method with using alpha as 0.25.
      eto = 0.051 * (1 - alpha) * Rs * sqrt(T_avg + 9.5) - 2.4 * (Rs/Ra)**2
          + [0.048 * (T_avg + 20) * ( 1- rh/100) * (0.5 + 0.536 * u2)] + (0.00012 * z)

    method_5:
      This is equation 30 in Valiantzas 2013c. This is when no wind data is available.
      eto = 0.0393 * Rs sqrt(T_avg + 9.5)
           - [2.46 * Rs * lat**0.15 / (4 * sin(2 * pi * J / 365 - 1.39) lat + 12)**2 + 0.92]**2
           - 0.024 * (T_avg + 20)(1 - rh/100) - (0.0268 * Rs)
           + (0.0984 * (T_avg + 17)) * (1.03 + 0.00055) * (T_max - T_min)**2 - rh/100

    method_6
      This method is when wind speed and solar radiation data is not available. This is equation 34 in
      Valiantzas, 2013c.
      eto = 0.0068 * Ra * sqrt[(T_avg + 9.5) * (T_max - T_min)]
          - 0.0696 * (T_max - T_min) - 0.024 * (T_avg + 20)
          * [ ((1-rh/100) - 0.00455 * Ra * sqrt(T_max - T_dew)
          + 0.0984 * (T_avg + 17) * (1.03 + 0.0055) * (T_max - T_min)**2)
          - rh/100

    method_7:
      This is equation 27 in Valiantzas, 2013c. This method requires all data. Djaman et al., (by mistake) used 0.0043
      in denominator instead of 0.00043.
      eto = 0.051 * (1-alpha) * Rs * (T_avg + 9.5)**0.5
            - 0.188 * (T_avg + 13) * (Rs/Ra - 0.194)
            * (1 - 0.00015) * (T_avg + 45)**2 * sqrt(rh/100)
            - 0.0165 * Rs * u**0.7 + 0.0585 * (T_avg + 17) * u**0.75
            * {[1 + 0.00043 * (T_max - T_min)**2]**2 - rh/100} / [1 + 0.00043 * (T_max - T_min)**2 + 0.0001*z]

    method_8:
      eto = 0.051 * (1-alpha) * Rs * (T_avg + 9.5)**0.5
            - 2.4 (Rs/Ra)**2 - 2.4 * (T_avg + 20) * (1 - rh/100)
            - 0.0165 * Rs * u**0.7 + 0.0585 * (T_avg + 17) * u**0.75
            * { [1 + 0.00043 (T_max - T_min)**2]**2 - rh/100} / ( 1 + 0.00043 * (T_max - T_min)**2 + (0.0001 * z)

    method_9:
      This must be equation 29 of Valiantzas, 2013c but method 9 in Djaman et al., 2017 used 2.46 instead of 22.46. This
      formulation does not require Ra.
      eto = [0.051 * (1-alpha) * Rs (T_avg + 9.5)**2
            * (2.46 * Rs * lat**0.15 / (4 sin(2 * pi J / 365 - 1.39) * lat + 12)**2 + 0.92)]**2
            - 0.024 * (T_avg + 20) * (1-rh/100) - 0.0165 * Rs * u**0.7
            + 0.0585 * (T_avg + 17) * u**0.75 * {[(1.03 + 0.00055) * (T_max - T_min)**2 - rh/100] + 0.0001*z}
    """
    def __call__(self, method='method_1', **kwargs):

        self.requirements(constants=['valiantzas_alpha'],
                          ts=['temp'])

        alpha = self.cons['valiantzas_alpha']
        z = self.cons['altitute']
        rh = self.input['rh']
        ta = self.input['temp']
        tmin = self.input['tmin']
        tmax = self.input['tmax']
        j = self.input['jday']
        ra = self._et_rad()
        u2 = self._wind_2m()
        w_aero = np.where(rh <= 65.0, 1.067, 0.78)  # empirical weighted factor
        rs_ra = (self.rs() / ra)**2
        tr = tmax - tmin
        tr_sq = tr**2
        lat_15 = self.lat_rad**0.15

        t_sqrt = np.sqrt(ta + 9.5)

        init = 0.0393 * self.rs() * t_sqrt

        rs_fact = 0.19 * (self.rs()**0.6) * lat_15

        t_20 = ta + 20

        rh_factor = 1.0 - (rh/100.0)

        if method == 'method_1':
            eto = init - rs_fact + 0.0061 * t_20 * (1.12 * (ta - tmin) - 2.0)**0.7

        elif method == 'method_2':
            eto = init - rs_fact + (0.078 * t_20 * rh_factor)

        elif method == 'method_3':
            eto = init - rs_ra - (t_20 * rh_factor * (0.024 - 0.1 * w_aero))

        elif method == 'method_4':
            eto = 0.051 * (1 - alpha) * self.rs() * t_sqrt - 2.4 * rs_ra + (0.048 * t_20 * rh_factor * (0.5 + 0.536 * u2)) + (0.00012 * z)

        elif method == 'method_5':
            eto = init

        elif method == 'method_6':
            pass

        elif method == 'method_7':
            pass

        elif method == 'method_8':
            pass

        elif method == 'method_9':
            eto = 0.051 * (1 - alpha) * self.rs() * (ta + 9.5)**2 * (2.46 * self.rs() * lat_15) / (4 * np.sin(2 * 3.14 * j / 365 - 1.39))

        else:
            raise ValueError

        self.post_process(eto, kwargs.get('transform', False))
        return eto

class Oudin(ETBase):
    """
    https://doi.org/10.1016/j.jhydrol.2004.08.026
    """
    pass

class RengerWessolek(ETBase):
    """
    RENGER, M. & WESSOLEK, G. (1990): Auswirkungen von Grundwasserabsenkung und Nutzungsänderungen auf die
     Grundwasserneubildung. – Mit. Inst. für Wasserwesen, Univ. der Bundeswehr München, 386: 295-307.
    """

class Black(ETBase):
    """
     https://doi.org/10.2136/sssaj1969.03615995003300050013x
    """

class McNaughtonBlack(ETBase):
    """
     https://doi.org/10.1029/WR009i006p01579
    """


def custom_resampler(array_like):
    """calculating heat index using monthly values of temperature."""
    return np.sum(np.power(np.divide(array_like, 5.0), 1.514))


def assign_yearly(data, index):
    # TODO for leap years or when first or final year is not complete, the results are not correct immitate
    # https://github.com/cran/Evapotranspiration/blob/master/R/Evapotranspiration.R#L1848
    """ assigns `step` summed data to whole data while keeping the length of data preserved."""
    n_ts = pd.DataFrame(data, index=index, columns=['N'])
    a = n_ts.resample('A').sum()  # annual sum
    ad = a.resample('D').backfill()  # annual sum backfilled
    # for case
    if len(ad) < 2:
        ad1 = pd.DataFrame(np.full(data.shape, np.nan), pd.date_range(n_ts.index[0], periods=len(data), freq='D'),
                           columns=['N'])
        ad1.loc[ad1.index[-1]] = ad.values
        ad2 = ad1.bfill()
        return ad2
    else:
        idx = pd.date_range(n_ts.index[0], ad.index[-1], freq="D")
        n_df_ful = pd.DataFrame(np.full(idx.shape, np.nan), index=idx, columns=['N'])
        n_df_ful['N'][ad.index] = ad.values.reshape(-1, )
        n_df_obj = n_df_ful[n_ts.index[0]: n_ts.index[-1]]
        n_df_obj1 = n_df_obj.bfill()
        return n_df_obj1
