import re
from weakref import WeakKeyDictionary

import numpy as np


metric_dict = {
    'Exa': 1e18,
    'Peta': 1e15,
    'Tera': 1e12,
    'Giga': 1e9,
    'Mega': 1e6,
    'Kilo': 1e3,
    'Hecto': 1e2,
    'Deca': 1e1,
    None: 1,
    'Deci':  1e-1,
    'Centi': 1e-2,
    'Milli': 1e-3,
    'Micro': 1e-6,
    'Nano': 1e-9,
    'Pico': 1e-12,
    'Femto': 1e-15,
    'Atto': 1e-18
}

time_dict = {
        'Year':   31540000,
        'Month':  2628000,
        'Weak':   604800,
        'Day':    86400,
        'Hour':   3600,
        'Minute': 60,
        'Second': 1,
    }

imperial_dist_dict = {
        'Mile': 63360,
        'Furlong': 7920,
        'Rod': 198,
        'Yard': 36,
        'Foot': 12,
        'Inch': 1
    }


unit_plurals = {
    "Inches": "Inch",
    "Miles": "Mile",
    "Meters": "Meter",
    "Feet": "Foot"
}


def check_plurals(unit):
    if unit in unit_plurals:
        unit = unit_plurals[unit]
    return unit


def split_speed_units(unit):
    dist = unit.split("Per")[0]
    zeit = unit.split("Per")[1]

    # distance and speed may contain underscore ('_') at start or at end e.g. when unit is "Meter_Per_Second"
    # we need to remove all such underscores
    dist = dist.replace("_", "")
    zeit = zeit.replace("_", "")
    if dist in unit_plurals:
        dist = unit_plurals[dist]
    return dist, zeit


class WrongUnitError(Exception):
    def __init__(self, u_type, qty, unit, allowed, prefix=None):
        self.u_type = u_type
        self.qty = qty
        self.unit = unit
        self.allowed = allowed
        self.pre = prefix

    def __str__(self):
        if self.pre is None:
            return '''
*
*   {} unit `{}` for {} is wrong. Use either of {}
*
'''.format(self.u_type, self.unit, self.qty, self.allowed)
# prefix {milli} provided for {input} unit of {temperature} is wrong. {input} unit is {millipascal}, allowed are {}}
        else:
            return """
*
* prefix `{}` provided for {} unit of {} is wrong.
* {} unit is: {}. Allowed units are
* {}.
*
""".format(self.pre, self.u_type, self.qty, self.u_type, self.unit, self.allowed)


def check_converter(converter):
    super_keys = converter.keys()

    for k, v in converter.items():
        sub_keys = v.keys()

        if all(x in super_keys for x in sub_keys):
            a = 1
        else:
            a = 0

        if all(x in sub_keys for x in super_keys):
            b = 1
        else:
            b = 0

        assert a == b


TempUnitConverter = {
    "FAHRENHEIT": {
        "Fahrenheit": lambda fahrenheit: fahrenheit * 1.0,  # fahrenheit to Centigrade
        "Kelvin": lambda fahrenheit: (fahrenheit + 459.67) * 5/9,  # fahrenheit to kelvin
        "Centigrade": lambda fahrenheit: (fahrenheit - 32.0) / 1.8  # fahrenheit to Centigrade
    },
    "KELVIN": {
        "Fahrenheit": lambda kelvin: kelvin * 9/5 - 459.67,  # kelvin to fahrenheit
        "Kelvin": lambda k: k*1.0,     # Kelvin to Kelvin
        "Centigrade": lambda kelvin: kelvin - 273.15  # kelvin to Centigrade}
    },
    "CENTIGRADE": {
        "Fahrenheit": lambda centigrade: centigrade * 1.8 + 32,  # Centigrade to fahrenheit
        "Kelvin": lambda centigrade: centigrade + 273.15,  # Centigrade to kelvin
        "Centigrade": lambda centigrade: centigrade * 1.0}
}


PressureConverter = {
    "Pascal": {  # Pascal to
        "Pascal": lambda pascal: pascal,
        "Bar": lambda pascal: pascal * 1e-5,
        "Atm": lambda pascal: pascal / 101325,
        "Torr": lambda pascal: pascal * 0.00750062,
        "Psi": lambda pascal: pascal / 6894.76,
        "Ta": lambda pascal: pascal * 1.01971621298E-5
    },
    "Bar": {  # Bar to
        "Pascal": lambda bar: bar / 0.00001,
        "Bar": lambda bar: bar,
        "Atm": lambda bar: bar / 1.01325,
        "Torr": lambda bar: bar * 750.062,
        "Psi": lambda bar: bar * 14.503,
        "Ta": lambda bar: bar * 1.01972
    },
    "Atm": {  # Atm to
        "Pascal": lambda atm: atm * 101325,
        "Bar": lambda atm: atm * 1.01325,
        "Atm": lambda atm: atm,
        "Torr": lambda atm: atm * 760,
        "Psi": lambda atm: atm * 14.6959,
        "At": lambda atm: atm * 1.03322755477
    },
    "Torr": {  # Torr to
        "Pascal": lambda torr: torr / 0.00750062,
        "Bar": lambda torr: torr / 750.062,
        "Atm": lambda torr: torr / 760,
        "Torr": lambda tor: tor,
        "Psi": lambda torr: torr / 51.7149,
        "Ta": lambda torr: torr * 0.00135950982242
    },
    "Psi": {  # Psi to
        "Pascal": lambda psi: psi * 6894.76,
        "Bar": lambda psi: psi / 14.5038,
        "Atm": lambda psi: psi / 14.6959,
        "Torr": lambda psi: psi * 51.7149,
        "Psi": lambda psi: psi,
        "Ta": lambda psi: psi * 0.0703069578296,
    },
    "Ta": {   # Ta to
        "Pascal": lambda at: at / 1.01971621298E-5,
        "Bar": lambda at: at / 1.0197,
        "Atm": lambda at: at / 1.03322755477,
        "Torr": lambda at: at / 0.00135950982242,
        "Psi": lambda at: at / 0.0703069578296,
        "Ta": lambda ta: ta
    }
}

DistanceConverter = {
    "Meter": {
        "Meter": lambda meter: meter,
        "Inch": lambda meter: meter * 39.3701
    },
    "Inch": {
        "Meter": lambda inch: inch * 0.0254,
        "Inch": lambda inch: inch
    }
}


class Pressure(object):
    """
    ```python
    p = Pressure(20, "Pascal")
    print(p.MilliBar)    #>> 0.2
    print(p.Bar)         #>> 0.0002
    p = Pressure(np.array([10, 20]), "KiloPascal")
    print(p.MilliBar)    # >> [100, 200]
    p = Pressure(np.array([1000, 2000]), "MilliBar")
    print(p.KiloPascal)  #>> [100, 200]
    print(p.Atm)         # >> [0.98692, 1.9738]
    ```
    """

    def __init__(self, val, input_unit):
        self.val = val
        check_converter(PressureConverter)
        self.input_unit = input_unit

    @property
    def allowed(self):
        return list(PressureConverter.keys())

    @property
    def input_unit(self):
        return self._input_unit

    @input_unit.setter
    def input_unit(self, in_unit):

        self._input_unit = in_unit

    def __getattr__(self, out_unit):
        # pycharm calls this method for its own working, executing default behaviour at such calls
        if out_unit.startswith('_'):
            return self.__getattribute__(out_unit)
        else:
            act_iu, iu_pf = self._preprocess(self.input_unit, "Input")

            act_ou, ou_pf = self._preprocess(out_unit, "Output")

            if act_iu not in self.allowed:
                raise WrongUnitError("Input", self.__class__.__name__, act_iu, self.allowed)
            if act_ou not in self.allowed:
                raise WrongUnitError("output", self.__class__.__name__, act_ou, self.allowed)

            ou_f = PressureConverter[act_iu][act_ou](self.val)

            val = np.round(np.array((iu_pf * ou_f) / ou_pf), 5)
            return val

    def _preprocess(self, given_unit, io_type="Input"):
        split_u = split_units(given_unit)
        if len(split_u) < 1:  # Given unit contained no capital letter so list is empty
            raise WrongUnitError(io_type, self.__class__.__name__, given_unit, self.allowed)

        pf, ou_pf = 1.0, 1.0
        act_u = split_u[0]
        if len(split_u) > 1:
            pre_u = split_u[0]  # prefix of input unit
            act_u = split_u[1]  # actual input unit

            if pre_u in metric_dict:
                pf = metric_dict[pre_u]  # input unit prefix factor
            else:
                raise WrongUnitError(io_type, self.__class__.__name__, act_u, self.allowed, pre_u)

        return act_u, pf


class NotString:
    def __init__(self):
        self.data = WeakKeyDictionary()

    def __get__(self, instance, owner):
        return self.data[instance]

    def __set__(self, instance, value):
        if isinstance(value, list) or isinstance(value, np.ndarray):
            value = np.array(value).astype(np.float32)

        self.data[instance] = value

    def __set_name__(self, owner, name):
        self.name = name


class Temp(object):
    """
    The idea is to write the conversion functions in a dictionary and then dynamically create attribute it the attribute
    is present in converter as key otherwise raise WongUnitError.
    converts temperature among units [kelvin, centigrade, fahrenheit]
    :param `temp`  a numpy array
    :param `input_unit` str, units of temp, should be "Kelvin", "Centigrade" or "Fahrenheit"

    Example:
    ```python
    temp = np.arange(10)
    T = Temp(temp, 'Centigrade')
    T.Kelvin
    >> array([273 274 275 276 277 278 279 280 281 282])
    T.Fahrenheit
    >> array([32. , 33.8, 35.6, 37.4, 39.2, 41. , 42.8, 44.6, 46.4, 48.2])
    T.Centigrade
    >>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ```
    """
    val = NotString()

    def __init__(self, val, input_unit):
        self.val = val
        check_converter(TempUnitConverter)
        self.input_unit = input_unit

    def __getattr__(self, out_unit):
        # pycharm calls this method for its own working, executing default behaviour at such calls
        if out_unit.startswith('_'):
            return self.__getattribute__(out_unit)
        else:
            if out_unit not in TempUnitConverter[self.input_unit]:
                raise WrongUnitError("output", self.__class__.__name__, out_unit, self.allowed)

            val = TempUnitConverter[self.input_unit][str(out_unit)](self.val)
            return val

    @property
    def allowed(self):
        return list(TempUnitConverter.keys())

    @property
    def input_unit(self):
        return self._input_unit

    @input_unit.setter
    def input_unit(self, in_unit):
        if in_unit.upper() == 'CELCIUS':
            in_unit = 'CENTIGRADE'
        if in_unit.upper() not in self.allowed:
            raise WrongUnitError("Input", self.__class__.__name__, in_unit, self.allowed)
        self._input_unit = in_unit.upper()


class Distance(object):
    """
    unit converter for distance or length between different imperial and/or metric units.
    ```python
    t = Distance(np.array([2.0]), "Mile")
    np.testing.assert_array_almost_equal(t.Inch, [126720], 5)
    np.testing.assert_array_almost_equal(t.Meter, [3218.688], 5)
    np.testing.assert_array_almost_equal(t.KiloMeter, [3.218688], 5)
    np.testing.assert_array_almost_equal(t.CentiMeter, [321869], 0)
    np.testing.assert_array_almost_equal(t.Foot, [10560.], 5)

    t = Distance(np.array([5000]), "MilliMeter")
    np.testing.assert_array_almost_equal(t.Inch, [196.85039], 5)
    np.testing.assert_array_almost_equal(t.Meter, [5.0], 5)
    np.testing.assert_array_almost_equal(t.KiloMeter, [0.005], 5)
    np.testing.assert_array_almost_equal(t.CentiMeter, [500.0], 5)
    np.testing.assert_array_almost_equal(t.Foot, [16.404199], 5)
    ```
    """

    def __init__(self, val, input_unit):
        self.val = val
        self.input_unit = input_unit

    @property
    def allowed(self):
        return list(imperial_dist_dict.keys()) + ['Meter']

    @property
    def input_unit(self):
        return self._input_unit

    @input_unit.setter
    def input_unit(self, in_unit):
        self._input_unit = in_unit

    def __getattr__(self, out_unit):
        # pycharm calls this method for its own working, executing default behaviour at such calls
        if out_unit.startswith('_'):
            return self.__getattribute__(out_unit)
        else:
            act_iu, iu_pf = self._preprocess(self.input_unit, "Input")

            act_ou, ou_pf = self._preprocess(out_unit, "Output")

            act_iu = check_plurals(act_iu)
            act_ou = check_plurals(act_ou)

            if act_iu not in self.allowed:
                raise WrongUnitError("Input", self.__class__.__name__, act_iu, self.allowed)
            if act_ou not in self.allowed:
                raise WrongUnitError("output", self.__class__.__name__, act_ou, self.allowed)

            out_in_meter = self._to_meters(ou_pf, act_ou)  # get number of meters in output unit
            input_in_meter = self.val * iu_pf  # for default case when input unit has Meter in it

            # if input unit is in imperial system, first convert it into inches and then into meters
            if act_iu in imperial_dist_dict:
                input_in_inches = imperial_dist_dict[act_iu] * self.val * iu_pf
                input_in_meter = DistanceConverter['Inch']['Meter'](input_in_inches)

            val = input_in_meter / out_in_meter

            return val

    def _to_meters(self, prefix, actual_unit):
        meters = prefix
        if actual_unit != "Meter":
            inches = imperial_dist_dict[actual_unit] * prefix
            meters = DistanceConverter['Inch']['Meter'](inches)
        return meters

    def _preprocess(self, given_unit, io_type="Input"):
        # TODO unit must not be split based on capital letters, it is confusing and prone to erros
        split_u = split_units(given_unit)
        if len(split_u) < 1:  # Given unit contained no capital letter so list is empty
            raise WrongUnitError(io_type, self.__class__.__name__, given_unit, self.allowed)

        pf, ou_pf = 1.0, 1.0
        act_u = split_u[0]
        if len(split_u) > 1:
            pre_u = split_u[0]  # prefix of input unit
            act_u = split_u[1]  # actual input unit

            if pre_u in metric_dict:
                pf = metric_dict[pre_u]  # input unit prefix factor
            else:
                raise WrongUnitError(io_type, self.__class__.__name__, act_u, self.allowed, pre_u)

        return act_u, pf


class Time(object):

    """
    ```python
    t = Time(np.array([100, 200]), "Hour")
    np.testing.assert_array_almost_equal(t.Day, [4.16666667, 8.33333333], 5)
    t = Time(np.array([48, 24]), "Day")
    np.testing.assert_array_almost_equal(t.Minute, [69120., 34560.], 5)
    ```
    """
    def __init__(self, val, input_unit):
        self.val = val
        self.input_unit = input_unit

    @property
    def allowed(self):
        return list(time_dict.keys())

    @property
    def input_unit(self):
        return self._input_unit

    @input_unit.setter
    def input_unit(self, in_unit):
        self._input_unit = in_unit

    def __getattr__(self, out_unit):
        # pycharm calls this method for its own working, executing default behaviour at such calls
        if out_unit.startswith('_'):
            return self.__getattribute__(out_unit)
        else:
            act_iu, iu_pf = self._preprocess(self.input_unit, "Input")

            act_ou, ou_pf = self._preprocess(out_unit, "Output")

            if act_iu not in self.allowed:
                raise WrongUnitError("Input", self.__class__.__name__, act_iu, self.allowed)
            if act_ou not in self.allowed:
                raise WrongUnitError("output", self.__class__.__name__, act_ou, self.allowed)

            in_sec = time_dict[act_iu] * self.val * iu_pf
            val = in_sec / (time_dict[act_ou]*ou_pf)
            return val

    def _preprocess(self, given_unit, io_type="Input"):
        split_u = split_units(given_unit)
        if len(split_u) < 1:  # Given unit contained no capital letter so list is empty
            raise WrongUnitError(io_type, self.__class__.__name__, given_unit, self.allowed)

        pf, ou_pf = 1.0, 1.0
        act_u = split_u[0]
        if len(split_u) > 1:
            pre_u = split_u[0]  # prefix of input unit
            act_u = split_u[1]  # actual input unit

            if pre_u in metric_dict:
                pf = metric_dict[pre_u]  # input unit prefix factor
            else:
                raise WrongUnitError(io_type, self.__class__.__name__, act_u, self.allowed, pre_u)

        return act_u, pf


class Speed(object):
    """
    converts between different units using Distance and Time classes which convert
    distance and time units separately. This class both classes separately and
    then does the rest of the work.
    ```python
    s = Speed(np.array([10]), "KiloMeterPerHour")
    np.testing.assert_array_almost_equal(s.MeterPerSecond, [2.77777778], 5)
    np.testing.assert_array_almost_equal(s.MilePerHour, [6.21371192], 5)
    np.testing.assert_array_almost_equal(s.FootPerSecond, [9.11344415], 5)
    s = Speed(np.array([14]), "FootPerSecond")
    np.testing.assert_array_almost_equal(s.MeterPerSecond, [4.2672], 5)
    np.testing.assert_array_almost_equal(s.MilePerHour, [9.54545], 5)
    np.testing.assert_array_almost_equal(s.KiloMeterPerHour, [15.3619], 4)
    s = Speed(np.arange(10), 'KiloMetersPerHour')
    o = np.array([ 0. , 10.936, 21.872, 32.808, 43.744, 54.680, 65.616 , 76.552, 87.489, 98.425])
    np.testing.assert_array_almost_equal(s.InchesPerSecond, o, 2)
    ```
    """

    def __init__(self, val, input_unit):
        self.val = val
        self.input_unit = input_unit

    @property
    def input_unit(self):
        return self._input_unit

    @input_unit.setter
    def input_unit(self, in_unit):
        if '_' in in_unit:
            raise ValueError("remove underscore from units {}".format(in_unit))
        self._input_unit = in_unit

    def __getattr__(self, out_unit):
        # pycharm calls this method for its own working, executing default behaviour at such calls
        if out_unit.startswith('_'):
            return self.__getattribute__(out_unit)
        else:
            in_dist, in_zeit = split_speed_units(self.input_unit)
            out_dist, out_zeit = split_speed_units(out_unit)

            d = Distance(np.array([1]), in_dist)
            dist_f = getattr(d, out_dist)  # distance factor
            t = Time(np.array([1]), in_zeit)
            time_f = getattr(t, out_zeit)   # time factor

            out_val = self.val * (dist_f/time_f)
            return out_val


def split_units(unit):
    """splits string `unit` based on capital letters"""
    return re.findall('[A-Z][^A-Z]*', unit)
