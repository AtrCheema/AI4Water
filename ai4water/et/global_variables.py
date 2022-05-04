
from ai4water.backend import np, random

# Latent heat of vaporisation [MJ.Kg-1]
LAMBDA = 2.45

#: Solar constant [ MJ m-2 min-1]
SOLAR_CONSTANT = 0.0820

SB_CONS = 3.405e-12  # per minute


ALLOWED_COLUMNS = ['temp', 'tmin', 'tmax', 'rel_hum', 'sol_rad', 'rn', 'wind_speed', 'es', 'sunshine_hrs',
                   'rh_min', 'rh_max', 'rh_mean', 'et_rad', 'cs_rad', 'half_hr', 'tdew', 'daylight_hrs']


def random_array(_length, lower=0.0, upper=0.5):
    """This creates a random array of length `length` and the floats vary between `lower` and `upper`."""
    rand_array = np.zeros(_length)
    for i in range(_length):
        rand_array[i] = random.uniform(lower, upper)
    return rand_array


def Colors(_name):

    colors = {'sol_rad': np.array([0.84067393, 0.66066663, 0.22888342]),
              'rel_hum': np.array([0.50832319, 0.53790088, 0.17337983]),
              'rh_mean': np.array([0.34068018, 0.65708722, 0.19501699]),
              'tmax': np.array([0.94943837, 0.34234137, 0.03188675]),
              'tmin': np.array([0.91051202, 0.65414968, 0.48220781]),
              'ra': np.array([0.64027147, 0.75848599, 0.59123481]),
              'rn': np.array([0.68802968, 0.38316639, 0.13177745]),
              'ea': np.array([0.19854081, 0.44556471, 0.35620562]),
              'wind_speed': np.array([0.40293934, 0.51160837, 0.1293387]),
              'jday': np.array([0.88396109, 0.14081036, 0.44402598]),
              'half_hr': np.array([0.53974518, 0.48519598, 0.11808065]),
              'sha': np.array([0.67873225, 0.67178641, 0.01953063])}

    if 'Daily' in _name:
        return np.array([0.63797563, 0.05503074, 0.07078517])
    elif 'Minute' in _name:
        return np.array([0.27822191, 0.7608274, 0.89536561])
    elif 'Hourly' in _name:
        return np.array([0.70670405, 0.71039014, 0.54375619])
    elif 'Monthly' in _name:
        return np.array([0.39865179, 0.61455622, 0.57515074])
    elif 'Yearly' in _name:
        return np.array([0.81158386, 0.182704, 0.93272506])
    elif 'Sub_hourly' in _name:
        return np.array([0.1844271, 0.70936978, 0.53026012])
    elif _name in colors:
        return colors[_name]
    else:
        c = random_array(3, 0.01, 0.99)
        print('for ', _name, c)
        return c


default_constants = {
    'lat': {'desc': 'latitude in decimal degrees', 'def_val': None, 'min': -90, 'max': 90},
    'long': {'desc': 'longitude in decimal degrees', 'def_val': None},
    'a_s': {'desc': 'fraction of extraterrestrial radiation reaching earth on sunless days', 'def_val': 0.23},
    'b_s': {'desc': """difference between fracion of extraterrestrial radiation reaching full-sun days and that 
                    on sunless days""", 'def_val': 0.5},
    'albedo': {'desc': """a numeric value between 0 and 1 (dimensionless), albedo of evaporative surface representing
     the portion of the incident radiation that is reflected back at the surface. Default is 0.23 for
    surface covered with short reference crop, which is for the calculation of Matt-Shuttleworth
     reference crop evaporation.""", 'def_val': 0.23, 'min': 0, 'max': 1},
    'abtew_k': {'desc': 'a coefficient used in Abtew', 'def_val': 0.52},
    'CH': {'desc': 'crop height', 'def_val': 0.12},
    'Ca': {'desc': 'Specific heat of air', 'def_val': 0.001013},
    'surf_res': {'desc': "surface resistance (s/m) depends on the type of reference crop. Default is 70 for short"
                         " reference crop", 'def_val': 70, 'min': 0, 'max': 9999},
    'alphaPT': {'desc': 'Priestley-Taylor coefficient', 'def_val': 1.26},
    'ritchie_a': {'desc': "Coefficient for Richie method", 'def_val': 0.002322},
    'ritchie_b': {'desc': "Coefficient for Richie method", 'def_val': 0.001548},
    'ritchie_c': {'desc': "Coefficient for Ritchie Method", 'def_val': 0.11223}
}
