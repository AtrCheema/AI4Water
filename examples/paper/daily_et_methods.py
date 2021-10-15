import os
import site   # so that dl4seq directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ai4water.ETUtil import HargreavesSamani, ETBase, Penman, PriestleyTaylor
from ai4water.datasets import CAMELS_AUS
from ai4water.utils.utils import process_axis


units = {'tmin': 'Centigrade',
         'tmax': 'Centigrade',
         'rh_min': 'percent',
         'rh_max': 'percent',
         'solar_rad': 'MegaJourPerMeterSquare'}

constants = dict()
constants['cts'] = 0.0055
constants['pen_ap'] = 2.4
constants['pan_ap'] = 2.4
constants['turc_k'] = 0.013
constants['wind_f'] = 'pen48'
constants['albedo'] = 0.23
constants['a_s'] = 0.23
constants['b_s'] = 0.5
constants['abtew_k'] = 0.52
constants['ct'] = 0.025
constants['tx'] = 3
constants['pan_coeff'] = 0.71
constants['pan_over_est'] = False
constants['pan_est'] = 'pot_et'
constants['CH'] = 0.12
constants['Ca'] = 0.001013
constants['surf_res'] = 70
constants['alphaPT'] = 1.28
constants['lat_rad'] = -37.293684
constants['lat_dec_deg'] = 63.506144
constants['altitude'] = 249
constants['alphaA'] = 0.14
constants['alpha_pt'] = 1.26

dataset = CAMELS_AUS(path=r"D:\mytools\AI4Water\AI4Water\utils\datasets\data\CAMELS\CAMELS_AUS")

inputs = ['mslp_SILO',
          'radiation_SILO',
          'rh_tmax_SILO',
          'tmin_SILO',
          'tmax_SILO',
          'rh_tmin_SILO',
          'vp_deficit_SILO',
          'vp_SILO',
          'et_morton_point_SILO'
          ]

data = dataset.fetch(['224206'], dynamic_attributes=inputs, categories=None, st='19700101', en='20141231')
data = data['224206']
data = data.rename(columns={
    'tmin_SILO': 'tmin',
    'tmax_SILO': 'tmax',
    'radiation_SILO': 'sol_rad',
    'vapor_pressure': 'vp_SILO',
    'rh_tmin_SILO': 'rh_min',
    'rh_tmax_SILO': 'rh_max',
    'vp_deficit_SILO': 'vp_def'
})

data1 = data[['tmin', 'tmax', 'sol_rad', 'rh_min', 'rh_max']]

eto_model = HargreavesSamani(data1, units=units, constants=constants, verbosity=2)
et_hs = eto_model()

eto_model = ETBase(data1, units=units, constants=constants, verbosity=2)
et_jh = eto_model()

eto_model = Penman(data1, units=units, constants=constants, verbosity=2)
et_penman = eto_model()
et_penman = np.where(et_penman<0.0, np.nan, et_penman)

eto_model = PriestleyTaylor(data1, units=units, constants=constants, verbosity=2)
et_pt = eto_model()

plt.close('all')
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='all')

process_axis(ax1,data['et_morton_point_SILO'], c= np.array([0.63797563, 0.05503074, 0.07078517]),
             label='Morton', ms=0.5, y_label='ETP (mm)', leg_ms=4, leg_pos="upper right")

process_axis(ax2, pd.Series(et_hs, data.index), ms=0.5, c=np.array([0.70670405, 0.71039014, 0.54375619]),
         label='Hargreaves and Samani', y_label='ETP (mm)', leg_ms=4, leg_pos="upper right")

process_axis(ax3, et_jh, ms=0.5, c=np.array([0.27822191, 0.7608274, 0.89536561]),
         label='Jensen and Haise', y_label='ETP (mm)', leg_ms=4, leg_pos="upper right")

process_axis(ax4, pd.Series(et_penman, index=data.index), ms=0.5, c=np.array([0.39865179, 0.61455622, 0.57515074]),
         label='Penman', y_label='ETP (mm)', leg_ms=4, leg_pos="upper right")

plt.show()