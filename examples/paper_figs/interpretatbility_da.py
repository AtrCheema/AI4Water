import os
import site
dl4seq_dir = "D:\\mytools\\AI4Water"
site.addsitedir(dl4seq_dir)

print(f'adding dl4seq_dir {dl4seq_dir}')

import random

import numpy as np
import tensorflow as tf

from AI4Water import DualAttentionModel
from AI4Water.utils.datasets import CAMELS_AUS
from AI4Water.utils.utils import dateandtime_now
from AI4Water.hyper_opt import HyperOpt, Categorical, Real, Integer
from AI4Water.utils.utils import Jsonize, plot, prepare_data
from AI4Water.utils import Visualizations

print(tf.test.is_gpu_available())
print(tf.__version__)
seed = 313
np.random.seed(seed)
random.seed(seed)


dataset = CAMELS_AUS()

inputs = ['et_morton_point_SILO',
           'precipitation_AWAP',
           'tmax_AWAP',
           'tmin_AWAP',
           'vprp_AWAP',
           'rh_tmax_SILO',
           'rh_tmin_SILO'
          ]

outputs = ['streamflow_MLd']

stations = ['401203']
data = dataset.fetch(stations, dynamic_attributes=inputs+outputs, categories=None)

for k,v in data.items():
    print(k, v.shape)
    data[k] = v['19700101':'20141231']

for k,v in data.items():
    if k in stations:
        target = v['streamflow_MLd']
        target[target < 0] = np.nan
        target[target == 0.0] = np.nan
        v['streamflow_MLd'] = target
        #v = v.fillna(-99)
        data[k] = v
        print(k, v.isna().sum().sum())

prefix = f"da_{dateandtime_now()}"

def objective_fn(**suggestion):

    suggestion = Jsonize(suggestion)()


    _model = DualAttentionModel(#model={'layers': layers},
                    data=data['401203'],
                    inputs=inputs,
                    outputs=outputs,
                    patience=50,
                    epochs=500,
                    lookback=int(suggestion['lookback']),
                    lr=suggestion['lr'],
                    batch_size=int(suggestion['batch_size']),
                    transformation=[{'method': 'robust', 'features': inputs},
                                    {'method': 'log', "replace_nans": True, "replace_zeros": True, 'features': outputs},
                                    {'method': 'robust', "replace_nans": True, 'features': outputs}
                                    ],
                    prefix=prefix
                    )

    # model.impute('interpolate', {'method': 'linear'}, cols=outputs)
    # model.impute(cols=outputs, method='SimpleImputer', imputer_args={})
    #
    h = _model.fit(indices='random')
    min_val_loss = float(np.min(h.history['val_loss']))
    print(f'with {suggestion} min val loss is {min_val_loss}')
    #return min_val_loss
    return _model


model = objective_fn(hidden_units=100, lookback=15, lr=0.0001, batch_size=64)
