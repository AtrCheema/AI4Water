import site
dl4seq_dir = "D:\\mytools\\AI4Water"
site.addsitedir(dl4seq_dir)

print(f'adding dl4seq_dir {dl4seq_dir}')

import numpy as np
import tensorflow as tf

from AI4Water import DualAttentionModel
from AI4Water.utils.datasets import CAMELS_AUS
from AI4Water.utils.utils import dateandtime_now
from AI4Water.utils.utils import Jsonize
from AI4Water.utils.visualizations import Interpret


tf.compat.v1.disable_eager_execution()


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
        data[k] = v
        print(k, v.isna().sum().sum())

prefix = f"da_{dateandtime_now()}"

def objective_fn(**suggestion):

    suggestion = Jsonize(suggestion)()

    _model = DualAttentionModel(
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

    h = _model.fit(indices='random')
    min_val_loss = float(np.min(h.history['val_loss']))
    print(f'with {suggestion} min val loss is {min_val_loss}')

    return _model

model = objective_fn(hidden_units=100, lookback=15, lr=0.0001, batch_size=64)

model.predict(indices=model.train_indices, prefix='train')

model.predict(indices=model.test_indices, prefix='test')

Interpret(model)
