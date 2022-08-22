"""
========================================
Building neural netowrks with tensorflow
========================================
"""

from ai4water import Model
from ai4water.datasets import busan_beach
import pandas as pd

# sphinx_gallery_thumbnail_number = -2

###########################################################


data = busan_beach()

layers ={
    "Input": {'config': {'shape': (14, 13)}},
    "LSTM": {'config': {'units': 14, 'activation': "elu"}},
    "Dense": 1
}
model = Model(
    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1:],
    model={'layers':layers},
    ts_args={"lookback": 14},
    lr=0.009919,
    batch_size=8,
    split_random=True,
    train_fraction=1.0,
    x_transformation="zscore",
    y_transformation={"method": "log", "replace_zeros": True, "treat_negatives": True},
    epochs=200
)

#%%
h = model.fit(data=data)

type(h)
#%%

p = model.predict(data=data)

#%%

p = model.predict_on_training_data(data=data, plots=['regression', 'residual', 'prediction'])

#%%

t, p = model.predict_on_all_data(data=data, return_true=True)