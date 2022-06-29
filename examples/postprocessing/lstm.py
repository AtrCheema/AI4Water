"""
=======================
Visualizing inside LSTM
=======================
"""

from ai4water import Model
from ai4water.models import LSTM
from ai4water.datasets import busan_beach
from ai4water.postprocessing import Visualize

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

#%%
data = busan_beach()
data.shape

#%%
input_features = data.columns.tolist()[0:-1]
output_features = data.columns.tolist()[-1:]

#%%

lookback = 14

#%%

model = Model(
    model=LSTM(
        units=13,
        input_shape=(lookback, len(input_features)),
        dropout=0.2
    ),
    input_features=input_features,
    output_features=output_features,
    ts_args={'lookback':lookback},
    epochs=2
)

#%%
model.fit(data=data)

#%%
visualizer = Visualize(model, save=False)

#%%
# gradients
#---------------------

visualizer.activations(layer_names="LSTM_0", examples_to_use=range(20, 50))

#%%
# gradients of activations
#-------------------------
visualizer.activation_gradients("LSTM_0", examples_to_use=range(20, 50))

#%%
# weights
#--------

visualizer.weights(layer_names="LSTM_0")


# %%
# gradients of weights
#---------------------

visualizer.weight_gradients(layer_names="LSTM_0")
