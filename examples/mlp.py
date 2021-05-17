# this file shows how to build a simple dense layer based model
# the input_features and outputs are columns and are present in the file
import pandas as pd
import tensorflow as tf

from AI4Water import Model
from AI4Water.utils.datasets import load_30min

tf.compat.v1.disable_eager_execution()

mlp_model = {'layers': {
    "Dense_0": {'units': 64, 'activation': 'relu'},
    "Flatten": {},
    "Dense_3": {'units': 1},
}}

input_features = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input8',
              'input11']
# column in dataframe to bse used as output/target
outputs = ['target7']

df = load_30min()

model = Model(data=df,
              batch_size=16,
              lookback=1,
              model = mlp_model,
              inputs=input_features,
              outputs=outputs,
              lr=0.0001
              )

history = model.fit(indices='random')

y, obs = model.predict()
model.view_model(st=0)
