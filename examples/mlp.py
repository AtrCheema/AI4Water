# this file shows how to build a simple dense layer based model
# the input_features and outputs are columns and are present in the file
import tensorflow as tf

from AI4Water import Model
from AI4Water.utils.datasets import arg_beach

tf.compat.v1.disable_eager_execution()

mlp_model = {'layers': {
    "Dense_0": {'units': 64, 'activation': 'relu'},
    "Flatten": {},
    "Dense_3": {'units': 1},
}}

df = arg_beach()

input_features = list(df.columns)[0:-1]

# column in dataframe to bse used as output/target
outputs = list(df.columns)[-1]

model = Model(data=df,
              batch_size=16,
              lookback=1,
              model = mlp_model,
              inputs=input_features,
              outputs=[outputs],
              lr=0.0001
              )

history = model.fit(indices='random')

y, obs = model.predict()
model.view_model(st=0)
