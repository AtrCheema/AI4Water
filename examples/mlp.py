# this file shows how to build a simple dense layer based model
# the input_features and outputs are columns and are present in the file
import tensorflow as tf

from ai4water import Model
from ai4water.utils.datasets import arg_beach

tf.compat.v1.disable_eager_execution()  # because we want to view model

mlp_model = {'layers': {
    "Dense_0": {'units': 64, 'activation': 'relu'},
    "Flatten": {},
    "Dense_3": 1,
}}

df = arg_beach()

input_features = list(df.columns)[0:-1]

# column in dataframe to bse used as output/target
outputs = list(df.columns)[-1]

model = Model(data=df,
              batch_size=16,
              lookback=1,
              model = mlp_model,
              input_features=input_features,
              output_features=[outputs],
              lr=0.0001,
              train_data='random',
              )

history = model.fit()

y, obs = model.predict()
model.view_model()
