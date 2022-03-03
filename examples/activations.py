"""
===========
Activations
===========
"""

from ai4water import Model
import tensorflow as tf
from ai4water.postprocessing import Visualize

tf.compat.v1.disable_eager_execution()
import pandas as pd
import numpy as np

# create dummy data
i = np.linspace(-20, 20, 100)
o = i + 1
data = pd.DataFrame(np.concatenate([i.reshape(-1,1), o.reshape(-1, 1)], axis=1),
                    columns=['input', 'output'])

data.head()

##############################################################

activation_layers = ['PReLU', "relu", "tanh", "ELU", "LeakyReLU",
                     "ThresholdedReLU", "selu", 'sigmoid', 'hardsigmoid', 'crelu',
            'relu6', 'softmax', 'softplus', 'softsign', "swish"]

layers = {
    "Input": {"config": {"shape": (1, ), "name": "CustomInputs"}},
    "PReLU": {"config": {},
              "inputs": "CustomInputs"},
    "relu": {"config": {},
              "inputs": "CustomInputs"},
    "tanh": {"config": {},
              "inputs": "CustomInputs"},
    "ELU": {"config": {},
              "inputs": "CustomInputs"},
    "LeakyReLU": {"config": {},
              "inputs": "CustomInputs"},
    "ThresholdedReLU": {"config": {},
              "inputs": "CustomInputs"},
    "selu": {"config": {},
              "inputs": "CustomInputs"},
    "sigmoid": {"config": {},
              "inputs": "CustomInputs"},
    "hardsigmoid": {"config": {},
              "inputs": "CustomInputs"},
    "crelu": {"config": {},
              "inputs": "CustomInputs"},
    "relu6": {"config": {},
              "inputs": "CustomInputs"},
    "softmax": {"config": {},
              "inputs": "CustomInputs"},
    "softplus": {"config": {},
              "inputs": "CustomInputs"},
    "softsign": {"config": {},
              "inputs": "CustomInputs"},
    "swish": {"config": {},
              "inputs": "CustomInputs"},
    "Concatenate": {"config": {},
               "inputs": activation_layers},
    "Dense": {"config": {"units": 1}},
          }

model = Model(model={'layers':layers},
            input_features = ['input'],
            output_features = ['output'],
            epochs=1,
            batch_size=32,
            #drop_remainder=True,
            shuffle=False)

###########################################################

model.fit(data=data)


###########################################################


vis = Visualize(model=model)

vis.activations(layer_names="CustomInputs", show=True)

###########################################################


vis.activations(layer_names="PReLU", show=True)

###########################################################

vis.activations(layer_names="relu", show=True)

###########################################################

vis.activations(layer_names="tanh", show=True)

###########################################################

vis.activations(layer_names="ELU", show=True)

###########################################################

vis.activations(layer_names="LeakyReLU", show=True)

###########################################################

vis.activations(layer_names="ThresholdedReLU", show=True)

###########################################################

vis.activations(layer_names="selu", show=True)

###########################################################

vis.activations(layer_names="sigmoid", show=True)

###########################################################

vis.activations(layer_names="hardsigmoid", show=True)

###########################################################

vis.activations(layer_names="crelu", show=True)

###########################################################

vis.activations(layer_names="relu6", show=True)

###########################################################

vis.activations(layer_names="softmax", show=True)

###########################################################

vis.activations(layer_names="softplus", show=True)

###########################################################

vis.activations(layer_names="softsign", show=True)

###########################################################

vis.activations(layer_names="swish", show=True)
