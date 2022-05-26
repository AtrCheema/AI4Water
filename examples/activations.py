"""
===========
Activations
===========
"""

# This notebook shows how to use activation functions as activation within
# layer or as activation functions.

from ai4water import Model
import tensorflow as tf
from ai4water.postprocessing import Visualize

tf.compat.v1.disable_eager_execution()

import pandas as pd
import numpy as np

assert tf.__version__ == "2.7.0"

print(np.__version__)

##############################################################

# create dummy data
i = np.linspace(-20, 20, 100)
o = i + 1
data = pd.DataFrame(np.concatenate([i.reshape(-1,1), o.reshape(-1, 1)], axis=1),
                    columns=['input', 'output'])

data.head()

##############################################################
# activations as layers
#-------------------------

# After 'Input' layer, all activations are used as tensorflow "Layer".

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
            shuffle=False)

###########################################################

model.fit(data=data)

vis = Visualize(model=model)

vis.activations(layer_names="CustomInputs", show=True)

###########################################################