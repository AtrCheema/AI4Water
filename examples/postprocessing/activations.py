"""
===========
Activations
===========
"""

# This notebook shows how to use activation functions as activation within
# layer or as activation functions.

# sphinx_gallery_thumbnail_number = 2
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from ai4water.functional import Model
from ai4water.postprocessing import Visualize


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

vis = Visualize(model=model, save=False, show=True)

vis.activations(layer_names="CustomInputs", examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names="PReLU", examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names="relu", examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names="tanh", examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names="ELU", examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names="LeakyReLU", examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names="ThresholdedReLU", examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names="selu", examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names="sigmoid", examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names="hardsigmoid", examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names="crelu", examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names="relu6", examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names="softmax", examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names="softplus", examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names="softsign", examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names="swish", examples_to_use=np.arange(50))

##############################################################
# As activation functions within layers
#---------------------------------------

layers = {"Input": {"config": {"shape": (1, ), "name": "CustomInputs"}}}

activation_layers = []
for idx, act_fn in enumerate(['tanh', 'relu', 'elu', 'leakyrelu', 'crelu', 'selu', 'relu6', 'sigmoid',
                              'hardsigmoid', 'swish']):

    # initializing the kernel/weight matrix of each dense layer with ones, so that it does not affect first forward propagation
    layers["Dense_" + act_fn] = {'config': {'units': 1, 'activation': act_fn, "kernel_initializer": "ones", "name": act_fn},
                                   'inputs': "CustomInputs"}
    activation_layers.append(act_fn)

layers["Concatenate"] = {"config": {"name": "concat"},
                    "inputs": activation_layers}

layers["Dense"] = {'config': {'units': 1}}

# We are building a neural network with 10 dense/fully connected layers
# each layer has a separate activation function. The name of the layer
# is same as the activation function used in the layer.

from ai4water.functional import Model

model = Model(model={'layers':layers},
            input_features = ['input'],
            output_features = ['output'],
            epochs=1,
              shuffle=False)

###########################################################

model.fit(data=data)

vis = Visualize(model=model)
vis.activations(layer_names="CustomInputs", examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names=activation_layers[0], examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names=activation_layers[1], examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names=activation_layers[2], examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names=activation_layers[3], examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names=activation_layers[4], examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names=activation_layers[5], examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names=activation_layers[6], examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names=activation_layers[7], examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names=activation_layers[8], examples_to_use=np.arange(50))

###########################################################

vis.activations(layer_names=activation_layers[9], examples_to_use=np.arange(50))