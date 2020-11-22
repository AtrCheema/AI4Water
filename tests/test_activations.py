# this file tests that given activations are working both as layers as well as activation functions withing a layer

from models import Model
from utils import make_model

import pandas as pd
import numpy as np

data_config, nn_config, _ = make_model()

layers = {}

for lyr in ['PRELU', "RELU", "TANH", "ELU", "LEAKYRELU", "THRESHOLDRELU", "SELU", 'sigmoid', 'hardsigmoid', 'crelu',
            'relu6', 'softmax', 'softplus', 'softsign', 'swish']:
    layers[lyr] = {'config': {}}

layers["Dense"] = {'config': {'units': 1}}

nn_config['layers'] = layers
nn_config['epochs'] = 2
data_config['lookback'] = 1

df = pd.read_csv("../data/nasdaq100_padding.csv")

model = Model(data_config, nn_config, df)

model.build_nn()

history = model.train_nn()

np.testing.assert_array_almost_equal(history.history['val_loss'], [0.1004275307059288, 0.09452582150697708])

layers = {}
for idx, act_fn in enumerate(['tanh', 'relu', 'elu', 'leakyrelu', 'crelu', 'selu', 'relu6', 'sigmoid',
                              'hardsigmoid', 'swish']):

    layers["Dense_" + str(idx)] = {'config': {'units': 1, 'activation': act_fn}}


nn_config['layers'] = layers
model = Model(data_config, nn_config, df)

model.build_nn()

history = model.train_nn()
np.testing.assert_array_almost_equal(history.history['val_loss'], [0.8970817923545837, 0.7911913394927979])
