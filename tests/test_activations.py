# this file tests that given activations are working both as layers as well as activation functions withing a layer
import unittest

from dl4seq.main import Model
from dl4seq.utils import make_model

import pandas as pd

data_config, nn_config, _ = make_model()

nn_config['epochs'] = 2
data_config['lookback'] = 1

df = pd.read_csv("../data/nasdaq100_padding.csv")

class TestActivations(unittest.TestCase):
    def test_as_layers(self):

        layers = {}

        for lyr in ['PRELU', "RELU", "TANH", "ELU", "LEAKYRELU", "THRESHOLDRELU", "SELU", 'sigmoid', 'hardsigmoid',
                    'crelu',
                    'relu6', 'softmax', 'softplus', 'softsign', 'swish']:
            layers[lyr] = {'config': {}}

        layers["Dense"] = {'config': {'units': 1}}

        nn_config['layers'] = layers

        model = Model(data_config,
                      nn_config,
                      df,
                      verbosity=0
                      )

        model.build_nn()

        history = model.train_nn()
        for t,p in zip(history.history['val_loss'], [0.1004275307059288, 0.09452582150697708]):
            self.assertAlmostEqual(t,p, 5)

    def test_as_fns(self):
        layers = {}
        for idx, act_fn in enumerate(['tanh', 'relu', 'elu', 'leakyrelu', 'crelu', 'selu', 'relu6', 'sigmoid',
                                      'hardsigmoid', 'swish']):

            layers["Dense_" + str(idx)] = {'config': {'units': 1, 'activation': act_fn}}


        nn_config['layers'] = layers
        model = Model(data_config,
                      nn_config,
                      df,
                      verbosity=0
                      )

        model.build_nn()

        history = model.train_nn()
        for t,p in zip(history.history['val_loss'], [0.8970817923545837, 0.7911913394927979]):
            self.assertAlmostEqual(t,p, 5)


if __name__ == "__main__":
    unittest.main()