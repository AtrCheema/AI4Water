# this file tests that given activations are working both as layers as well as activation functions withing a layer
import unittest
import os
import tensorflow as tf
from inspect import getsourcefile
from os.path import abspath
import pandas as pd

import site   # so that dl4seq directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

from dl4seq import Model
from dl4seq.utils import make_model


def make_model_local(**kwargs):
    config = make_model(
        epochs=2,
        lookback=1,
        **kwargs
    )
    return config

file_path = abspath(getsourcefile(lambda:0))
dpath = os.path.join(os.path.join(os.path.dirname(os.path.dirname(file_path)), "dl4seq"), "data")
fname = os.path.join(dpath, "nasdaq100_padding.csv")

df = pd.read_csv(fname)

version = tf.__version__.split('.')[0] + tf.__version__.split('.')[1]

class TestActivations(unittest.TestCase):
    def test_as_layers(self):

        layers = {}

        for lyr in ['PRELU', "RELU", "TANH", "ELU", "LEAKYRELU", "THRESHOLDRELU", "SELU", 'sigmoid', 'hardsigmoid',
                    'crelu',
                    'relu6', 'softmax', 'softplus', 'softsign', 'swish']:
            layers[lyr] = {'config': {}}

        layers["Dense"] = {'config': {'units': 1}}
        layers["reshape"] = {'config': {'target_shape': (1, 1)}}

        model = Model(epochs=2,
                      lookback=1,
                      layers=layers,
                      data=df,
                      verbosity=0
                      )

        val = {
            '21': [0.09297575600513237, 0.09400989675627566],
            '23': [0.0870760977268219, 0.1053781732916832]
        }

        history = model.fit()
        if int(tf.__version__.split('.')[0]) > 1:
            for t,p in zip(history.history['val_loss'], val[version]):
                self.assertAlmostEqual(t,p, 5)
        return

    def test_as_fns(self):
        layers = {}
        for idx, act_fn in enumerate(['tanh', 'relu', 'elu', 'leakyrelu', 'crelu', 'selu', 'relu6', 'sigmoid',
                                      'hardsigmoid', 'swish']):

            layers["Dense_" + str(idx)] = {'config': {'units': 1, 'activation': act_fn}}

        layers["reshape"] = {'config': {'target_shape': (1, 1)}}

        model = Model(epochs=2,
                      lookback=1,
                      layers=layers,
                      data=df,
                      verbosity=0
                      )

        history = model.fit()
        val = {
            '21': [0.8971164431680119, 0.7911620726129243],
            '23': [0.10781528055667877, 0.09552989155054092]
        }

        if int(tf.__version__.split('.')[0]) > 1:
            for t,p in zip(history.history['val_loss'], val[version]):
                self.assertAlmostEqual(t,p, 5)
        return


if __name__ == "__main__":
    unittest.main()