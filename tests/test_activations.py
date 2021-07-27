# this file tests that given activations are working both as layers as well as activation functions withing a layer
import unittest
import os
import site   # so that AI4Water directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

import tensorflow as tf

from AI4Water import Model
from AI4Water.utils.datasets import load_nasdaq


df = load_nasdaq()

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
                      model={'layers': layers},
                      data=df,
                      transformation='minmax',
                      verbosity=0
                      )

        val_losses = {
            '21_nt': [0.09297575600513237, 0.09400989675627566],
            '23_posix': [0.0870760977268219, 0.1053781732916832],
            '24_posix': [0.0870760977268219, 0.1053781732916832],
            '21_posix': [0.09297575600513237, 0.095427157656984],
            '20_posix': [0.09297575600513237, 0.095427157656984],
            '23_nt': [0.0870760977268219, 0.1053781732916832],
            '24_nt': [0.0870760977268219, 0.1053781732916832],
            '25_nt_subclassing': [0.049483511596918106, 0.04080097749829292],
            '25_nt_functional': [0.049483511596918106, 0.04080097749829292],
            '24_nt_functional': [0.04127213731408119, 0.04080097749829292],
        }

        history = model.fit()
        if int(tf.__version__.split('.')[0]) > 1:
            for t,p in zip(history.history['val_loss'], val_losses[f"{version}_{os.name}_{model.api}"]):
                self.assertAlmostEqual(t, p, 2)
        return

    def test_as_fns(self):
        layers = {}
        for idx, act_fn in enumerate(['tanh', 'relu', 'elu', 'leakyrelu', 'crelu', 'selu', 'relu6', 'sigmoid',
                                      'hardsigmoid', 'swish']):

            layers["Dense_" + str(idx)] = {'config': {'units': 1, 'activation': act_fn}}

        layers["reshape"] = {'config': {'target_shape': (1, 1)}}

        model = Model(epochs=2,
                      lookback=1,
                      model={'layers': layers},
                      data=df,
                      transformation='minmax',
                      verbosity=0
                      )

        history = model.fit()
        val_losses = {
            '21_nt': [0.8971164431680119, 0.7911620726129243],
            '23_nt': [0.10781528055667877, 0.09552989155054092],
            '24_nt': [0.10781528055667877, 0.09552989155054092],
            '23_posix': [0.10781528055667877, 0.09552989155054092],
            '24_posix': [0.10781528055667877, 0.09552989155054092],
            '21_posix': [0.10688107734841351, 0.0938945620801094],
            '20_posix': [0.8971164431680119, 0.10688107734841351],
            '25_nt_subclassing': [0.025749117136001587, 0.040039800107479095],
            '25_nt_functional': [0.025749117136001587, 0.040039800107479095],
            '24_nt_functional': [0.05252218618988991, 0.040802694857120514],
        }

        if int(tf.__version__.split('.')[0]) > 1:
            for t,p in zip(history.history['val_loss'], val_losses[f"{version}_{os.name}_{model.api}"]):
                self.assertAlmostEqual(t,p, 2)
        return


if __name__ == "__main__":
    unittest.main()