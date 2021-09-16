# this file tests that given activations are working both as layers as well as activation functions withing a layer
import unittest
import os
import site   # so that ai4water directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

import tensorflow as tf

if 230 <= int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 250:
    from ai4water.functional import Model
    print(f"Switching to functional API due to tensorflow version {tf.__version__}")
else:
    from ai4water import Model

from ai4water.datasets import load_nasdaq


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
            '20_posix_functional': [0.09297575600513237, 0.095427157656984],
            '21_posix_functional': [0.09297575600513237, 0.095427157656984],
            '23_posix_functional': [ 0.04948361963033676, 0.043167594820261],
            '24_posix_functional': [0.04948361963033676, 0.043167594820261],
            '25_posix_subclassing': [0.0870760977268219, 0.1053781732916832],
            '25_posix_functional': [0.0870760977268219, 0.1053781732916832],
            '26_posix_functional': [0.04948361963033676, 0.043167594820261],
            '26_posix_subclassing': [0.0870760977268219, 0.1053781732916832],
            '23_nt': [0.0870760977268219, 0.1053781732916832],
            '24_nt': [0.0870760977268219, 0.1053781732916832],
            '21_nt_subclassing': [0.038706957432889856, 0.043269214379164936],
            '23_nt_functional': [0.049483511596918106, 0.043167594820261],
            '25_nt_subclassing': [0.049483511596918106, 0.04080097749829292],
            '26_nt_subclassing': [0.049483511596918106, 0.04080097749829292],
            '25_nt_functional': [0.049483511596918106, 0.04080097749829292],
            '24_nt_functional': [0.049483511596918106, 0.04080097749829292],
        }

        history = model.fit()
        if int(tf.__version__.split('.')[0]) > 1:
            print(f"{version}_{os.name}_{model.api}")
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
            '20_posix_functional': [0.8971164431680119, 0.10688107734841351],
            '21_posix_functional': [0.10688107734841351, 0.0938945620801094],
            '23_posix_functional': [0.025749117136001587, 0.037755679339170456],
            '24_posix_functional': [0.025749117136001587, 0.037755679339170456],
            '25_posix_functional': [0.10781528055667877, 0.09552989155054092],
            '26_posix_functional': [0.025749117136001587, 0.037755679339170456],

            '21_nt_subclassing': [0.02578388827527157, 0.03749684586972845],
            '23_nt_functional': [0.025749117136001587, 0.037755679339170456],
            '24_nt': [0.10781528055667877, 0.09552989155054092],
            '25_nt_subclassing': [0.025749117136001587, 0.040039800107479095],
            '26_nt_subclassing': [0.025749117136001587, 0.040039800107479095],
            '25_nt_functional': [0.025749117136001587, 0.040039800107479095],
            '24_nt_functional': [0.025749117136001587, 0.040802694857120514],
        }

        if int(tf.__version__.split('.')[0]) > 1:
            print(f"{version}_{os.name}_{model.api}")
            for t,p in zip(history.history['val_loss'], val_losses[f"{version}_{os.name}_{model.api}"]):
                self.assertAlmostEqual(t,p, 2)
        return


if __name__ == "__main__":
    unittest.main()