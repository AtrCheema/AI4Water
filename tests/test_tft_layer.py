
import os
import unittest
import site  # so that AI4Water directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

import tensorflow as tf
import numpy as np

from AI4Water.models.tft_layer import TemporalFusionTransformer
from AI4Water import Model

tf.compat.v1.disable_eager_execution()
np.random.seed(313)
tf.random.set_seed(313)

num_encoder_steps = 168
params = {
    'total_time_steps': 192,  # 8 * 24,
    'num_encoder_steps': num_encoder_steps,  # 7 * 24,
    'num_inputs': 5,
    'category_counts': [2],
    'input_obs_loc': [0],  # leave empty if not available
    'static_input_loc': [4],  # if not static inputs, leave this empty
    'known_regular_inputs': [1, 2, 3],
    'known_categorical_inputs': [0],  # leave empty if not applicable
    'hidden_units': 8,
    'dropout_rate': 0.1,
    'num_heads': 4,
    'stack_size': 1,
    'use_cudnn': False,
    'future_inputs': True,
    'return_sequences': True,
}

output_size = 1
quantiles = [0.25, 0.5, 0.75]
n = 500
tot_steps = int(params['total_time_steps'])
x = np.random.random((n, tot_steps, int(params['num_inputs'])))
y = np.random.random((n, tot_steps - num_encoder_steps, len(quantiles)))

class Test_TFT(unittest.TestCase):

    def test_as_model(self):
        params['total_time_steps'] = 192
        params['future_inputs'] = True

        time_dim = params['total_time_steps']
        if params['total_time_steps'] > params['num_encoder_steps']:
            if not params['future_inputs']:
                time_dim = params['total_time_steps'] - params['num_encoder_steps']

        # Inputs.
        all_inputs = tf.keras.layers.Input(
            shape=(
                time_dim,
                params['num_inputs'],
            ))

        tft = TemporalFusionTransformer(**params)
        transformer_layer = tft(all_inputs)

        if params['total_time_steps'] == num_encoder_steps:
            outputs = tf.keras.layers.Dense(output_size * len(quantiles))(transformer_layer[:, -1])
        else:
            outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_size * len(quantiles)))(
                transformer_layer[Ellipsis, num_encoder_steps:, :])

        model = tf.keras.Model(inputs=all_inputs, outputs=outputs)

        model.compile(optimizer='adam', loss='mse')
        num_paras = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
        self.assertEqual(num_paras, 7411)

        h = model.fit(x, y, validation_split=0.3)

        # The following test is passing if only one test is run in this file and it fails if multiple tests are run
        # Thus if will fail due to randomness.
        #np.testing.assert_almost_equal(h.history['loss'][0], 0.4319019560303007)  # 0.4319019560303007/0.49118527361324854
        return

    def test_as_layer(self):
        params['total_time_steps'] = 192
        params['future_inputs'] = True
        layers = {
            "Input": {"config": {"shape": (params['total_time_steps'], params['num_inputs'])}},
            "TemporalFusionTransformer": {"config": params},
            "lambda": {"config": tf.keras.layers.Lambda(lambda _x: _x[Ellipsis, num_encoder_steps:, :])},
            "TimeDistributed": {"config": {}},
            "Dense": {"config": {"units": output_size * len(quantiles)}}
        }
        model = Model(model={'layers':layers},
                      inputs=['inp1', 'inp2', 'inp3', 'inp4', 'inp5'],
                      outputs=['out1', 'out2', 'out3'],
                      verbosity=0)
        h = model._model.fit(x=x,y=y, validation_split=0.3)  # TODO, this h['loss'] is different than what we got from other test
        #np.testing.assert_almost_equal(h.history['loss'][0], 0.4319019560303007)

        num_paras = np.sum([np.prod(v.get_shape().as_list()) for v in model._model.trainable_variables])
        self.assertEqual(num_paras, 7411)
        return

    def test_as_layer_for_nowcasting(self):
        params['total_time_steps'] = num_encoder_steps
        params['future_inputs'] = False
        layers = {
            "Input": {"config": {"shape": (params['total_time_steps'], params['num_inputs'])}},
            "TemporalFusionTransformer": {"config": params},
            "lambda": {"config": tf.keras.layers.Lambda(lambda _x: _x[Ellipsis, -1, :])},
            "Dense": {"config": {"units": output_size * len(quantiles)}}
        }
        model = Model(model={'layers':layers},
                      inputs=['inp1', 'inp2', 'inp3', 'inp4', 'inp5'],
                      outputs=['out1', 'out2', 'out3'],
                      verbosity=1)
        x = np.random.random((n,  int(params['total_time_steps']), int(params['num_inputs'])))
        y = np.random.random((n, len(quantiles), 1))
        model._model.fit(x=x,y=y, validation_split=0.3)

        num_paras = np.sum([np.prod(v.get_shape().as_list()) for v in model._model.trainable_variables])
        assert model.forecast_len == 1
        assert model.forecast_step == 0
        assert model.outs == len(quantiles)
        self.assertEqual(num_paras, 5484)
        return

if __name__ == "__main__":
    unittest.main()
