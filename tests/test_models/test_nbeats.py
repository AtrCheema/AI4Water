import unittest

import numpy as np
import tensorflow as tf

np.random.seed(313)


tf_version = int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0'))

if 230 <= tf_version < 250:
    from ai4water.functional import Model
    print(f"Switching to functional API due to tensorflow version {tf.__version__}")
else:
    from ai4water import Model

if tf_version>=210:
    tf.random.set_seed(313)
    tf.compat.v1.disable_eager_execution()
else:
    tf.random.set_random_seed(313)


from tensorflow.keras.models import Model as KModel
from tensorflow.keras.layers import Input, Flatten

from ai4water import Model
from ai4water.datasets import arg_beach
from ai4water.models.tensorflow import NBeats

"""
tf115, subclassing and functional both working
tf21, tf26 subclassing is only working by disabling eager mode while functional is fine
"""

input_features = arg_beach().columns.tolist()[0:-1]
output_features = arg_beach().columns.tolist()[-1:]

class TestNBeats(unittest.TestCase):

    def test_as_layer(self):
        inp = Input(shape=(10, 3))
        nb = NBeats(lookback=10, forecast_length=1,
                    num_exo_inputs=2)(inp)
        out = Flatten()(nb)

        model = KModel(inputs=[inp], outputs=out)
        model.compile(optimizer='adam', loss='mse')

        x = np.random.random((100, 10, 3))
        y = np.random.random((100, 1))
        h = model.fit(x=x, y=y, verbose=0)
        assert model.count_params() == 1236224

        # todo, the following test passes if only this test is run in this fil
        # if tf_version == 115:
        #     assert np.allclose(sum(h.history['loss']), 0.25868335), f"{h.history['loss']}"

        return

    def test_with_ai4water_and_external_data(self):
        x = np.random.random((100, 10, 3))
        y = np.random.random((100, 1))

        model = Model(model={"layers":
                                 {"Input": {"shape": (10, 3)},
                                 "NBeats": {"lookback": 10, "forecast_length": 1, "num_exo_inputs": 2},
                                  "Flatten": {},
                             }},
                      lookback=10,
                      verbosity=0
                      )

        model.fit(x=x, y=y)
        return


    def test_ai4water_with_inherent_data(self):
        model = Model(model={"layers": {
            "NBeats": {"lookback": 10, "forecast_length": 1, "num_exo_inputs": 12},
            "Flatten": {},
        }},
            lookback=10,
            input_features=input_features,
            output_features=output_features,
            forecast_step=1,
            epochs=2,
            verbosity=0
        )

        model.fit(data=arg_beach())
        return

if __name__ == "__main__":

    unittest.main()
