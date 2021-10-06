import os
import unittest
import site  # so that ai4water directory is in path
import sys
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from ai4water.datasets import arg_beach, load_nasdaq
from ai4water import InputAttentionModel, DualAttentionModel


def make_and_run(input_model, _layers=None, lookback=12, epochs=3, **kwargs):

    model = input_model(
        verbosity=0,
        batch_size=64,
        lookback=lookback,
        lr=0.001,
        epochs=epochs,
        train_data='random',
        **kwargs
    )

    _ = model.fit()

    pred_y = model.predict()
    model.interpret()

    return pred_y

class TestModels(unittest.TestCase):

    # InputAttention based model does not conform reproducibility so just testing that it runs.

    def test_InputAttentionModel(self):

        prediction = make_and_run(InputAttentionModel, data=arg_beach())
        self.assertGreater(float(abs(prediction[0].sum())), 0.0)

    # def test_InputAttentionModel_with_drop_remainder(self):
    #
    #     prediction = make_and_run(InputAttentionModel, drop_remainder=True)
    #     self.assertGreater(float(prediction[0].sum()), 0.0)

    def test_DualAttentionModel(self):
        # DualAttentionModel based model

        prediction = make_and_run(
            DualAttentionModel,
            data=load_nasdaq(inputs=['AAL', 'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK'])
        )
        self.assertGreater(float(abs(prediction[0].sum())), 0.0)


if __name__ == "__main__":
    unittest.main()