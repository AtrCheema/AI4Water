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

arg_beach = arg_beach()
arg_input_features = arg_beach.columns.tolist()[0:-1]
arg_output_features = arg_beach.columns.tolist()[-1:]

nasdaq= load_nasdaq(inputs=['AAL', 'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK'])
nasdaq_input_features = nasdaq.columns.tolist()[0:-1]
nasdaq_output_features = nasdaq.columns.tolist()[-1:]


def make_and_run(input_model, data, _layers=None, lookback=12, batch_size=64, epochs=3, **kwargs):

    model = input_model(
        verbosity=0,
        batch_size=batch_size,
        lookback=lookback,
        lr=0.001,
        epochs=epochs,
        train_data='random',
        **kwargs
    )

    _ = model.fit(data=data)

    pred_y = model.predict(data='training')
    eval_score = model.evaluate(data='training')
    pred_y = model.predict()

    return pred_y

class TestModels(unittest.TestCase):

    # InputAttention based model does not conform reproducibility so just testing that it runs.

    def test_InputAttentionModel(self):

        prediction = make_and_run(InputAttentionModel,
                                  data=arg_beach,
                                  input_features=arg_input_features,
                                  output_features=arg_output_features)
        self.assertGreater(float(abs(prediction[0].sum())), 0.0)
        return

    # def test_InputAttentionModel_with_drop_remainder(self):
    #
    #     prediction = make_and_run(InputAttentionModel, drop_remainder=True)
    #     self.assertGreater(float(prediction[0].sum()), 0.0)

    def test_DualAttentionModel(self):
        # DualAttentionModel based model

        prediction = make_and_run(
            DualAttentionModel,
            data=nasdaq,
            input_features=nasdaq_input_features,
            output_features=nasdaq_output_features
        )

        self.assertGreater(float(abs(prediction[0].sum())), 0.0)
        return

    def test_da_without_prev_y(self):
        prediction = make_and_run(
            DualAttentionModel,
            data=arg_beach,
            teacher_forcing=False,
            batch_size=8,
            drop_remainder=True,
            input_features=arg_input_features,
            output_features=arg_output_features
        )
        self.assertGreater(float(abs(prediction[0].sum())), 0.0)
        return



if __name__ == "__main__":
    unittest.main()