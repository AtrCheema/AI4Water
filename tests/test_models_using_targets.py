import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from dl4seq.utils import make_model
from dl4seq.models import InputAttentionModel, DualAttentionModel
import unittest

import pandas as pd


def make_and_run(input_model, _layers=None, lookback=12, epochs=4, **kwargs):

    data_config, nn_config, total_intervals = make_model(batch_size=16,
                                                         lookback=lookback,
                                                         lr=0.001,
                                                         epochs=epochs,
                                                         **kwargs)
    nn_config['layers'] = _layers

    df = pd.read_csv("../data/nasdaq100_padding.csv")

    model = input_model(
        data_config=data_config,
        nn_config=nn_config,
        data=df,
        verbosity=0
    )

    model.build_nn()

    _ = model.train_nn(indices='random')

    _, pred_y = model.predict(use_datetime_index=False)

    return pred_y

class TestModels(unittest.TestCase):

    # InputAttention based model does not conform reproducibility so just testing that it runs.

    def test_InputAttentionModel(self):

        prediction = make_and_run(InputAttentionModel)
        self.assertGreater(float(prediction[0].sum()), 0.0)

    def test_DualAttentionModel(self):
        # DualAttentionModel based model

        prediction = make_and_run(DualAttentionModel)
        self.assertGreater(float(prediction[0].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()