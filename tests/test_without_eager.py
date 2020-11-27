import tensorflow as tf
import os

tf.compat.v1.disable_eager_execution()

import site  # so that dl4seq directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)))

from dl4seq.utils import make_model
from dl4seq import InputAttentionModel, DualAttentionModel
import unittest

import pandas as pd
from inspect import getsourcefile
from os.path import abspath

def make_and_run(input_model, _layers=None, lookback=12, epochs=1, **kwargs):

    data_config, nn_config, _ = make_model(batch_size=64,
                                           lookback=lookback,
                                           lr=0.001,
                                           epochs=epochs,
                                           **kwargs)
    nn_config['layers'] = _layers

    file_path = abspath(getsourcefile(lambda: 0))
    dpath = os.path.join(os.path.dirname(os.path.dirname(file_path)), "data")
    fname = os.path.join(dpath, "nasdaq100_padding.csv")
    df = pd.read_csv(fname)

    model = input_model(
        data_config=data_config,
        nn_config=nn_config,
        data=df,
        verbosity=0
    )

    model.build()

    _ = model.train(indices='random')

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