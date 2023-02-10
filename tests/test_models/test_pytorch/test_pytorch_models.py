import os
import sys
import site   # so that AI4Water directory is in path
import math
import unittest

import numpy as np

ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

from ai4water.datasets import busan_beach
from ai4water.pytorch_models import HARHNModel, IMVModel

lookback = 10
epochs = 50
df = busan_beach()
input_features = df.columns.tolist()[0:-1]
output_features = df.columns.tolist()[-1:]


class TestPytorchModels(unittest.TestCase):

    def test_hrhnmodel(self):
        model = HARHNModel(teacher_forcing=True,
                           input_features=input_features,
                           output_features=output_features,
                           epochs=3,
                           model={'layers': {'n_conv_lyrs': 3, 'enc_units': 4, 'dec_units': 4}},
                           verbosity=0,
                           #use_cuda=False,
                           ts_args={"lookback": 4}
                           )
        # doping na will be wrong but it is just for test purpose
        model.fit(data=df.dropna())
        p = model.predict_on_test_data(data=df.dropna())
        assert isinstance(p, np.ndarray)
        s = model.evaluate_on_training_data(data=df.dropna())
        assert math.isfinite(s)
        return

    def test_imvmodel(self):

        model = IMVModel( # use_cuda=False, todo
                         input_features=input_features,
                         output_features=output_features,
                         val_fraction=0.0,
                         epochs=2,
                         lr=0.0001,
                         batch_size=4,
                         split_random=True,
                         x_transformation={'method': 'minmax', 'features': input_features},
                         y_transformation={'method': 'log2', 'features': ['tetx_coppml'], 'replace_zeros': True},
                         model={'layers': {'hidden_units': 4}},
                         verbosity=0,
            ts_args={"lookback": 4}
                         )

        model.fit(data=df)
        p = model.predict_on_training_data(data=df)
        assert isinstance(p, np.ndarray)
        val_score = model.evaluate_on_training_data(data=df)
        assert math.isfinite(val_score)
        model.interpret()
        return


if __name__ == "__main__":
    unittest.main()