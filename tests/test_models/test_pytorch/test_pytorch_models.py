import os
import sys
import site   # so that AI4Water directory is in path
import math
import unittest

import numpy as np

ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

from ai4water.datasets import arg_beach
from ai4water.pytorch_models import HARHNModel, IMVModel

lookback = 10
epochs = 50
df = arg_beach()
input_features = df.columns.tolist()[0:-1]
output_features = df.columns.tolist()[-1:]


class TestPytorchModels(unittest.TestCase):

    def test_hrhnmodel(self):
        model = HARHNModel(teacher_forcing=True,
                           input_features=input_features,
                           output_features=output_features,
                           epochs=3,
                           model={'layers': {'n_conv_lyrs': 3, 'enc_units': 4, 'dec_units': 4}},
                           verbosity=0
                           )
        # doping na will be wrong but it is just for test purpose
        model.fit(data=arg_beach().dropna())
        p = model.predict()
        assert isinstance(p, np.ndarray)
        s = model.evaluate(data='training')
        assert math.isfinite(s)
        return

    def test_imvmodel(self):

        model = IMVModel(val_data="same",
                         input_features=input_features,
                         output_features=output_features,
                         val_fraction=0.0,
                         epochs=2,
                         lr=0.0001,
                         batch_size=4,
                         train_data='random',
                         x_transformation={'method': 'minmax', 'features': list(arg_beach().columns)[0:-1]},
                         y_transformation={'method': 'log2', 'features': ['tetx_coppml'], 'replace_zeros': True},
                         model={'layers': {'hidden_units': 4}},
                         verbosity=0
                         )

        model.fit(data=arg_beach())
        p = model.predict()
        assert isinstance(p, np.ndarray)
        val_score = model.evaluate(data='training')
        assert math.isfinite(val_score)
        model.interpret()
        return


if __name__ == "__main__":
    unittest.main()