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


class TestPytorchModels(unittest.TestCase):
    def test_hrhnmodel(self):
        model = HARHNModel(data=arg_beach().dropna(),  # doping na will be wrong but it is just for test purpose
                           teacher_forcing=True,
                           epochs=3,
                           model={'layers': {'n_conv_lyrs': 3, 'enc_units': 4, 'dec_units': 4}},
                           verbosity=0
                           )
        model.fit()
        p = model.predict()
        assert isinstance(p, np.ndarray)
        s = model.evaluate('training')
        assert math.isfinite(s)
        return

    def test_imvmodel(self):
        model = IMVModel(data=arg_beach(),
                         val_data="same",
                         val_fraction=0.0,
                         epochs=2,
                         lr=0.0001,
                         batch_size=4,
                         train_data='random',
                         transformation=[
                             {'method': 'minmax', 'features': list(arg_beach().columns)[0:-1]},
                             {'method': 'log2', 'features': ['tetx_coppml'], 'replace_zeros': True, 'replace_nans': True}
                         ],
                         model={'layers': {'hidden_units': 4}},
                         verbosity=0
                         )

        model.fit()
        p = model.predict()
        assert isinstance(p, np.ndarray)
        val_score = model.evaluate('training')
        assert math.isfinite(val_score)
        model.interpret()
        return

if __name__ == "__main__":
    unittest.main()