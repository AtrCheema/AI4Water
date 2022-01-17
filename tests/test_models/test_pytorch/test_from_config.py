
import os
import unittest

from ai4water.datasets import busan_beach
from ai4water.pytorch_models import HARHNModel, IMVModel

lookback = 10
epochs = 50
df = busan_beach()
input_features = df.columns.tolist()[0:-1]
output_features = df.columns.tolist()[-1:]


def test_imvmodel():
    model = IMVModel(input_features=input_features,
                     output_features=output_features,
                     epochs=2,
                     model={'layers': {'hidden_units': 4}},
                     verbosity=0
                     )

    model.fit(data=df)
    config_path = os.path.join(model.path, "config.json")


    new_model = IMVModel.from_config_file(config_path)
    new_model.verbosity = 1
    wfile = os.listdir(os.path.join(model.path, "weights"))[-1]
    wpath = os.path.join(model.path, "weights", wfile)
    new_model.update_weights(wpath)
    return


if __name__ == "__main__":
    unittest.main()



