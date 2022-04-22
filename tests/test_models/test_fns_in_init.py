import unittest

from ai4water import Model
from ai4water.datasets import busan_beach
from ai4water.models import MLP, LSTM, CNN


data = busan_beach()
input_features = data.columns.tolist()[0:-1]
output_features = data.columns.tolist()[-1:]


class TestModels(unittest.TestCase):

    def test_mlp(self):
        model = Model(model=MLP(32),
                      input_features=input_features,
                      output_features=output_features,
                      epochs=1,
                      verbosity=0
                      )
        assert model.category == "DL"
        return

    def test_lstm(self):
        model = Model(model=LSTM(32),
                      input_features=input_features,
                      output_features=output_features,
                      ts_args={'lookback': 5},
                      verbosity=0)
        assert model.category == "DL"
        return

    def test_cnn(self):
        model = Model(model=CNN(32, 2),
                      input_features=input_features,
                      output_features=output_features,
                      ts_args={'lookback': 5},
                      verbosity=0)
        assert model.category == "DL"
        return


if __name__ == "__main__":
    unittest.main()
