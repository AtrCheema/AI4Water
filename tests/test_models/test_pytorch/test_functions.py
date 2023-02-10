
import unittest

from ai4water import Model
from ai4water.models import MLP, LSTM
from ai4water.datasets import busan_beach


data = busan_beach()
input_features = data.columns.tolist()[0:-1]
output_features = data.columns.tolist()[-1:]


class TestFunctions(unittest.TestCase):

    def test_mlp_reg(self):
        model = Model(model=MLP(32, 2, (13,), backend="pytorch"),
                  backend="pytorch",
                      input_features = input_features,
                      output_features = output_features,
                      epochs=1,
                      verbosity=0
                      )
        model.fit(data=data)
        return

    def test_lstm_reg(self):
        model = Model(model=LSTM(32, 2, (5, 13), backend="pytorch"),
                  backend="pytorch",
                      input_features = input_features,
                      output_features = output_features,
                      ts_args={'lookback':5},
                      epochs=1,
                      verbosity=0
                      )
        model.fit(data=data)


if __name__ == "__main__":
    unittest.main()