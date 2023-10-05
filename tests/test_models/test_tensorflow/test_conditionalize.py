import unittest
import os
import sys
import site   # so that ai4water directory is in path

import numpy as np

ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

from ai4water import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model as KModel
from ai4water.models._tensorflow.private_layers import Conditionalize
from ai4water.datasets import busan_beach


class TestConditionalize(unittest.TestCase):

    def test_multiple_conds(self):
        i = Input(shape=(10, 3))
        h = Input(shape=(14,))
        c = Conditionalize(32)([h, h, h])
        rnn = LSTM(32)(i, initial_state=[c,c])
        o = Dense(1)(rnn)

        model = KModel(inputs=[i, h], outputs=o)
        model.compile('adam', loss='mse')
        self.assertEqual(model.count_params(), 6085)

    def test_single_cond(self):
        i = Input(shape=(10, 3))
        h = Input(shape=(14,))
        c = Conditionalize(32)(h)
        rnn = LSTM(32)(i, initial_state=[c, c])
        o = Dense(1)(rnn)

        model = KModel(inputs=[i, h], outputs=o)
        model.compile('adam', loss='mse')
        self.assertEqual(model.count_params(), 5121)

    def test_ai4water_model(self):

        model = Model(model={"layers": {
                "Input": {"shape": (10, 3)},
                "Input_cat": {"shape": (10,)},
                 "Conditionalize": {"config": {"units": 32, "name": "h_state"},
                                    "inputs": "Input_cat"},
                 "LSTM": {"config": {"units": 32},
                          "inputs": "Input",
                          'call_args': {'initial_state': ['h_state', 'h_state']}},
                 "Dense": {"units": 1}}},
            ts_args={"lookback": 10}, verbosity=0, epochs=1)

        x1 = np.random.random((100, 10, 3))
        x2 = np.random.random((100, 10))
        y = np.random.random(100)
        h = model.fit(x=[x1, x2], y=y)
        return


if __name__ == "__main__":

    unittest.main()