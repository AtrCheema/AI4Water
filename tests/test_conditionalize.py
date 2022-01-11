import unittest
import os
import sys
import site   # so that ai4water directory is in path
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
from ai4water.models.tensorflow.private_layers import Conditionalize
from ai4water.datasets import busan_beach



class TestConditionalize(unittest.TestCase):

    def test_multiple_conds(self):
        i = Input(shape=(10, 3))
        h = Input(shape=(14,))
        c = Conditionalize(32)([h, h, h])
        rnn = LSTM(32)(i, initial_state=[c,c])
        o = Dense(1)(rnn)

        model = Model(inputs=[i, h], outputs=o)
        model.compile('adam', loss='mse')
        self.assertEqual(model.count_params(), 6085)

    def test_single_cond(self):
        i = Input(shape=(10, 3))
        h = Input(shape=(14,))
        c = Conditionalize(32)(h)
        rnn = LSTM(32)(i, initial_state=[c, c])
        o = Dense(1)(rnn)

        model = Model(inputs=[i, h], outputs=o)
        model.compile('adam', loss='mse')
        self.assertEqual(model.count_params(), 5121)

    def todo_test_ai4water_model(self):
        #
        # model = Model(model={"layers":
        #                          {"Input": {"shape": (10, 3)},
        #                           "Conditionalize": {"units": 32, "name": "h_state"},
        #                           "LSTM": {"config": {"units": 32},
        #                                    "inputs": ["Input"],
        #                                    'call_args': {'initial_state': ['h_state', 'h_state']}
        #                                    },
        #                           "Dense": {"units": 1}
        #                           }},
        #               data=arg_beach()
        #               )
        return


if __name__ == "__main__":

    unittest.main()