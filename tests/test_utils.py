import pandas as pd
import numpy as np
import unittest
import os

import site   # so that dl4seq directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

from dl4seq.utils import make_model
from dl4seq import Model

examples = 2000
ins = 5
outs = 1
in_cols = ['input_'+str(i) for i in range(5)]
out_cols = ['output']
cols=  in_cols + out_cols

data = pd.DataFrame(np.arange(int(examples*len(cols))).reshape(-1,examples).transpose(),
                    columns=cols,
                    index=pd.date_range("20110101", periods=examples, freq="D"))

lookback=7
batch_size=16
layers = {
    "LSTM": {"config": {"units": 1}},
    "Dense": {"config": {"units": 1}}
}

data_config, nn_config, _ = make_model(
    layers=layers,
    inputs=in_cols,
    outputs=out_cols,
    batch_size=batch_size,
    lookback=lookback,
    normalize=False,
    epochs=1
)

model = Model(data_config,
              nn_config,
              data=data,
              verbosity=0
              )

def build_train_predict(_model):
    model.build_nn()
    x, y = model.train_data(st=10, en=500)

    model.train_nn()
    model.predict()

    return x,y


class TestUtils(unittest.TestCase):

    """
    Given following sample consisting of input/output paris
    input, input, output1, output2, output 3
    1,     11,     21,       31,     41
    2,     12,     22,       32,     42
    3,     13,     23,       33,     43
    4,     14,     24,       34,     44
    5,     15,     25,       35,     45
    6,     16,     26,       36,     46
    7,     17,     27,       37,     47

    if we predict
    27, 37, 47     outs=3, forecast_length=1,  horizon/forecast_step=0,

    if we predict
    28, 38, 48     outs=3, forecast_length=1,  horizon/forecast_step=1,

    if we predict
    27, 37, 47
    28, 38, 48     outs=3, forecast_length=2,  horizon/forecast_step=0,

    if we predict
    28, 38, 48
    29, 39, 49   outs=3, forecast_length=3,  horizon/forecast_step=1,
    30, 40, 50

    if we predict
    38            outs=1, forecast_length=3, forecast_step=0
    39
    40

    if we predict
    39            outs=1, forecast_length=1, forecast_step=2

    if we predict
    39            outs=1, forecast_length=3, forecast_step=2
    40
    41

    output/target/label shape
    (examples, outs, forecast_length)

    Additonally I also build, train and predict from the model so that it is confirmed that everything works
    with different input/output shapes.
    """

    def test_ForecastStep0_Outs(self):
        # test x, y output is fom t not t+1 and number of outputs are 1
        # forecast_length = 1 i.e we are predicting one horizon
        model.nn_config['layers'] = {
            "LSTM": {"config": {"units": 1}},
            "Dense": {"config": {"units": 1}},
            "Reshape": {"config": {"target_shape": (1, 1)}}
        }
        x, y = build_train_predict(model)

        self.assertEqual(int(x[0][0].sum()), 140455,)
        self.assertEqual(int(y[0]), 10016)
        self.assertEqual(model.outs, 1)
        self.assertEqual(model.forecast_step, 0)
        return

    def test_ForecastStep0_Outs5(self):
        # test x, y when output is fom t not t+1 and number of inputs are 1 and outputs > 1
        # forecast_length = 1 i.e we are predicting one horizon
        model.nn_config['layers'] = {
            "LSTM": {"config": {"units": 1}},
            "Dense": {"config": {"units": 5}},
            "Reshape": {"config": {"target_shape": (5, 1)}}
        }
        model.in_cols = ['input_0']
        model.out_cols = ['input_1', 'input_2',  'input_3', 'input_4', 'output']

        x, y = build_train_predict(model)

        self.assertEqual(model.outs, 5)
        self.assertEqual(model.ins, 1)
        self.assertEqual(model.forecast_step, 0)

        self.assertEqual(int(x[0][0].sum()), 91)
        self.assertEqual(int(y[0].sum()), 30080)
        return

    def test_ForecastStep1_Outs1(self):
        # when we want to predict t+1 given number of outputs are 1
        # forecast_length = 1 i.e we are predicting one horizon
        model.nn_config['layers'] = {
            "LSTM": {"config": {"units": 1}},
            "Dense": {"config": {"units": 1}},
            "Reshape": {"config": {"target_shape": (1, 1)}}
        }
        model.data_config['forecast_step'] = 1
        model.in_cols = ['input_0', 'input_1', 'input_2',  'input_3', 'input_4']
        model.out_cols = ['output']

        x, y = build_train_predict(model)

        self.assertEqual(model.outs, 1)
        self.assertEqual(model.forecast_step, 1)
        self.assertEqual(int(x[0][-1].sum()), 157325)
        self.assertEqual(int(y[-1].sum()), 10499)
        return


    def test_ForecastStep10_Outs1(self):
        # when we want to predict value t+10 given number of outputs are 1
        # forecast_length = 1 i.e we are predicting one horizon
        model.data_config['forecast_step'] = 10
        model.in_cols = ['input_0', 'input_1', 'input_2',  'input_3', 'input_4']
        model.out_cols = ['output']

        x, y = build_train_predict(model)

        self.assertEqual(model.forecast_step, 10)
        self.assertEqual(int(x[0][-1].sum()), 157010)
        self.assertEqual(int(y[-1].sum()), 10499)
        self.assertEqual(int(x[0][0].sum()), 140455)
        self.assertEqual(int(y[0].sum()), 10026)
        return


    def test_ForecastStep10_Outs5(self):
        # when we want to predict t+10 given number of inputs are 1 and outputs are > 1
        # forecast_length = 1 i.e we are predicting one horizon
        model.nn_config['layers'] = {
            "LSTM": {"config": {"units": 2}},
            "Dense": {"config": {"units": 5}},
            "Reshape": {"config": {"target_shape": (5, 1)}}
        }
        model.data_config['forecast_step'] = 10
        model.in_cols = ['input_0']
        model.out_cols = ['input_1', 'input_2',  'input_3', 'input_4', 'output']

        x, y = build_train_predict(model)

        self.assertEqual(model.forecast_step, 10)
        self.assertEqual(model.outs, 5)
        self.assertEqual(int(x[0][-1].sum()), 3402)
        self.assertEqual(int(y[-1].sum()), 32495)
        self.assertEqual(int(x[0][0].sum()), 91)
        self.assertEqual(int(y[0].sum()), 30130)
        return

    def test_ForecastStep2_Outs1_ForecastLen3(self):
        """
        39
        34
        31
        """
        model.nn_config['layers'] = {
            "LSTM": {"config": {"units": 1}},
            "Dense": {"config": {"units": 3}},
            "Reshape": {"config": {"target_shape": (1, 3)}}
        }

        model.data_config['forecast_step'] = 2
        model.in_cols = ['input_0', 'input_1', 'input_2',  'input_3', 'input_4']
        model.out_cols = ['output']
        model.data_config['forecast_length'] = 3

        x, y = build_train_predict(model)

        self.assertEqual(model.outs, 1)
        self.assertEqual(model.forecast_step, 2)
        self.assertEqual(model.forecast_len, 3)
        self.assertEqual(int(x[0][-1].sum()), 157220)
        self.assertEqual(int(y[-1].sum()), 31494)
        self.assertEqual(int(x[0][0].sum()), 140455)
        self.assertEqual(int(y[0].sum()), 30057)
        self.assertEqual(y[0].shape, (1, 3))
        return

    def test_ForecastStep1_Outs3_ForecastLen3(self):
        """
        we predict
        28, 38, 48
        29, 39, 49   outs=3, forecast_length=3,  horizon/forecast_step=1,
        30, 40, 50
        """
        model.nn_config['layers'] = {
            "LSTM": {"config": {"units": 1}},
            "Dense": {"config": {"units": 9}},
            "Reshape": {"config": {"target_shape": (3, 3)}}
        }
        model.data_config['forecast_step'] = 1
        model.in_cols = ['input_0', 'input_1', 'input_2']
        model.out_cols = ['input_3', 'input_4', 'output']
        model.data_config['forecast_length'] = 3

        x, y = build_train_predict(model)

        self.assertEqual(model.outs, 3)
        self.assertEqual(model.forecast_step, 1)
        self.assertEqual(model.forecast_len, 3)
        self.assertEqual(int(x[0][-1].sum()), 52353)
        self.assertEqual(int(y[-1].sum()), 76482)
        self.assertEqual(int(x[0][0].sum()), 42273)
        self.assertEqual(int(y[0].sum()), 72162)
        return

    def test_InputStep3(self):
        """
        input_step: 3
        outs = 3
        forecast_step = 2
        forecast_length = 3
        """
        model.nn_config['layers'] = {
            "LSTM": {"config": {"units": 1}},
            "Dense": {"config": {"units": 9}},
            "Reshape": {"config": {"target_shape": (3, 3)}}
        }
        model.data_config['input_step'] = 3
        model.data_config['forecast_step'] = 2
        model.data_config['forecast_length'] = 3
        model.in_cols = ['input_0', 'input_1', 'input_2']
        model.out_cols = ['input_3', 'input_4', 'output']

        x, y = build_train_predict(model)

        self.assertEqual(int(x[0][0].sum()), 42399)
        self.assertEqual(int(y[0].sum()), 72279)
        self.assertEqual(int(x[0][-1].sum()), 52164)
        self.assertEqual(int(y[-1].sum()), 76464)
        return

    def test_HighDim(self):
        """
        input_step: 10
        outs = 3
        forecast_step = 10
        forecast_length = 10
        """
        model.nn_config['layers'] = {
            "LSTM": {"config": {"units": 1}},
            "Dense": {"config": {"units": 30}},
            "Reshape": {"config": {"target_shape": (3, 10)}}
        }
        model.data_config['input_step'] = 10
        model.data_config['forecast_step'] = 10
        model.data_config['forecast_length'] = 10
        model.in_cols = ['input_0', 'input_1', 'input_2']
        model.out_cols = ['input_3', 'input_4', 'output']

        x,y = build_train_predict(model)

        self.assertEqual(int(x[0][0].sum()), 42840)
        self.assertEqual(int(y[0].sum()), 242535)
        self.assertEqual(int(x[0][-1].sum()), 51261)
        self.assertEqual(int(y[-1].sum()), 254565)
        return


if __name__ == "__main__":
    unittest.main()