from inspect import getsourcefile
from os.path import abspath
import pandas as pd
import numpy as np
import unittest
import os
from sklearn.model_selection import train_test_split
import site   # so that dl4seq directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

from dl4seq.utils import make_model
from dl4seq import Model
from dl4seq.utils.utils import get_sklearn_models, split_by_indices

seed = 313
np.random.seed(seed)

file_path = abspath(getsourcefile(lambda:0))
dpath = os.path.join(os.path.join(os.path.dirname(os.path.dirname(file_path)), "dl4seq"), "data")
fname = os.path.join(dpath, "nasdaq100_padding.csv")
nasdaq_df = pd.read_csv(fname)

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

def get_layers(o=1, forecast_len=1):

    return {
            "LSTM": {"config": {"units": 1}},
            "Dense": {"config": {"units": o*forecast_len }},
            "Reshape": {"config": {"target_shape": (o, forecast_len)}}
        }


def build_model(**kwargs):

    config = make_model(
        batch_size=batch_size,
        lookback=lookback,
        transformation=None,
        epochs=1,
        **kwargs
    )

    model = Model(config,
                  data=data,
                  verbosity=0
                  )

    return model


def train_predict(model):

    x, y = model.train_data(st=10, en=500)

    model.train()
    model.predict()

    return x,y


def run_same_train_val_data(**kwargs):
    config = make_model(
        val_data="same",
        test_fraction=0.2,
        epochs=1,
    )

    model = Model(config,
                  data=nasdaq_df,
                  verbosity=0)

    model.train(**kwargs)

    x, y = model.train_data(indices=model.train_indices)
    return x, y


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
        model = build_model(
            inputs = in_cols,
            outputs= out_cols,
            layers=get_layers()
        )

        x, y = train_predict(model)

        self.assertEqual(int(x[0][0].sum()), 140455,)
        self.assertEqual(int(y[0]), 10016)
        self.assertEqual(model.outs, 1)
        self.assertEqual(model.forecast_step, 0)
        return

    def test_ForecastStep0_Outs5(self):
        # test x, y when output is fom t not t+1 and number of inputs are 1 and outputs > 1
        # forecast_length = 1 i.e we are predicting one horizon
        model = build_model(
            inputs = ['input_0'],
            outputs= ['input_1', 'input_2',  'input_3', 'input_4', 'output'],
            layers=get_layers(5)
        )

        x, y = train_predict(model)

        self.assertEqual(model.outs, 5)
        self.assertEqual(model.ins, 1)
        self.assertEqual(model.forecast_step, 0)

        self.assertEqual(int(x[0][0].sum()), 91)
        self.assertEqual(int(y[0].sum()), 30080)
        return

    def test_ForecastStep1_Outs1(self):
        # when we want to predict t+1 given number of outputs are 1
        # forecast_length = 1 i.e we are predicting one horizon

        model = build_model(
            inputs = ['input_0', 'input_1', 'input_2',  'input_3', 'input_4'],
            outputs= ['output'],
            forecast_step=1,
            layers=get_layers()
        )

        x, y = train_predict(model)

        self.assertEqual(model.outs, 1)
        self.assertEqual(model.forecast_step, 1)
        self.assertEqual(int(x[0][-1].sum()), 157325)
        self.assertEqual(int(y[-1].sum()), 10499)
        return

    def test_ForecastStep10_Outs1(self):
        # when we want to predict value t+10 given number of outputs are 1
        # forecast_length = 1 i.e we are predicting one horizon

        model = build_model(
            inputs = in_cols,
            outputs= out_cols,
            forecast_step=10,
            layers={
            "LSTM": {"config": {"units": 1}},
            "Dense": {"config": {"units": 1}},
            "Reshape": {"config": {"target_shape": (1, 1)}}
        }
        )

        x, y = train_predict(model)

        self.assertEqual(model.forecast_step, 10)
        self.assertEqual(int(x[0][-1].sum()), 157010)
        self.assertEqual(int(y[-1].sum()), 10499)
        self.assertEqual(int(x[0][0].sum()), 140455)
        self.assertEqual(int(y[0].sum()), 10026)
        return


    def test_ForecastStep10_Outs5(self):
        # when we want to predict t+10 given number of inputs are 1 and outputs are > 1
        # forecast_length = 1 i.e we are predicting one horizon
        model = build_model(
            inputs = ['input_0'],
            outputs= ['input_1', 'input_2',  'input_3', 'input_4', 'output'],
            forecast_step=10,
            layers=get_layers(5)
        )

        x, y = train_predict(model)

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
        model = build_model(
            inputs = in_cols,
            outputs= out_cols,
            forecast_step=2,
            forecast_length = 3,
            layers=get_layers(1, 3)
        )


        x, y = train_predict(model)

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

        model = build_model(
            inputs = ['input_0', 'input_1', 'input_2'],
            outputs= ['input_3', 'input_4', 'output'],
            forecast_step=1,
            forecast_length = 3,
            layers=get_layers(3,3)
        )


        x, y = train_predict(model)

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
        model = build_model(
            inputs = ['input_0', 'input_1', 'input_2'],
            outputs= ['input_3', 'input_4', 'output'],
            forecast_step=2,
            forecast_length = 3,
            input_step=3,
            layers=get_layers(3,3)
        )

        x, y = train_predict(model)

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
        model = build_model(
            inputs = ['input_0', 'input_1', 'input_2'],
            outputs= ['input_3', 'input_4', 'output'],
            forecast_step=10,
            forecast_length = 10,
            input_step=10,
            layers=get_layers(3,10)
        )

        x,y = train_predict(model)

        self.assertEqual(int(x[0][0].sum()), 42840)
        self.assertEqual(int(y[0].sum()), 242535)
        self.assertEqual(int(x[0][-1].sum()), 51261)
        self.assertEqual(int(y[-1].sum()), 254565)
        return

    def test_plot_feature_importance(self):

        model = build_model(inputs=in_cols,
                            outputs=out_cols)
        model.plot_feature_importance(np.random.randint(1, 10, 5))

    def test_get_attributes(self):
        sk = get_sklearn_models()
        rf = sk["RANDOMFORESTREGRESSOR"]
        gb = sk["GRADIENTBOOSTINGREGRESSOR"]
        self.assertGreater(len(sk), 1)
        return

    def test_split_by_indices(self):
        x = np.arange(1000).reshape(100, 5, 2)
        y = np.arange(100)
        tr_indices, test_indices = train_test_split(np.arange(100), test_size=0.2, random_state=seed)
        testx, testy = split_by_indices(x,y, test_indices)
        np.allclose(testx[0][0], [600, 601])
        np.allclose(testy, test_indices)

        tr_x1, tr_y = split_by_indices(x, y, tr_indices)
        np.allclose(tr_y, tr_indices)
        tr_x2, tr_y = split_by_indices([x, x, x], [y], tr_indices)
        self.assertEqual(len(tr_x2), 3)
        self.assertEqual(len(tr_y), 1)
        np.allclose(tr_y[0], tr_indices)
        np.allclose(tr_x1, tr_x2[0])

        return

    def test_same_test_val_data_train_random(self):
        #TODO not a good test, must check that individual elements in returned arrayare correct

        x,y = run_same_train_val_data(indices='random')
        self.assertEqual(len(x[0]), len(y))

        return


    def test_same_test_val_data_with_chunk(self):
        #TODO not a good test, must check that individual elements in returned arrayare correct

        x, y = run_same_train_val_data(st=0, en=3000)

        self.assertEqual(len(x[0]), len(y))

        return

    def test_same_test_val_data(self):

        x,y = run_same_train_val_data()
        self.assertEqual(len(x[0]), len(y))
        return

if __name__ == "__main__":
    unittest.main()