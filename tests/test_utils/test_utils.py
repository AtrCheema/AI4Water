import os
import json
import time
import random
import warnings
import unittest
import site   # so that ai4water directory is in path
cwd = os.path.dirname(os.path.abspath(__file__))
ai4_dir = os.path.dirname(os.path.dirname(cwd))
site.addsitedir(ai4_dir)

import numpy as np
import pandas as pd
import tensorflow as tf

if 230 <= int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 250:
    from ai4water.functional import Model
else:
    from ai4water import Model

from ai4water.utils import jsonize
from ai4water.functional import Model as FModel
from ai4water.datasets import load_nasdaq, busan_beach
from ai4water.utils.utils import TrainTestSplit, ts_features, prepare_data


seed = 313
np.random.seed(seed)
random.seed(seed)

nasdaq_df = load_nasdaq()

examples = 2000
ins = 5
outs = 1
in_cols = ['input_'+str(i) for i in range(5)]
out_cols = ['output']
columns=  in_cols + out_cols

data2 = np.arange(int(50 * 5)).reshape(-1, 50).transpose()
data1 = pd.DataFrame(np.arange(int(examples*len(columns))).reshape(-1,examples).transpose(),
                    columns=columns,
                    index=pd.date_range("20110101", periods=examples, freq="D"))

lookback=7
batch_size=16
busan_beach = busan_beach()


def get_df_with_nans(n=1000, inputs=True, outputs=False, frac=0.8, output_cols=None,
                     input_cols=None):
    np.random.seed(seed)

    if output_cols is None:
        output_cols=['out1']
    if input_cols is None:
        input_cols = ['in1', 'in2']

    cols=[]
    if inputs:
        cols += input_cols
    if outputs:
        cols += output_cols

    df = pd.DataFrame(np.random.random((n, len(input_cols) + len(output_cols))),
                      columns=input_cols + output_cols)
    for col in cols:
        df.loc[df.sample(frac=frac).index, col] = np.nan

    return df


def get_layers(o=1, forecast_len=1):

    _layers = {
            "LSTM": {"units": 1},
            "Dense": {"units": o*forecast_len }
        }

    if forecast_len>1:
        _layers.update({"Reshape": {"target_shape": (o, forecast_len)}})
    return _layers


default_model = {
    'layers': {
        "Dense_0": {'units': 64, 'activation': 'relu'},
        "Flatten": {},
        "Dense_3": 1,
    }
}

def build_model(**kwargs):

    model = Model(
        verbosity=0,
        batch_size=batch_size,
        epochs=1,
        **kwargs
    )

    return model


def train_predict(model):

    model.fit(data=data1.astype(np.float32))
    x, y = model.training_data()

    test_evaluation(model, data1.astype(np.float32))

    return x,y


def test_train_val_test_data(data, **kwargs):

    model = Model(
        model=default_model,
        input_features=data.columns.to_list()[0:-1],
        output_features=data.columns.to_list()[-1:],
        epochs=2,
        verbosity=0,
        **kwargs
    )

    model.fit(data=data)

    train_x, train_y = model.training_data()
    _val_data = model.validation_data()

    _, _ = model.validation_data()

    test_x, test_y = model.test_data()

    if kwargs.get('indices', None) is not None:
        assert train_x.shape[1:] == (model.lookback, data.shape[1]-1), train_x.shape
        assert test_x.shape[1:] == (model.lookback, data.shape[1]-1), test_x.shape

    elif kwargs.get('en', 0) == 400:
        assert train_x.shape[1:] == (model.lookback, data.shape[1]-1), train_x.shape
        assert test_x.shape[1:] == (model.lookback, data.shape[1]-1), test_x.shape

    return train_x, train_y


def test_evaluation(model, data):

    model.evaluate_on_training_data(data=data)
    _, _ = model.training_data()

    model.evaluate_on_validation_data(data=data)
    _, _ = model.validation_data()

    model.evaluate_on_test_data(data=data)
    _, _ = model.test_data()

    return


class TestUtils(unittest.TestCase):

    """
    I also build the model so that it is confirmed that everything works
    with different input/output shapes.
    """

    def test_ForecastStep0_Outs(self):
        # test x, y output is fom t not t+1 and number of outputs are 1
        # forecast_len = 1 i.e we are predicting one horizon
        model = build_model(
            input_features = in_cols,
            output_features = out_cols,
            model={'layers':get_layers()},
            ts_args={'lookback': lookback},
            indices={'training': np.arange(10, 500)},
        )

        x, y = train_predict(model)

        self.assertEqual(int(x[0].sum()), 140455,)
        self.assertEqual(int(y[0]), 10016)
        self.assertEqual(model.num_outs, 1)
        self.assertEqual(model.forecast_step, 0)
        return

    def test_ForecastStep0_Outs5(self):
        # test x, y when output is fom t not t+1 and number of inputs are
        # 1 and outputs > 1
        # forecast_len = 1 i.e we are predicting one horizon
        model = build_model(
            input_features = ['input_0'],
            output_features= ['input_1', 'input_2',  'input_3', 'input_4', 'output'],
            model={'layers':get_layers(5)},
            ts_args={'lookback': lookback},
            indices={'training': np.arange(10, 500).tolist()},
        )

        x, y = train_predict(model)

        self.assertEqual(model.num_outs, 5)
        self.assertEqual(model.num_ins, 1)
        self.assertEqual(model.forecast_step, 0)

        self.assertEqual(int(x[0].sum()), 91)
        self.assertEqual(int(y[0].sum()), 30080)
        return

    def test_ForecastStep1_Outs1(self):
        # when we want to predict t+1 given number of outputs are 1
        # forecast_len = 1 i.e we are predicting one horizon

        model = build_model(
            input_features = ['input_0', 'input_1', 'input_2',  'input_3', 'input_4'],
            output_features= ['output'],
            ts_args={'lookback': lookback, 'forecast_step': 1},
            model={'layers':get_layers()},
            indices={'training': np.arange(3, 615).tolist()},
        )

        x, y = train_predict(model)

        self.assertEqual(model.num_outs, 1)
        self.assertEqual(model.forecast_step, 1)
        self.assertEqual(int(x[-1].sum()), 157290)
        self.assertEqual(int(y[-1].sum()), 10498)
        return

    def test_ForecastStep10_Outs1(self):
        # when we want to predict value t+10 given number of outputs are 1
        # forecast_len = 1 i.e we are predicting one horizon

        model = build_model(
            input_features = in_cols,
            output_features= out_cols,
            ts_args={'lookback': lookback, 'forecast_step': 10},
            model={'layers':{"LSTM": 1}},
            indices={'training': np.arange(10, 602)})

        x, y = train_predict(model)

        self.assertEqual(model.forecast_step, 10)
        self.assertEqual(int(x[-1].sum()), 156975)
        self.assertEqual(int(y[-1].sum()), 10498)
        self.assertEqual(int(x[0].sum()), 140455)
        self.assertEqual(int(y[0].sum()), 10026)
        return

    def test_ForecastStep10_Outs5(self):
        # when we want to predict t+10 given number of input_features are 1
        # and outputs are > 1
        # forecast_len = 1 i.e we are predicting one horizon
        model = build_model(
            input_features = ['input_0'],
            output_features= ['input_1', 'input_2',  'input_3', 'input_4', 'output'],
            ts_args={'lookback': lookback, 'forecast_step': 10},
            model={'layers':get_layers(5)},
            indices={'training': np.arange(10, 602)},
        )

        x, y = train_predict(model)

        self.assertEqual(model.forecast_step, 10)
        self.assertEqual(model.num_outs, 5)
        self.assertEqual(int(x[-1].sum()), 3395)
        self.assertEqual(int(y[-1].sum()), 32490)
        self.assertEqual(int(x[0].sum()), 91)
        self.assertEqual(int(y[0].sum()), 30130)
        return

    def test_ForecastStep2_Outs1_ForecastLen3(self):
        """
        39
        34
        31
        """
        model = build_model(
            input_features = in_cols,
            output_features= out_cols,
            ts_args={'lookback': lookback, 'forecast_step': 2,
                     'forecast_len': 3},
            model={'layers':get_layers(1, 3)},
            indices={'training': np.arange(10, 610)},
        )

        x, y = train_predict(model)

        self.assertEqual(model.num_outs, 1)
        self.assertEqual(model.forecast_step, 2)
        self.assertEqual(model.forecast_len, 3)
        self.assertEqual(int(x[-1].sum()), 157220)
        self.assertEqual(int(y[-1].sum()), 31494)
        self.assertEqual(int(x[0].sum()), 140455)
        self.assertEqual(int(y[0].sum()), 30057)
        self.assertEqual(y[0].shape, (1, 3))
        return

    def test_ForecastStep1_Outs3_ForecastLen3(self):
        """
        we predict
        28, 38, 48
        29, 39, 49   outs=3, forecast_len=3,  horizon/forecast_step=1,
        30, 40, 50
        """

        model = build_model(
            input_features = ['input_0', 'input_1', 'input_2'],
            output_features= ['input_3', 'input_4', 'output'],
            ts_args={'lookback': lookback, 'forecast_step':1, 'forecast_len': 3},
            model={'layers':get_layers(3,3)},
            indices={'training': np.arange(10, 611)},
        )

        x, y = train_predict(model)

        self.assertEqual(model.num_outs, 3)
        self.assertEqual(model.forecast_step, 1)
        self.assertEqual(model.forecast_len, 3)
        self.assertEqual(int(x[-1].sum()), 52332)
        self.assertEqual(int(y[-1].sum()), 76473)
        self.assertEqual(int(x[0].sum()), 42273)
        self.assertEqual(int(y[0].sum()), 72162)
        return

    def test_InputStep3(self):
        """
        input_step: 3
        outs = 3
        forecast_step = 2
        forecast_len = 3
        """
        time.sleep(1)
        model = build_model(
            input_features = ['input_0', 'input_1', 'input_2'],
            output_features= ['input_3', 'input_4', 'output'],
            ts_args = {'lookback': lookback,
                       'forecast_step': 2,
                       'forecast_len': 3,
                       'input_steps': 3,
        },
            model={'layers':get_layers(3,3)},
            indices={'training': np.arange(10, 592)}
        )

        x, y = train_predict(model)

        self.assertEqual(int(x[0].sum()), 42399)
        self.assertEqual(int(y[0].sum()), 72279)
        self.assertEqual(int(x[-1].sum()), 52143)
        self.assertEqual(int(y[-1].sum()), 76455)
        return

    def test_HighDim(self):
        """
        input_step: 10
        outs = 3
        forecast_step = 10
        forecast_len = 10
        """
        time.sleep(1)
        model = build_model(
            input_features = ['input_0', 'input_1', 'input_2'],
            output_features= ['input_3', 'input_4', 'output'],

            ts_args={'lookback': lookback,
                     'forecast_step': 10,
                     'forecast_len': 10,
                     'input_steps': 10,
        },
            model={'layers':get_layers(3,10)},
            indices={'training': np.arange(10, 513)}
        )

        x,y = train_predict(model)

        self.assertEqual(int(x[0].sum()), 42840)
        self.assertEqual(int(y[0].sum()), 242535)
        self.assertEqual(int(x[-1].sum()), 51261)
        self.assertEqual(int(y[-1].sum()), 254565)
        return

    def test_same_test_val_data_with_chunk(self):
        #TODO not a good test, must check that individual elements in returned
        # arrayare correct

        x, y = test_train_val_test_data(data=nasdaq_df,
                                        indices={'training': np.arange(3000)},
                                        ts_args={'lookback': lookback})

        self.assertEqual(len(x), len(y))

        return

    def test_same_test_val_data(self):  # todo, failing in plotting
        time.sleep(1)
        x,y = test_train_val_test_data(data=nasdaq_df,
                                       ts_args={'lookback': lookback})
        self.assertEqual(len(x), len(y))
        return

    def test_same_val_data_with_st_en_defined(self):
        test_train_val_test_data(data=busan_beach,
                                        indices={'training': np.arange(165)},
                                        ts_args={'lookback': lookback})
        return

    def test_same_val_data_with_random(self):
        test_train_val_test_data(data=busan_beach,
                                        split_random=True,
                                        ts_args={'lookback': lookback})
        return

    def test_with_st_en_defined(self):
        time.sleep(1)
        test_train_val_test_data(data=busan_beach,
                                 indices={'training': np.arange(165)},
                                 ts_args={'lookback': lookback},
                                 val_fraction=0.0)
        return

    def test_with_random(self):
        test_train_val_test_data(data=busan_beach,
                                        split_random=True,
                                        ts_args={'lookback': lookback},
                                        val_fraction=0.0)
        return

    def test_train_val_split(self):

        n1 = 175
        n2 = 380
        x1 = np.random.random((n1, 10))
        x3 = np.random.random((n1, 10, 9))
        x4 = np.random.random((n2, 10))
        x = [x1,  x3, x4]

        y1 = np.random.random((n1, 1))
        y2 = np.random.random((n2, 1))
        y = [y1, y2]
        # should raise error
        # train_test_split(x, y, 0.3)

        x1 = np.random.random((n1, 10))
        x2 = np.random.random((n1, 9))
        x3 = np.random.random((n1, 10, 9))
        x4 = np.random.random((n1, 10))
        x = [x1, x2, x3, x4]

        y1 = np.random.random((n1, 1))
        y2 = np.random.random((n2, 1))
        y = [y1, y2]
        #
        # train_test_split(x, y, 0.3)

        x1 = np.arange(100).reshape(10, 10)
        x2 = np.arange(90).reshape(10, 9)
        x3 = np.arange(900).reshape(10, 10, 9)
        x4 = np.arange(100).reshape(10, 10, 1)
        xx = [x1, x2, x3, x4]

        y1 = np.arange(1000, 1010).reshape(10, 1)
        y2 = np.arange(1000, 1200).reshape(10, 10, 2)
        yy = [y1, y2]

        tr_x, testx, tr_y, testy = TrainTestSplit().split_by_slicing(xx, yy)

        a = set([len(i) for i in tr_x])
        assert a.pop() == 7

        a = set([len(i) for i in testx])
        assert a.pop() == 3

        tr_y1, tr_y2 = tr_y
        assert tr_y1[-1].item() == 1006
        test_y1, test_y2 = testy
        assert test_y2[-1][-1][-1].item() == 1199

        tr_x, testx, tr_y, testy = TrainTestSplit(seed=313).split_by_random(xx, yy)
        tr_y1, tr_y2 = tr_y
        assert tr_y1[-1].item() == 1001
        test_y1, test_y2 = testy
        assert test_y2[-1][-1][-1].item() == 1179

        tr_x, testx, tr_y, testy = TrainTestSplit().split_by_slicing(x1, y1)
        assert tr_y[0].item() == 1000

        tr_x, testx, tr_y, testy = TrainTestSplit().split_by_slicing(x1, y2)
        assert tr_y[0][0][0].item() == 1000

        splitter = TrainTestSplit().KFold(x1, y1, 4)

        for idx, ((tr_x, tr_y), (test_x, test_y)) in enumerate(splitter): pass
        assert idx == 3
        assert tr_x.shape[-1] == x1.shape[-1]

        splitter = TrainTestSplit().ShuffleSplit(x1, y1, 4)

        for idx, ((tr_x, tr_y), (test_x, test_y)) in enumerate(splitter): pass
        return

    def test_datetimeindex(self):
        # makes sure that using datetime_index=True during prediction, the returned
        # values are in correct order
        time.sleep(1)
        model = Model(
            input_features=in_cols,
            output_features=out_cols,
            epochs=2,
            model={'layers':
                       {"LSTM": 2, "Dense": 1}
                   },
            ts_args={'lookback':lookback},
            split_random=True,
            verbosity=0)

        t,p = model.predict_on_test_data(data=data1.astype(np.float32),return_true=True)
        # the values in t must match the corresponding indices after adding 10000,
        # because y column starts from 100000
        for i in range(100):
            idx = model.dh_.test_indices[i] + model.lookback - 1
            true = int(round(t[i].item()))
            self.assertEqual(true, idx + 10000)
        test_evaluation(model, data1.astype(np.float32))
        return

    def test_random_idx_with_nan_in_outputs(self):
        # testing that if output contains nans and we use random indices, then correct
        # examples are assinged
        # for training and testing given val_data is 'same'.
        df = get_df_with_nans(inputs=False, outputs=True, frac=0.8)

        model = Model(input_features=['in1', 'in2'],
                      output_features=['out1'],
                      model=default_model,
                      epochs=1,
                      split_random=True,
                      verbosity=0
                      )

        model.fit(data=df)
        idx5 = [50,   0,  72, 153,  39,  31, 170,   8]  # last 8 train indices
        self.assertTrue(np.allclose(idx5, model.dh_.train_indices[-8:]))

        x, y = model.training_data()

        eighth_non_nan_val_4m_st = df['out1'][df['out1'].notnull()].iloc[8]
        # the last training index is 8, so the last y value must be 8th non-nan value
        #self.assertAlmostEqual(float(y[-1]), eighth_non_nan_val_4m_st)  # todo

        # checking that x values are also correct
        eighth_non_nan_val_4m_st = df[['in1', 'in2']][df['out1'].notnull()].iloc[8]
        self.assertTrue(np.allclose(df[['in1', 'in2']].iloc[86], eighth_non_nan_val_4m_st))
        #self.assertTrue(np.allclose(x[-1, -1], eighth_non_nan_val_4m_st)) # todo

        xx, yy = model.test_data()
        # the second test index is 9, so second value of yy must be 9th non-nan value
        # self.assertEqual(model.test_indices[2], 10) # todo
        #self.assertAlmostEqual(float(yy[2]), df['out1'][df['out1'].notnull()].iloc[10]) # todo
        #self.assertTrue(np.allclose(xx[2, -1], df[['in1', 'in2']][df['out1'].notnull()].iloc[10])) # todo

        test_indices = model.dh_.test_indices
        train_indices = model.dh_.train_indices
        _data = model.dh_.data

        exp_train_idx = (_data.shape[0] - int(_data[model.output_features].isna().sum()))
        exp_test_idx = (_data.shape[0] - int(_data[model.output_features].isna().sum()))
        assert np.max(test_indices) < exp_test_idx
        assert np.max(train_indices) < exp_train_idx

        test_evaluation(model, df)

        return

    def test_random_idx_with_nan_inputs(self):
        """
        Test that when nans are present in inputs and we use random indices, then
        x,y data is correctly made.
        """

        df = get_df_with_nans(inputs=True, frac=0.1)

        model = Model(input_features=['in1', 'in2'],
                      model=default_model,
                      output_features=['out1'],
                      #val_data='same',
                      #test_fraction=0.3,
                      epochs=1,
                      nan_filler={'method': 'fillna', 'imputer_args': {'method': 'bfill'}},
                      split_random = True,
                      verbosity=0)

        model.fit(data=df)
        _, _ = model.training_data()

        test_evaluation(model, df)

        # for i in range(100):
        #     idx = model.dh_.train_indices[i]
        #     df_x = df[['in1', 'in2']].iloc[idx]
        #     if idx > model.lookback and int(df_x.isna().sum()) == 0:
        #         self.assertAlmostEqual(float(df['out1'].iloc[idx+14]), float(y[i]), 6)
                #self.assertTrue(np.allclose(df[['in1', 'in2']].iloc[idx], x[0][i, -1])) # todo

        return

    def test_random_idx_with_nan_inputs_outputs(self):
        """
        Test that when nans are present in inputs and outputs and we use random
        indices, then x,y data is correctly made.
        """

        df = get_df_with_nans(inputs=True, outputs=True, frac=0.1)

        model = Model(input_features=['in1', 'in2'],
                      output_features=['out1'],
                      model=default_model,
                      #val_data='same',
                      #test_fraction=0.3,
                      epochs=1,
                      split_random=True,
                      nan_filler={'method': 'fillna', 'imputer_args': {'method': 'bfill'}},
                      verbosity=0)

        model.fit(data=df)

        x, y = model.training_data()

        test_evaluation(model, df)

        # for i in range(100):
        #     idx = model.train_indices[i]
        #     df_x = df[['in1', 'in2']].iloc[idx]
        #     if idx > model.lookback and int(df_x.isna().sum()) == 0:
        #         self.assertAlmostEqual(float(df['out1'].iloc[idx]), y[i], 6)
        #         self.assertTrue(np.allclose(df[['in1', 'in2']].iloc[idx], x[0][i, -1]))

        test_indices = model.dh_.test_indices
        train_indices = model.dh_.train_indices
        _data = model.dh_.data

        exp_test_idx = (_data.shape[0] - int(_data[model.output_features].isna().sum()))
        exp_train_idx = (_data.shape[0] - int(_data[model.output_features].isna().sum()))
        assert np.max(test_indices) < exp_test_idx
        assert np.max(train_indices) < exp_train_idx

        return

    def test_multi_out_nans(self):
        """
        Test that when multiple outputs are the target and they contain nans, then we
        ignore these nans during loss calculation.
        """
        if int(''.join(tf.__version__.split('.')[0:2])) < 23 or int(tf.__version__[0])<2:
            warnings.warn(f"test with ignoring nan in labels can not be done in tf "
                          f"version {tf.__version__}")
        else:
            df = get_df_with_nans(200, inputs=False, outputs=True,
                                  output_cols=['out1', 'out2'], frac=0.5)

            layers = {
                "Flatten": {},
                "Dense": 2,
            }

            model = FModel(allow_nan_labels=True,
                          model={'layers':layers},
                          input_features=['in1', 'in2'],
                          output_features=['out1', 'out2'],
                          epochs=10,
                          verbosity=0,
                          )

            history = model.fit(data=df)

            self.assertTrue(np.abs(np.sum(history.history['loss'])) > 0.0)
            self.assertTrue(np.abs(np.sum(history.history['val_loss'])) > 0.0)

            test_evaluation(model, df)

            return

    def test_nan_labels1(self):
        if int(''.join(tf.__version__.split('.')[0:2])) < 23 or int(tf.__version__[0])<2:
            warnings.warn(f"test with ignoring nan in labels can not be done in tf "
                          f"version {tf.__version__}")
        else:
            df = get_df_with_nans(500, inputs=False, outputs=True,
                                  output_cols=['out1', 'out2'], frac=0.9)

            layers = {
                "Flatten": {},
                "Dense": 2,
            }

            # todo, make sure that model-subclassing also work
            model = FModel(allow_nan_labels=1,
                          model={'layers': layers},
                          input_features=['in1', 'in2'],
                          output_features=['out1', 'out2'],
                          epochs=10,
                          verbosity=0,
                          batch_size=4,
                          split_random = True,
                          )

            history = model.fit(data=df.copy())

            self.assertFalse(any(np.isin(model.dh_.train_indices ,model.dh_.test_indices)))
            self.assertTrue(np.abs(np.sum(history.history['val_loss'])) > 0.0)

            test_evaluation(model, df.copy())

            return

    def test_ignore_nan1_and_data(self):  # todo failing on linux
        if int(''.join(tf.__version__.split('.')[0:2])) < 23 or int(tf.__version__[0])<2:
            warnings.warn(f"test with ignoring nan in labels can not be done in tf version {tf.__version__}")
        else:
            df = get_df_with_nans(500, inputs=False, outputs=True, output_cols=['out1', 'out2'], frac=0.9)

            layers = {
                "Flatten": {"config": {}},
                "Dense": {"config": {"units": 2}},
                #"Reshape": {"config": {"target_shape": (2, 1)}}
            }

            model = Model(allow_nan_labels=1,
                          val_fraction=0.0,
                          model={'layers':layers},
                          input_features=['in1', 'in2'],
                          output_features=['out1', 'out2'],
                          ts_args={'lookback': 15},
                          epochs=10,
                          verbosity=0,
                          split_random=True)

            history = model.fit(data=df)

            self.assertTrue(np.abs(np.sum(history.history['loss'])) > 0.0)

            testx, testy = model.test_data()

            np.allclose(testy[4][0], df[['out1']].iloc[29])
            return


class TestTSFeatures(unittest.TestCase):

    def test_stats(self):
        d = ts_features(np.random.random(10))
        self.assertGreater(len(d), 1)
        return

    def test_stats_pd(self):
        d = ts_features(pd.Series(np.random.random(10)))
        self.assertGreater(len(d), 1)
        return

    def test_stats_list(self):
        d = ts_features(np.random.random(10).tolist())
        self.assertGreater(len(d), 1)
        return

    def test_stats_slice(self):
        d = ts_features(np.random.random(100), st=10, en=50)
        self.assertEqual(d['Counts'], 40)
        return

    def test_ts_features_numbers(self):
        # test that all the features are calculated
        d = ts_features(np.random.random(100))
        self.assertEqual(len(d), 21)
        return

    def test_ts_features_single_feature(self):
        # test that only one feature can be calculated
        d = ts_features(np.random.random(10), features=['Shannon entropy'])
        self.assertEqual(len(d), 1)
        return


class TestPrepareData(unittest.TestCase):

    def test_prepare_data0(self):
        # vanilla case of time series forecasting
        x, _, y = prepare_data(data2,
                               num_outputs=2,
                               lookback=4,
                               forecast_step=1)
        self.assertAlmostEqual(y[0].sum(), 358.0, 6)
        return

    def test_prepare_data1(self):
        # multi_step ahead at multiple horizons with known future inputs
        x, prevy, y = prepare_data(data2,
                                   num_outputs=2,
                                   lookback=4,
                                   forecast_len=3,
                                   forecast_step=1,
                                   known_future_inputs = True)
        self.assertAlmostEqual(y[0].sum(), 1080.0, 6)
        return

    def test_prepare_data2(self):
        # multi_step ahead at multiple horizons without known_future_inputs
        x, prevy, y = prepare_data(data2,
                                   num_outputs=2,
                                   lookback=4,
                                   forecast_len=3,
                                   forecast_step=1,
                                   known_future_inputs=False)
        self.assertAlmostEqual(y[0].sum(), 1080.0, 6)
        return

    def test_prepare_data3(self):
        # multistep ahead at single horizon
        x, prevy, y = prepare_data(data2,
                                   num_outputs=2,
                                   lookback=4,
                                   forecast_len=1,
                                   forecast_step=3)
        self.assertAlmostEqual(y[0].sum(), 362.0, 6)
        return

    def test_prepare_data4(self):
        # multi output, with strides in input, multi step ahead at multiple horizons
        # without future inputs
        x, prevy, label = prepare_data(data2,
                                       num_outputs=2,
                                       lookback=4,
                                       input_steps=2,
                                       forecast_step=2,
                                       forecast_len=4)
        self.assertEqual(x.shape, (38, 4, 3))
        self.assertEqual(label.shape, (38, 2, 4))
        self.assertTrue(np.allclose(label[0], np.array([[158., 159., 160., 161.],
                                                        [208., 209., 210., 211.]])))
        return

    def test_prepare_data_no_outputs(self):
        """Test when all the columns are used as inputs and thus make_3d_batches does
        not produce any label data."""
        exs = 100
        d = np.arange(int(exs * 5)).reshape(-1, exs).transpose()
        x, prevy, label = prepare_data(d, num_inputs=5, lookback=4, input_steps=2,
                                       forecast_step=2,
                                       forecast_len=4)
        self.assertEqual(label.sum(), 0.0)
        return

    def test_prepare_data_with_mask(self):
        data = np.arange(int(50 * 5)).reshape(-1, 50).transpose()
        idx = random.choices(np.arange(49), k=20)
        data[idx, -1] = -99
        x, prevy, y = prepare_data(data, num_outputs=1, lookback=4,
                                   mask=-99)
        self.assertEqual(len(x),  len(y))
        self.assertGreater(len(data)-len(idx) + 4, len(x))
        return

    def test_prepare_data_with_mask1(self):
        data = np.arange(int(50 * 5), dtype=np.float32).reshape(-1, 50).transpose()
        idx = [9, 14, 24, 36, 36, 43, 0, 3, 24, 11, 48, 25, 46, 40, 42, 2, 42, 37, 2, 38]
        data[idx, -1] = np.nan
        x, prevy, y = prepare_data(data, num_outputs=1, lookback=4,
                                   mask=np.nan)
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(x), 33)
        return



class TestJsonize(unittest.TestCase):

    def test_1(self):
        with open('jsonize.json', 'w') as fp:
            json.dump(jsonize('a'), fp)
        return

    def test_2(self):
        with open('jsonize.json', 'w') as fp:
            json.dump(jsonize(1), fp)
        return

    def test_3(self):

        with open('jsonize.json', 'w') as fp:
            json.dump(jsonize(1.0), fp)
        return

    def test_4(self):
        with open('jsonize.json', 'w') as fp:
            json.dump(jsonize(['1', 2, 2.0]), fp)
        return

    def test_5(self):
        with open('jsonize.json', 'w') as fp:
            json.dump(jsonize(np.array(1.0)), fp)
        return

    def test_6(self):
        with open('jsonize.json', 'w') as fp:
            json.dump(jsonize(['1', 2, np.array([1.0, 2.0])]), fp)
        return

    def test_7(self):
        with open('jsonize.json', 'w') as fp:
            json.dump(jsonize(['1', 2, [np.array([1.0, 2.0])]]), fp)
        return

    def test_8(self):
        with open('jsonize.json', 'w') as fp:
            json.dump(jsonize(['1', [1,2,3, [np.array(1.0)]], [np.array([1.0, 2.0])]]), fp)
        return

    def test_9(self):
        # lambda function
        with open('jsonize.json', 'w') as fp:
            json.dump(jsonize(lambda i: i**i), fp)
        return

    def test_10(self):
        # named function
        with open('jsonize.json', 'w') as fp:
            json.dump(jsonize(jsonize), fp)
        return

    def test_11(self):
        # dictionary with function as key and value
        with open('jsonize.json', 'w') as fp:
            json.dump(jsonize({jsonize: jsonize}), fp)
        return

    def test_12(self):
        with open('jsonize.json', 'w') as fp:
            json.dump(jsonize({1: ['1', 2, np.array([1.0, 2.0])]}), fp)
        return

    def test_13(self):
        with open('jsonize.json', 'w') as fp:
            json.dump(jsonize({1: ['1', [1,2,3, [np.array(1.0)]], [np.array([1.0, 2.0])]],
                           2: {'a': ['1', [1,2,3, [np.array(1.0)]], [np.array([1.0, 2.0])]]}}), fp)
        return

    def test_14(self):
        with open('jsonize.json', 'w') as fp:
            json.dump(jsonize({1: ['1', [1,2,3, [np.array(1.0)]], [np.array([1.0, 2.0])]],
                               2: {'a': ['1', [1,2,3, [np.array(1.0)]], [np.array([1.0, 2.0])]],
                                   jsonize: [np.array(1.0), jsonize]}}), fp)
        return

    def test_15(self):
        with open('jsonize.json', 'w') as fp:
            json.dump(jsonize({1: ['1', [1,2,3, [np.array(1.0)]], [np.array([1.0, 2.0])]],
                               2: {'a': ['1', [1,2,3, [np.array(1.0)]], [np.array([1.0, 2.0])]],
                                   jsonize: [np.array(1.0), jsonize, list, busan_beach]}}), fp)
        return

    def test_type(self):
        for _type in [list, str, dict, tuple]:
            with open('jsonize.json', 'w') as fp:
                json.dump(jsonize(_type), fp)
        return

    def test_tuple1(self):
        assert isinstance(jsonize((1,2,3)), tuple)
        return

    def test_list1(self):
        assert isinstance(jsonize([1,2,3]), list)
        return

    def test_df(self):
        with open('jsonize.json', 'w') as fp:
            json.dump(jsonize(busan_beach), fp)
        return

    def test_series(self):
        with open('jsonize.json', 'w') as fp:
            json.dump(jsonize(busan_beach.iloc[0]), fp)
        return

    def test_bool(self):
        assert isinstance(jsonize(True), bool)
        return

    def test_none(self):
        # None should be None
        assert jsonize(None) is None
        return


if __name__ == "__main__":
    unittest.main()