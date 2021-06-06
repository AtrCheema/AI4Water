import os
import random
import warnings
import unittest
import site   # so that AI4Water directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from AI4Water import Model
from AI4Water.backend import get_sklearn_models
from AI4Water.utils.imputation import Imputation
from AI4Water.utils.datasets import load_nasdaq
from AI4Water.utils.visualizations import Interpret
from AI4Water.utils.utils import split_by_indices, train_val_split, ts_features, prepare_data, Jsonize

tf.compat.v1.disable_eager_execution()

seed = 313
np.random.seed(seed)
random.seed(seed)

nasdaq_df = load_nasdaq()

examples = 2000
ins = 5
outs = 1
in_cols = ['input_'+str(i) for i in range(5)]
out_cols = ['output']
cols=  in_cols + out_cols

data2 = np.arange(int(50 * 5)).reshape(-1, 50).transpose()
data1 = pd.DataFrame(np.arange(int(examples*len(cols))).reshape(-1,examples).transpose(),
                    columns=cols,
                    index=pd.date_range("20110101", periods=examples, freq="D"))

lookback=7
batch_size=16


def get_df_with_nans(n=1000, inputs=True, outputs=False, frac=0.8, output_cols=None, input_cols=None):
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

    df = pd.DataFrame(np.random.random((n, len(input_cols) + len(output_cols))), columns=input_cols + output_cols)
    for col in cols:
        df.loc[df.sample(frac=frac).index, col] = np.nan

    return df


def get_layers(o=1, forecast_len=1):

    return {
            "LSTM": {"config": {"units": 1}},
            "Dense": {"config": {"units": o*forecast_len }},
            "Reshape": {"config": {"target_shape": (o, forecast_len)}}
        }


def build_model(**kwargs):

    model = Model(
        data=data1,
        verbosity=0,
        batch_size=batch_size,
        lookback=lookback,
        transformation=None,
        epochs=1,
        **kwargs
    )

    return model


def train_predict(model):

    x, _, y = model.train_data(st=10, en=500)

    model.fit()
    model.predict()

    return x,y


def run_same_train_val_data(**kwargs):

    model = Model(
        data=nasdaq_df,
        val_data="same",
        test_fraction=0.2,
        epochs=1,
        verbosity=0)

    model.fit(**kwargs)

    x, _, y = model.train_data(indices=model.train_indices)
    return x, y


class TestUtils(unittest.TestCase):

    """
    I also build, train and predict from the model so that it is confirmed that everything works
    with different input/output shapes.
    """

    def test_ForecastStep0_Outs(self):
        # test x, y output is fom t not t+1 and number of outputs are 1
        # forecast_length = 1 i.e we are predicting one horizon
        model = build_model(
            inputs = in_cols,
            outputs= out_cols,
            model={'layers':get_layers()}
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
            model={'layers':get_layers(5)}
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
            model={'layers':get_layers()}
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
            model={'layers':{
            "LSTM": {"config": {"units": 1}},
            "Dense": {"config": {"units": 1}},
            "Reshape": {"config": {"target_shape": (1, 1)}}
        }}
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
            model={'layers':get_layers(5)}
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
            model={'layers':get_layers(1, 3)}
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
            model={'layers':get_layers(3,3)}
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
            model={'layers':get_layers(3,3)}
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
            model={'layers':get_layers(3,10)}
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
        Interpret(model).plot_feature_importance(np.random.randint(1, 10, 5))

    def test_same_test_val_data_with_chunk(self):
        #TODO not a good test, must check that individual elements in returned arrayare correct

        x, y = run_same_train_val_data(st=0, en=3000)

        self.assertEqual(len(x[0]), len(y))

        return

    def test_same_test_val_data(self):

        x,y = run_same_train_val_data()
        self.assertEqual(len(x[0]), len(y))
        return


    def test_train_val_split(self):
        # This should raise error
        # This should raise error because all arrays are not of equal length
        n1 = 175
        n2 = 380
        x1 = np.random.random((n1, 10))
        x2 = np.random.random((n1, 9))
        x3 = np.random.random((n1, 10, 9))
        x4 = np.random.random((n2, 10))
        x5 = np.random.random((n2, 9))
        x6 = np.random.random((n2, 10, 9))
        x = [x1, x2, x3, x4, x5, x6]

        y1 = np.random.random((n1, 1))
        y2 = np.random.random((n2, 1))
        y = [y1, y2]

        tr_x, tr_y, val_x, val_y = train_val_split(x,y, 0.33)

        return

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

    def test_datetimeindex(self):
        # makes sure that using datetime_index=True during prediction, the returned values are in correct order

        model = Model(
                      data=data1,
                      inputs=in_cols,
                      outputs=out_cols,
                      epochs=2,
                      model={'layers':{
                          "LSTM": {"config": {"units": 2}},
                          "Dense": {"config": {"units": 1}},
                          "Reshape": {"config": {"target_shape": (1, 1)}}
                      }
                      },
                      lookback=lookback,
                      verbosity=0)

        model.fit(indices="random")
        t,p = model.predict(indices=model.train_indices, use_datetime_index=True)
        # the values in t must match the corresponding indices after adding 10000, because y column starts from 100000
        for i in range(100):
            self.assertEqual(int(t[i]), model.train_indices[i] + 10000)
        return

    def test_random_idx_with_nan_in_outputs(self):
        # testing that if output contains nans and we use random indices, then correct examples are assinged
        # for training and testing given val_data is 'same'.
        df = get_df_with_nans(inputs=False, outputs=True, frac=0.8)

        model = Model(inputs=['in1', 'in2'],
                                   outputs=['out1'],
                                   transformation=None,
                                   val_data='same',
                                   test_fraction=0.3,
                                   epochs=1,
                                   data=df,
                                   verbosity=0)

        model.fit(indices='random')
        idx5 = [50,   0,  72, 153,  39,  31, 170,   8]  # last 8 train indices
        self.assertTrue(np.allclose(idx5, model.train_indices[-8:]))

        x, _, y = model.train_data(indices=model.train_indices)

        eighth_non_nan_val_4m_st = df['out1'][df['out1'].notnull()].iloc[8]
        # the last training index is 8, so the last y value must be 8th non-nan value
        self.assertAlmostEqual(float(y[-1]), eighth_non_nan_val_4m_st)

        # checking that x values are also correct
        eighth_non_nan_val_4m_st = df[['in1', 'in2']][df['out1'].notnull()].iloc[8]
        self.assertTrue(np.allclose(df[['in1', 'in2']].iloc[86], eighth_non_nan_val_4m_st))
        self.assertTrue(np.allclose(x[0][-1, -1], eighth_non_nan_val_4m_st))

        xx, _, yy = model.test_data(indices=model.test_indices)
        # the second test index is 9, so second value of yy must be 9th non-nan value
        self.assertEqual(model.test_indices[2], 10)
        self.assertAlmostEqual(float(yy[2]), df['out1'][df['out1'].notnull()].iloc[10])
        self.assertTrue(np.allclose(xx[0][2, -1], df[['in1', 'in2']][df['out1'].notnull()].iloc[10]))

        assert np.max(model.test_indices) < (model.data.shape[0] - int(model.data[model.out_cols].isna().sum()))
        assert np.max(model.train_indices) < (model.data.shape[0] - int(model.data[model.out_cols].isna().sum()))
        return

    def test_random_idx_with_nan_inputs(self):
        """
        Test that when nans are present in inputs and we use random indices, then x,y data is correctly made.
        """

        df = get_df_with_nans(inputs=True, frac=0.1)

        model = Model(inputs=['in1', 'in2'],
                      outputs=['out1'],
                      transformation=None,
                      val_data='same',
                      test_fraction=0.3,
                      epochs=1,
                      data=df,
                      input_nans={'fillna': {'method': 'bfill'}},
                      verbosity=1)

        model.fit(indices='random')

        x, _, y = model.train_data(indices=model.train_indices)

        for i in range(100):
            idx = model.train_indices[i]
            df_x = df[['in1', 'in2']].iloc[idx]
            if idx > model.lookback and int(df_x.isna().sum()) == 0:
                self.assertAlmostEqual(float(df['out1'].iloc[idx]), y[i], 6)
                self.assertTrue(np.allclose(df[['in1', 'in2']].iloc[idx], x[0][i, -1]))

        return

    def test_random_idx_with_nan_inputs_outputs(self):
        """
        Test that when nans are present in inputs and outputs and we use random indices, then x,y data is correctly made.
        """

        df = get_df_with_nans(inputs=True, outputs=True, frac=0.1)

        model = Model(inputs=['in1', 'in2'],
                                   outputs=['out1'],
                                   transformation=None,
                      val_data='same',
                      test_fraction=0.3,
                      epochs=1,
                      data=df,
                                   input_nans={'fillna': {'method': 'bfill'}},
                      verbosity=1)

        model.fit(indices='random')

        x, _, y = model.train_data(indices=model.train_indices)

        # for i in range(100):
        #     idx = model.train_indices[i]
        #     df_x = df[['in1', 'in2']].iloc[idx]
        #     if idx > model.lookback and int(df_x.isna().sum()) == 0:
        #         self.assertAlmostEqual(float(df['out1'].iloc[idx]), y[i], 6)
        #         self.assertTrue(np.allclose(df[['in1', 'in2']].iloc[idx], x[0][i, -1]))

        assert np.max(model.test_indices) < (model.data.shape[0] - int(model.data[model.out_cols].isna().sum()))
        assert np.max(model.train_indices) < (model.data.shape[0] - int(model.data[model.out_cols].isna().sum()))
        return

    def test_ffill(self):
        """Test that filling nan by ffill method works"""
        orig_df = get_df_with_nans()
        imputer = Imputation(data=orig_df, method='fillna', imputer_args={'method': 'ffill'})
        imputed_df = imputer()
        self.assertAlmostEqual(sum(imputed_df.values[2:8, 1]), 0.1724 * 6, 4)

        return

    def test_interpolate_cubic(self):
        """Test that fill nan by interpolating using cubic method works."""
        orig_df = get_df_with_nans()
        imputer = Imputation(data=orig_df, method='interpolate', imputer_args={'method': 'cubic'})
        imputed_df = imputer()
        self.assertAlmostEqual(imputed_df.values[8, 0], 0.6530285703060589)

        return

    def test_knn_imputation(self):
        """Test that knn imputation works seamlessly"""
        orig_df = get_df_with_nans(frac=0.5)
        imputer = Imputation(data=orig_df, method='KNNImputer', imputer_args={'n_neighbors': 3})
        imputed_df = imputer()
        self.assertEqual(sum(imputed_df.isna().sum()),  0)

        return

    def test_multi_out_nans(self):
        """
        Test that when multiple outputs are the target and they contain nans, then we ignore these nans during
        loss calculation.
        """
        if int(''.join(tf.__version__.split('.')[0:2])) < 23 or int(tf.__version__[0])<2:
            warnings.warn(f"test with ignoring nan in labels can not be done in tf version {tf.__version__}")
        else:
            df = get_df_with_nans(200, inputs=False, outputs=True, output_cols=['out1', 'out2'], frac=0.5)

            layers = {
                "Flatten": {"config": {}},
                "Dense": {"config": {"units": 2}},
                "Reshape": {"config": {"target_shape": (2,1)}}}

            model = Model(allow_nan_labels=True,
                          model={'layers':layers},
                          inputs=['in1', 'in2'],
                          outputs=['out1', 'out2'],
                          epochs=10,
                          verbosity=0,
                          data=df)

            history = model.fit()

            self.assertTrue(np.abs(np.sum(history.history['nse'])) > 0.0)
            self.assertTrue(np.abs(np.sum(history.history['val_nse'])) > 0.0)
            return

    def test_nan_labels1(self):
        if int(''.join(tf.__version__.split('.')[0:2])) < 23 or int(tf.__version__[0])<2:
            warnings.warn(f"test with ignoring nan in labels can not be done in tf version {tf.__version__}")
        else:
            df = get_df_with_nans(500, inputs=False, outputs=True, output_cols=['out1', 'out2'], frac=0.9)

            layers = {
                "Flatten": {"config": {}},
                "Dense": {"config": {"units": 2}},
                "Reshape": {"config": {"target_shape": (2 ,1)}}}

            model = Model(allow_nan_labels=1,
                          transformation=None,
                          model={'layers': layers},
                          inputs=['in1', 'in2'],
                          outputs=['out1', 'out2'],
                          epochs=10,
                          verbosity=0,
                          data=df.copy())

            history = model.fit(indices='random')

            self.assertFalse(any(np.isin(model.train_indices ,model.test_indices)))
            self.assertTrue(np.abs(np.sum(history.history['val_nse'])) > 0.0)
            return

    # def test_ignore_nan1_and_data(self):  # todo failing on linux
    #     if int(''.join(tf.__version__.split('.')[0:2])) < 23 or int(tf.__version__[0])<2:
    #         warnings.warn(f"test with ignoring nan in labels can not be done in tf version {tf.__version__}")
    #     else:
    #         df = get_df_with_nans(500, inputs=False, outputs=True, output_cols=['out1', 'out2'], frac=0.9)
    #
    #         layers = {
    #             "Flatten": {"config": {}},
    #             "Dense": {"config": {"units": 2}},
    #             "Reshape": {"config": {"target_shape": (2, 1)}}}
    #
    #         model = Model(allow_nan_labels=1,
    #                       transformation=None,
    #                       val_data="same",
    #                       val_fraction=0.0,
    #                       model={'layers':layers},
    #                       inputs=['in1', 'in2'],
    #                       outputs=['out1', 'out2'],
    #                       epochs=10,
    #                       verbosity=1,
    #                       data=df.copy())
    #
    #         history = model.fit(indices='random')
    #
    #         self.assertTrue(np.abs(np.sum(history.history['val_nse'])) > 0.0)
    #
    #         testx, _, testy = model.test_data(indices=model.test_indices)
    #
    #         np.allclose(testy[4][0], df[['out1']].iloc[29])
    #         return

    def test_jsonize(self):
        a = [np.array([2.0])]
        b = Jsonize(a)()
        self.assertTrue(isinstance(b, list))
        self.assertTrue(isinstance(b[0], float))
        return

    def test_jsonize_nested_dict(self):
        a = {'a': {'b': {'c': {'d': {'e': np.array([2])}}}}}
        b = Jsonize(a)()
        self.assertTrue(isinstance(b, dict))
        self.assertTrue(isinstance(b['a']['b']['c']['d']['e'], int))
        return

    def test_jsonize_none(self):
        a = [None]
        b = Jsonize(a)()
        self.assertTrue(isinstance(b, list))
        self.assertTrue(isinstance(b[0], type(None)))
        return

    def test_prepare_data0(self):
        # vanilla case of time series forecasting
        x, _, y = prepare_data(data2,
                               num_outputs=2,
                               lookback_steps=4,
                               forecast_step=1)
        self.assertAlmostEqual(y[0].sum(), 358.0, 6)
        return

    def test_prepare_data1(self):
        # multi_step ahead at multiple horizons with known future inputs
        x, prevy, y = prepare_data(data2,
                                   num_outputs=2,
                                   lookback_steps=4,
                                   forecast_len=3,
                                   forecast_step=1,
                                   known_future_inputs = True)
        self.assertAlmostEqual(y[0].sum(), 1080.0, 6)
        return

    def test_prepare_data2(self):
        # multi_step ahead at multiple horizons without known_future_inputs
        x, prevy, y = prepare_data(data2,
                                   num_outputs=2,
                                   lookback_steps=4,
                                   forecast_len=3,
                                   forecast_step=1,
                                   known_future_inputs=False)
        self.assertAlmostEqual(y[0].sum(), 1080.0, 6)
        return

    def test_prepare_data3(self):
        # multistep ahead at single horizon
        x, prevy, y = prepare_data(data2,
                                   num_outputs=2,
                                   lookback_steps=4,
                                   forecast_len=1,
                                   forecast_step=3)
        self.assertAlmostEqual(y[0].sum(), 362.0, 6)
        return

    def test_prepare_data4(self):
        # multi output, with strides in input, multi step ahead at multiple horizons without future inputs
        x, prevy, label = prepare_data(data2,
                                       num_outputs=2,
                                       lookback_steps=4,
                                       input_steps=2,
                                       forecast_step=2,
                                       forecast_len=4)
        self.assertEqual(x.shape, (38, 4, 3))
        self.assertEqual(label.shape, (38, 2, 4))
        self.assertTrue(np.allclose(label[0], np.array([[158., 159., 160., 161.],
                                                        [208., 209., 210., 211.]])))
        return

    def test_prepare_data_no_outputs(self):
        """Test when all the columns are used as inputs and thus make_3d_batches does not produce any label data."""
        exs = 100
        d = np.arange(int(exs * 5)).reshape(-1, exs).transpose()
        x, prevy, label = prepare_data(d, num_inputs=5, lookback_steps=4, input_steps=2, forecast_step=2,
                                       forecast_len=4)
        self.assertEqual(label.sum(), 0.0)
        return

    def test_prepare_data_with_mask(self):
        data = np.arange(int(50 * 5)).reshape(-1, 50).transpose()
        idx = random.choices(np.arange(49), k=20)
        data[idx, -1] = -99
        x, prevy, y = prepare_data(data, num_outputs=1, lookback_steps=4, mask=-99)
        self.assertEqual(len(x),  len(y))
        self.assertGreater(len(data)-len(idx) + 4, len(x))
        return

    def test_prepare_data_with_mask1(self):
        data = np.arange(int(50 * 5), dtype=np.float32).reshape(-1, 50).transpose()
        idx = random.choices(np.arange(49), k=20)
        data[idx, -1] = np.nan
        x, prevy, y = prepare_data(data, num_outputs=1, lookback_steps=4, mask=np.nan)
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(x), 33)
        return


if __name__ == "__main__":
    unittest.main()