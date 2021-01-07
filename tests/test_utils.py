from inspect import getsourcefile
from os.path import abspath
import pandas as pd
import numpy as np
import unittest
import os

from sklearn.model_selection import train_test_split
import site   # so that dl4seq directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

from dl4seq import Model
from dl4seq.utils.utils import split_by_indices, train_val_split, stats
from dl4seq.backend import get_sklearn_models

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


def make_dummy_model(imputation, inputs=True, frac=0.8):
    orig_df = get_df_with_nans(inputs=inputs, frac=frac)

    class DummyModel:
        in_cols = ['in1', 'in2']
        out_cols = ['out1']
        data_config = {'input_nans': imputation}

    _df = Model.imputation(DummyModel, orig_df, len(in_cols), len(out_cols))
    return _df, orig_df

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

    x, y = model.train_data(st=10, en=500)

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

    x, y = model.train_data(indices=model.train_indices)
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

    def test_make_3d_batches(self):
        class MyModel:
            def check_nans(self, xx, _y, y, o):
                return xx, _y, y
        exs = 50
        d = np.arange(int(exs * 5)).reshape(-1, exs).transpose()
        x, prevy, label = Model.make_3d_batches(MyModel, d,outs=2,
                          lookback=4, in_step=2, forecast_step=2, forecast_len=4)
        self.assertEqual(x.shape, (38, 4, 3))
        self.assertEqual(label.shape, (38, 2, 4))
        self.assertTrue(np.allclose(label[0], np.array([[158., 159., 160., 161.],
                                                        [208., 209., 210., 211.]])))
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
        d = stats(np.random.random(10))
        self.assertGreater(len(d), 1)
        return

    def test_stats_pd(self):
        d = stats(pd.Series(np.random.random(10)))
        self.assertGreater(len(d), 1)
        return

    def test_stats_list(self):
        d = stats(np.random.random(10).tolist())
        self.assertGreater(len(d), 1)
        return

    def test_datetimeindex(self):
        # makes sure that using datetime_index=True during prediction, the returned values are in correct order

        model = Model(
                      data=data1,
                      inputs=in_cols,
                      outputs=out_cols,
                      epochs=2,
                      layers={
                          "LSTM": {"config": {"units": 2}},
                          "Dense": {"config": {"units": 1}}},
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

        x,y = model.train_data(indices=model.train_indices)

        eighth_non_nan_val_4m_st = df['out1'][df['out1'].notnull()].iloc[8]
        # the last training index is 8, so the last y value must be 8th non-nan value
        self.assertAlmostEqual(float(y[-1]), eighth_non_nan_val_4m_st)

        # checking that x values are also correct
        eighth_non_nan_val_4m_st = df[['in1', 'in2']][df['out1'].notnull()].iloc[8]
        self.assertTrue(np.allclose(df[['in1', 'in2']].iloc[86], eighth_non_nan_val_4m_st))
        self.assertTrue(np.allclose(x[0][-1, -1], eighth_non_nan_val_4m_st))

        xx,yy  = model.test_data(indices=model.test_indices)
        # the second test index is 9, so second value of yy must be 9th non-nan value
        self.assertEqual(model.test_indices[2], 9)
        self.assertAlmostEqual(float(yy[2]), df['out1'][df['out1'].notnull()].iloc[9])
        self.assertTrue(np.allclose(xx[0][2, -1], df[['in1', 'in2']][df['out1'].notnull()].iloc[9]))
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
                      verbosity=0)

        model.fit(indices='random')

        x,y = model.train_data(indices=model.train_indices)

        for i in range(100):
            idx = model.train_indices[i]
            df_x = df[['in1', 'in2']].iloc[idx]
            if idx > model.lookback and int(df_x.isna().sum()) == 0:
                self.assertAlmostEqual(float(df['out1'].iloc[idx]), y[i], 6)
                self.assertTrue(np.allclose(df[['in1', 'in2']].iloc[idx], x[0][i, -1]))

        return

    def test_ffill(self):
        """Test that filling nan by ffill method works"""
        out, _df = make_dummy_model({'fillna': {'method': 'ffill'}})

        self.assertAlmostEqual(sum(out[2:8,1]), 0.1724 * 6, 4)

        return

    def test_interpolate_cubic(self):
        """Test that fill nan by interpolating using cubic method works."""
        out, _ = make_dummy_model({'interpolate': {'method': 'cubic'}})

        self.assertAlmostEqual(float(out[8, 0]), 0.6530285703060589, 4)

        return

    def test_knn_imputation(self):
        """Test that knn imputation works seamlessly"""
        out, _df = make_dummy_model({'KNNImputer': {'n_neighbors': 3}}, frac=0.5)

        self.assertEqual(np.isnan(out).sum(), 0)

        return

    def test_multi_out_nans(self):
        """
        Test that when multiple outputs are the target and they contain nans, then we ignore these nans during
        loss calculation.
        """
        df = get_df_with_nans(200, inputs=False, outputs=True, output_cols=['out1', 'out2'], frac=0.5)

        layers = {
            "Flatten": {"config": {}},
            "Dense": {"config": {"units": 2}},
            "Reshape": {"config": {"target_shape": (2,1)}}}

        model = Model(allow_nan_labels=True,
                      layers=layers,
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
        df = get_df_with_nans(500, inputs=False, outputs=True, output_cols=['out1', 'out2'], frac=0.9)

        layers = {
            "Flatten": {"config": {}},
            "Dense": {"config": {"units": 2}},
            "Reshape": {"config": {"target_shape": (2 ,1)}}}

        model = Model(allow_nan_labels=1,
                      transformation=None,
                      layers=layers,
                      inputs=['in1', 'in2'],
                      outputs=['out1', 'out2'],
                      epochs=10,
                      verbosity=0,
                      data=df.copy())

        history = model.fit(indices='random')

        self.assertFalse(any(np.isin(model.train_indices ,model.test_indices)))
        self.assertTrue(np.abs(np.sum(history.history['val_nse'])) > 0.0)
        return

    def test_ignore_nan1_and_data(self):
        df = get_df_with_nans(500, inputs=False, outputs=True, output_cols=['out1', 'out2'], frac=0.9)

        layers = {
            "Flatten": {"config": {}},
            "Dense": {"config": {"units": 2}},
            "Reshape": {"config": {"target_shape": (2, 1)}}}

        model = Model(allow_nan_labels=1,
                      transformation=None,
                      val_data="same",
                      val_fraction=0.0,
                      layers=layers,
                      inputs=['in1', 'in2'],
                      outputs=['out1', 'out2'],
                      epochs=10,
                      verbosity=0,
                      data=df.copy())

        history = model.fit(indices='random')

        self.assertTrue(np.abs(np.sum(history.history['val_nse'])) > 0.0)

        testx, testy = model.test_data(indices=model.test_indices)

        np.allclose(testy[4][0], df[['out1']].iloc[29])
        return

if __name__ == "__main__":
    unittest.main()