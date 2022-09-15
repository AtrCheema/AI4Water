import time
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input

if 230 <= int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 250:
    from ai4water.functional import Model
    print(f"Switching to functional API due to tensorflow version {tf.__version__}")
else:
    from ai4water import Model


from ai4water.functional import Model as FModel
from ai4water.datasets import load_nasdaq, busan_beach
from ai4water.preprocessing import DataSet, DataSetUnion

nasdaq_input_features = load_nasdaq().columns.tolist()[0:-1],
nasdaq_output_features = load_nasdaq().columns.tolist()[-1:],

examples = 200
lookback = 10
inp_1d = ['prec', 'temp']
inp_2d = ['lai', 'etp']
w = 5
h = 5

# input for 2D data, radar images
# 5 rows 5 columns, and 100 images
pcp = np.random.randint(0, 5, examples*w*h).reshape(examples, w, h)
lai = np.random.randint(6, 10, examples*w*h).reshape(examples, w, h)

# np.shape(data)==(100,5,5,5); 5 rows, 5 columns, 5 variables, and 100 images in each variable
data_2d = np.full((len(pcp), len(inp_2d), w, h), np.nan)

##output
flow = np.full((len(pcp), 1), np.nan)  # np.shape(flow)==(100,1)

for i in range(len(flow)): #len(flow)==100
    flow[i] = pcp[i, :].sum() + lai[i, :].sum()
    data_2d[i, :] = np.stack([pcp[i], lai[i]])


def make_1d(outputs):
    data_1d = np.random.random((examples, len(inp_1d) + len(outputs)))
    # output of model
    if len(outputs) == 1:
        data_1d[:, -1] = np.add(np.add(np.add(data_1d[:, 0], data_1d[:, 1]), data_1d[:, 2]), flow.reshape(-1,))
    else:
        data_1d[:, -1] = np.add(np.add(np.add(data_1d[:, 0], data_1d[:, 1]), data_1d[:, 2]), flow.reshape(-1, ))
        data_1d[:, -2] = np.multiply(np.add(np.add(data_1d[:, 0], data_1d[:, 1]), data_1d[:, 2]), flow.reshape(-1, ))
    data_1d = pd.DataFrame(data_1d, columns=inp_1d+ outputs)
    return data_1d


def make_layers(outs):
    layers = {
        "Input_1d": {"shape": (lookback, len(inp_1d)), "name": "inp_1d"},
        "LSTM_1d": {"units": 12},

        "Input_2d": {"shape": (lookback, len(inp_2d), w, h), "name": "inp_2d"},
        "ConvLSTM2D": {"filters": 32, "kernel_size": (3,3), "data_format": "channels_first"},
        "Flatten": {"config": {},
                    "inputs": "ConvLSTM2D"},
        "Concatenate": {"config": {},
                   "inputs": ["LSTM_1d", "Flatten"]},
        "Dense": {"units": outs},
    }
    return layers


def build_and_run(outputs, x_transformation=None, indices=None, **kwargs):
    model = Model(
        model={"layers": make_layers(len(outputs['inp_1d']))},
        ts_args={'lookback':lookback},
        input_features = {"inp_1d": inp_1d, "inp_2d": inp_2d},
        output_features=outputs,
        x_transformation = x_transformation,
        indices={'training':indices},
        epochs=2,
        verbosity=0,
        **kwargs
    )

    ds1 = DataSet(make_1d(outputs['inp_1d']),
                  input_features=inp_1d,
                  ts_args={'lookback':lookback},
                  verbosity=0)
    ds2 = DataSet(data_2d,
                  input_features=inp_2d,
                  ts_args={'lookback':lookback},
                  verbosity=0)
    ds = DataSetUnion(inp_1d=ds1, inp_2d=ds2, verbosity=0)

    model.fit(data=ds)

    return model.predict_on_test_data(data=ds)


class test_MultiInputModels(unittest.TestCase):

    def test_with_no_transformation(self):
        time.sleep(1)
        outputs = {"inp_1d": ['flow']}
        build_and_run(outputs, x_transformation=None)
        return

    def test_with_transformation(self):
        outputs = {"inp_1d": ['flow']}
        transformation={'inp_1d': 'minmax', 'inp_2d': None}
        build_and_run(outputs, x_transformation=transformation)
        return

    def test_with_random_sampling(self):
        time.sleep(1)
        outputs = {"inp_1d": ['flow']}
        build_and_run(outputs, split_random=True)
        return

    def test_with_multiple_outputs(self):
        outputs = {'inp_1d': ['flow', 'suro']}
        build_and_run(outputs)
        return

    def test_add_output_layer1(self):
        # check if it adds both dense and reshapes it correctly or not
        model = Model(model={'layers': {'LSTM': 64, "Dense": 1}},
                      input_features=nasdaq_input_features,
                      output_features=nasdaq_output_features,
                      ts_args = {"lookback": 15},
                      verbosity=0)

        self.assertEqual(model.ai4w_outputs[0].shape[1], model.num_outs)
        self.assertEqual(model.ai4w_outputs[0].shape[-1], model.forecast_len)

        return

    def test_same_no_test_data(self):
        # test that we can use val_data="same" with multiple inputs. Execution of model.fit() below means that
        # tf.data was created successfully and keras Model accepted it to train as well.
        _examples = 200
        time.sleep(1)
        class MyModel(FModel):
            def add_layers(self, layers_config:dict, inputs=None):
                input_layer_names = ['input1', 'input2', 'input3', 'input4']

                inp1 = Input(shape=(10, 5), name=input_layer_names[0])
                inp2 = Input(shape=(10, 4), name=input_layer_names[1])
                inp3 = Input(shape=(10,), name=input_layer_names[2])
                inp4 = Input(shape=(9,), name=input_layer_names[3])
                conc1 = tf.keras.layers.Concatenate()([inp1, inp2])
                dense1 = Dense(1)(conc1)
                out1 = tf.keras.layers.Flatten()(dense1)
                conc2 = tf.keras.layers.Concatenate()([inp3, inp4])
                s = tf.keras.layers.Concatenate()([out1, conc2])
                out = Dense(1, name='output')(s)
                #out = tf.keras.layers.Reshape((1,1))(out)
                return [inp1, inp2, inp3, inp4], out

            def training_data(self, data=None, data_keys=None, **kwargs):
                print('using customized training data')
                in1 = np.random.random((_examples, 10, 5))
                in2 = np.random.random((_examples, 10, 4))
                in3 = np.random.random((_examples, 10))
                in4 = np.random.random((_examples, 9))
                o = np.random.random((_examples, 1))
                return [in1, in2, in3, in4], o

            def validation_data(self, *args, **kwargs):
                return self.training_data(*args, **kwargs)

            def test_data(self, *args, **kwargs):
                return self.training_data(*args, **kwargs)

        model = MyModel(train_fraction=1.0, verbosity=0, category="DL")

        hist = model.fit()
        assert model.mode == "regression"
        self.assertGreater(len(hist.history['loss']), 1)

        return

    def test_customize_loss(self):
        class QuantileModel(Model):

            def loss(self):
                return qloss

        def qloss(y_true, y_pred):
            # Pinball loss for multiple quantiles
            qs = quantiles
            q = tf.constant(np.array([qs]), dtype=tf.float32)
            e = y_true - y_pred
            v = tf.maximum(q * e, (q - 1) * e)
            return tf.keras.backend.mean(v)

        # Define a dummy dataset consisting of 6 time-series.
        cols = 6
        data = np.arange(int(2000 * cols)).reshape(-1, 2000).transpose()
        data = pd.DataFrame(data, columns=['input_' + str(col) for col in range(cols)],
                            index=pd.date_range('20110101', periods=len(data), freq='H'))

        # Define Model
        layers = {'Dense_0': {'units': 8, 'activation': 'relu'},
                  'Dense_3': {'units': 3},
                  }

        # Define Quantiles
        quantiles = [0.005, 0.025,   0.995]

        # Initiate Model
        model = QuantileModel(
            input_features=data.columns.tolist()[0:-1],
            output_features=data.columns.tolist()[-1:],
            ts_args={'lookback':1},
            verbosity=0,
            model={'layers': layers},
            epochs=2,
            y_transformation='log10',
            indices={'training': np.arange(1500)},
            quantiles=quantiles)

        # Train the model on first 1500 examples/points, 0.2% of which will be used for validation
        model.fit(data=data.astype(np.float32))

        t,p = model.predict_on_test_data(data=data.astype(np.float32), return_true=True)
        assert p.shape[1]==3
        return

    def test_predict_without_y(self):
        # make sure that .predict method can be called without `y`.

        model = Model(model='RandomForestRegressor',
                      verbosity=0
                      )
        model.fit(data=busan_beach())
        y = model.predict(x=np.random.random((10, model.num_ins)))
        assert len(y) == 10

        # check also for functional model
        model = FModel(model='RandomForestRegressor',
                      verbosity=0
                      )
        model.fit(data=busan_beach())
        y = model.predict(x=np.random.random((10, model.num_ins)))
        assert len(y) == 10

        time.sleep(1)
        return


if __name__ == "__main__":
    unittest.main()