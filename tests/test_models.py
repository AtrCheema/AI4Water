import os
import unittest
import site  # so that AI4Water directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import make_classification
from sklearn.datasets import make_multilabel_classification

from AI4Water import Model
from AI4Water.functional import Model as FModel
from AI4Water.utils.datasets import load_nasdaq


examples = 200
lookback = 10
inp_1d = ['prec', 'temp']
inp_2d = ['lai', 'etp']
w = 5
h = 5

# input for 2D data, radar images
pcp = np.random.randint(0, 5, examples*w*h).reshape(examples, w, h)  # 5 rows 5 columns, and 100 images
lai = np.random.randint(6, 10, examples*w*h).reshape(examples, w, h)

data_2d = np.full((len(pcp), len(inp_2d), w, h), np.nan) #np.shape(data)==(100,5,5,5); 5 rows, 5 columns, 5 variables, and 100 images in each variable

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
        "ConvLSTM2d": {"filters": 32, "kernel_size": (3,3), "data_format": "channels_first"},
        "Flatten": {"config": {},
                    "inputs": "ConvLSTM2d"},
        "Concat": {"config": {},
                   "inputs": ["LSTM_1d", "Flatten"]},
        "Dense": {"units": outs},
        "Reshape": {"target_shape": (outs, 1)}
    }
    return layers


def test_evaluation(model):

    model.evaluate('training')
    train_x, _, train_y = model.training_data()

    model.evaluate('validation')
    val_data = model.validation_data()

    model.evaluate('test')
    test_data = model.test_data()
    if not isinstance(test_data, tf.data.Dataset):
        test_x, _, test_y = test_data

    if model.config['val_data'] == 'same' and not isinstance(val_data, tf.data.Dataset):
        val_x,_,y = val_data
        assert test_x[0].shape == val_x[0].shape

    return


def build_and_run(outputs, transformation=None, indices=None):
    model = Model(
        model={"layers": make_layers(len(outputs['inp_1d']))},
        lookback=lookback,
        input_features = {"inp_1d": inp_1d, "inp_2d": inp_2d},
        output_features=outputs,
        data={'inp_1d': make_1d(outputs['inp_1d']), 'inp_2d': data_2d},
        transformation = transformation,
        epochs=2,
        verbosity=0
    )

    model.fit(indices=indices)

    test_evaluation(model)

    return model.predict(indices=model.test_indices if indices else None)

class test_MultiInputModels(unittest.TestCase):

    def test_with_no_transformation(self):
        outputs = {"inp_1d": ['flow']}
        build_and_run(outputs, transformation=None)
        return

    def test_with_transformation(self):
        outputs = {"inp_1d": ['flow']}
        transformation={'inp_1d': 'minmax', 'inp_2d': None}
        build_and_run(outputs, transformation=transformation)
        return

    def test_with_random_sampling(self):
        outputs = {"inp_1d": ['flow']}
        build_and_run(outputs, indices="random")
        return

    def test_with_multiple_outputs(self):
        outputs = {'inp_1d': ['flow', 'suro']}
        build_and_run(outputs)
        return

    def test_add_output_layer1(self):
        # check if it adds both dense and reshapes it correctly or not
        model = Model(model={'layers': {'lstm': 64}},
                      data=load_nasdaq(),
                      verbosity=0)

        self.assertEqual(model.ai4w_outputs[0].shape[1], model.num_outs)
        self.assertEqual(model.ai4w_outputs[0].shape[-1], model.forecast_len)

        return

    def test_add_output_layer2(self):
        # check if it reshapes the output correctly
        model = Model(model={'layers': {'lstm': 64,
                                        'Dense': 1}},
                      data=load_nasdaq(),
                      verbosity=0)

        self.assertEqual(model.ai4w_outputs[0].shape[1], model.num_outs)
        self.assertEqual(model.ai4w_outputs[0].shape[-1], model.forecast_len)
        return

    def test_add_no_output_layer(self):
        # check if it does not add layers when it does not have to
        model = Model(model={'layers': {'lstm': 64,
                                        'Dense': 1,
                                        'Reshape': {'target_shape': (1,1)}}},
                      data=load_nasdaq(),
                      verbosity=0)

        self.assertEqual(model.ai4w_outputs[0].shape[1], model.num_outs)
        self.assertEqual(model.ai4w_outputs[0].shape[-1], model.forecast_len)
        return

    def test_same_val_data(self):
        # test that we can use val_data="same" with multiple inputs. Execution of model.fit() below means that
        # tf.data was created successfully and keras Model accepted it to train as well.
        _examples = 200

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
                out = tf.keras.layers.Reshape((1,1))(out)
                return [inp1, inp2, inp3, inp4], out

            def training_data(self, data=None, data_keys=None, **kwargs):

                in1 = np.random.random((_examples, 10, 5))
                in2 = np.random.random((_examples, 10, 4))
                in3 = np.random.random((_examples, 10))
                in4 = np.random.random((_examples, 9))
                o = np.random.random((_examples, 1, 1))
                return [in1, in2, in3, in4], None, o

        model = MyModel(val_data='same', verbosity=0, category="DL")

        hist = model.fit()
        self.assertGreater(len(hist.history['loss']), 1)
        test_evaluation(model)
        return

    def test_customize_loss(self):
        class QuantileModel(Model):

            def denormalize_data(self,
                                 inputs: np.ndarray,
                                 predicted: np.ndarray,
                                 true: np.ndarray,
                                 in_cols, out_cols,
                                 scaler_key: str,
                                 transformation=None):
                return predicted, true

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
        data = pd.DataFrame(data, columns=['input_' + str(i) for i in range(cols)],
                            index=pd.date_range('20110101', periods=len(data), freq='H'))

        # Define Model
        layers = {'Dense_0': {'units': 64, 'activation': 'relu'},
                  'Dense_3': {'units': 3}}

        # Define Quantiles
        quantiles = [0.005, 0.025,   0.995]

        # Initiate Model
        model = QuantileModel(
            input_features=['input_' + str(i) for i in range(cols - 1)],
            output_features=['input_' + str(cols - 1)],
            lookback=1,
            verbosity=0,
            model={'layers': layers},
            epochs=2,
            data=data.astype(np.float32),
            quantiles=quantiles)

        # Train the model on first 1500 examples/points, 0.2% of which will be used for validation
        model.fit(st=0, en=1500)
        test_evaluation(model)
        return


def build_and_run_class_problem(n_classes, loss, is_multilabel=False, activation='softmax'):

    input_features = [f'input_{n}' for n in range(10)]

    if is_multilabel:
        outputs = [f'target_{n}' for n in range(n_classes)]
        X, y = make_multilabel_classification(n_samples=100, n_features=len(input_features), n_classes=n_classes,
                                              n_labels=2, random_state=0)
        y = y.reshape(-1, n_classes)

    else:
        outputs = ['target']
        X, y = make_classification(n_samples=100, n_features=len(input_features), n_informative=n_classes, n_classes=n_classes,
                               random_state=1)
        y = y.reshape(-1, 1)

    df = pd.DataFrame(np.concatenate([X, y], axis=1), columns=input_features + outputs)

    model = Model(data=df,
                  model={'layers': {
                      'Dense_0': 10,
                      'Flatten': {},
                      'Dense_1': n_classes, 'activation': activation}},
                  input_features=input_features,
                  loss=loss,
                  output_features=outputs,
                  verbosity=0,
                  )
    model.fit()
    test_evaluation(model)

    assert model.problem == 'classification'
    assert len(model.classes) == n_classes
    assert model.num_classes == n_classes
    return model


class TestClassifications(unittest.TestCase):

    def test_binary_classification(self):

        model = build_and_run_class_problem(2, 'binary_crossentropy')

        assert model.is_binary
        assert not model.is_multiclass
        assert not model.is_multilabel

        return

    def test_multiclass_classification(self):

        model = build_and_run_class_problem(3, 'binary_crossentropy')

        assert not model.is_binary
        assert model.is_multiclass
        assert not model.is_multilabel

        return

    def test_multilabel_classification(self):

        model = build_and_run_class_problem(5, 'binary_crossentropy', is_multilabel=True)

        assert not model.is_binary
        assert not model.is_multiclass
        assert model.is_multilabel

        return

    def test_multilabel_classification_with_categorical(self):

        model = build_and_run_class_problem(5, 'categorical_crossentropy', is_multilabel=True)

        assert not model.is_binary
        assert not model.is_multiclass
        assert model.is_multilabel

        return

    def test_multilabel_classification_with_binary_sigmoid(self):

        model = build_and_run_class_problem(5, 'binary_crossentropy', is_multilabel=True, activation='sigmoid')

        assert not model.is_binary
        assert not model.is_multiclass
        assert model.is_multilabel

        return

    def test_multilabel_classification_with_categorical_sigmoid(self):

        model = build_and_run_class_problem(5, 'categorical_crossentropy', is_multilabel=True, activation='sigmoid')

        assert not model.is_binary
        assert not model.is_multiclass
        assert model.is_multilabel

        return

if __name__ == "__main__":
    unittest.main()