import os
import site
import unittest
# so that dl4seq directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )

import tensorflow as tf

from dl4seq import Model
from dl4seq.data import load_30min

data = load_30min()

inputs = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input7',
       'input8']
outputs = ['target5']


def build_model(layers, lookback):
    model = Model(model={'layers': layers},
        inputs=inputs,
        outputs=outputs,
        lookback=lookback,
        data=data,
        verbosity=0
    )
    return model


class TestBuiltTFConfig(unittest.TestCase):

    def test_mlp(self):

        model = build_model({
        "Dense_0": {'units': 64, 'activation': 'relu'},
        "Dropout_0": 0.3,
        "Dense_1": {'units': 32, 'activation': 'relu'},
        "Dropout_1": 0.3,
        "Dense_2": {'units': 16, 'activation': 'relu'},
        "Dense_3": 1
    }, 1)
        assert model._model.count_params() == 3201
        assert model._model.layers.__len__() == 8
        return

    def test_lstm(self):

        model = build_model({
            "LSTM_0": {'units': 64, 'return_sequences': True},
            "LSTM_1": 32,
            "Dropout": 0.3,
            "Dense": 1
                  },
            12)

        assert model._model.count_params() == 31137
        assert model._model.layers.__len__() == 6

    def test_1dcnn(self):

        model = build_model({
            "Conv1D_9": {'filters': 64, 'kernel_size': 2},
            "dropout": 0.3,
            "Conv1D_1": {'filters': 32, 'kernel_size': 2},
            "maxpool1d": 2,
            'flatten': {},  # This layer does not receive any input arguments
            'leakyrelu': {},  # activation function can also be used as a separate layer
            "Dense": 1
                  },
        12)
        assert model._model.count_params() == 5377
        assert model._model.layers.__len__() == 9
        return

    def test_lstmcnn(self):

        model = build_model({
            "LSTM":{'units': 64, 'return_sequences': True},
            "Conv1D_0": {'filters': 64, 'kernel_size': 2},
            "dropout": 0.3,
            "Conv1D_1": {'filters': 32, 'kernel_size': 2},
            "maxpool1d": 2,
            'flatten': {},
            'leakyrelu': {},
            "Dense": 1
        },
            12)
        assert model._model.count_params() == 31233
        assert model._model.layers.__len__() == 10
        return

    def test_convlstm(self):

        model = build_model({
            'Input': {'shape': (3, 1, 4, 8)},
            'convlstm2d': {'filters': 64, 'kernel_size': (1, 3), 'activation': 'relu'},
            'flatten': {},
            'repeatvector': 1,
            'lstm': {'units': 128, 'activation': 'relu', 'dropout': 0.3, 'recurrent_dropout': 0.4},
            'Dense': 1
        },
            12)
        assert model._model.count_params() == 187265
        assert model._model.layers.__len__() == 7

    def test_cnnlstm(self):
        sub_sequences = 3
        lookback = 15
        time_steps = lookback // sub_sequences

        model = build_model({
            "Input": {'shape': (sub_sequences, time_steps, 10)},
            "TimeDistributed_0": {},
            'conv1d_0': {'filters': 64, 'kernel_size': 2},
            'LeakyRelu_0': {},
            "TimeDistributed_1": {},
            'conv1d_1': {'filters': 32, 'kernel_size': 2},
            'elu_1': {},
            "TimeDistributed_2": {},
            'conv1d_2': {'filters': 16, 'kernel_size': 2},
            'tanh_2': {},
            "TimeDistributed_3": {},
            "maxpool1d": 2,
            "TimeDistributed_4": {},
            'flatten': {},
            'lstm_0':  {'units': 64, 'activation': 'relu', 'dropout': 0.4, 'recurrent_dropout': 0.5, 'return_sequences': True,
                       'name': 'lstm_0'},
            'Relu_1': {},
            'lstm_1':   {'units': 32, 'activation': 'relu', 'dropout': 0.4, 'recurrent_dropout': 0.5, 'name': 'lstm_1'},
            'sigmoid_2': {},
            'Dense': 1
        },
        15)

        assert model._model.count_params() == 39697
        assert model._model.layers.__len__() == 15
        return

    def test_lstm_autoenc(self):

        model = build_model({
            'lstm_0': {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4},
            "leakyrelu_0":{},
            'RepeatVector': 11,
            'lstm_1': {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4},
            "relu_1": {},
            'Dense': 1
        }, 12)
        assert model._model.count_params() == 124101
        assert model._model.layers.__len__() == 8
        return

    def test_tcn(self):

        model = build_model({
            "tcn": {'nb_filters': 64,
                               'kernel_size': 2,
                               'nb_stacks': 1,
                               'dilations': [1, 2, 4, 8, 16, 32],
                               'padding': 'causal',
                               'use_skip_connections': True,
                               'return_sequences': False,
                               'dropout_rate': 0.0},
            'Dense': 1
        }, 12)
        assert model._model.count_params() == 92545
        assert model._model.layers.__len__() == 4

    def test_multi_inputs(self):

        model = build_model({
            "Input_0": {"shape": (5, 10), "name": "cont_inputs"},
            "lstm_0": {"config": {"units": 62, "activation": "leakyrelu", "dropout": 0.4, "recurrent_dropout": 0.4,
                                  "return_sequences": False, "name": "lstm_0"},
                       "inputs": "cont_inputs"},

            "Input_1": {"shape": 10, "name": "disc_inputs"},
            "Dense_0": {"config": {"units": 64, "activation": "leakyrelu", "name": "Dense_0"},
                        "inputs": "disc_inputs"},
            "flatten_0": {"config": {"name": "flatten_0"},
                          "inputs": "Dense_0"},

            "Concat": {"config": {"name": "Concat"},
                       "inputs": ["lstm_0", "flatten_0"]},

            "Dense_1": {"units": 16, "activation": "leakyrelu", "name": "Dense_1"},
            "Dropout": 0.4,
            "Dense_2": 1
        }, 5)
        assert model._model.count_params() == 20857
        assert model._model.layers.__len__() == 10

    def test_multi_output(self):

        model = build_model({
            "LSTM": {'config': {'units': 64, 'return_sequences': True, 'return_state': True},
                     'outputs': ['junk', 'h_state', 'c_state']},

            "Dense_0": {'config': {'units': 1, 'name': 'MyDense'},
                      'inputs': 'h_state'},

            "Conv1D_1": {'config': {'filters': 64, 'kernel_size': 3, 'name': 'myconv'},
                        'inputs': 'junk'},
            "MaxPool1D": {'config': {'name': 'MyMaxPool'},
                        'inputs': 'myconv'},
            "Flatten": {'config': {'name': 'MyFlatten'},
                        'inputs': 'MyMaxPool'},

            "Concat": {'config': {'name': 'MyConcat'},
                    'inputs': ['MyDense', 'MyFlatten']},

            "Dense": 1
        },
        15)
        assert model._model.count_params() == 31491
        assert model._model.layers.__len__() == 9
        return

    def test_add_args(self):

        model = build_model({
            "Input": {'shape': (15, len(inputs)), 'name': "MyInputs"},
            "LSTM": {'config': {'units': 64, 'return_sequences': True, 'return_state': True, 'name': 'MyLSTM1'},
                     'inputs': 'MyInputs',
                     'outputs': ['junk', 'h_state', 'c_state']},

            "Dense_0": {'config': {'units': 1, 'name': 'MyDense'},
                      'inputs': 'h_state'},

            "Conv1D_1": {'config': {'filters': 64, 'kernel_size': 3, 'name': 'myconv'},
                        'inputs': 'junk'},
            "MaxPool1D": {'config': {'name': 'MyMaxPool'},
                        'inputs': 'myconv'},
            "Flatten": {'config': {'name': 'MyFlatten'},
                        'inputs': 'MyMaxPool'},

            "LSTM_3": {"config": {'units': 64, 'name': 'MyLSTM2'},
                       'inputs': 'MyInputs',
                       'call_args': {'initial_state': ['h_state', 'c_state']}},

            "Concat": {'config': {'name': 'MyConcat'},
                    'inputs': ['MyDense', 'MyFlatten', 'MyLSTM2']},

            "Dense": 1
        }, 15)
        assert model._model.count_params() == 50243
        assert model._model.layers.__len__() == 10
        return

    def test_lambda(self):

        model = build_model({
            "LSTM_0": {"units": 32, "return_sequences": True},
            "lambda": {"config": tf.keras.layers.Lambda(lambda x: x[:, -1, :])},
            "Dense": 1
        },
        10)
        assert model._model.count_params() == 5281
        assert model._model.layers.__len__() == 5
        return


if __name__ == "__main__":
    unittest.main()

