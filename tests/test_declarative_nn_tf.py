import os
import site
import unittest
import sys
# so that ai4water directory is in path
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

import tensorflow as tf

if 230 <= int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 250:
    from ai4water.functional import Model
    print(f"Switching to functional API due to tensorflow version {tf.__version__}")
else:
    from ai4water import Model

from ai4water.datasets import arg_beach
from ai4water.private_layers import ConditionalRNN

inputs = ["tide_cm", "wat_temp_c", "sal_psu", "air_temp_c", "pcp_mm", "pcp3_mm", "pcp6_mm" ,"pcp12_mm"]
outputs = ['tetx_coppml']

data = arg_beach(inputs, outputs)


def build_model(layers, lookback):
    model = Model(model={'layers': layers},
        input_features=inputs,
        output_features=outputs,
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


        assert model.trainable_parameters() == 3201
        assert model.nn_layers().__len__() == 8
        return

    def test_lstm(self):

        model = build_model({
            "LSTM_0": {'units': 64, 'return_sequences': True},
            "LSTM_1": 32,
            "Dropout": 0.3,
            "Dense": 1
                  },
            12)

        assert model.trainable_parameters() == 31137
        assert model.nn_layers().__len__() == 6

    def test_1dcnn(self):

        model = build_model({
            "Conv1D_9": {'filters': 64, 'kernel_size': 2},
            "Dropout": 0.3,
            "Conv1D_1": {'filters': 32, 'kernel_size': 2},
            "MaxPool1D": 2,
            'Flatten': {},  # This layer does not receive any input arguments
            'LeakyReLU': {},  # activation function can also be used as a separate layer
            "Dense": 1
                  },
        12)
        assert model.trainable_parameters() == 5377
        assert model.nn_layers().__len__() == 9
        return

    def test_lstmcnn(self):

        model = build_model({
            "LSTM":{'units': 64, 'return_sequences': True},
            "Conv1D_0": {'filters': 64, 'kernel_size': 2},
            "Dropout": 0.3,
            "Conv1D_1": {'filters': 32, 'kernel_size': 2},
            "MaxPool1D": 2,
            'Flatten': {},
            'LeakyReLU': {},
            "Dense": 1
        },
            12)
        assert model.trainable_parameters() == 31233
        assert model.nn_layers().__len__() == 10
        return

    def test_convlstm(self):

        model = build_model({
            'Input': {'shape': (3, 1, 4, 8)},
            'ConvLSTM2D': {'filters': 64, 'kernel_size': (1, 3), 'activation': 'relu'},
            'Flatten': {},
            'RepeatVector': 1,
            'LSTM': {'units': 128, 'activation': 'relu', 'dropout': 0.3, 'recurrent_dropout': 0.4},
            'Dense': 1
        },
            12)
        assert model.trainable_parameters() == 187265
        assert model.nn_layers().__len__() == 7

    def test_cnnlstm(self):
        sub_sequences = 3
        lookback = 15
        time_steps = lookback // sub_sequences

        model = build_model({
            "Input": {'shape': (sub_sequences, time_steps, 10)},
            "TimeDistributed_0": {},
            'Conv1D_0': {'filters': 64, 'kernel_size': 2},
            'LeakyReLU_0': {},
            "TimeDistributed_1": {},
            'Conv1D_1': {'filters': 32, 'kernel_size': 2},
            'ELU_1': {},
            "TimeDistributed_2": {},
            'Conv1D_2': {'filters': 16, 'kernel_size': 2},
            'tanh_2': {},
            "TimeDistributed_3": {},
            "MaxPool1D": 2,
            "TimeDistributed_4": {},
            'Flatten': {},
            'LSTM_0':  {'units': 64, 'activation': 'relu', 'dropout': 0.4, 'recurrent_dropout': 0.5, 'return_sequences': True,
                       'name': 'lstm_0'},
            'relu_1': {},
            'LSTM_1':   {'units': 32, 'activation': 'relu', 'dropout': 0.4, 'recurrent_dropout': 0.5, 'name': 'lstm_1'},
            'sigmoid_2': {},
            'Dense': 1
        },
        15)

        assert model.trainable_parameters() == 39697
        assert model.nn_layers().__len__() == 15
        return

    def test_lstm_autoenc(self):

        model = build_model({
            'LSTM_0': {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4},
            "LeakyReLU_0":{},
            'RepeatVector': 11,
            'LSTM_1': {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4},
            "relu_1": {},
            'Dense': 1
        }, 12)
        assert model.trainable_parameters() == 124101
        assert model.nn_layers().__len__() == 8
        return

    def test_tcn(self):

        model = build_model({
            "TCN": {'nb_filters': 64,
                               'kernel_size': 2,
                               'nb_stacks': 1,
                               'dilations': [1, 2, 4, 8, 16, 32],
                               'padding': 'causal',
                               'use_skip_connections': True,
                               'return_sequences': False,
                               'dropout_rate': 0.0},
            'Dense': 1
        }, 12)
        assert model.trainable_parameters() == 92545
        assert model.nn_layers().__len__() == 4

    def test_multi_inputs(self):

        model = build_model({
            "Input_0": {"shape": (5, 10), "name": "cont_inputs"},
            "LSTM_0": {"config": {"units": 62, "activation": "leakyrelu", "dropout": 0.4, "recurrent_dropout": 0.4,
                                  "return_sequences": False, "name": "lstm_0"},
                       "inputs": "cont_inputs"},

            "Input_1": {"shape": 10, "name": "disc_inputs"},
            "Dense_0": {"config": {"units": 64, "activation": "leakyrelu", "name": "Dense_0"},
                        "inputs": "disc_inputs"},
            "Flatten_0": {"config": {"name": "flatten_0"},
                          "inputs": "Dense_0"},

            "Concatenate": {"config": {"name": "Concat"},
                       "inputs": ["lstm_0", "flatten_0"]},

            "Dense_1": {"units": 16, "activation": "leakyrelu", "name": "Dense_1"},
            "Dropout": 0.4,
            "Dense_2": 1
        }, 5)
        assert model.trainable_parameters() == 20857
        assert model.nn_layers().__len__() == 10

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

            "Concatenate": {'config': {'name': 'MyConcat'},
                    'inputs': ['MyDense', 'MyFlatten']},

            "Dense": 1
        },
        15)
        assert model.trainable_parameters() == 31491
        assert model.nn_layers().__len__() == 9
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

            "Concatenate": {'config': {'name': 'MyConcat'},
                    'inputs': ['MyDense', 'MyFlatten', 'MyLSTM2']},

            "Dense": 1
        }, 15)
        assert model.trainable_parameters() == 50243
        assert model.nn_layers().__len__() == 10
        return

    def test_custom_layer(self):
        num_hrus = 7
        lookback = 5

        if int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) in [210, 250, 115]:
            # todo, write __call__ for custom layer for tf 2.1 and 2.5
            return

        class SharedRNN(tf.keras.layers.Layer):

            def __init__(self, *args, **kwargs):
                super().__init__()
                self.rnn = ConditionalRNN(*args, **kwargs)

            def __call__(self, inputs, conditions, *args, **kwargs):
                assert len(inputs.shape) == 4
                assert inputs.shape[1] == conditions.shape[1]

                outputs = []

                for i in range(inputs.shape[1]):
                    rnn_output = self.rnn([inputs[:, i], conditions[:, i]])
                    rnn_output = tf.keras.layers.Dense(1, name=f'HRU_{i}')(rnn_output)
                    outputs.append(rnn_output)

                return outputs

        layers = {
            'Input_cont': {'shape': (num_hrus, lookback, 3)},  # 3 because a,b,c are input parameters
            'Input_static': {'shape': (num_hrus, 3)},
            SharedRNN: {'config': {'units': 32},
                        'inputs': ['Input_cont', 'Input_static'],
                        'outputs': 'rnn_outputs'},
            'Add': {'name': 'total_flow'},
            'Reshape_output': {'target_shape': (1, 1)}
        }

        model = Model(
            model={'layers': layers},
            verbosity=0
        )
        assert model.trainable_parameters() == 4967

        return

    def test_lambda(self):

        model = build_model({
            "LSTM_0": {"units": 32, "return_sequences": True},
            "lambda": {"config": tf.keras.layers.Lambda(lambda x: x[:, -1, :])},
            "Dense": 1
        },
        10)
        assert model.trainable_parameters() == 5281
        assert model.nn_layers().__len__() == 5
        return


if __name__ == "__main__":
    unittest.main()

