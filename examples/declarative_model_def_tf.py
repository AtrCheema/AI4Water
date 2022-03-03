"""
========================================
Building neural netowrks with tensorflow
========================================
"""

from ai4water import Model
from ai4water.datasets import busan_beach
import pandas as pd

###########################################################

layers = {"Dense_0": {'units': 64, 'activation': 'relu'},
          "Dropout_0": 0.3,  # 0.3 here refers to 'rate' keyword argument in Dropout layer in Tensorflow API
          "Dense_1": {'units': 32, 'activation': 'relu'},
          "Dropout_1": 0.3,
          "Dense_2": {'units': 16, 'activation': 'relu'},
          "Dense_3": 1     # 1 refers to 'units' keyword argument in Dense layer in Tensorflow
          }

df = busan_beach()

model = Model(
            input_features=df.columns.tolist()[0:-1],
            output_features=df.columns.tolist()[-1:],
            model={'layers':layers},
              )

###########################################################

layers = {"LSTM_0": {'units': 64, 'return_sequences': True},
          "LSTM_1": 32,
          "Dense": 1
          }

Model(input_features=df.columns.tolist()[0:-1],
      output_features=df.columns.tolist()[-1:],
      model={'layers':layers},
      ts_args={"lookback": 12}
      )


###########################################################

# 1d CNN based model
# If a layer does not receive any input arguments for its initialization,
# still an empty dictioanry must be provided. Activation functions can
# also be used as a separate layer.

layers = {"Conv1D_9": {'filters': 64, 'kernel_size': 2},
          "Dropout": 0.3,
          "Conv1D_1": {'filters': 32, 'kernel_size': 2},
          "MaxPool1D": 2,
          'Flatten': {}, # This layer does not receive any input arguments
          'LeakyReLU': {},  # activation function can also be used as a separate layer
          "Dense": 1
          }
Model(input_features=df.columns.tolist()[0:-1],
      output_features=df.columns.tolist()[-1:],
      model={'layers':layers},
      ts_args={"lookback": 12}
      )


###########################################################
# LSTM -> CNN based model
# -----------------------

layers = {"LSTM": {'units': 64, 'return_sequences': True},
          "Conv1D_0": {'filters': 64, 'kernel_size': 2},
          "Dropout": 0.3,
          "Conv1D_1": {'filters': 32, 'kernel_size': 2},
          "MaxPool1D": 2,
          'Flatten': {},
          'LeakyReLU': {},
          "Dense": 1
          }

Model(input_features=df.columns.tolist()[0:-1],
      output_features=df.columns.tolist()[-1:],
      model={'layers':layers},
      ts_args={"lookback": 12}
      )


###########################################################
# ConvLSTM based model
# -----------------------

# AI4Water will infer input shape for general cases however it is better to
# explicitly define the Input layer when the input is > 3d or the number of inputs are more than one.

layers = {'Input': {'shape': (3, 1, 4, 8)},
          'ConvLSTM2D': {'filters': 64, 'kernel_size': (1, 3), 'activation': 'relu'},
          'Flatten': {},
          'RepeatVector': 1,
          'LSTM':   {'units': 128,   'activation': 'relu', 'dropout': 0.3, 'recurrent_dropout': 0.4 },
          'Dense': 1
          }
Model(input_features=df.columns.tolist()[0:-1],
      output_features=df.columns.tolist()[-1:],
      model={'layers':layers},
      ts_args={"lookback": 12}
      )

###########################################################
# CNN -> LSTM
# -----------------------

# If a layer is to be enclosed in TimeDistributed layer, just add the layer followed by
# TimeDistributed as shown below. In following, 3 Conv1D layers are enclosed in
# TimeDistributed layer. Similary Flatten and MaxPool1D are also wrapped in TimeDistributed layer.

sub_sequences = 3
lookback = 15
time_steps = lookback // sub_sequences
layers = {
    "Input": {'config': {'shape': (None, time_steps, 10)}},
    "TimeDistributed_0": {},
    'Conv1D_0': {'filters': 64, 'kernel_size': 2},
    'LeakyReLU_0': {},
    "TimeDistributed_1":{},
    'Conv1D_1': {'filters': 32, 'kernel_size': 2},
    'ELU_1': {},
        "TimeDistributed_2": {},
    'Conv1D_2': {'filters': 16, 'kernel_size': 2},
    'tanh_2': {},
    "TimeDistributed_3": {},
    "MaxPool1D": {'pool_size': 2},
    "TimeDistributed_4": {},
    'Flatten': {},
    'LSTM_0':   {'units': 64, 'activation': 'relu', 'dropout': 0.4, 'recurrent_dropout': 0.5,
                  'return_sequences': True,  'name': 'lstm_0'},
    'relu_1': {},
    'LSTM_1':   {'units': 32, 'activation': 'relu', 'dropout': 0.4,
                  'recurrent_dropout': 0.5, 'name': 'lstm_1'},
    'sigmoid_2': {},
    'Dense': 1
}
Model(input_features=df.columns.tolist()[0:-1],
      output_features=df.columns.tolist()[-1:],
      model={'layers':layers},
      ts_args={"lookback": 12}
      )

###########################################################
# LSTM based auto-encoder
# -----------------------

layers = {
    'LSTM_0': {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4},
    "LeakyReLU_0": {},
    'RepeatVector': 11,
    'LSTM_1': {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4},
    "relu_1": {},
    'Dense': 1
}
Model(input_features=df.columns.tolist()[0:-1],
      output_features=df.columns.tolist()[-1:],
      model={'layers':layers},
      ts_args={"lookback": 12}
      )

###########################################################
# TCN layer
# -----------------------

layers = {"TCN": {'nb_filters': 64,
                  'kernel_size': 2,
                  'nb_stacks': 1,
                  'dilations': [1, 2, 4, 8, 16, 32],
                  'padding': 'causal',
                  'use_skip_connections': True,
                  'return_sequences': False,
                  'dropout_rate': 0.0},
          'Dense': 1
          }

Model(input_features=df.columns.tolist()[0:-1],
      output_features=df.columns.tolist()[-1:],
      model={'layers':layers},
      ts_args={"lookback": 12}
      )

###########################################################
# Multiple Inputs
# -----------------------

# In order to build more complex models, where a layer takes more than one inputs, you can specify the inputs key
# for the layer and specify which inputs the layer uses. The value of the inputs dictionary must be a list in this
# case whose members must be the names of the layers which must have been defined earlier. The input/initializating
# arguments in the layer must be enclosed in a config dictionary within the layer in such cases.
class MyModel(Model):

    def training_data(self, **kwargs) -> (list, list):
        """ write code which returns x and y where x consits of [(samples, 5, 10), (samples, 10)] and y consits of
            list [(samples, 1)]
         """
        return

    def test_data(self, **kwargs):
        return


layers = {"Input_0": {"shape": (5, 10), "name": "cont_inputs"},
          "LSTM_0": {"config": { "units": 62,  "activation": "leakyrelu", "dropout": 0.4,
                                 "recurrent_dropout": 0.4, "return_sequences": False,  "name": "lstm_0"},
                     "inputs": "cont_inputs"},

          "Input_1": {"shape": 10, "name": "disc_inputs"},
          "Dense_0": {"config": {"units": 64,"activation": "leakyrelu", "name": "Dense_0"},
                      "inputs": "disc_inputs"},
          "Flatten_0": {"config": {"name": "flatten_0" },
                        "inputs": "Dense_0"},

          "Concatenate": {"config": {"name": "Concat" },
                     "inputs": ["lstm_0", "flatten_0"]},

          "Dense_1": {"units": 16, "activation": "leakyrelu", "name": "Dense_1"},
          "Dropout": 0.4,
          "Dense_2": 1
        }

Model(input_features=df.columns.tolist()[0:-1],
      output_features=df.columns.tolist()[-1:],
      model={'layers':layers},
      ts_args={"lookback": 12}
      )

###########################################################
# Multiple Output Layers
# -----------------------

layers = {
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
}

Model(input_features=df.columns.tolist()[0:-1],
      output_features=df.columns.tolist()[-1:],
      model={'layers':layers},
      ts_args={"lookback": 12}
      )

###########################################################
# Additional call args
# -----------------------

# We might be tempted to provide additional call arguments to a layer. For example, in tensorflow’s LSTM layer,
# we can provide initial state of an LSTM. Suppose we want to use hidden and cell state of one LSTM as initial state
# for next LSTM. In such cases we can make use of call_args as key. The value of call_args must a dictionary.
# In this way we can provide keyword arguments while calling a layer.

layers ={
    "Input": {'config': {'shape': (15, 8), 'name': "MyInputs"}},
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
}
Model(input_features=df.columns.tolist()[0:-1],
      output_features=df.columns.tolist()[-1:],
      model={'layers':layers},
      ts_args={"lookback": 12}
      )


###########################################################
# lambda layers
# -----------------------

import tensorflow as tf


layers = {
    "LSTM_0": {"config": {"units": 32, "return_sequences": True}},
    "lambda": {"config": tf.keras.layers.Lambda(lambda x: x[:, -1, :])},
    "Dense": {"config": {"units": 1}}
}
Model(input_features=df.columns.tolist()[0:-1],
      output_features=df.columns.tolist()[-1:],
      model={'layers':layers},
      ts_args={"lookback": 12}
      )
# The model can be seelessly loaded from the saved json file using

# config_path = "path like"
# model = Model.from_config(config_path=config_path)