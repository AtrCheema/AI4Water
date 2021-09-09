# this file shows how to use `nn_config` to build a model consisting of multiple back to back layers
# We build a CNN->LSTM model consisting of 3 CNN layers followed by max pooling and then feeding its output
# to two LSTM layers.

from ai4water import Model
from ai4water.datasets import arg_beach

df = arg_beach()
input_features = list(df.columns)[0:-1]

# column in dataframe to bse used as output/target
outputs = list(df.columns)[-1]

sub_sequences = 3
lookback = 15
time_steps = lookback // sub_sequences
layers = {
    "Input": {'shape': (sub_sequences, time_steps, len(input_features))},
    "TimeDistributed_0": {'name': 'td_for_conv1'},
    'conv1d_0': {'filters': 64, 'kernel_size': 2, 'name': 'first_conv1d'},
    'LeakyRelu_0': {},
    "TimeDistributed_1": {},
    'conv1d_1': {'filters': 32, 'kernel_size': 2},
    'elu_1': {},
    "TimeDistributed_2": {},
    'conv1d_2': {'filters': 16, 'kernel_size': 2},
    'tanh_2': {},
    "TimeDistributed_3": {},
    "maxpool1d": {'pool_size': 2},
    "TimeDistributed_4": {},
    'flatten': {},
    'lstm_0': {'units': 64, 'activation': 'relu', 'dropout': 0.4, 'recurrent_dropout': 0.5,
               'return_sequences': True, 'name': 'lstm_0'},
    'Relu_1': {},
    'lstm_1': {'units': 32, 'activation': 'relu', 'dropout': 0.4, 'recurrent_dropout': 0.5, 'name': 'lstm_1'},
    'sigmoid_2': {},
    'Dense': 1
}

model = Model(
    data=df,
    batch_size=16,
    lookback=lookback,
    input_features=input_features,
    output_features=outputs,
    model={'layers':layers},
    lr=0.0001
              )

# This model is built only to showcase how to build multi layer model by manipulating config
# history = model.fit()

#y, obs = model.predict()
