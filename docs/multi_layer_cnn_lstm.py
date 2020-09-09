# this file shows how to use `nn_config` to build a model consisting of multiple back to back layers
# We build a CNN->LSTM model consisting of 3 CNN layers followed by max pooling and then feeding its output
# to two LSTM layers.

from run_model import make_model
from models import CNNLSTMModel

import pandas as pd

input_features = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm', 'wind_speed_mps',
                  'rel_hum']
# column in dataframe to bse used as output/target
outputs = ['blaTEM_coppml']

data_config, nn_config, total_intervals = make_model(batch_size=16,
                                                     lookback=15,
                                                     inputs=input_features,
                                                     outputs=outputs,
                                                     lr=0.0001)

nn_config['cnn_config'] = {'n_layers': 3,
                           'cnn_0': {
                            'filters': 64,
                            'kernel_size': 2,
                            'act_layer': 'LeakyRelu'
                            },
                           'cnn_1': {
                            'filters': 32,
                            'kernel_size': 2,
                            'act_layer': 'LeakyRelu'
                            },
                           'cnn_2': {
                            'filters': 16,
                            'kernel_size': 2,
                            'act_layer': 'LeakyRelu'
                            },
                           'max_pool_size': 2}

nn_config['lstm_config'] = {'n_layers': 2,
                            'lstm_1': {  # for more options https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
                                'units': 64,
                                'activation': 'relu',  # activation inside LSTM
                                'dropout': 0.4,
                                'recurrent_dropout': 0.5,
                                'act_layer': 'relu',
                                'return_sequences': True,  # since multiple LSTMs, all must return sequences except last
                                'name': 'lstm_1'
                            },
                            'lstm_2': {
                                'units': 32,
                                'activation': 'relu',  # activation inside LSTM
                                'dropout': 0.4,
                                'recurrent_dropout': 0.5,
                                'act_layer': 'relu',
                                'name': 'lstm_2'
                            }
                            }

df = pd.read_csv('../data/all_data_30min.csv')

model = CNNLSTMModel(data_config=data_config,
                     nn_config=nn_config,
                     data=df,
                     intervals=total_intervals
                     )

model.build_nn()

# This model is built only to showcase how to build multi layer model by manipulating nn_config
history = model.train_nn(indices='random')

y, obs = model.predict(st=0, use_datetime_index=False, marker='.', linestyle='')
