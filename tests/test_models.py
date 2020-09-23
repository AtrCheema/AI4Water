import pandas as pd

from utils import make_model
from models import Model, LSTMModel, CNNLSTMModel, ConvLSTMModel, InputAttentionModel
from models import DualAttentionModel, AutoEncoder


def make_and_run(input_model, _layers=None, lookback=12, epochs=4, **kwargs):

    input_features = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm', 'wind_speed_mps',
                      'rel_hum']
    # column in dataframe to bse used as output/target
    outputs = ['blaTEM_coppml']

    data_config, nn_config, total_intervals = make_model(batch_size=16,
                                                         lookback=lookback,
                                                         lr=0.001,
                                                         inputs=input_features,
                                                         outputs = outputs,
                                                         epochs=epochs,
                                                         **kwargs)
    nn_config['layers'] = _layers

    df = pd.read_csv('../data/all_data_30min.csv')

    _model = input_model(data_config=data_config,
                  nn_config=nn_config,
                  data=df,
                  intervals=total_intervals
                  )

    _model.build_nn()

    _ = _model.train_nn(indices='random')

    _ = _model.predict(use_datetime_index=False)

    return _model


layers = {"Dense_0": {'units': 64, 'activation': 'relu'},
          "Dropout_0": {'rate': 0.3},
          "Dense_1": {'units': 32, 'activation': 'relu'},
          "Dropout_1": {'rate': 0.3},
          "Dense_2": {'units': 16, 'activation': 'relu'},
          "Dense_3": {'units': 1}
          }

model = make_and_run(Model, lookback=1, _layers=layers)

##
# LSTM based model
layers = {"LSTM_0": {'units': 64, 'return_sequences': True},
          "LSTM_1": {'units': 32},
          "Dropout": {'rate': 0.3},
          "Dense": {'units': 1}
          }
make_and_run(LSTMModel, layers)

##
# CNN based model
layers = {"Conv1D_9": {'filters': 64, 'kernel_size': 2},
          "dropout": {'rate': 0.3},
          "Conv1D_1": {'filters': 32, 'kernel_size': 2},
          "maxpool1d": {'pool_size': 2},
          'flatten': {},
          'leakyrelu': {},
          "Dense": {'units': 1}
          }
make_and_run(LSTMModel, layers)

##
# LSTMCNNModel based model
layers = {"LSTM": {'units': 64, 'return_sequences': True},
          "Conv1D_0": {'filters': 64, 'kernel_size': 2},
          "dropout": {'rate': 0.3},
          "Conv1D_1": {'filters': 32, 'kernel_size': 2},
          "maxpool1d": {'pool_size': 2},
          'flatten': {},
          'leakyrelu': {},
          "Dense": {'units': 1}
          }
make_and_run(LSTMModel, layers)

##
# ConvLSTMModel based model
layers = {'convlstm2d': {'filters': 64, 'kernel_size': (1, 3), 'activation': 'relu'},
          'flatten': {},
          'repeatvector': {'n': 1},
          'lstm':   {'units': 128,   'activation': 'relu', 'dropout': 0.3, 'recurrent_dropout': 0.4 },
          'Dense': {'units': 1}
          }
make_and_run(ConvLSTMModel, layers)


##
# CNNLSTM based model
layers = {"TimeDistributed_0": {},
          "Conv1D_0": {'filters': 64, 'kernel_size': 2},
          "leakyrelu": {},
          "TimeDistributed_1": {},
          "maxpool1d": {'pool_size': 2},
          "TimeDistributed_2": {},
          'flatten': {},
          'lstm':   {'units': 64,   'activation': 'relu', 'dropout': 0.4, 'recurrent_dropout': 0.5 },
          'Dense': {'units': 1}
               }
make_and_run(CNNLSTMModel, layers)


##
# LSTM auto-encoder
layers = {
    'lstm_0': {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4},
    "leakyrelu_0": {},
    'RepeatVector': {'n': 11},
    'lstm_1': {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4},
    "relu_1": {},
    'Dense': {'units': 1}
}
make_and_run(AutoEncoder, layers, lookback=12)


##
# TCN based model auto-encoder
layers = {"tcn": {'nb_filters': 64,
                  'kernel_size': 2,
                  'nb_stacks': 1,
                  'dilations': [1, 2, 4, 8, 16, 32],
                  'padding': 'causal',
                  'use_skip_connections': True,
                  'return_sequences': False,
                  'dropout_rate': 0.0},
          'Dense': {'units': 1}
          }
make_and_run(LSTMModel, layers)


##
# InputAttentionModel based model
make_and_run(InputAttentionModel)

##
# DualAttentionModel based model
make_and_run(DualAttentionModel)