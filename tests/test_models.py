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


layers = {"Dense_0": {'config': {'units': 64, 'activation': 'relu'}},
          "Dropout_0": {'config': {'rate': 0.3}},
          "Dense_1": {'config': {'units': 32, 'activation': 'relu'}},
          "Dropout_1": {'config': {'rate': 0.3}},
          "Dense_2": {'config': {'units': 16, 'activation': 'relu'}},
          "Dense_3": {'config': {'units': 1}}
          }

model = make_and_run(Model, lookback=1, _layers=layers)

##
# LSTM based model
layers = {"LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
          "LSTM_1": {'config': {'units': 32}},
          "Dropout": {'config': {'rate': 0.3}},
          "Dense": {'config': {'units': 1, 'name': 'output'}}
          }
make_and_run(LSTMModel, layers)


##
# LSTM  + Raffel Attention
layers = {"LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
          "LSTM_1": {'config': {'units': 32, 'return_sequences': True}},
          "AttentionRaffel": {'config': {'step_dim': 12}},
          "Dropout": {'config': {'rate': 0.3}},
          "Dense": {'config': {'units': 1, 'name': 'output'}}
          }
make_and_run(LSTMModel, layers)

##
# LSTM + SelfAttention model
layers = {"LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
          "SelfAttention": {'config': {}},
          "Dropout": {'config': {'rate': 0.3}},
          "Dense": {'config': {'units': 1, 'name': 'output'}}
          }
make_and_run(LSTMModel, layers)


##
# LSTM + HierarchicalAttention model
layers = {"LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
          "HierarchicalAttention": {'config': {}},
          "Dropout": {'config': {'rate': 0.3}},
          "Dense": {'config': {'units': 1, 'name': 'output'}}
          }
make_and_run(LSTMModel, layers)

##
# CNN based model
layers = {"Conv1D_9": {'config': {'filters': 64, 'kernel_size': 2}},
          "dropout": {'config': {'rate': 0.3}},
          "Conv1D_1": {'config': {'filters': 32, 'kernel_size': 2}},
          "maxpool1d": {'config': {'pool_size': 2}},
          'flatten': {'config': {}},
          'leakyrelu': {'config': {}},
          "Dense": {'config': {'units': 1}}
          }
make_and_run(LSTMModel, layers)

##
# LSTMCNNModel based model
layers = {"LSTM": {'config': {'units': 64, 'return_sequences': True}},
          "Conv1D_0": {'config': {'filters': 64, 'kernel_size': 2}},
          "dropout": {'config': {'rate': 0.3}},
          "Conv1D_1": {'config': {'filters': 32, 'kernel_size': 2}},
          "maxpool1d": {'config': {'pool_size': 2}},
          'flatten': {'config': {}},
          'leakyrelu': {'config': {}},
          "Dense": {'config': {'units': 1}}
          }
make_and_run(LSTMModel, layers)

##
# ConvLSTMModel based model
ins = 8
lookback = 12
sub_seq = 3
sub_seq_lens = int(lookback / sub_seq)
layers = {'Input' : {'config': {'shape':(sub_seq, 1, sub_seq_lens, ins)}},
          'convlstm2d': {'config': {'filters': 64, 'kernel_size': (1, 3), 'activation': 'relu'}},
          'flatten': {'config': {}},
          'repeatvector': {'config': {'n': 1}},
          'lstm':   {'config': {'units': 128,   'activation': 'relu', 'dropout': 0.3, 'recurrent_dropout': 0.4 }},
          'Dense': {'config': {'units': 1}}
          }
make_and_run(ConvLSTMModel, layers, subsequences=sub_seq, lookback=lookback)


##
# CNNLSTM based model
subsequences = 3
timesteps = lookback // subsequences
layers = {'Input' : {'config': {'shape':(None, timesteps, ins)}},
          "TimeDistributed_0": {'config': {}},
          "Conv1D_0": {'config': {'filters': 64, 'kernel_size': 2}},
          "leakyrelu": {'config': {}},
          "TimeDistributed_1": {'config': {}},
          "maxpool1d": {'config': {'pool_size': 2}},
          "TimeDistributed_2": {'config': {}},
          'flatten': {'config': {}},
          'lstm':   {'config': {'units': 64,   'activation': 'relu', 'dropout': 0.4, 'recurrent_dropout': 0.5 }},
          'Dense': {'config': {'units': 1}}
               }
make_and_run(CNNLSTMModel, layers, subsequences=subsequences)


##
# LSTM auto-encoder
layers = {
    'lstm_0': {'config': {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4}},
    "leakyrelu_0": {'config': {}},
    'RepeatVector': {'config': {'n': 11}},
    'lstm_1': {'config': {'units': 100,  'dropout': 0.3, 'recurrent_dropout': 0.4}},
    "relu_1": {'config': {}},
    'Dense': {'config': {'units': 1}}
}
make_and_run(LSTMModel, layers, lookback=12)


##
# TCN based model auto-encoder
layers = {"tcn":  {'config': {'nb_filters': 64,
                  'kernel_size': 2,
                  'nb_stacks': 1,
                  'dilations': [1, 2, 4, 8, 16, 32],
                  'padding': 'causal',
                  'use_skip_connections': True,
                  'return_sequences': False,
                  'dropout_rate': 0.0}},
          'Dense':  {'config': {'units': 1}}
          }
make_and_run(LSTMModel, layers)


##
# InputAttentionModel based model
make_and_run(InputAttentionModel)

##
# DualAttentionModel based model
make_and_run(DualAttentionModel)