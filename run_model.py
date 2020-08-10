import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from main import TCNModel


def make_model(lstm_units,
               dropout,
               rec_dropout,
               lstm_act,
               batch_size,
               lookback,
               lr,
               enc_lstm1=20,
               enc_lstm2=20):

    nn_config = dict()

    nbeats_options = {
        'backcast_length': lookback,
        'forecast_length': 1,
        'stack_types': ('generic', 'generic'),
        'nb_blocks_per_stack': 2,
        'thetas_dim': (4, 4),
        'share_weights_in_stack': True,
        'hidden_layer_units': lstm_units
    }
    nn_config['nbeats_options'] = nbeats_options

    nn_config['enc_config'] = {'n_h': enc_lstm1,  # length of hidden state m
                               'n_s': enc_lstm1,  # length of hidden state m
                               'm': enc_lstm2  # length of hidden state m
                               }
    nn_config['dec_config'] = {
        'p': 30,
        'n_hde0': 30,
        'n_sde0': 30
    }

    tcn_options = {'nb_filters': 64,
                   'kernel_size': 2,
                   'nb_stacks': 1,
                   'dilations': [1, 2, 4, 8, 16, 32],
                   'padding': 'causal',
                   'use_skip_connections': True,
                   'dropout_rate': 0.0}
    nn_config['tcn_options'] = tcn_options

    nn_config['lr'] = lr
    nn_config['optimizer'] = 'adam'
    nn_config['loss'] = 'mse'
    nn_config['epochs'] = 10

    nn_config['subsequences'] = 3  # used for cnn_lst structure

    nn_config['lstm_config'] = {'lstm_units': lstm_units,
                                'lstm_act': lstm_act,
                                'dropout': dropout,
                                'rec_dropout': rec_dropout,
                                }
    nn_config['cnn_config'] = {'filters': 64,
                               'kernel_size': 2,
                               'activation': 'LeakyRelu',
                               'max_pool_size': 2}

    data_config = dict()
    data_config['lookback'] = lookback
    data_config['batch_size'] = batch_size
    data_config['val_size'] = 0.2
    data_config['CACHEDATA'] = True
    data_config['data_path'] = os.path.join(os.getcwd(), 'data.csv')


    data_config['inputs'] = ['tmin', 'tmax', 'slr', 'WTEMP(C)', 'FLOW_OUTcms', 'SED_OUTtons', 'NO3_OUTkg']
    data_config['outputs'] = ['obs_chla']

    total_intervals = (
        (0, 146,),
        (145, 386,),
        (385, 628,),
        (625, 821,),
        (821, 1110),
        (1110, 1447))



    return data_config, nn_config, total_intervals

