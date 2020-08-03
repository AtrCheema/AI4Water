import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from main import Model


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
    nn_config['epochs'] = 100

    nn_config['subsequences'] = 100  # used for cnn_lst structure

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
    data_config['data_path'] = os.path.join(os.getcwd(), 'nk_data.csv')
    df = pd.read_csv('nk_data.csv')
    # cols = list(df.columns)
    # cols.remove('NDX')
    # cols = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm', 'wind_speed_mps', 'rel_hum']

    data_config['inputs'] = ['cum oprtv time', '1st yoib ablyag', '2nd nongchog ablyag', '2nd nongchogsu yolyang', 'ondo']
    data_config['outputs'] = ['FLUX SFX']

    # total_intervals = (
    #     (0, 146,),
    #     (145, 386,),
    #     (385, 628,),
    #     (625, 821,),
    #     (821, 1110),
    #     (1110, 1447))

    _model = Model(data_config=data_config,
                   nn_config=nn_config,
                   # intervals=total_intervals
                   )

    _model.build_nn(method='simple_lstm')  # 'lstm_cnn', 'simple_lstm', 'dual_attention', 'input_attention'

    idx = np.arange(1600)
    tr_idx, test_idx = train_test_split(idx, test_size=0.5, random_state=313)
    history = _model.train_nn(indices=list(tr_idx))

    return _model, tr_idx, test_idx


model, tr_idx, test_idx = make_model(lstm_units=64,
                             dropout=0.4,
                             rec_dropout=0.5,
                             lstm_act='relu',
                             batch_size=8,
                             lookback=20,
                             lr=8.95e-5)


y, obs = model.predict()


