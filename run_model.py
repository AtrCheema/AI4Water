import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from models import Model

def make_model(**kwargs):
    """ This functions fills the default arguments needed to run all the models. The input parameters for each
    model can be overwritten by providing their name either for nn_config or for data_config
    """
    _nn_config = dict()

    nbeats_options = {
        'backcast_length': 15,
        'forecast_length': 1,
        'stack_types': ('generic', 'generic'),
        'nb_blocks_per_stack': 2,
        'thetas_dim': (4, 4),
        'share_weights_in_stack': True,
        'hidden_layer_units': 64
    }
    _nn_config['nbeats_options'] = nbeats_options

    _nn_config['enc_config'] = {'n_h': 20,  # length of hidden state m
                               'n_s': 20,  # length of hidden state m
                               'm': 20  # length of hidden state m
                               }
    _nn_config['dec_config'] = {
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
    _nn_config['tcn_options'] = tcn_options

    _nn_config['lr'] = 0.0001
    _nn_config['optimizer'] = 'adam'
    _nn_config['loss'] = 'mse'
    _nn_config['epochs'] = 10
    _nn_config['min_val_loss'] = 9999
    _nn_config['patience'] = 15

    _nn_config['subsequences'] = 3  # used for cnn_lst structure

    _nn_config['lstm_config'] = {'lstm_units': 64,
                                'lstm_act': 'relu',
                                'dropout': 0.4,
                                'rec_dropout': 0.5,
                                }
    _nn_config['cnn_config'] = {'filters': 64,
                               'kernel_size': 2,
                               'activation': 'LeakyRelu',
                               'max_pool_size': 2}

    _nn_config['HARHN_config'] = {'n_conv_lyrs': 3,
                                  'enc_units': 64,
                                  'dec_units': 64}

    _data_config = dict()
    _data_config['lookback'] = 15
    _data_config['batch_size'] = 32
    _data_config['val_fraction'] = 0.2
    _data_config['CACHEDATA'] = True



    # data_config['inputs'] = ['tmin', 'tmax', 'slr', 'WTEMP(C)', 'FLOW_OUTcms', 'SED_OUTtons', 'NO3_OUTkg']
    # data_config['outputs'] = ['obs_chla']
    _data_config['inputs'] = ['cum oprtv time', '1st yoib ablyag', '2nd nongchog ablyag', '2nd nongchogsu yolyang',
                             'ondo']
    _data_config['outputs'] = ['FLUX SFX']

    for key, val in kwargs.items():
        if key in _data_config:
            _data_config[key] = val
        if key in _nn_config:
            _nn_config[key] = val

    _total_intervals = (
        (0, 146,),
        (145, 386,),
        (385, 628,),
        (625, 821,),
        (821, 1110),
        (1110, 1447))



    return _data_config, _nn_config, _total_intervals

if __name__=="__main__":

    data_config, nn_config, total_intervals = make_model(lstm_units=64,
                                                         dropout=0.4,
                                                         rec_dropout=0.5,
                                                         lstm_act='relu',
                                                         batch_size=32,
                                                         lookback=15,
                                                         lr=8.95e-5)

    df = pd.read_csv('data/nk_data.csv')

    model = Model(data_config=data_config,
                  nn_config=nn_config,
                  data=df,
                  # intervals=total_intervals
                  )


    model.build_nn()

    idx = np.arange(720)
    tr_idx, test_idx = train_test_split(idx, test_size=0.5, random_state=313)

    history = model.train_nn(indices=list(tr_idx))