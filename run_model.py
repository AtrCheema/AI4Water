import pandas as pd

from models import HARHNModel

def make_model(**kwargs):
    """ This functions fills the default arguments needed to run all the models. The input parameters for each
    model can be overwritten by providing their name either for nn_config or for data_config
    """
    _nn_config = dict()

    _nn_config['autoenc_config'] = {'enc_config':
                                        {'lstm_units': 100,
                                         'act_fn': 'relu',
                                         'dropout': 0.3,
                                         'rec_dropout': 0.4,
                                         'return_sequences': False},
                                    'dec_config':
                                        {'lstm_units': 100,
                                          'act_fn': 'relu',
                                          'dropout': 0.3,
                                          'rec_dropout': 0.4,
                                          'return_sequences': False},
                                    'composite': False
                               }

    _nn_config['nbeats_options'] = {
        'backcast_length': 15,
        'forecast_length': 1,
        'stack_types': ('generic', 'generic'),
        'nb_blocks_per_stack': 2,
        'thetas_dim': (4, 4),
        'share_weights_in_stack': True,
        'hidden_layer_units': 64
    }

    _nn_config['enc_config'] = {'n_h': 20,  # length of hidden state m
                               'n_s': 20,  # length of hidden state m
                               'm': 20,  # length of hidden state m
                               'enc_lstm1_act': None,
                                'enc_lstm2_act': None,
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
    _nn_config['epochs'] = 40
    _nn_config['min_val_loss'] = 0.0001
    _nn_config['patience'] = 100

    _nn_config['subsequences'] = 3  # used for cnn_lst structure

    _nn_config['lstm_config'] = {'lstm_units': 64,
                                'act_fn': 'relu',
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
    _data_config['val_fraction'] = 0.3  # fraction of data to be used for validation
    _data_config['CACHEDATA'] = True
    _data_config['ignore_nans'] = False  # if True, and if target values contain Nans, those samples will not be ignored
    _data_config['use_predicted_output'] = True  # if true, model will use previous predictions as input


    # input features in data_frame
    _data_config['inputs'] = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm', 'wind_speed_mps',
                      'rel_hum']
    # column in dataframe to bse used as output/target
    _data_config['outputs'] = ['blaTEM_coppml']

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
    input_features = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm', 'wind_speed_mps',
                      'rel_hum']
    # column in dataframe to bse used as output/target
    outputs = ['blaTEM_coppml']

    data_config, nn_config, total_intervals = make_model(batch_size=16,
                                                         lookback=15,
                                                         inputs = input_features,
                                                         outputs = outputs,
                                                         lr=0.001)

    df = pd.read_csv('data/all_data_30min.csv')

    model = HARHNModel(data_config=data_config,
                  nn_config=nn_config,
                  data=df,
                  intervals=total_intervals
                  )


    model.build_nn()

    history = model.train_nn(indices='random')

    # y, obs = model.predict()
    acts = model.activations(st=0, en=1400)