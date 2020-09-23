import pandas as pd
import os

from models import Model


def make_model(**kwargs):
    """ This functions fills the default arguments needed to run all the models. The input parameters for each
    model can be overwritten by providing their name either for nn_config or for data_config.
    """
    _nn_config = dict()

    _nn_config['layers'] = {"Dense_0": {'units': 64, 'activation': 'relu'},
                                 "Dropout_0": {'rate': 0.3},
                                 "Dense_1": {'units': 32, 'activation': 'relu'},
                                 "Dropout_1": {'rate': 0.3},
                                 "Dense_2": {'units': 16, 'activation': 'relu'},
                                 "Dense_3": {'units': 1}
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

    _nn_config['composite'] = False  # for auto-encoders

    _nn_config['lr'] = 0.0001
    _nn_config['optimizer'] = 'adam'
    _nn_config['loss'] = 'mse'
    _nn_config['epochs'] = 14
    _nn_config['min_val_loss'] = 0.0001
    _nn_config['patience'] = 100

    _nn_config['subsequences'] = 3  # used for cnn_lst structure

    _nn_config['HARHN_config'] = {'n_conv_lyrs': 3,
                                  'enc_units': 64,
                                  'dec_units': 64}

    _data_config = dict()
    _data_config['lookback'] = 15
    _data_config['batch_size'] = 32
    _data_config['val_fraction'] = 0.2  # fraction of data to be used for validation
    _data_config['test_fraction'] = 0.2
    _data_config['CACHEDATA'] = True
    _data_config['ignore_nans'] = False  # if True, and if target values contain Nans, those samples will not be ignored
    _data_config['use_predicted_output'] = True  # if true, model will use previous predictions as input

    # input features in data_frame
    dpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    _df = pd.read_csv(os.path.join(dpath, "nasdaq100_padding.csv"))
    in_cols = list(_df.columns)
    in_cols.remove("NDX")
    _data_config['inputs'] = in_cols
    # column in dataframe to bse used as output/target
    _data_config['outputs'] = ["NDX"]

    _nn_config['dense_config'] = {1: {'units':1}}

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


if __name__ == "__main__":
    input_features = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm', 'wind_speed_mps',
                      'rel_hum']
    # column in dataframe to bse used as output/target
    outputs = ['blaTEM_coppml']

    data_config, nn_config, total_intervals = make_model(batch_size=16,
                                                         lookback=1,
                                                         inputs=input_features,
                                                         outputs=outputs,
                                                         lr=0.0001)

    df = pd.read_csv('data/all_data_30min.csv')

    model = Model(data_config=data_config,
                  nn_config=nn_config,
                  data=df,
                  intervals=total_intervals
                  )

    model.build_nn()

    history = model.train_nn(indices='random')

    # y, obs = model.predict(st=0, use_datetime_index=False, marker='.', linestyle='')
    # model.view_model(st=0)
