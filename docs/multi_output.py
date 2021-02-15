# This file shows how to build a naive model for multiple-outputs
import pandas as pd
import numpy as np

from models import InputAttentionModel
from models.global_variables import keras
from run_model import make_model


class MultiSite(InputAttentionModel):

    def run_paras(self, **kwargs):

        outputs = []
        train_x, train_y, train_label = self.fetch_data(self.data, **kwargs)
        inputs = [train_x]
        for out in range(self.outs):
            config0 = self.nn_config['enc_config']['enc_config_' + str(out)]
            s0_train = np.zeros((train_x.shape[0], config0['n_s']))
            h0_train = np.zeros((train_x.shape[0], config0['n_h']))
            inputs = inputs + [s0_train, h0_train]
            outputs.append(train_label[:, out])

        return inputs, outputs

    def build_nn(self):

        setattr(self, 'method', 'input_attention')
        print('building input attention')

        predictions = []
        enc_input = keras.layers.Input(shape=(self.lookback, self.ins), name='enc_input1')  # Enter time series data
        inputs = [enc_input]

        for out in range(self.outs):

            config = self.nn_config['enc_config']['enc_config_'+str(out)]

            lstm_out1, h0, s0 = self._encoder(enc_input, config, lstm2_seq=False, suf=str(out))
            act_out = keras.layers.LeakyReLU(name='leaky_relu_'+str(out))(lstm_out1)
            predictions.append(keras.layers.Dense(1)(act_out))
            inputs = inputs + [s0, h0]

        self.k_model = self.compile(model_inputs=inputs, outputs=predictions)

        return

def make_multiout_model(from_config=False, config_file=None, **kwargs):

    input_features = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm',
                      # 'wind_dir_deg',
                      'wind_speed_mps', 'rel_hum']
    # column in dataframe to bse used as output/target
    outputs = ['aac_coppml', 'blaTEM_coppml', 'sul1_coppml', 'tetx_coppml']

    data_config, nn_config, total_intervals = make_model(inputs=input_features,
                                                         outputs=outputs,
                                                         **kwargs)

    df = pd.read_excel('data/all_data_30min.xlsx')
    df.index = pd.to_datetime(df['Date_Time2'])

    if from_config:
        _model = MultiSite.from_config(config_file, data=df)


    else:

        _model = MultiSite(data_config=data_config,
                          nn_config=nn_config,
                          data=df,
                          intervals=total_intervals
                          )

    return _model

if __name__=="__main__":

    enc_config = {
        "enc_config_0": {
            "enc_lstm1_act": "elu",
            "enc_lstm2_act": "elu",
            "m": 32,
            "n_h": 32,
            "n_s": 32
        },
        "enc_config_1": {
            "enc_lstm1_act": "LeakyRelu",
            "enc_lstm2_act": "LeakyRelu",
            "m": 64,
            "n_h": 64,
            "n_s": 64
        },
        "enc_config_2": {
            "enc_lstm1_act": "relu",
            "enc_lstm2_act": "relu",
            "m": 16,
            "n_h": 16,
            "n_s": 16
        },
        "enc_config_3": {
            "enc_lstm1_act": "relu",
            "enc_lstm2_act": "relu",
            "m": 64,
            "n_h": 64,
            "n_s": 64
        }
    }

    # model = make_multiout_model(batch_size=12, lookback=6, lr=0.0008547086,
    #                      enc_config=enc_config)

    def loss(x, _y):

        mse1 = keras.losses.MSE(x[0], _y[0])
        mse2 = keras.losses.MSE(x[1], _y[1])
        mse3 = keras.losses.MSE(x[2], _y[2])
        mse4 = keras.losses.MSE(x[3], _y[3])

        return mse1 + mse2 + mse3 + mse4

    # model.loss = loss
    #
    # model.build_nn()
    #
    # history = model.train_nn(indices='random')
    #
    # y, obs = model.predict(use_datetime_index=False)
    #acts = model.activations(st=0, en=1400)

    cpath = "D:\\experiements\\exp\\dl_ts_prediction\\results\\AttnRNN_multi_opt\\10_20200912_0636\\config.json"
    model = make_multiout_model(from_config=True, config_file=cpath)
    model.loss = loss
    model.build_nn()
    model.load_weights("weights_284_0.0618.hdf5")

    # tr_y, tr_obs = model.predict(indices=model.train_indices,
    #                        pref='train_',
    #                        # use_datetime_index=False
    #                              )
    # test_y, test_obs = model.predict(indices=model.test_indices,
    #                        pref='test_',
    #                        # use_datetime_index=False
    #                              )
    # all_y, all_obs = model.predict(pref='all_',
    #                        # use_datetime_index=False
    #                              )

    model.data_config['ignore_nans'] = True
    y, obs = model.predict(pref='all_den',
                           # use_datetime_index=False
                          )
    tr_y, tr_obs = model.predict(indices=model.train_indices,
                                 pref='train_den',
                                  # use_datetime_index=False
                                )
    test_y, test_obs=model.predict(indices=model.test_indices,
                                   pref='test_den',
                                   # use_datetime_index=False
                                  )