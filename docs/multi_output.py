# This file shows how to build a naive model for multiple-outputs
import pandas as pd
import numpy as np
import os

from models import InputAttentionModel
from models.global_variables import keras
from run_model import make_model


class MultiSite(InputAttentionModel):

    def run_paras(self, **kwargs):
        train_x, train_y, train_label = self.fetch_data(self.data, **kwargs)

        s0_train = np.zeros((train_x.shape[0], self.nn_config['enc_config']['n_s']))
        h0_train = np.zeros((train_x.shape[0], self.nn_config['enc_config']['n_h']))

        s2_train = np.zeros((train_x.shape[0], self.nn_config['enc_config']['n_s']))
        h2_train = np.zeros((train_x.shape[0], self.nn_config['enc_config']['n_h']))
        return [train_x, s0_train, h0_train, s2_train, h2_train], [train_label[:,0], train_label[:,1]]

    def build_nn(self):

        setattr(self, 'method', 'input_attention')
        print('building input attention')

        predictions = []
        enc_input = keras.layers.Input(shape=(self.lookback, self.ins), name='enc_input1')  # Enter time series data
        inputs = [enc_input]

        for out in range(self.outs):

            lstm_out1, h0, s0 = self._encoder(enc_input, self.nn_config['enc_config'], lstm2_seq=False, suf=str(out))
            act_out = keras.layers.LeakyReLU(name='leaky_relu_'+str(out))(lstm_out1)
            predictions.append(keras.layers.Dense(1)(act_out))
            inputs = inputs + [s0, h0]

        print('predictions: ', predictions)

        self.k_model = self.compile(model_inputs=inputs, outputs=predictions)

        return


if __name__ == "__main__":
    input_features = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm', 'wind_speed_mps',
                      'rel_hum']
    # column in dataframe to bse used as output/target
    outputs = ['blaTEM_coppml', 'sul1_coppml']

    data_config, nn_config, total_intervals = make_model(batch_size=4,
                                                         lookback=15,
                                                         inputs=input_features,
                                                         outputs=outputs,
                                                         lr=0.0001,
                                                         epochs=200)

    cwd = os.getcwd()
    df = pd.read_csv(os.path.join(os.path.dirname(cwd), "data\\all_data_30min.csv"))

    model = MultiSite(data_config=data_config,
                      nn_config=nn_config,
                      data=df,
                      intervals=total_intervals
                      )

    def loss(x, _y):

        mse1 = keras.losses.MSE(x[0], _y[0])
        mse2 = keras.losses.MSE(x[1], _y[1])

        return mse1 + mse2

    model.loss = loss

    model.build_nn()

    history = model.train_nn(indices='random', tensorboard=True)

    y, obs = model.predict()
    # acts = model.activations(st=0, en=1400)
