# This file shows how to build a naive model for multiple-outputs. Following assumptions are made
# All outputs are modelled by a separate parallel NN(InputAttention in this case).
# All outputs are different and their losses are summed to be used as final loss for back-propagation.
# Each of the parallel NN receives same input.
# The loss function is also customized although it is not necessary

import numpy as np
import tensorflow as tf
from tensorflow import keras

from AI4Water import InputAttentionModel
from AI4Water.utils.datasets import arg_beach

tf.compat.v1.disable_eager_execution()

class MultiSite(InputAttentionModel):
    """ This is only for two outputs currently. """

    def training_data(self, data=None, data_keys=None, **kwargs):
        data= self.data if data is None else data
        train_x, train_y, train_label = self.fetch_data(data,
                                                        inps=self.in_cols,
                                                        outs=self.out_cols,
                                                        transformation=self.config['transformation'],
                                                        **kwargs)

        inputs = [train_x]
        for out in range(self.outs):
            s0_train = np.zeros((train_x.shape[0], self.config['enc_config']['n_s']))
            h0_train = np.zeros((train_x.shape[0], self.config['enc_config']['n_h']))

            inputs = inputs + [s0_train, h0_train]

        return inputs, train_label

    def build(self):

        setattr(self, 'method', 'input_attention')
        print('building input attention')

        self.config['enc_config'] = self.enc_config

        predictions = []
        enc_input = keras.layers.Input(shape=(self.lookback, self.ins), name='enc_input1')  # Enter time series data
        inputs = [enc_input]

        for out in range(self.outs):
            lstm_out1, h0, s0 = self._encoder(enc_input, self.enc_config, lstm2_seq=False, suf=str(out))
            act_out = keras.layers.LeakyReLU(name='leaky_relu_' + str(out))(lstm_out1)
            predictions.append(keras.layers.Dense(1)(act_out))
            inputs = inputs + [s0, h0]

        predictions = keras.layers.Concatenate()(predictions)
        predictions = keras.layers.Reshape(target_shape=(2, 1))(predictions)

        print('predictions: ', predictions)

        self._model = self.compile(model_inputs=inputs, outputs=predictions)

        return


if __name__ == "__main__":
    # column in dataframe to bse used as output/target
    outputs = ['blaTEM_coppml', 'aac_coppml']

    df = arg_beach(target=outputs)
    input_features = list(df.columns)[0:-2]

    model = MultiSite(
        data=df,
        batch_size=4,
        lookback=15,
        inputs=input_features,
        outputs=outputs,
        lr=0.0001,
        epochs=2,
        val_fraction=0.3,  # TODO why less than 0.3 give error here?
        test_fraction=0.3,
        steps_per_epoch=38
    )


    def loss(x, _y):
        mse1 = keras.losses.MSE(x[0], _y[0])
        mse2 = keras.losses.MSE(x[1], _y[1])

        return mse1 + mse2


    model.loss = loss

    history = model.fit(indices='random', tensorboard=True)

    y, obs = model.predict()
    activations = model.activations(st=0, en=1400)
