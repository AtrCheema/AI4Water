# This file shows how to build a naive model for multiple-outputs. Following assumptions are made
# All outputs are modelled by a separate parallel NN(InputAttention in this case).
# All outputs are different and their losses are summed to be used as final loss for back-propagation.
# Each of the parallel NN receives same input.
# The loss function is also customized although it is not necessary

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ai4water import InputAttentionModel
from ai4water.datasets import busan_beach

tf.compat.v1.disable_eager_execution()

class MultiSite(InputAttentionModel):
    """ This is only for two outputs currently. """

    def training_data(self, data='training', data_keys=None, **kwargs):
        #data= self.data if data is None else data
        train_x, train_y, train_label = self.fetch_data(data,
                                                        **kwargs)
        train_x = train_x[0]
        inputs = [train_x]
        for out in range(self.num_outs):
            s0_train = np.zeros((train_x.shape[0], self.config['enc_config']['n_s']))
            h0_train = np.zeros((train_x.shape[0], self.config['enc_config']['n_h']))

            inputs = inputs + [s0_train, h0_train]

        return inputs, train_label

    def test_data(self, data='test', **kwargs):
        return self.training_data(data=data, **kwargs)

    def validation_data(self, data='validation', **kwargs):
        return self.training_data(data=data, **kwargs)

    def build(self, input_shape=None):

        setattr(self, 'method', 'input_attention')
        print('building input attention')

        self.config['enc_config'] = self.enc_config
        setattr(self, 'batch_size', self.config['batch_size'])
        setattr(self, 'drop_remainder', self.config['drop_remainder'])
        setattr(self, 'teacher_forcing', self.config['teacher_forcing'])

        predictions = []
        enc_input = keras.layers.Input(shape=(self.lookback, self.num_ins), name='enc_input1')  # Enter time series data
        inputs = [enc_input]

        for out in range(self.num_outs):
            lstm_out1, h0, s0 = self._encoder(enc_input, self.enc_config, lstm2_seq=False, suf=str(out))
            act_out = keras.layers.LeakyReLU(name='leaky_relu_' + str(out))(lstm_out1)
            predictions.append(keras.layers.Dense(1)(act_out))
            inputs = inputs + [s0, h0]

        predictions = keras.layers.Concatenate()(predictions)
        predictions = keras.layers.Reshape(target_shape=(self.num_outs, 1))(predictions)

        print('predictions: ', predictions)

        self._model = self.compile(model_inputs=inputs, outputs=predictions)

        return


if __name__ == "__main__":
    # column in dataframe to bse used as output/target
    outputs = ['blaTEM_coppml', 'aac_coppml']

    df = busan_beach(target=outputs)
    input_features = list(df.columns)[0:-2]

    model = MultiSite(
        batch_size=4,
        lookback=15,
        input_features=input_features,
        output_features=outputs,
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

    history = model.fit(data=df, callbacks={'tensorboard': {}})

    y = model.predict()
    activations = model.activations()
