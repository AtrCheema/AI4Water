# This file shows how to customize the loss function. We want to predict uncertainty by using quantile loss.
# In this problem, instead of predicted actual observation, we predict quantiles
# The loss value function is customized. We use pinball loss. https://www.lokad.com/pinball-loss-function-definition
# Inspired from https://www.kaggle.com/ulrich07/quantile-regression-with-keras

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from AI4Water import Model

class QuantileModel(Model):

    def inverse_transform(self, data, key):
        # todo
        return data

    def loss(self):

        return qloss

def qloss(y_true, y_pred):
    # Pinball loss for multiple quantiles
    qs = quantiles
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return keras.backend.mean(v)


# Define a dummy dataset consisting of 6 time-series.
rows = 2000
cols = 6
data = np.arange(int(rows*cols)).reshape(-1, rows).transpose()
data = pd.DataFrame(data, columns=['input_' + str(i) for i in range(cols)],
                    index=pd.date_range('20110101', periods=len(data), freq='H'))

# Define Model
layers = {'Dense_0': {'config':  {'units': 64, 'activation': 'relu'}},
          'Dropout_0': {'config':  {'rate': 0.3}},
          'Dense_1': {'config':  {'units': 32, 'activation': 'relu'}},
          'Dropout_1': {'config':  {'rate': 0.3}},
          'Dense_2': {'config':  {'units': 16, 'activation': 'relu'}},
          'Dense_3': {'config':  {'units': 9}}}

# Define Quantiles
quantiles = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]

# Initiate Model
model = QuantileModel(
    input_features=['input_' + str(i) for i in range(cols - 1)],
    output_features=['input_' + str(cols - 1)],
    lookback=1,
    model={'layers':layers},
    epochs=10,
    data=data,
    quantiles=quantiles)

# Train the model on first 1500 examples/points, 0.2% of which will be used for validation
model.fit()

# make predictions on a chunk of test data, which was retained while training
true_y, pred_y = model.predict()

model.plot_quantile(true_y, pred_y, 3, 5)
