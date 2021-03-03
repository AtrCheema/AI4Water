import os

import tensorflow as tf
from tensorflow import keras
import pandas as pd

from dl4seq import Model


class CustomModel(keras.Model):
    def train_step(self, data):
        print('custom train_step')
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


fname = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dl4seq/data/nasdaq100_padding.csv")
df = pd.read_csv(fname)

model = Model(
    batch_size=32,
    lookback=1,
    lr=8.95e-5,
    data=df
)

model.KModel = CustomModel

history = model.fit(indices='random')

y, obs = model.predict()
