# This file shows a minimal example how to customize 'train_step' using the functional api of AI4water

import tensorflow as tf

assert int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) >= 230, f"""This example is only tested with
        tensorflow versions above 2.3.0. Your version is {tf.__version__}"""

from ai4water.functional import Model
from ai4water.datasets import arg_beach


class CustomModel(tf.keras.models.Model):

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


model = Model(
    model={"layers": {"Dense": 8}},
    batch_size=32,
    lookback=1,
    lr=8.95e-5,
    data=arg_beach(),
    epochs=2,
    KModel=CustomModel,
    train_data='random',
)

history = model.fit()

# since the statement 'custom train_step' is printed, we have verified that tensorflow
# used our own customized train_step during training.
