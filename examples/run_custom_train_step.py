# This file shows a minimal example how to customize 'train_step' using the Model class of AI4water

import tensorflow as tf

from ai4water import Model
from ai4water.datasets import arg_beach


class CustomModel(Model):

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


model = CustomModel(
    model={"layers": {"Dense_0": 8,
                      "Dense_1": 1,
                      }
           },
    batch_size=32,
    lookback=1,
    lr=8.95e-5,
    data=arg_beach(),
    epochs=2,
    train_data='random',
)



history = model.fit()

#y, obs = model.predict()
