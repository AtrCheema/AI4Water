from run_model import make_model
from models import Model
from models.global_variables import keras, tf

import pandas as pd


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


data_config, nn_config, total_intervals = make_model(lstm_units=64,
                                                     dropout=0.4,
                                                     rec_dropout=0.5,
                                                     lstm_act='relu',
                                                     batch_size=32,
                                                     lookback=15,
                                                     lr=8.95e-5)

df = pd.read_csv('data/all_data_30min.csv')

model = Model(data_config=data_config,
              nn_config=nn_config,
              data=df
              )

model.KModel = CustomModel

model.build_nn()

history = model.train_nn(indices='random')

# y, obs = model.predict()
