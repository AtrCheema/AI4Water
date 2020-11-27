from dl4seq.utils import make_model
from dl4seq.main import Model

import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd

# TODO put code in @tf.function
# TODO write validation code


class CustomModel(Model):

    def train(self, st=0, en=None, indices=None, **callbacks):
        # Instantiate an optimizer.
        optimizer = keras.optimizers.Adam(learning_rate=self.nn_config['lr'])
        # Instantiate a loss function.
        loss_fn = self.loss

        # Prepare the training dataset.
        batch_size = self.data_config['batch_size']

        indices = self.get_indices(indices)

        train_x, train_y, train_label = self.fetch_data(data=self.data, st=st, en=en, shuffle=True,
                                                        write_data=self.data_config['CACHEDATA'],
                                                        indices=indices)

        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_label))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        for epoch in range(self.nn_config['epochs']):
            print("\nStart of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, full_outputs) in enumerate(train_dataset):

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables autodifferentiation.
                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    mask = tf.greater(tf.reshape(full_outputs, (-1,)), 0.0)  # (batch_size,)
                    y_obj = full_outputs[mask]  # (vals_present, 1)

                    if y_obj.shape[0] < 1:  # no observations present for this batch so skip this
                        continue
                    logits = self._model(x_batch_train, training=True)  # Logits for this minibatch

                    logits_obj = logits[mask]
                    # Compute the loss value for this minibatch.
                    loss_value = keras.backend.mean(loss_fn(y_obj, logits_obj))

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, self._model.trainable_weights)

                # grads = [tf.clip_by_norm(g, 1.0) for g in grads]
                grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, self._model.trainable_weights))

                # Log every 200 batches.
                if step % 20 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %s samples" % ((step + 1) * 64))
        return loss_value


# input features in data_frame
input_features = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input8',
                  'input11']
# column in dataframe to bse used as output/target
outputs = ['target7']

layers = {"LSTM_0": {'config': {'units': 64, 'return_sequences': True}},
          "LSTM_1": {'config':  {'units': 32}},
          "Dropout": {'config':  {'rate': 0.3}},
          "Dense": {'config':  {'units': 1}}
          }

data_config, nn_config, total_intervals = make_model(layers=layers,
                                                     batch_size=12,
                                                     lookback=15,
                                                     lr=8.95e-5,
                                                     ignore_nans=False,
                                                     inputs=input_features,
                                                     outputs=outputs,
                                                     epochs=10)

fname = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data\\data_30min.csv")
df = pd.read_csv(fname)  # must be 2d dataframe


model = CustomModel(data_config=data_config,
                    nn_config=nn_config,
                    data=df
                    )

model.build()

history = model.train(indices='random', tensorboard=True)

test_pred, test_obs = model.predict(indices=model.test_indices)
train_pred, train_obs = model.predict(indices=model.train_indices)
