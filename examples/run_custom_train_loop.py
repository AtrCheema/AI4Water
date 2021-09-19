# This file describes the minimal example of customizing the whole training function ('fit')
# using AI4Water's Model class.

import tensorflow as tf

from ai4water import Model
from ai4water.datasets import arg_beach

# TODO put code in @tf.function
# TODO write validation code


class CustomModel(Model):

    def fit(self, data='training', callbacks=None, **kwargs):
        self.is_training = True
        # Instantiate an optimizer.
        optimizer = self.get_optimizer()
        # Instantiate a loss function.
        if self.api == 'functional':
            loss_fn = self.loss()
            _model = self._model
        else:
            loss_fn = self.loss
            _model = self

        # Prepare the training dataset.
        batch_size = self.config['batch_size']

        train_x, train_label = self.training_data()

        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_label))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        for epoch in range(self.config['epochs']):
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
                    logits = _model(x_batch_train, training=True)  # Logits for this minibatch

                    logits_obj = logits[mask]
                    # Compute the loss value for this minibatch.
                    loss_value = tf.keras.backend.mean(loss_fn(y_obj, logits_obj))

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, _model.trainable_weights)

                # grads = [tf.clip_by_norm(g, 1.0) for g in grads]
                grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, _model.trainable_weights))

                # Log every 200 batches.
                if step % 20 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %s samples" % ((step + 1) * 64))
        return loss_value


layers = {"LSTM_0": {'units': 64, 'return_sequences': True},
          "LSTM_1": 32,
          "Dropout": 0.3,
          "Dense": 1
          }

model = CustomModel(model={'layers': layers},
                    batch_size=12,
                    lookback=15,
                    lr=8.95e-5,
                    allow_nan_labels=2,
                    epochs=10,
                    data=arg_beach(),
                    train_data='random'
                    )
history = model.fit(callbacks={'tensorboard': True})

test_pred, test_obs = model.predict()
train_pred, train_obs = model.predict(data='training')
