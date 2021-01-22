# https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit

import tensorflow as tf

#@tf.function
def train_step(keras_model, data):
    if int(''.join(tf.__version__.split('.')[0:2])) < 23:
        raise NotImplementedError(f"ignoring nan in labels can not be done in tf version {tf.__version__}")
    # Unpack the data. Its structure depends on your model and
    # on what you pass to `fit()`.
    x, y = data

    with tf.GradientTape() as tape:
        y_pred = keras_model(x, training=True)  # Forward pass

        mask = tf.greater(y, 0.0)
        true_y = tf.boolean_mask(y, mask)
        pred_y = tf.boolean_mask(y_pred, mask)

        # Compute the loss value
        # (the loss function is configured in `compile()`)
        loss = keras_model.compiled_loss(true_y, pred_y, regularization_losses=keras_model.losses)

    # Compute gradients
    trainable_vars = keras_model.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # Update weights
    keras_model.optimizer.apply_gradients(zip(gradients, trainable_vars))
    # Update metrics (includes the metric that tracks the loss)
    keras_model.compiled_metrics.update_state(true_y, pred_y)
    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in keras_model.metrics}


#@tf.function
def test_step(keras_model, data):
    if int(''.join(tf.__version__.split('.')[0:2])) < 23:
        raise NotImplementedError(f"ignoring nan in labels can not be done in tf version {tf.__version__}")
    print('custom test_step')
    x, y = data

    y_pred = keras_model(x, training=False)  # compute predictions

    mask = tf.greater(y, 0.0)
    true_y = tf.boolean_mask(y, mask)
    pred_y = tf.boolean_mask(y_pred, mask)

    keras_model.compiled_loss(true_y, pred_y, regularization_losses=keras_model.losses)

    keras_model.compiled_metrics.update_state(true_y, pred_y)

    return {m.name: m.result() for m in keras_model.metrics}