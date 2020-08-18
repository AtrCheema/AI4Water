import tensorflow as tf


def tf_nse(true, pred, name='NSE'):
    """ Nash-Sutcliff efficiency to be used as loss function. It is subtracted from one before being returned"""
    neum = tf.reduce_sum(tf.square(tf.subtract(pred, true)))
    denom = tf.reduce_sum(tf.square(tf.subtract(true, tf.math.reduce_mean(true))))
    const = tf.constant(1.0, dtype=tf.float32)
    _nse = tf.subtract(const, tf.math.divide(neum, denom), name=name)
    return tf.subtract(const, _nse, name=name + '_LOSS')
