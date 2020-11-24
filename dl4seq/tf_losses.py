import tensorflow as tf
import tensorflow.keras.backend as K
from TSErrors import FindErrors
import numpy as np


def reset_graph(seed=313):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)  # tf.random.set_seed(seed)  #
    np.random.seed(seed)


def tf_nse(true, _pred, name='NSE'):
    """ Nash-Sutcliff efficiency to be used as loss function. It is subtracted from one before being returned"""
    neum = tf.reduce_sum(tf.square(tf.subtract(_pred, true)))
    denom = tf.reduce_sum(tf.square(tf.subtract(true, tf.math.reduce_mean(true))))
    const = tf.constant(1.0, dtype=tf.float32)
    _nse = tf.subtract(const, tf.math.divide(neum, denom), name=name)
    return tf.subtract(const, _nse, name=name + '_LOSS')


def corr_coeff(true, predicted):
    """ Pearson correlation coefficient

    https://stackoverflow.com/a/58890795/5982232
    """
    mx = tf.math.reduce_mean(true)
    my = tf.math.reduce_mean(predicted)
    xm, ym = true - mx, predicted - my
    r_num = tf.math.reduce_mean(tf.multiply(xm, ym))
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return r_num / r_den


def tf_kge(true, predicted):
    """ Kling Gupta efficiency. It is not being subtracted from 1.0 so that it can be used as loss"""
    tf_cc = corr_coeff(true, predicted)
    tf_alpha = tf.math.reduce_std(predicted) / tf.math.reduce_std(true)
    tf_beta = K.sum(predicted) / K.sum(true)
    return K.sqrt(K.square(tf_cc - 1.0) + K.square(tf_alpha - 1.0) + K.square(tf_beta - 1.0))


def tf_r2(true, predicted):
    """
    https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/34019
    it is like r2_square score of sklearn which can be negative
    Not being subtracted from 1.0
    """
    r = corr_coeff(true, predicted)
    return r ** 2


def tf_r2_mod(true, predicted):
    """
    https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/34019
    it is like r2_square score of sklearn which can be negative
    Not being subtracted from 1.0
    """
    ss_res = K.sum(K.square(true - predicted))
    ss_tot = K.sum(K.square(true - K.mean(true)))
    return ss_res / (ss_tot + K.epsilon())


def tf_nse_beta(true, predicted, name='nse_beta'):
    """
    Beta decomposition of NSE. See Gupta et. al 2009
    used in kratzert et al., 2018
    """
    const = tf.constant(1.0, dtype=tf.float32)
    nse_beta = (K.mean(predicted) - K.mean(true)) / K.std(true)
    return tf.subtract(const, nse_beta, name=name + '_LOSS')


def tf_nse_alpha(true, predicted, name='nse_alpha'):
    """
    Alpha decomposition of NSE. See Gupta et. al 2009
    used in kratzert et al., 2018
    It is being subtracted from 1.0
    """
    const = tf.constant(1.0, dtype=tf.float32)
    nse_alpha = K.std(predicted) / K.std(true)
    return tf.subtract(const, nse_alpha, name=name + '_LOSS')

def pbias(true, predicted):

    _sum = K.sum(tf.subtract(predicted, true))
    _a = tf.divide(_sum, K.sum(true))
    return 100.0 * _a


def nse(true, _pred, name='NSE'):
    """Nash-Sutcliff efficiency to be used as loss function. It is subtracted from one before being returned"""
    neum = tf.reduce_sum(tf.square(tf.subtract(_pred, true)))
    denom = tf.reduce_sum(tf.square(tf.subtract(true, tf.math.reduce_mean(true))))
    const = tf.constant(1.0, dtype=tf.float32)
    _nse = tf.subtract(const, tf.math.divide(neum, denom), name=name)
    return 1.0 - tf.subtract(const, _nse, name=name + '_LOSS')


def kge(true, predicted):
    """ Kling Gupta efficiency. It is not being subtracted from 1.0 so that it can be used as loss"""
    tf_cc = corr_coeff(true, predicted)
    tf_alpha = tf.math.reduce_std(predicted) / tf.math.reduce_std(true)
    tf_beta = K.sum(predicted) / K.sum(true)
    return 1.0 - K.sqrt(K.square(tf_cc - 1.0) + K.square(tf_alpha - 1.0) + K.square(tf_beta - 1.0))


if __name__ == "__main__":
    reset_graph()
    _true = np.random.random(10)
    pred = np.random.random(10)

    t = tf.convert_to_tensor(_true, dtype=tf.float32)
    p = tf.convert_to_tensor(pred, dtype=tf.float32)

    np_errors = FindErrors(_true, pred)

    print('corr_coeff {:<10.5f} {:<10.5f}'.format(np_errors.corr_coeff(), K.eval(corr_coeff(t, p))))
    print('r2 {:<10.5f} {:<10.5f}'.format(np_errors.r2(), 1.0-K.eval(tf_r2(t, p))))
    print('nse {:<10.5f} {:<10.5f}'.format(np_errors.nse(), 1.0 - K.eval(tf_nse(t, p))))
    print('kge {:<10.5f} {:<10.5f}'.format(np_errors.kge(), 1.0 - K.eval(tf_kge(t, p))))
    print('r2_mod {:<10.5f} {:<10.5f}'.format(np_errors.r2_mod(), 1.0 - K.eval(tf_r2_mod(t, p))))
    print('nse_beta {:<10.5f} {:<10.5f}'.format(np_errors.nse_beta(), 1.0 - K.eval(tf_nse_beta(t, p))))
    print('nse_alpha {:<10.5f} {:<10.5f}'.format(np_errors.nse_alpha(), 1.0 - K.eval(tf_nse_alpha(t, p))))
    print('pbias {:<10.5f} {:<10.5f}'.format(np_errors.pbias(), K.eval(pbias(t, p))))
