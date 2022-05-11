"""
================
Attention LSTM
================
"""

"""
This example illustrates how to use 'self-attention' mechanism op top of LSTM to make
the prediction of LSTM interpretable.
"""

import ai4water
ai4water.__version__

#########################################

import tensorflow as tf
tf.__version__

#########################################

import matplotlib.pyplot as plt
from ai4water import Model

from easy_mpl import imshow

import numpy as np
np.__version__

#########################################

seq_len = 20
num_inputs = 2
model = Model(
    model = {"layers": {
        "Input_1": {"shape": (seq_len, num_inputs)},
        "AttentionLSTM": {"num_inputs": num_inputs, "lstm_units": 16},
        "Dense": 1
    }},
)

#########################################

model.inputs

#########################################

model.outputs

#########################################

def add_numbers_before_delimiter(n: int,
                                 seq_length: int,
                                 index_1: int = None) -> (np.array, np.array):
    """
    Task: Add all the numbers that come before the delimiter.
    x = [1, 2, 3, 0, 4, 5, 6, 7, 8, 9]. Result is y =  6.
    @param n: number of samples in (x, y).
    @param seq_length: length of the sequence of x.
    @param index_1: index of the number that comes after the first 0.
    @return: returns two numpy.array x and y of shape (n, seq_length, 1) and (n, 1).
    """
    x = np.random.uniform(0, 1, (n, seq_length))
    y = np.zeros(shape=(n, 1))
    for i in range(len(x)):
        if index_1 is None:
            a = np.random.choice(range(1, len(x[i])), size=1, replace=False)[0]
        else:
            a = index_1
        y[i] =  np.sum(x[i, 0:a])
        x[i, a] = 0.0

    x = np.expand_dims(x, axis=-1)
    return x, y

#########################################

x_train1, y_train1 = add_numbers_before_delimiter(20_00, seq_len)
x_train2, y_train2 = add_numbers_before_delimiter(20_00, seq_len)
x_train = np.concatenate([x_train1, x_train2], axis=2)
y_train = y_train1 + y_train2
x_train.shape, y_train.shape

x_val1, y_val1 = add_numbers_before_delimiter(5_00, seq_len)
x_val2, y_val2 = add_numbers_before_delimiter(5_00, seq_len)
x_val = np.concatenate([x_val1, x_val2], axis=2)
y_val = y_val1 + y_val2
x_val.shape, y_val.shape

#########################################

h = model.fit(x=x_train, y=y_train,
              validation_data=(x_val, y_val),
              epochs=1000, verbose=1
              )

#########################################

# Now prepare test data

x_test1, y_test1 = add_numbers_before_delimiter(5_00, seq_len)
x_test2, y_test2 = add_numbers_before_delimiter(5_00, seq_len)
x_test = np.concatenate([x_test1, x_test2], axis=2)
y_test = y_test1 + y_test2
x_test.shape, y_test.shape

#########################################

attention_weights = model.get_attention_lstm_weights(x_test)


attention_weights.keys()

#########################################

num_examples = 10  # number of examples to show

fig, axis = plt.subplots(2, sharex="all")

imshow(attention_weights["self_attention"][0:num_examples],
       ylabel="Examples",
       title="Predicted important steps", cmap="hot",
      ax=axis[0], show=False)

a = x_test1[0:num_examples].reshape(-1, seq_len)
a = np.where(a==0.0, 1.0, 0.0)
imshow(a, ylabel="Examples",
       xlabel="Sequence Length (lookback steps)",
       xticklabels=np.arange(20),
       title="Actual important steps", cmap="hot",
      ax=axis[1])

#########################################

fig, axis = plt.subplots(2, sharex="all")

imshow(attention_weights["self_attention_1"][0:num_examples],
       ylabel="Examples",
       title="Predicted important steps", cmap="hot",
      ax=axis[0], show=False)

a = x_test2[0:num_examples].reshape(-1, seq_len)
a = np.where(a==0.0, 1.0, 0.0)
imshow(a, ylabel="Examples",
       xlabel="Sequence Length (lookback steps)",
       xticklabels=np.arange(20),
       title="Actual important steps", cmap="hot",
      ax=axis[1])

