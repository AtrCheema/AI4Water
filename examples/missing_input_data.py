# Sometimes we have some missing values in input data.
# This file shows how to build and set time series prediction problem when we have missing values as a chunk in target
# data. # We will insert some nan values (as a representative of missing values) in chunks i.e. continuous missing
# values after certain values. These nan values will then be ignored using the feature `intervals`.
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from dl4seq import DualAttentionModel

tf.compat.v1.disable_eager_execution()

fname = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dl4seq/data/nasdaq100_padding.csv")
df = pd.read_csv(fname)

# df.index = pd.date_range("20110101", periods=len(df), freq='H')
out = df["NDX"]
print(out.isna().sum())

# put four chunks of missing intervals
intervals = [(100, 200), (1000, 8000), (10000, 31000)]

for interval in intervals:
    st, en = interval[0], interval[1]
    out[st:en] = np.nan

df["NDX"] = out
print("{} nan values created in NDX column".format(out.isna().sum()))

# verify the missing values
print(df[98:108])

model = DualAttentionModel(
                    data=df,
                    intervals=[(0, 99), (200, 999), (8000, 9999), (31000, 40560)],
                    batch_size=32,
                    lookback=5,
                    lr=0.0001
                           )

history = model.fit(indices='random')

y, obs = model.predict(indices=model.test_indices, use_datetime_index=False)
# tr_y, tr_obs = model.predict(indices=model.train_indices, prefix='train', use_datetime_index=False)

model.view_model(st=0, save=True)  # takes a lot of time to save all plots

# Since we are using DualAttentionModel which requires observations at previous steps, we can not make
# predictions at steps which are skipped from `intervals`. However, for a model which does not require previous
# observations such as simple LSTM (`Model`), we can set the intervals to `None` using `model.intervals=None` and then
# make predictions at all time steps because we have missing values only in target values and not in input values.
