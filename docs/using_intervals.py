# This file shows how to build and set time series prediction problem when we have missing values as a chunk in target
# data. # We will insert some nan values (as a representative of missing values) in chunks i.e. continuous missing
# values after certain values. These nan values will then be ignored using the feature `intervals`.
import numpy as np
import pandas as pd

from models import DualAttentionModel
from run_model import make_model


df = pd.read_csv("../data/nasdaq100_padding.csv")
out = df["NDX"]
print(out.isna().sum())

# put four chunks of missing intervals
intervals = [(100, 200), (1000, 2000), (10000, 11000)]

for interval in intervals:
    st,en = interval[0], interval[1]
    out[st:en] = np.nan

df["NDX"] = out
print("{} nan values created in NDX column".format(out.isna().sum()))

# verify the missing values
print(df[98:108])

data_config, nn_config, _ = make_model(batch_size=16,
                                                     lookback=5,
                                                     lr=0.0001)

model = DualAttentionModel(data_config=data_config,
                           nn_config=nn_config,
                           data=df,
                           intervals=[(0, 99), (200, 999), (2000, 9999), (11000, 40561)]
                           )

# model.build_nn()

#history = model.train_nn(indices='random')

#y, obs = model.predict(indices=model.test_indices)

#model.plot_activations(st=0,save=True)

# Since we are using DualAttentionModel which requires observations at previous steps, we can not make
# predictions at steps which are skipped from `intervals`. However, for a model which does not require previous
# observations such as simple LSTM (`Model`), we can set the intervals to `None` using `model.intervals=None` and then
# make predictions at all time steps because we have missing values only in target values and not in input values.