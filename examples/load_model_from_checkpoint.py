# this example shows how to build the Models from `from_checkout` class method
# first we will train and save a simple model and load it from config file

import pandas as pd
import os

from AI4Water import Model


fname = os.path.join(os.path.dirname(os.path.dirname(__file__)), "AI4Water/data/nasdaq100_padding.csv")
df = pd.read_csv(fname)

model = Model(lookback=1, epochs=2,
              data=df,
              )

history = model.fit(indices='random')

w_path = model.path
# for clarity, delete the model, although it is overwritten
del model

# Load the `Model` from checkpoint, provide the checkpoint
cpath = os.path.join(w_path, "config.json") # "provide complete path of config file"
model = Model.from_config(cpath, data=df)

w_file = "weights_002_0.0010.hdf5"  # The file name of weights
model.load_weights(w_file)
x, y = model.predict(indices=model.test_indices, use_datetime_index=False)
