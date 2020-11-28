# this example shows how to build the Models from `from_checkout` class method
# first we will train and save a simple model and load it from config file

import pandas as pd
import os

from dl4seq.utils import make_model
from dl4seq import Model


data_config, nn_config = make_model(lookback=1)

fname = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dl4seq/data/nasdaq100_padding.csv")
df = pd.read_csv(fname)

model = Model(data_config=data_config,
              nn_config=nn_config,
              data=df,
              )

model.build()

history = model.train(indices='random')

# for clarity, delete the model, although it is overwritten
del model

# Load the `Model` from checkpoint, provide the checkpoint
cpath = "provide complete path of config file"
model = Model.from_config(cpath, data=df)

model.build()

w_file = "file_name.hdf5"
model.load_weights(w_file)
x, y = model.predict(indices=model.test_indices, use_datetime_index=False)
