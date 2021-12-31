# this example shows how to build the Models from `from_checkout` class method
# first we will train and save a simple model and load it from config file

import os

from ai4water import Model
from ai4water.datasets import arg_beach

model = Model(lookback=1,
              epochs=2,
              model={"layers": {"Dense": 8, "Dense_1": 1}},
              train_data='random'
              )

history = model.fit(data=arg_beach())

w_path = model.path
# for clarity, delete the model, although it is overwritten
del model

# Load the `Model` from checkpoint, provide the checkpoint
cpath = os.path.join(w_path, "config.json") # "provide complete path of config file"
model = Model.from_config(cpath)

model.update_weights()
y = model.predict(data=arg_beach())
