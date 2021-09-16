# this example shows how to build the Models from `from_checkout` class method
# first we will train and save a simple model and load it from config file

import os

from ai4water import Model
from ai4water.datasets import arg_beach
from ai4water.utils.utils import find_best_weight

model = Model(lookback=1,
              epochs=2,
              model={"layers": {"Dense": 8, "Dense_1": 1}},
              train_data='random',
              data=arg_beach(),
              )

history = model.fit()

w_path = model.path
# for clarity, delete the model, although it is overwritten
del model

# Load the `Model` from checkpoint, provide the checkpoint
cpath = os.path.join(w_path, "config.json") # "provide complete path of config file"
model = Model.from_config(cpath, data=arg_beach())

w_file = find_best_weight(os.path.join(w_path, "weights"))  # The file name of weights
model.update_weights(w_file)
x, y = model.predict()
