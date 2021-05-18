# this example shows how to build the Models from `from_checkout` class method
# first we will train and save a simple model and load it from config file

import os

from AI4Water import Model
from AI4Water.utils.datasets import load_nasdaq
from AI4Water.utils.utils import find_best_weight

df = load_nasdaq()

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

w_file = find_best_weight(os.path.join(w_path, "weights"))  # The file name of weights
model.load_weights(w_file)
x, y = model.predict(indices=model.test_indices, use_datetime_index=False)
