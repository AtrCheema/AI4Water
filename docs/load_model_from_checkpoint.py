# this example shows how to build the Models from `from_checkout` class method
# first we will train and save a simple model and load it from config file

import pandas as pd

import os
from run_model import make_model
from models import Model


data_config, nn_config, total_intervals = make_model()

cwd = os.getcwd()
df = pd.read_csv(os.path.join(os.path.dirname(cwd), "data\\all_data_30min.csv"))

model = Model(data_config=data_config,
              nn_config=nn_config,
              data=df,
              )

model.build_nn()

history = model.train_nn(indices='random')

# for clarity, delete the model, although it is overwritten
del model

# Load the `Model` from checkpoint, provide the checkpoint
cpath = "provide exact path of config file"
model = Model.from_config(cpath, data=df)

model.build_nn()

w_file = "file_name.hdf5"
model.load_weights(w_file)
x, y = model.predict(indices=model.test_indices)
