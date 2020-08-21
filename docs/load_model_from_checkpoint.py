# this example shows how to build the Models from `from_checkout` class method

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