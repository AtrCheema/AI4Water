#How to use dl4seq for classification problems

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

from dl4seq.utils import make_model
from dl4seq import Model

data_class = load_diabetes()
cols = data_class['feature_names'] + ['target']
df = pd.DataFrame(np.concatenate([data_class['data'], data_class['target'].reshape(-1,1)], axis=1), columns=cols)

config = make_model(
    inputs=data_class['feature_names'],
    outputs=['target'],
    lookback=1,
    batches="2d",
    val_fraction=0.0,
    ml_model = 'DecisionTreeRegressor',
    ml_model_args = {"max_depth":3, "criterion":"mae"},
    transformation=None
)

model = Model(config,
              data=df)

h = model.train()

x,y = model.train_data()
