# How to use AI4Water for regression problems using classifical ML algorithms

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

from ai4water import Model

data_class = load_diabetes()
cols = data_class['feature_names'] + ['target']
df = pd.DataFrame(np.concatenate([data_class['data'], data_class['target'].reshape(-1, 1)], axis=1), columns=cols)

model = Model(
    data=df,
    input_features=data_class['feature_names'],
    output_features=['target'],
    lookback=1,
    batches="2d",
    val_fraction=0.0,
    model={'DecisionTreeRegressor': {"max_depth": 3, "criterion": "mae"}},
    transformation=None
)

h = model.fit()

x, y = model.training_data()
