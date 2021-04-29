#How to use dl4seq for classification problems

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

from AI4Water import Model

data_class = load_diabetes()
cols = data_class['feature_names'] + ['target']
df = pd.DataFrame(np.concatenate([data_class['data'], data_class['target'].reshape(-1,1)], axis=1), columns=cols)

model = Model(
              data=df,
              inputs=data_class['feature_names'],
              outputs=['target'],
              lookback=1,
              batches="2d",
              val_fraction=0.0,
              model= {'DecisionTreeRegressor': {"max_depth": 3, "criterion": "mae"}},
              transformation=None
              )

h = model.fit()

x, _, y = model.train_data()
