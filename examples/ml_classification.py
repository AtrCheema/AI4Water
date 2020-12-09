#How to use dl4seq for classification problems

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

from dl4seq.utils import make_model
from dl4seq import Model

data_class = load_breast_cancer()
cols = data_class['feature_names'].tolist() + ['target']
df = pd.DataFrame(np.concatenate([data_class['data'], data_class['target'].reshape(-1,1)], axis=1), columns=cols)

config = make_model(
    inputs=data_class['feature_names'].tolist(),
    outputs=['target'],
    lookback=1,
    batches="2d",
    val_fraction=0.0,
    ml_model = 'DecisionTreeClassifier',
    ml_model_args = {"max_depth":4, "random_state":313},
    transformation=None,
    problem="classification"
)

model = Model(config,
              data=df
              )

h = model.train()

model.view_model()
