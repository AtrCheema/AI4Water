#How to use AI4Water for classification problems

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

from AI4Water import Model

data_class = load_breast_cancer()
cols = data_class['feature_names'].tolist() + ['target']
df = pd.DataFrame(np.concatenate([data_class['data'], data_class['target'].reshape(-1,1)], axis=1), columns=cols)

model = Model(
    data=df,
    input_features=data_class['feature_names'].tolist(),
    output_features=['target'],
    val_fraction=0.0,
    model={"DecisionTreeClassifier":{"max_depth": 4, "random_state": 313}},
    transformation=None,
    problem="classification"
)

h = model.fit()

# make prediction on test data
t,p = model.predict()

# get some useful plots
model.interpret()

#**********Evaluate the model on test data using only input
x,y = model.test_data()
pred = model.evaluate(x=x) # using only `x`

