"""
==================================
neural networks for classification
==================================
This file shows how to build neural networks for a classification problem.

"""
import numpy as np
import pandas as pd

from ai4water import Model
from ai4water.models import MLP
from sklearn.datasets import load_breast_cancer

#%%

bunch = load_breast_cancer()

data = pd.DataFrame(np.column_stack([
    bunch['data'][0:1000, :], bunch['target'][0:1000, :]
]),
    columns=bunch['feature_names'].tolist() + ['diagnostic'])

del bunch

data.shape

#%%

model = Model(
    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1:],
    model=MLP(units=10, mode="classification"),
    lr=0.009919,
    batch_size=8,
    split_random=True,
    x_transformation="zscore",
    epochs=200,
    loss="binary_crossentropy"
)
#
#%%
h = model.fit(data=data)

#%%
p = model.predict_on_validation_data(data=data)