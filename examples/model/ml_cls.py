"""
===================================
machine learning for classification
===================================
"""

import numpy as np
import pandas as pd
import ai4water
from ai4water import Model
from sklearn.datasets import load_breast_cancer

ai4water.__version__

#%%

bunch = load_breast_cancer()

data = pd.DataFrame(np.column_stack([bunch['data'], bunch['target']]),
                    columns=bunch['feature_names'].tolist() + ['diagnostic'])

#

model = Model(
    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1:],
    model="XGBClassifier",
    split_random=True,
    x_transformation="zscore",
)

#%%
h = model.fit(data=data)

#%%
p = model.predict_on_validation_data(data=data)