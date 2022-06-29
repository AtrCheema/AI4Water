"""
===================================
machine learning for classification
===================================
"""

from ai4water import Model
from ai4water.datasets import MtropicsLaos

#%%

laos = MtropicsLaos()
data = laos.make_classification(lookback_steps=1)
data.shape

#

model = Model(
    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1:],
    model="XGBClassifier",
    split_random=True,
    train_fraction=1.0,
    x_transformation="zscore",
)

#%%
h = model.fit(data=data)

#%%
p = model.predict_on_validation_data(data=data)