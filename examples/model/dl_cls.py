"""
==================================
neural networks for classification
==================================
"""


from ai4water import Model
from ai4water.models import MLP
from ai4water.datasets import MtropicsLaos

#%%

laos = MtropicsLaos()
data = laos.make_classification(lookback_steps=1)
data.shape

#

model = Model(
    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1:],
    model=MLP(units=10, mode="classification"),
    lr=0.009919,
    batch_size=8,
    split_random=True,
    train_fraction=1.0,
    x_transformation="zscore",
    epochs=200,
    loss="binary_crossentropy"
)

#%%
h = model.fit(data=data)

#%%
p = model.predict_on_validation_data(data=data)