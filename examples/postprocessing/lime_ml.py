"""
==========================
Explaining model with LIME
==========================
This file shows how to use ``LimeExplainer`` class of AI4Water. The LimeExplainer
class is a wrapper around ``LIME`` library. It helps in making plots which explain
individual examples/samples of data.
"""

from ai4water import Model
from ai4water.datasets import busan_beach
from ai4water.postprocessing.explain import LimeExplainer

# sphinx_gallery_thumbnail_number = 3

#%%

data =busan_beach()
data.shape

#%%
model = Model(model="XGBRegressor",
              val_fraction=0.0,
             )

#%%
model.fit(data=data)

#%%
y_true, y_pred = model.predict_on_test_data(data=data, return_true=True)

#%%
for idx,t,p in zip(range(len(y_true)), y_true, y_pred):
    print(idx, t, p, abs(t.item() - p))

#%%
test_x, test_y = model.test_data(data=data)

#%%
train_x, train_y = model.training_data(data=data)

#%%
explainer = LimeExplainer(model,
                          test_x,
                          train_data=train_x, # the data on which model was trained
                          path=model.path,
                          feature_names=model.input_features, # names of features
                          mode=model.mode,
                          save=False
                          )

#%%
_ = explainer.explain_example(0, annotate=True, num_samples=10000)

#%%
_ = explainer.explain_example(19, annotate=True)

#%%
_ = explainer.explain_example(26, annotate=True)

#%%
_ = explainer.explain_example(40, annotate=True)

#%%
_ = explainer.explain_example(41, annotate=True)

#%%
_ = explainer.explain_example(42, annotate=True)

#%%
_ = explainer.explain_example(43, annotate=True)

#%%
_ = explainer.explain_example(44, annotate=True)

#%%
_ = explainer.explain_example(51, annotate=True)
