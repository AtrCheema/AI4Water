"""
==========================
Explaining model with SHAP
==========================
This file shows how to use ``SHAPExplainer`` class of AI4Water. The LimeExplainer
class is a wrapper around ``SHAP`` library. It helps in making plots which explain
individual examples/samples of data.
"""

from ai4water import Model
from ai4water.datasets import MtropicsLaos
from ai4water.postprocessing.explain import ShapExplainer

# sphinx_gallery_thumbnail_number = 5

#%%

laos = MtropicsLaos()
data = laos.make_regression()
data.shape

#%%

model = Model(model="XGBRegressor",
              prefix="ecoli_shap",  # folder name to save results
              val_fraction=0.0,
             )

#%%
model.fit(data=data)

#%%
y_true, y_pred = model.predict_on_test_data(data=data, return_true=True)

#%%
for idx,t,p in zip(range(len(y_true)), y_true, y_pred):
    print(idx, t, p)

#%%
train_x, train_y = model.training_data()

#%%
test_x, test_y = model.test_data()

#%%
explainer = ShapExplainer(
    model,
    test_x,
    train_data=train_x, # the data on which model was trained
    path=model.path,
    feature_names=model.input_features, # names of features
    num_means=10,
    explainer="TreeExplainer",
    save=False
)

#%%
explainer.plot_shap_values()

#%%
explainer.heatmap()

#%%
explainer.beeswarm_plot()

#%%
explainer.summary_plot()

#%%
explainer.waterfall_plot_single_example(0)

#%%
explainer.waterfall_plot_single_example(40)

#%%
explainer.waterfall_plot_single_example(41)

#%%
explainer.waterfall_plot_single_example(42)

#%%
explainer.waterfall_plot_single_example(43)

#%%
explainer.waterfall_plot_single_example(44)

#%%
# force plots
#--------------

#%%
explainer.force_plot_single_example(40)

#%%
explainer.force_plot_single_example(41)

#%%
explainer.force_plot_single_example(42)

#%%
explainer.force_plot_single_example(43)

#%%
explainer.force_plot_single_example(44)

#%%
# dependence plots
#-----------------

explainer.dependence_plot_single_feature('susp_pm')

#%%

explainer.dependence_plot_all_features()

#%%
explainer.decision_plot()

#%%

explainer.decision_plot(indices=range(39, 45), highlight=3)

