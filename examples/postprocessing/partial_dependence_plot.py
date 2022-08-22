"""
=======================
partial dependence plot
=======================
"""

from ai4water import Model
from ai4water.datasets import busan_beach
from ai4water.postprocessing.explain import PartialDependencePlot

# sphinx_gallery_thumbnail_number = 5
#%%

data = busan_beach()
input_features = data.columns.tolist()[0:-1]
output_features = data.columns.tolist()[-1:]

############################################################

model = Model(model="XGBRegressor",
              verbosity=0)

# train model

model.fit(data=data)

# get data

x, _ = model.training_data()

# initiate plotter

pdp = PartialDependencePlot(model.predict,
                            x,
                            model.input_features,
                            save=False,
                            num_points=14)

pdp.plot_1d("tide_cm")

############################################################

pdp.plot_1d("tide_cm", show_dist_as="grid")

############################################################

pdp.plot_1d("tide_cm", show_dist=False)

############################################################


pdp.plot_1d("tide_cm", show_dist_as="grid", ice=False)

############################################################

pdp.plot_1d("tide_cm", show_dist=False, ice=False,
            model_expected_value=True)

############################################################

pdp.plot_1d("tide_cm", show_dist=False, ice=False,
            feature_expected_value=True)
