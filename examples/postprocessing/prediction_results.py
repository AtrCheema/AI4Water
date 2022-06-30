"""
====================================
postprocessing of prediction results
====================================
This file shows how to post-process prediction results
"""

from ai4water.datasets import busan_beach
from ai4water import Model

# sphinx_gallery_thumbnail_number = -1

#%%

model = Model(model="XGBRegressor")
#%%

model.fit(data=busan_beach())

#%%
model.prediction_distribution(feature="tide_cm", data=busan_beach(), show_percentile=True)

#%%

model.feature_interaction(
    ['tide_cm', 'sal_psu'],
    data=busan_beach(),
    plot_type="circles",
    cust_grid_points=[[-41.4, -20.0, 0.0, 20.0, 42.0],
                      [33.45, 33.7, 33.9, 34.05, 34.4]],
)

#%%

model.feature_interaction(
    ['tide_cm', 'sal_psu'],
    data=busan_beach(),
    annotate_counts=True,
    annotate_colors=("black", "black"),
    annotate_fontsize=10,
    cust_grid_points=[[-41.4, -20.0, 0.0, 20.0, 42.0],
                      [33.45, 33.7, 33.9, 34.05, 34.4]],
)

#%%