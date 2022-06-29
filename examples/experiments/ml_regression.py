"""
=========================================
Comparison of machine learning algorithms
=========================================
"""


from ai4water.datasets import busan_beach
from ai4water.experiments import MLRegressionExperiments

########################################################

data = busan_beach()


comparisons = MLRegressionExperiments(
    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1:],
    split_random=True,
    verbosity=0
)


comparisons.fit(data=data,
                run_type="dry_run",
                include=['RandomForestRegressor',
                         'XGBRegressor',
                         'GaussianProcessRegressor',
                         'HistGradientBoostingRegressor',
                         "LGBMRegressor",
                         "GradientBoostingRegressor",
                         "CatBoostRegressor",
                         "XGBRFRegressor"
                         ])

comparisons.compare_errors('r2')

###############################################

best_models = comparisons.compare_errors('r2',
                                         cutoff_type='greater',
                                         cutoff_val=0.01)

################################################

comparisons.taylor_plot()
