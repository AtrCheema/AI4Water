"""
=========================================================
Comparison of XGBRegressor with different transformations
=========================================================
"""

from ai4water.experiments import TransformationExperiments
from ai4water.hyperopt import Categorical, Integer, Real
from ai4water.utils.utils import dateandtime_now

from ai4water.datasets import busan_beach

data = busan_beach()
input_features = data.columns.tolist()[0:-1]
output_features = data.columns.tolist()[-1:]


class MyTransformationExperiments(TransformationExperiments):

    def update_paras(self, **kwargs):

        y_transformation = kwargs.pop('y_transformation')
        if y_transformation == "none":
            y_transformation = None

        return {
            'model': {"XGBRegressor": kwargs},
            'y_transformation': y_transformation
        }

cases = {
    'model_None': {'y_transformation': 'none'},
    'model_minmax': {'y_transformation': 'minmax'},
    'model_zscore': {'y_transformation': 'zscore'},
    'model_center': {'y_transformation': 'center'},
    'model_scale': {'y_transformation': 'scale'},
    'model_robust': {'y_transformation': 'robust'},
    'model_quantile': {'y_transformation': 'quantile'},
    'model_box_cox': {'y_transformation': {'method': 'box-cox', 'treat_negatives': True, 'replace_zeros': True}},
    'model_yeo-johnson': {'y_transformation': 'yeo-johnson'},
    'model_sqrt': {'y_transformation': 'sqrt'},
    'model_log': {'y_transformation': {'method':'log', 'treat_negatives': True, 'replace_zeros': True}},
    'model_log10': {'y_transformation': {'method':'log10', 'treat_negatives': True, 'replace_zeros': True}},
    "model_pareto": {"y_transformation": "pareto"},
    "model_vast": {"y_transformation": "vast"},
    "model_mmad": {"y_transformation": "mmad"}
         }


num_samples=10
search_space = [
# maximum number of trees that can be built
Integer(low=10, high=30, name='iterations', num_samples=num_samples),
# Used for reducing the gradient step.
Real(low=0.09, high=0.3, prior='log', name='learning_rate', num_samples=num_samples),
# Coefficient at the L2 regularization term of the cost function.
Real(low=0.5, high=5.0, name='l2_leaf_reg', num_samples=num_samples),
# arger the value, the smaller the model size.
Real(low=0.1, high=10, name='model_size_reg', num_samples=num_samples),
# percentage of features to use at each split selection, when features are selected over again at random.
Real(low=0.1, high=0.5, name='rsm', num_samples=num_samples),
# number of splits for numerical features
Integer(low=32, high=50, name='border_count', num_samples=num_samples),
# The quantization mode for numerical features.  The quantization mode for numerical features.
Categorical(categories=['Median', 'Uniform', 'UniformAndQuantiles',
                        'MaxLogSum', 'MinEntropy', 'GreedyLogSum'], name='feature_border_type')
]

x0 = [10, 0.11, 1.0, 1.0, 0.2, 45, "Uniform"]
experiment = MyTransformationExperiments(cases=cases,
                                         input_features=input_features,
                                         output_features = output_features,
                                         param_space=search_space,
                                         x0=x0,
                                         verbosity=0,
                                         epochs=5,
                                         exp_name = f"xgb_y_exp_{dateandtime_now()}")

experiment.fit(data = data,
               run_type='dry_run'
               )

experiment.plot_improvement('nse')

