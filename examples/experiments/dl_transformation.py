"""
=================================================
Comparison of LSTM with different transformations
=================================================
"""

from ai4water.experiments import TransformationExperiments
from ai4water.models import LSTM
from ai4water.hyperopt import Categorical, Integer
from ai4water.utils.utils import dateandtime_now

from ai4water.datasets import busan_beach

lookback = 14

data = busan_beach()
input_features = data.columns.tolist()[0:-1]
output_features = data.columns.tolist()[-1:]


class MyTransformationExperiments(TransformationExperiments):

    def update_paras(self, **kwargs):
        _layers = LSTM(units=kwargs['units'],
                       input_shape=(lookback, len(input_features)),
                       activation=kwargs['activation'])

        y_transformation = kwargs['y_transformation']
        if y_transformation == "none":
            y_transformation = None

        return {
            'model': _layers,
            'ts_args': {'lookback': lookback},
            'batch_size': int(kwargs['batch_size']),
            'lr': float(kwargs['lr']),
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

search_space = [
    Integer(low=10, high=30, name='units', num_samples=10),
    Categorical(categories=['relu', 'elu', 'tanh', "linear"], name='activation'),
    Categorical(categories=[4, 8, 12, 16, 24, 32], name='batch_size'),
    Categorical(categories=[0.05, 0.02, 0.009, 0.007, 0.005,
                            0.003, 0.001, 0.0009, 0.0007, 0.0005, 0.0003,
                            0.0001, 0.00009, 0.00007, 0.00005], name='lr'),

]

x0 = [16, "relu", 32, 0.0001]
experiment = MyTransformationExperiments(cases=cases,
                                         input_features=input_features,
                                         output_features = output_features,
                                         param_space=search_space,
                                         x0=x0,
                                         verbosity=0,
                                         epochs=5,
                                         exp_name = f"ecoli_lstm_y_exp_{dateandtime_now()}")

experiment.fit(data = data,
               run_type='dry_run'
               )

experiment.plot_improvement('nse')

