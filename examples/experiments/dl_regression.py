"""
=========================================
Comparison of deep learning architectures
=========================================
"""


from ai4water.datasets import busan_beach
from ai4water.experiments import DLRegressionExperiments

########################################################

data = busan_beach()


comparisons = DLRegressionExperiments(
    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1:],
    split_random=True,
    train_fraction=1.0,
    epochs=20,
    ts_args={"lookback": 12},
    verbosity=0
)


comparisons.fit(data=data,
                include=['MLP',
                         'LSTM',
                         'CNNLSTM',
                         'TCN',
                         #"TFT",
                         "LSTMAutoEncoder",
                         ])

comparisons.compare_errors('r2')

###############################################

best_models = comparisons.compare_errors('r2',
                                         cutoff_type='greater',
                                         cutoff_val=0.01)

################################################

comparisons.taylor_plot()
