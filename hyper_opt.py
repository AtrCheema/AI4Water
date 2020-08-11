from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_evaluations, plot_objective, plot_convergence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from main import Model
from run_model import make_model


with open('opt_results.txt', 'w') as f:
    f.write('Hyper OPT Results')

dim_batch_size = Categorical(categories=[4, 8, 12, 24, 32], name='batch_size')
dim_lookback = Integer(low=5, high=20, prior='uniform', name='lookback')
dim_learning_rate = Real(low=1e-7, high=1e-4, prior='uniform', name='lr')
dim_lstm_units = Integer(low=32, high=256, prior='uniform', name='lstm_units')
dim_act_f = Categorical(categories=['relu', 'tanh', 'elu', 'LeakyRelu'], name='act_f')
dim_dropout = Real(low=0.1, high=0.5, prior='uniform', name='dropout')

dim_lstm1_units = Categorical(categories=[16, 32, 64, 128], name='lstm1_units')
dim_lstm2_units = Categorical(categories=[16, 32, 64, 128], name='lstm2_units')

default_values = [12, 10, 0.00001, 32, 32,  # 'relu', 0.4
                  ]

dimensions = [dim_batch_size, dim_lookback, dim_learning_rate,
              # dim_lstm_units, dim_act_f, dim_dropout,
              dim_lstm1_units, dim_lstm2_units]

def objective_fn(**kwargs):
    data_config, nn_config, total_intervals = make_model(**kwargs)

    df = pd.read_csv('data/nk_data.csv')

    model = Model(data_config=data_config,
                  nn_config=nn_config,
                  data=df,
                  # intervals=total_intervals
                  )


    model.build_nn()

    idx = np.arange(720)
    tr_idx, test_idx = train_test_split(idx, test_size=0.5, random_state=313)

    history = model.train_nn(indices=list(tr_idx))
    return np.min(history['val_loss'])

@use_named_args(dimensions=dimensions)
def fitness(batch_size, lookback, lr,  # lstm_units, act_f, dropout
            lstm1_units, lstm2_units
            ):

    mse = objective_fn(lstm_units=32,
                       dropout=0.4,
                       rec_dropout=0.4,
                       lstm_act='relu',
                       batch_size=batch_size,
                       lookback=lookback,
                       lr=lr,
                       enc_lstm1=lstm1_units,
                       enc_lstm2=lstm2_units)

    error = mse

    msg = """\nwith lstm1_units {}, lstm2_units {}, batch_size {} lookback {} lr {} val loss is {}
          """.format(lstm1_units, lstm2_units, batch_size, lookback, lr, error)
    print(msg)
    with open('opt_results.txt', 'a') as fp:
        fp.write(msg)

    return error


search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI',   # Expected Improvement.
                            n_calls=80,
                            # acq_optimizer='auto',
                            x0=default_values,
                            random_state=10)

_ = plot_evaluations(search_result)
plt.savefig('evaluations', dpi=400, bbox_inches='tight')
plt.show()

_ = plot_objective(search_result)
plt.savefig('objective', dpi=400, bbox_inches='tight')
plt.show()


_ = plot_convergence(search_result)
plt.savefig('convergence', dpi=400, bbox_inches='tight')
plt.show()