from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_evaluations, plot_objective, plot_convergence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import Model, InputAttentionModel
from run_model import make_model


with open('opt_results.txt', 'w') as f:
    f.write('Hyper OPT Results')

dim_batch_size = Categorical(categories=[4, 8, 12, 24, 32], name='batch_size')
dim_lookback = Integer(low=5, high=20, prior='uniform', name='lookback')
dim_learning_rate = Real(low=1e-7, high=1e-3, prior='uniform', name='lr')

dim_lstm0_units = Categorical(categories=[16, 32, 64, 128], name='lstm0_units')
dim_lstm1_units = Categorical(categories=[16, 32, 64, 128], name='lstm1_units')
dim_lstm2_units = Categorical(categories=[16, 32, 64, 128], name='lstm2_units')
dim_lstm3_units = Categorical(categories=[16, 32, 64, 128], name='lstm3_units')

dim_act0_f = Categorical(categories=['relu', 'tanh', 'elu', 'LeakyRelu'], name='lstm0_act')
dim_act1_f = Categorical(categories=['relu', 'tanh', 'elu', 'LeakyRelu'], name='lstm1_act')
dim_act2_f = Categorical(categories=['relu', 'tanh', 'elu', 'LeakyRelu'], name='lstm2_act')
dim_act3_f = Categorical(categories=['relu', 'tanh', 'elu', 'LeakyRelu'], name='lstm3_act')

default_values = [12, 10, 0.00001,
                  32, # 32, 32, 32,
                  'tanh', 'tanh' #, 'tanh', 'tanh'
                  ]

dimensions = [dim_batch_size, dim_lookback, dim_learning_rate,
              dim_lstm0_units, # dim_lstm1_units, dim_lstm2_units, dim_lstm3_units,
              dim_act1_f, dim_act2_f #, dim_act2_f, dim_act3_f
              ]

def objective_fn(**kwargs)->float:

    data_config, nn_config, total_intervals = make_model(**kwargs)

    df = pd.read_excel('data/all_data_30min.xlsx')

    model = InputAttentionModel(data_config=data_config,
                                nn_config=nn_config,
                                intervals=total_intervals,
                                data=df)
    model.build_nn()

    history = model.train_nn(indices='random')
    return np.min(history.history['val_loss'])

@use_named_args(dimensions=dimensions)
def fitness(batch_size, lookback, lr,
            lstm0_units, #lstm1_units, lstm2_units, lstm3_units,
            lstm1_act, lstm2_act,  # lstm2_act, lstm3_act
            ):



    enc_config =  {'n_h': lstm0_units,  # length of hidden state m
                                 'n_s': lstm0_units,  # length of hidden state m
                                 'm': lstm0_units,  # length of hidden state m
                                 'enc_lstm1_act': lstm1_act,
                                 'enc_lstm2_act': lstm2_act,
                                 }


    error = objective_fn(batch_size=int(batch_size),
                         lookback=int(lookback),
                         lr=lr,
                         enc_config=enc_config),

                         # epochs=2000

    lstm_units = str([lstm0_units])
    lstm_acts = str([lstm1_act, lstm2_act])

    msg = """\nwith lstm_units {}, lstm1_act {},  batch_size {}, lookback {}, lr {}, val loss is {}
          """.format(lstm_units, lstm_acts,batch_size, lookback, lr, error)
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