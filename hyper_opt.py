from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
import numpy as np
import pandas as pd

from utils import skopt_plots
from models import Model
from run_model import make_model


with open('opt_results.txt', 'w') as f:
    f.write('Hyper OPT Results')

dim_batch_size = Categorical(categories=[4, 8, 12, 24, 32], name='batch_size')
dim_lookback = Integer(low=5, high=20, prior='uniform', name='lookback')
dim_learning_rate = Real(low=1e-7, high=1e-3, prior='uniform', name='lr')
dim_lstm_units = Categorical(categories=[16, 32, 64, 128], name='lstm_units')
dim_act1_f = Categorical(categories=['relu', 'tanh', 'elu', 'LeakyRelu', 'none'], name='lstm1_act')
dim_act2_f = Categorical(categories=['relu', 'tanh', 'elu', 'LeakyRelu', 'none'], name='lstm2_act')

default_values = [12, 10, 0.00001, 32, 'none', 'none']

dimensions = [dim_batch_size, dim_lookback, dim_learning_rate,
              dim_lstm_units, dim_act1_f, dim_act2_f]


def objective_fn(**kwargs):

    data_config, nn_config, total_intervals = make_model(**kwargs)

    df = pd.read_csv('data/all_data_30min.csv')

    model = Model(data_config=data_config,
                  nn_config=nn_config,
                  data=df,
                  intervals=total_intervals
                  )

    model.build_nn()

    history = model.train_nn(indices='random')
    return np.min(history.history['val_loss'])


@use_named_args(dimensions=dimensions)
def fitness(batch_size, lookback, lr,
            lstm_units, lstm1_act, lstm2_act
            ):

    if lstm1_act == 'none':
        lstm1_act = None
    if lstm2_act == 'none':
        lstm2_act = None

    enc_config = {'n_h': lstm_units,  # length of hidden state m
                  'n_s': lstm_units,  # length of hidden state m
                  'm': lstm_units,  # length of hidden state m
                  'enc_lstm1_act': lstm1_act,
                  'enc_lstm2_act': lstm2_act,
                  }

    error = objective_fn(batch_size=batch_size,
                         lookback=lookback,
                         lr=lr,
                         enc_config=enc_config,
                         epochs=10)

    msg = """\nwith lstm_units {}, lstm1_act {}, lstm2_act {}, batch_size {} lookback {} lr {} val loss is {}
          """.format(lstm_units, lstm1_act, lstm2_act, batch_size, lookback, lr, error)
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

skopt_plots(search_result)
