from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
import numpy as np


from utils import skopt_plots
from docs.MultiOutputParallel import make_multi_model, LSTMAutoEncMultiOutput


with open('opt_results.txt', 'w') as f:
    f.write('Hyper OPT Results')

dim_batch_size = Categorical(categories=[4, 8, 12, 24, 32], name='batch_size')
# dim_lookback = Integer(low=5, high=20, prior='uniform', name='lookback')
dim_lookback = Categorical(categories=[3, 6, 9, 12, 15], name='lookback')
dim_learning_rate = Real(low=1e-7, high=1e-3, prior='uniform', name='lr')
# dim_filters0 = Categorical(categories=[16, 32, 64, 128], name='filters0')
# dim_filters1 = Categorical(categories=[16, 32, 64, 128], name='filters1')
# dim_filters2 = Categorical(categories=[16, 32, 64, 128], name='filters2')
# dim_filters3 = Categorical(categories=[16, 32, 64, 128], name='filters3')
# dim_kernel_size0 = Categorical(categories=[2, 3, 4, 5, 6], name='kernel_size0')
# dim_kernel_size1 = Categorical(categories=[2, 3, 4, 5, 6], name='kernel_size1')
# dim_kernel_size2 = Categorical(categories=[2, 3, 4, 5, 6], name='kernel_size2')
# dim_kernel_size3 = Categorical(categories=[2, 3, 4, 5, 6], name='kernel_size3')

dim_lstm0_units = Categorical(categories=[32, 64, 128], name='lstm0_units')
dim_lstm1_units = Categorical(categories=[32, 64, 128], name='lstm1_units')
dim_lstm2_units = Categorical(categories=[32, 64, 128], name='lstm2_units')
dim_lstm3_units = Categorical(categories=[32, 64, 128], name='lstm3_units')

dim_act_fn0 = Categorical(categories=['relu', 'leakyrelu', 'elu', 'tanh'], name='act_fn0')
dim_act_fn1 = Categorical(categories=['relu', 'leakyrelu', 'elu', 'tanh'], name='act_fn1')
dim_act_fn2 = Categorical(categories=['relu', 'leakyrelu', 'elu', 'tanh'], name='act_fn2')
dim_act_fn3 = Categorical(categories=['relu', 'leakyrelu', 'elu', 'tanh'], name='act_fn3')

dim_act_fn0_1 = Categorical(categories=['relu', 'leakyrelu', 'elu', 'tanh'], name='act_fn0_1')
dim_act_fn1_1 = Categorical(categories=['relu', 'leakyrelu', 'elu', 'tanh'], name='act_fn1_1')
dim_act_fn2_1 = Categorical(categories=['relu', 'leakyrelu', 'elu', 'tanh'], name='act_fn2_1')
dim_act_fn3_1 = Categorical(categories=['relu', 'leakyrelu', 'elu', 'tanh'], name='act_fn3_1')

default_values = [12, 3, 0.00001,
                  # 64, 128, 64, 64,
                  # 2, 3, 4, 5,
                  'relu', 'leakyrelu', 'elu', 'tanh',
                  'relu', 'leakyrelu', 'elu', 'tanh',
                  32, 32, 32, 32
                  ]

dimensions = [dim_batch_size, dim_lookback, dim_learning_rate,
              # dim_filters0, dim_filters1, dim_filters2, dim_filters3,
              # dim_kernel_size0, dim_kernel_size1, dim_kernel_size2, dim_kernel_size3,
              dim_act_fn0, dim_act_fn1, dim_act_fn2, dim_act_fn3,
              dim_act_fn0_1, dim_act_fn1_1, dim_act_fn2_1, dim_act_fn3_1,
              dim_lstm0_units, dim_lstm1_units, dim_lstm2_units, dim_lstm3_units
              ]


def objective_fn(**kwargs):

    model, train_idx, test_idx = make_multi_model(LSTMAutoEncMultiOutput,
                                                  **kwargs)

    history = model.train_nn(indices=train_idx)

    return model.path, np.min(history['val_loss'])


@use_named_args(dimensions=dimensions)
def fitness(batch_size, lookback, lr,
            # filters0, filters1, filters2, filters3,
            # kernel_size0, kernel_size1, kernel_size2, kernel_size3,
            act_fn0, act_fn1, act_fn2, act_fn3,
            act_fn0_1, act_fn1_1, act_fn2_1, act_fn3_1,
            lstm0_units, lstm1_units, lstm2_units, lstm3_units,
            ):


    autoenc_config = {'nn_0': {'lstm_0': {'units': int(lstm0_units),  'dropout': 0.3, 'recurrent_dropout': 0.4},
                               act_fn0: {},
                               'RepeatVector': {'n': lookback - 1},
                               'lstm_1': {'units': int(lstm0_units),  'dropout': 0.3, 'recurrent_dropout': 0.4},
                               act_fn0_1: {},
                               'Dense': {'units': 1}
                               },
                      'nn_1': {'lstm_0': {'units': int(lstm1_units),  'dropout': 0.3, 'recurrent_dropout': 0.4},
                               act_fn1: {},
                               'RepeatVector': {'n': lookback - 1},
                               'lstm_1': {'units': int(lstm1_units),  'dropout': 0.3, 'recurrent_dropout': 0.4},
                               act_fn1_1: {},
                               'Dense': {'units': 1}
                               },
                      'nn_2': {'lstm_0': {'units': int(lstm2_units),  'dropout': 0.3, 'recurrent_dropout': 0.4},
                               act_fn2: {},
                               'RepeatVector': {'n': lookback -1 },
                               'lstm_1': {'units': int(lstm2_units),  'dropout': 0.3, 'recurrent_dropout': 0.4},
                               act_fn2_1: {},
                               'Dense': {'units': 1}
                               },
                      'nn_3': {'lstm_0': {'units': int(lstm3_units),  'dropout': 0.3, 'recurrent_dropout': 0.4},
                               act_fn3: {},
                               'RepeatVector': {'n': lookback - 1},
                               'lstm_1': {'units': int(lstm3_units),  'dropout': 0.3, 'recurrent_dropout': 0.4},
                               act_fn3_1: {},
                               'Dense': {'units': 1}
                               }
                      }

    _path, error = objective_fn(batch_size=int(batch_size),
                         lookback=int(lookback),
                         lr=lr,
                         layers=autoenc_config,
                         epochs=500)

    #filters = str([filters0, filters1, filters2, filters3])
    #kernel_sizes = str([kernel_size0, kernel_size1, kernel_size2, kernel_size3])
    lstm_units = str([lstm0_units, lstm1_units, lstm2_units, lstm3_units])
    act_fns = str([act_fn0, act_fn1, act_fn2, act_fn3])
    act_fns1 = str([act_fn0_1, act_fn1_1, act_fn2_1, act_fn3_1])
    msg = """\nwith lstm_units {}, act_lyrs1 {}, act_lyrs {},  batch_size {} lookback {} lr {} val loss is {}
          in folder {}
          """.format(lstm_units, act_fns1, act_fns,  batch_size, lookback, lr, error, _path.split("\\")[-1])
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
                            random_state=2)

skopt_plots(search_result)
