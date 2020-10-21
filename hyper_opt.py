from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
import numpy as np
import json
from collections import OrderedDict
import os
from shutil import rmtree, copyfile

from utils import skopt_plots
from docs.MultiInputSharedModel import make_multi_model, MultiInputSharedModel


title = 'tcn_dense_hpo'
RESULTS = {}

dim_batch_size = Categorical(categories=[8, 16, 24, 32], name='batch_size')
dim_lookback = Integer(low=3, high=12, prior='uniform', name='lookback')
dim_learning_rate = Real(low=1e-7, high=1e-3, prior='uniform', name='lr')

dim_lstm1_units = Categorical(categories=[64, 128, 256], name='lstm1_units')
dim_lstm2_units = Categorical(categories=[64, 128, 256], name='lstm2_units')
dim_dense_units = Categorical(categories=[32, 64, 128], name='dense_units')

dim_lstm1_act = Categorical(categories=['relu', 'leakyrelu', 'elu', 'tanh'], name='lstm1_act')
dim_lstm2_act = Categorical(categories=['relu', 'leakyrelu', 'elu', 'tanh'], name='lstm2_act')
dim_dense_act = Categorical(categories=['relu', 'leakyrelu', 'elu', 'tanh'], name='dense_act')
dim_out_act = Categorical(categories=['relu', 'none'], name='out_act')

default_values = [24, 10, 0.0005286,
                  'tanh', 'tanh', 'relu', 'none',
                  64, 64, 32,
                  ]

dimensions = [dim_batch_size, dim_lookback, dim_learning_rate,
              dim_lstm1_act, dim_lstm2_act, dim_dense_act, dim_out_act,
              dim_lstm1_units, dim_lstm2_units, dim_dense_units
              ]

def objective_fn(**kwargs):
    """ This function will build and train the NN with given parameters. It will return the minimum of validation loss,
    which we want to minimize using Optimization algorithm.
    """
    model = make_multi_model(MultiInputSharedModel,
                          ignore_nans=False,
                          metrics=['nse', 'kge', 'pbias'],
                          prefix=title,
                          **kwargs)

    model.build_nn()
    history = model.train_nn(st=0, en=5500)

    return model.path, np.min(history['val_loss'])


def clear_weigths(_res:dict, opt_dir, keep=3):
    """ Optimization will save weights of all the trained models, not all of them are useful. Here removing weights
    of all except top 3. The number of models whose weights to be retained can be set by `keep` para."""
    od = OrderedDict(sorted(_res.items()))

    idx = 0
    for k,v in od.items():
        folder = v['folder']
        _path = os.path.join(opt_dir, folder)
        w_path = os.path.join(_path, 'weights')

        if idx > keep-1:
            if os.path.exists(w_path):
                rmtree(w_path)

        idx += 1

    # move json file to opt_path
    json_file = os.path.basename(opt_dir) + ".json"
    src = os.path.join(os.getcwd(), json_file)
    if os.path.exists(src):
        copyfile(src, os.path.join(opt_path, json_file))

    sorted_fname = os.path.join(opt_dir, 'sorted.json')
    with open(sorted_fname, 'w') as sfp:
        json.dump(od, sfp, sort_keys=True, indent=True)
    return

@use_named_args(dimensions=dimensions)
def fitness(batch_size, lookback, lr,
            lstm1_act, lstm2_act, dense_act, out_act,
            lstm1_units, lstm2_units, dense_units,
            ):

    lookback = int(lookback)
    if out_act == "none":
        out_act = None

    layers = {
        'lstm_0': {'config': {'units': int(lstm1_units),  'activation': lstm1_act,  'dropout': 0.2, 'recurrent_dropout': 0.4,
                                        'return_sequences': True, 'return_state': False, 'name': 'lstm_0'}},
        'lstm_1': {'config':  {'units': int(lstm2_units),  'activation': lstm2_act,  'dropout': 0.2, 'recurrent_dropout': 0.4,
                                        'return_sequences': False, 'return_state': False, 'name': 'lstm_1'}},
        'Dense_0': {'config':  {'units': int(dense_units), 'activation': dense_act}},
        'Dropout': {'config':  {'rate': 0.4}},
        'Dense_1': {'config':  {'units': 1, "activation": out_act}}
               }

    _path, error = objective_fn(batch_size=int(batch_size),
                         lookback=int(lookback),
                         lr=lr,
                         layers=layers,
                         epochs=10)


    error = round(error, 7)

    result = {'lstm1_act': lstm1_act,
              'lstm2_act': lstm2_act,
              'dense_act': dense_act,
              'out_act': out_act,

              'lstm1_units': int(lstm1_units),
              'lstm2_units': int(lstm2_units),
              'dense_units': int(dense_units),

              'lookback': int(lookback),
              'lr': float(lr),
              'batch_size': int(batch_size),
              'folder': _path.split("\\")[-1],
              'val_loss': error
              }

    RESULTS[error] = result

    # instead of appending, writing the new file, so that all the results are saved as one dictionary, which can be
    # useful if we want to reload the results.
    with open(title + '.json', 'w') as rfp:
        json.dump(RESULTS, rfp, sort_keys=True, indent=4)

    return error


search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI',   # Expected Improvement.
                            n_calls=12,
                            # acq_optimizer='auto',
                            x0=default_values,
                            random_state=2)

opt_path = os.path.join(os.getcwd(), "results\\" + title)

skopt_plots(search_result, pref=opt_path)

fname = os.path.join(opt_path, 'gp_parameters')
sr = {}
for attr in dir(search_result):
    sr[attr] = str(getattr(search_result, attr))

with open(fname + '.json', 'w') as fp:
    json.dump(sr, fp, sort_keys=True, indent=4)

clear_weigths(RESULTS, opt_path)
