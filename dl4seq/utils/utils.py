import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import OrderedDict
import os
from shutil import rmtree
import datetime
import json
import pandas as pd
import sklearn

plt.rcParams["font.family"] = "Times New Roman"


def _plot(*args, **kwargs):
    plt.close('all')
    plt.plot(*args, **kwargs)
    plt.legend(loc="best")
    plt.show()
    return


def plot_results(true, predicted, name=None, **kwargs):
    """
    # kwargs can be any/all of followings
        # fillstyle:
        # marker:
        # linestyle:
        # markersize:
        # color:
    """

    regplot_using_searborn(true, predicted, name)
    fig, axis = plt.subplots()
    set_fig_dim(fig, 12, 8)
    axis.plot(true, **kwargs, color='b', label='True')
    axis.plot(predicted, **kwargs, color='r', label='predicted')
    axis.legend(loc="best", fontsize=22, markerscale=4)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    if name is not None:
        plt.savefig(name, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return


def regplot_using_searborn(true, pred, name):
    # https://seaborn.pydata.org/generated/seaborn.regplot.html
    plt.close('all')
    sns.regplot(x=true, y=pred, color="g")
    plt.xlabel('Observed', fontsize=14)
    plt.ylabel('Predicted', fontsize=14)

    if name is not None:
        plt.savefig(name + '_reg', dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_loss(history: dict, name=None):


    plt.clf()
    plt.close('all')
    fig = plt.figure()
    plt.style.use('ggplot')
    i = 1

    sub_plots = {1: (1,1),
                 2: (1, 1),
                 3: (1, 2),
                 4: (1, 2),
                 5: (1, 3),
                 6: (1, 3),
                 7: (2, 2),
                 8: (2, 2),
                 9: (3, 2),
                 10: (3, 2)
                 }

    epochs = range(1, len(history['loss']) + 1)
    axis_cache = {}

    for key, val in history.items():

        m_name = key.split('_')[1] if '_' in key else key

        if m_name in list(axis_cache.keys()):
            axis = axis_cache[m_name]
            axis.plot(epochs, val, color=[0.96707953, 0.46268314, 0.45772886], label= 'Validation ' + m_name)
            axis.legend()
        else:
            axis = fig.add_subplot(*sub_plots[len(history)], i)
            axis.plot(epochs, val, color=[0.13778617, 0.06228198, 0.33547859], label= 'Training ' + key)
            axis.legend()
            axis_cache[key] = axis
            i += 1

    if name is not None:
        plt.savefig(name, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return


def set_fig_dim(fig, width, height):
    fig.set_figwidth(width)
    fig.set_figheight(height)


def plot_train_test_pred(y, obs, tr_idx, test_idx):
    yf_tr = np.full(len(y), np.nan)
    yf_test = np.full(len(y), np.nan)
    yf_tr[tr_idx.reshape(-1, )] = y[tr_idx.reshape(-1, )].reshape(-1, )
    yf_test[test_idx.reshape(-1, )] = y[test_idx.reshape(-1, )].reshape(-1, )

    fig, axis = plt.subplots()
    set_fig_dim(fig, 14, 8)
    axis.plot(obs, '-', label='True')
    axis.plot(yf_tr, '.', label='predicted train')
    axis.plot(yf_test, '.', label='predicted test')
    axis.legend(loc="best", fontsize=18, markerscale=4)

    plt.show()


def maybe_create_path(prefix=None, path=None):
    if path is None:
        save_dir = dateandtime_now()
        model_dir = os.path.join(os.getcwd(), "results")

        if prefix:
            model_dir = os.path.join(model_dir, prefix)

        save_dir = os.path.join(model_dir, save_dir)
    else:
        save_dir = path

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    act_dir = os.path.join(save_dir, 'activations')
    if not os.path.exists(act_dir):
        os.makedirs(act_dir)

    weigth_dir = os.path.join(save_dir, 'weights')
    if not os.path.exists(weigth_dir):
        os.makedirs(weigth_dir)

    return save_dir, act_dir, weigth_dir


def dateandtime_now():
    """ returns the datetime in following format
    YYYYMMDD_HHMMSS
    """
    jetzt = datetime.datetime.now()
    dt = str(jetzt.year)
    for time in ['month', 'day', 'hour', 'minute', 'second']:
        _time = str(getattr(jetzt, time))
        if len(_time) < 2:
            _time = '0' + _time
        if time == 'hour':
            _time = '_' + _time
        dt += _time
    return dt


def save_config_file(path, config=None, errors=None, indices=None, name=''):

    sort_keys = True
    if errors is not None:
        suffix = dateandtime_now()
        fpath = path + "/errors_" + name +  suffix + ".json"
        data = errors
    elif config is not None:
        fpath = path + "/config.json"
        data = config
        sort_keys = False
    elif indices is not None:
        fpath = path + "/indices.json"
        data = indices
    else:
        raise ValueError("")

    with open(fpath, 'w') as fp:
        json.dump(data, fp, sort_keys=sort_keys, indent=4)

    return


def skopt_plots(search_result, pref=os.getcwd()):

    from skopt.plots import plot_evaluations, plot_objective, plot_convergence

    plt.close('all')
    _ = plot_evaluations(search_result)
    plt.savefig(os.path.join(pref , 'evaluations'), dpi=400, bbox_inches='tight')

    plt.close('all')
    _ = plot_objective(search_result)
    plt.savefig(os.path.join(pref , 'objective'), dpi=400, bbox_inches='tight')

    plt.close('all')
    _ = plot_convergence(search_result)
    plt.savefig(os.path.join(pref , 'convergence'), dpi=400, bbox_inches='tight')


def check_min_loss(epoch_losses, epoch, msg:str, save_fg:bool, to_save=None):
    epoch_loss_array = epoch_losses[:-1]

    current_epoch_loss = epoch_losses[-1]

    if len(epoch_loss_array) > 0:
        min_loss = np.min(epoch_loss_array)
    else:
        min_loss = current_epoch_loss

    if np.less(current_epoch_loss, min_loss):
        msg = msg + "    {:10.5f} ".format(current_epoch_loss)

        if to_save is not None:
            save_fg = True
    else:
        msg = msg + "              "

    return msg, save_fg


def make_model(**kwargs):
    """
    This functions fills the default arguments needed to run all the models. All the input arguments can be overwritten
    by providing their name.
    :return
      nn_config: `dict`, contais parameters to build and train the neural network such as `layers`
      data_config: `dict`, contains parameters for data preparation/pre-processing/post-processing etc.
    """

    dpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    df = pd.read_csv(os.path.join(dpath, "nasdaq100_padding.csv"))
    in_cols = list(df.columns)
    in_cols.remove("NDX")

    nn_args = {
        'enc_config': {'type': dict, 'default': {'n_h': 20,  # length of hidden state m
                                'n_s': 20,  # length of hidden state m
                                'm': 20,  # length of hidden state m
                                'enc_lstm1_act': None,
                                'enc_lstm2_act': None,
                                }},
        'dec_config': {'': dict, 'default': {
                                            'p': 30,
                                            'n_hde0': 30,
                                            'n_sde0': 30
                                            }},
        'layers': {'type': dict, 'default': {
                                "Dense_0": {'config': {'units': 64, 'activation': 'relu'}},
                                "Dense_3": {'config':  {'units': 1}}
                                 }},
        'composite':    {'type': bool, 'default': False},   # for auto-encoders
        'lr':           {'type': float, 'default': 0.0001},
        'optimizer':    {'type': str, 'default': 'adam'},  # can be any of valid keras optimizers https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
        'loss':         {'type': str, 'default': 'mse'},
        'epochs':       {'type': int, 'default': 14},
        'min_val_loss': {'type': float, 'default': 0.0001},
        'patience':     {'type': int, 'default': 100},
        'shuffle':      {'type': bool, 'default': True},
        'save_model':   {'type': bool, 'default': True},  # to save the best models using checkpoints
        'subsequences': {'type': int, 'default': 3},  # used for cnn_lst structure
        'harhn_config': {'type': dict, 'default': {'n_conv_lyrs': 3,
                                                  'enc_units': 64,
                                                  'dec_units': 64}},
        'nbeats_options': {'type': dict, 'default': {
                                        'backcast_length': 15 if 'lookback' not in kwargs else int(kwargs['lookback']),
                                        'forecast_length': 1,
                                        'stack_types': ('generic', 'generic'),
                                        'nb_blocks_per_stack': 2,
                                        'thetas_dim': (4, 4),
                                        'share_weights_in_stack': True,
                                        'hidden_layer_units': 62
                                    }}
    }

    data_args = {
        'forecast_length':   {"type": int, "default": 1},   # how many future values we want to predict
        'batches_per_epoch': {"type": int, "default": None},  # comes handy if we want to skip certain batches from last
    # if the shape of last batch is smaller than batch size and if we want to skip this last batch, set following to True.
    # Useful if we have fixed batch size in our model but the number of samples is not fully divisble by batch size
        'drop_remainder':    {"type": bool,  "default": False},
        'normalize':         {"type": [str, type(None), dict],   "default": 'minmax'},  # can be None or any of the method defined in scalers.py
        'lookback':          {"type": int,   "default": 15},
        'batch_size':        {"type": int,   "default": 32},
        'val_fraction':      {"type": float, "default": 0.2}, # fraction of data to be used for validation
        'val_data':          {"type": None,  "default": None}, # If this is not string and not None, this will overwite `val_fraction`
        'steps_per_epoch':   {"type": int,   "default": None},  # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        'test_fraction':     {"type": float, "default": 0.2},   # fraction of data to be used for test
        'cache_data':        {"type": bool,  "default": False},   # write the data/batches as hdf5 file
        'ignore_nans':       {"type": bool,  "default": False},  # if True, and if target values contain Nans, those samples will not be ignored
        'metrics':           {"type": list,  "default": ['nse']},  # can be string or list of strings such as 'mse', 'kge', 'nse', 'pbias'
        'use_predicted_output': {"type": bool , "default": True},  # if true, model will use previous predictions as input
    # If the model takes one kind of inputs that is it consists of only 1 Input layer, then the shape of the batches
    # will be inferred from this Input layer but for cases,  the model takes more than 1 Input, then there can be two
    # cases, either all the inputs are of same shape or they  are not. In second case, we should overwrite `train_paras`
    # method. In former case, define whether the batches are 2d or 3d. 3d means it is for an LSTM and 2d means it is
    # for Dense layer.
        'batches':           {"type": str, "default": '3d'},
        'seed':              {"type": int, "default": 313},  # for reproducability
        'forecast_step':     {"type": int, "default": 0},  # how many steps ahead we want to predict
        'input_step':        {"type": int, "default": 1},  # step size of input data
        'inputs':            {"type": list, "default": in_cols}, # input features in data_frame
        'outputs':           {"type": list, "default": ["NDX"]}, # column in dataframe to bse used as output/target
    # tuple of tuples where each tuple consits of two integers, marking the start and end of interval. An interval here
    # means chunk/rows from the input file/dataframe to be skipped when when preparing data/batches for NN. This happens
    # when we have for example some missing values at some time in our data. For further usage see `examples/using_intervals`
        "intervals":         {"type": tuple, "default": (
                                                        (0, 146,),
                                                        (145, 386,),
                                                        (385, 628,),
                                                        (625, 821,),
                                                        (821, 1110),
                                                        (1110, 1447)
        )}
    }

    nn_config=  {key:val['default'] for key,val in nn_args.items()}

    data_config=  {key:val['default'] for key,val in data_args.items()}

    for key, val in kwargs.items():
        arg_name = key.lower()

        if arg_name in nn_config:
            update_dict(arg_name, val, nn_args[arg_name]['type'], nn_config)

        elif arg_name in data_config:
            update_dict(arg_name, val, data_args[arg_name]['type'], data_config)

        else:
            raise ValueError(f"Unknown keyworkd argument {key} provided")

    return data_config, nn_config


def update_dict(arg_name, val, dtype, dict_to_update):

    if dtype is not None:
        if isinstance(dtype, list):
            val_type = type(val)
            if val_type not in dtype:
                raise TypeError("{} must be any of the type {} but it is of type {}"
                                .format(arg_name, dtype, type(val)))
        elif not isinstance(val, dtype):
            raise TypeError(f"{arg_name} must be of type {dtype} but it is of type {type(val)}")
    dict_to_update[arg_name] = val
    return


def get_index(idx_array, fmt='%Y%m%d%H%M'):
    """ converts a numpy 1d array into pandas DatetimeIndex type."""

    if not isinstance(idx_array, np.ndarray):
        raise TypeError

    return pd.to_datetime(idx_array.astype(str), format=fmt)


def jsonize_skopt_results(skopt_results):
    """ just converts skopt_results in a way that when written to json file, they look pretty/more understandable"""
    gp_sr = {}
    for attr in ['fun', 'func_vals', 'models', 'random_state', 'space', 'specs', 'x', 'x_iters']:
        val = getattr(skopt_results, attr)
        if attr == 'fun':
            gp_sr[attr] = float(val)

        elif attr == 'func_vals':
            gp_sr[attr] = val.tolist()

        elif attr == 'models':
            models = {}
            for idx, mod in enumerate(val):
                models[idx] = str(mod)
            gp_sr[attr] = models

        elif attr == 'random_state':
            gp_sr[attr] = str(val)

        elif attr == 'space':
            space = {}
            for idx, dim in enumerate(val.dimensions):
                space[idx] = str(dim)
            gp_sr[attr] = space

        elif attr == 'specs':
            specs = {}
            for k, v in val.items():
                if k == 'args':
                    args = {}
                    for _k, _v in v.items():
                        if _k == 'dimensions':
                            dims = {}
                            for idx, dim in enumerate(_v):
                                dims[idx] = str(dim)
                            args[_k] = dims
                        else:
                            args[_k] = str(_v)
                    specs[k] = args
                else:
                    specs[k] = str(v)
            gp_sr[attr] = specs

        elif attr == "x":
            gp_sr[attr] = val

        elif attr == "x_iters":
            x_iters = val
            nx_iters = []

            for xiter in x_iters:
                nxiter = []
                for x in xiter:
                    if isinstance(x, np.int32) or isinstance(x, np.int64):
                        nxiter.append(int(x))
                    elif isinstance(x, str):
                        nxiter.append(x)
                    else:
                        nxiter.append(float(x))
                nx_iters.append(nxiter)
            gp_sr[attr] = nx_iters

    return gp_sr


def clear_weights(_res:dict, opt_dir, keep=3):
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

    # append ranking of models to folder_names
    idx = 0
    for k,v in od.items():
        folder = v['folder']
        old_path = os.path.join(opt_dir, folder)
        new_path = os.path.join(opt_dir, str(idx+1) + "_" + folder)
        os.rename(old_path, new_path)

        idx += 1

    sorted_fname = os.path.join(opt_dir, 'sorted.json')
    with open(sorted_fname, 'w') as sfp:
        json.dump(od, sfp, sort_keys=True, indent=True)
    return

def get_attributes(aus, what:str='losses') ->dict:
    """ gets all callable attributes of aus e.g. from tf.keras.what and saves them in dictionary with their names all
    capitalized so that calling them becomes case insensitive. It is possible that some of the attributes of tf.keras.layers
    are callable but still not a valid `layer`, sor some attributes of tf.keras.losses are callable but still not valid
    losses, in that case the error will be generated from tensorflow. We are not catching those error right now."""
    all_attrs = {}
    for l in dir(getattr(aus, what)):
        attr = getattr(getattr(aus, what), l)
        if callable(attr) and not l.startswith('_'):
            all_attrs[l.upper()] = attr

    return all_attrs


def get_sklearn_models():

    # the following line must be executed in order for get_attributes to work, don't know why
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import multilayer_perceptron
    from sklearn.multioutput import MultiOutputRegressor

    sklearn_models = get_attributes(sklearn, "ensemble")
    sklearn_models.update(get_attributes(sklearn, "linear_model"))
    sklearn_models.update(get_attributes(sklearn, "multioutput"))
    sklearn_models.update(get_attributes(sklearn, "neighbors"))
    sklearn_models.update(get_attributes(sklearn, "neural_network"))
    sklearn_models.update(get_attributes(sklearn, "svm"))
    sklearn_models.update(get_attributes(sklearn, "tree"))

    return sklearn_models