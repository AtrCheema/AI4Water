import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import datetime
import json
import pandas as pd

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
    axis.plot(true, **kwargs, label='True')
    axis.plot(predicted, **kwargs, label='predicted')
    axis.legend(loc="best", fontsize=22, markerscale=4)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    if name is not None:
        plt.savefig(name, dpi=300, bbox_inches='tight')
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
    plt.show()


def plot_loss(history: dict, name=None):

    loss = history['loss']

    epochs = range(1, len(loss) + 1)

    fig, axis = plt.subplots()

    axis.plot(epochs, history['loss'], color=[0.13778617, 0.06228198, 0.33547859], label='Training loss')

    if 'val_loss' in history:
        axis.plot(epochs, history['val_loss'],
                  color=[0.96707953, 0.46268314, 0.45772886], label='Validation loss')

    axis.set_xlabel('Epochs')
    axis.set_ylabel('Loss value')
    axis.set_yscale('log')
    plt.title('Loss Curve')
    plt.legend()

    if name is not None:
        plt.savefig(name, dpi=300, bbox_inches='tight')
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
    jetzt = datetime.datetime.now()
    jahre = str(jetzt.year)
    month = str(jetzt.month)
    if len(month) < 2:
        month = '0' + month
    tag = str(jetzt.day)
    if len(tag) < 2:
        tag = '0' + tag
    datum = jahre + month + tag

    stunde = str(jetzt.hour)
    if len(stunde) < 2:
        stunde = '0' + stunde
    minute = str(jetzt.minute)
    if len(minute) < 2:
        minute = '0' + minute

    return datum + '_' + stunde + str(minute)


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


def skopt_plots(search_result):

    from skopt.plots import plot_evaluations, plot_objective, plot_convergence

    _ = plot_evaluations(search_result)
    plt.savefig('evaluations', dpi=400, bbox_inches='tight')
    plt.show()

    _ = plot_objective(search_result)
    plt.savefig('objective', dpi=400, bbox_inches='tight')
    plt.show()

    _ = plot_convergence(search_result)
    plt.savefig('convergence', dpi=400, bbox_inches='tight')
    plt.show()


def check_min_loss(epoch_losses, _epoch, _msg:str, _save_fg:bool, to_save=None):
    epoch_loss_array = epoch_losses[:-1]

    current_epoch_loss = epoch_losses[-1]

    if len(epoch_loss_array) > 0:
        min_loss = np.min(epoch_loss_array)
    else:
        min_loss = current_epoch_loss

    if np.less(current_epoch_loss, min_loss):
        _msg = _msg + "    {:10.5f} ".format(current_epoch_loss)

        if to_save is not None:
            _save_fg = True
    else:
        _msg = _msg + "              "

    return _msg, _save_fg


def make_model(**kwargs):
    """
    This functions fills the default arguments needed to run all the models. All the input arguments can be overwritten
    by providing their name.
    :return
      nn_config: `dict`, contais parameters to build and train the neural network such as `layers`
      data_config: `dict`, contains parameters for data preparation/pre-processing/post-processing etc.
      intervals:  `tuple` tuple of tuples whiere each tuple consits of two integers, marking the start and end of
                   interval. An interval here means chunk/rows from the input file/dataframe to be skipped when
                   when preparing data/batches for NN. This happens when we have for example some missing values at some
                   time in our data. For further usage see `docs/using_intervals`.
    """
    _nn_config = dict()

    _nn_config['layers'] = {"Dense_0": {'units': 64, 'activation': 'relu'},
                                 "Dropout_0": {'rate': 0.3},
                                 "Dense_1": {'units': 32, 'activation': 'relu'},
                                 "Dropout_1": {'rate': 0.3},
                                 "Dense_2": {'units': 16, 'activation': 'relu'},
                                 "Dense_3": {'units': 1}
                                 }

    _nn_config['enc_config'] = {'n_h': 20,  # length of hidden state m
                                'n_s': 20,  # length of hidden state m
                                'm': 20,  # length of hidden state m
                                'enc_lstm1_act': None,
                                'enc_lstm2_act': None,
                                }
    _nn_config['dec_config'] = {
        'p': 30,
        'n_hde0': 30,
        'n_sde0': 30
    }

    _nn_config['composite'] = False  # for auto-encoders

    _nn_config['lr'] = 0.0001
    _nn_config['optimizer'] = 'adam'
    _nn_config['loss'] = 'mse'
    _nn_config['epochs'] = 14
    _nn_config['min_val_loss'] = 0.0001
    _nn_config['patience'] = 100

    _nn_config['subsequences'] = 3  # used for cnn_lst structure

    _nn_config['HARHN_config'] = {'n_conv_lyrs': 3,
                                  'enc_units': 64,
                                  'dec_units': 64}

    _data_config = dict()
    _data_config['lookback'] = 15
    _data_config['batch_size'] = 32
    _data_config['val_fraction'] = 0.2  # fraction of data to be used for validation
    _data_config['val_data'] = None # If this is not string and not None, this will overwite `val_fraction`
    _data_config['test_fraction'] = 0.2
    _data_config['CACHEDATA'] = True
    _data_config['ignore_nans'] = False  # if True, and if target values contain Nans, those samples will not be ignored
    _data_config['use_predicted_output'] = True  # if true, model will use previous predictions as input

    # input features in data_frame
    dpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    _df = pd.read_csv(os.path.join(dpath, "nasdaq100_padding.csv"))
    in_cols = list(_df.columns)
    in_cols.remove("NDX")
    _data_config['inputs'] = in_cols
    # column in dataframe to bse used as output/target
    _data_config['outputs'] = ["NDX"]

    _nn_config['dense_config'] = {1: {'units':1}}

    for key, val in kwargs.items():
        if key in _data_config:
            _data_config[key] = val
        if key in _nn_config:
            _nn_config[key] = val

    _total_intervals = (
        (0, 146,),
        (145, 386,),
        (385, 628,),
        (625, 821,),
        (821, 1110),
        (1110, 1447))

    return _data_config, _nn_config, _total_intervals

def get_index(idx_array, fmt='%Y%m%d%H%M'):
    """ converts a numpy 1d array into pandas DatetimeIndex type."""

    if not isinstance(idx_array, np.ndarray):
        raise TypeError

    return pd.to_datetime(idx_array.astype(str), format=fmt)