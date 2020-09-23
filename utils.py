import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import datetime
import json

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

    return save_dir


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

    current_epoch_loss = epoch_losses[-1] # np.mean(batch_loss_array)

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