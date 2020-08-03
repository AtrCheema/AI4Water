import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def _plot(*args, **kwargs):
    plt.close('all')
    plt.plot(*args, **kwargs)
    plt.legend(loc="best")
    plt.show()
    return


def plot_results(true, predicted, name=None):

    regplot_using_searborn(true, predicted, name)
    fig, axis = plt.subplots()
    set_fig_dim(fig, 12, 8)
    axis.plot(true, '-', label='True')
    axis.plot(predicted, '-', label='predicted')
    axis.legend(loc="best", fontsize=22, markerscale=4)

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


def plot_loss(history, name=None):

    loss = history.history['loss']

    epochs = range(1, len(loss) + 1)

    fig, axis = plt.subplots()

    axis.plot(epochs, history.history['loss'], color=[0.13778617, 0.06228198, 0.33547859], label='Training loss')

    if 'val_loss' in history.history:
        axis.plot(epochs, history.history['val_loss'],
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
