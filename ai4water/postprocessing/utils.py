
from itertools import zip_longest

from ai4water.backend import np, mpl, plt
from ai4water.utils.visualizations import Plot

from easy_mpl.utils import create_subplots


class LossCurve(Plot):

    def __init__(self, path=None, show=1, save:bool=True):
        self.path = path
        self.show = show

        super().__init__(path, save=save)

    def plot_loss(self,
                  history: dict,
                  name="loss_curve",
                  figsize:tuple=None)->plt.Axes:
        """Considering history is a dictionary of different arrays, possible
        training and validation loss arrays, this method plots those arrays."""

        plt.clf()
        plt.close('all')
        plt.style.use('ggplot')

        legends = {
            'mean_absolute_error': 'Mean Absolute Error',
            'mape': 'Mean Absolute Percentage Error',
            'mean_squared_logarithmic_error': 'Mean Squared Logarithmic Error',
            'pbias': "Percent Bias",
            "nse": "Nash-Sutcliff Efficiency",
            "kge": "Kling-Gupta Efficiency",
            "tf_r2": "$R^{2}$",
            "r2": "$R^{2}$"
        }

        epochs = range(1, len(history['loss']) + 1)

        nplots = len(history)
        val_losses = {}
        losses = history.copy()
        for k in history.keys():
            val_key = f"val_{k}"
            if val_key in history:
                nplots -= 1
                val_losses[val_key] = losses.pop(val_key)

        fig, axis = create_subplots(nplots, figsize=figsize)

        if not isinstance(axis, np.ndarray):
            axis = np.array([axis])

        axis = axis.flat

        for idx, (key, loss), val_data in zip_longest(range(nplots), losses.items(), val_losses.items()):

            ax = axis[idx]

            if val_data is not None:
                val_key, val_loss = val_data
                ax.plot(epochs, val_loss, color=[0.96707953, 0.46268314, 0.45772886],
                      label='Validation ')
                ax.legend()

            ax.plot(epochs, loss, color=[0.13778617, 0.06228198, 0.33547859],
                      label='Training ')
            ax.legend()
            ax.set_xlabel("Epochs")
            ax.set_ylabel(legends.get(key, key))

        ax.set(frame_on=True)

        self.save_or_show(fname=name, show=self.show)
        mpl.rcParams.update(mpl.rcParamsDefault)
        return ax


def choose_examples(x, examples_to_use, y=None):
    """Chooses examples from x and y"""
    if isinstance(examples_to_use, int):
        x = x[examples_to_use]
        x = np.expand_dims(x, 0)  # dimension must not decrease
        index = np.array([examples_to_use])

    elif isinstance(examples_to_use, float):
        assert examples_to_use < 1.0
        # randomly choose x fraction from test_x
        x, index = choose_n_imp_exs(x, int(examples_to_use * len(x)), y)

    elif hasattr(examples_to_use, '__len__'):
        index = np.array(examples_to_use)
        x = x[index]
    else:
        raise ValueError(f"unrecognized value of examples_to_use: {examples_to_use}")

    return x, index


def choose_n_imp_exs(x: np.ndarray, n: int, y=None):
    """Chooses the n important examples from x and y"""

    n = min(len(x), n)

    st = n // 2
    en = n - st

    if y is None:
        idx = np.random.randint(0, len(x), n)
    else:
        st = np.argsort(y, axis=0)[0:st].reshape(-1, )
        en = np.argsort(y, axis=0)[-en:].reshape(-1, )
        idx = np.hstack([st, en])

    x = x[idx]

    return x, idx
