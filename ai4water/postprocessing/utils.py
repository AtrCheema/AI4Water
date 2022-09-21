
from ai4water.backend import np, mpl, plt
from ai4water.utils.visualizations import Plot


class LossCurve(Plot):

    def __init__(self, path=None, show=1, save:bool=True):
        self.path = path
        self.show = show

        super().__init__(path, save=save)

    def plot_loss(self, history: dict, name="loss_curve"):
        """Considering history is a dictionary of different arrays, possible
        training and validation loss arrays, this method plots those arrays."""

        plt.clf()
        plt.close('all')
        fig = plt.figure()
        plt.style.use('ggplot')
        i = 1

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

        sub_plots = {1: {'axis': (1, 1), 'width': 9, 'height': 6},
                     2: {'axis': (1, 2), 'width': 9, 'height': 6},
                     3: {'axis': (1, 3), 'width': 9, 'height': 6},
                     4: {'axis': (2, 2), 'width': 9, 'height': 6},
                     5: {'axis': (5, 1), 'width': 8, 'height': 12},
                     6: {'axis': (3, 2), 'width': 8, 'height': 12},
                     7: {'axis': (3, 2), 'width': 20, 'height': 20},
                     8: {'axis': (4, 2), 'width': 20, 'height': 20},
                     9: {'axis': (5, 2), 'width': 20, 'height': 20},
                     10: {'axis': (5, 2), 'width': 20, 'height': 20},
                     12: {'axis': (4, 3), 'width': 20, 'height': 20},
                     }

        epochs = range(1, len(history['loss']) + 1)
        axis_cache = {}

        for key, val in history.items():

            m_name = key.split('_')[1:] if 'val' in key and '_' in key else key

            if isinstance(m_name, list):
                m_name = '_'.join(m_name)
            if m_name in list(axis_cache.keys()):
                axis = axis_cache[m_name]
                axis.plot(epochs, val, color=[0.96707953, 0.46268314, 0.45772886],
                          label='Validation ')
                axis.legend()
            else:
                axis = fig.add_subplot(*sub_plots[len(history)]['axis'], i)
                axis.plot(epochs, val, color=[0.13778617, 0.06228198, 0.33547859],
                          label='Training ')
                axis.legend()
                axis.set_xlabel("Epochs")
                axis.set_ylabel(legends.get(key, key))
                axis_cache[key] = axis
                i += 1
            axis.set(frame_on=True)

        fig.set_figheight(sub_plots[len(history)]['height'])
        fig.set_figwidth(sub_plots[len(history)]['width'])
        self.save_or_show(fname=name, show=self.show)
        mpl.rcParams.update(mpl.rcParamsDefault)
        return


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
