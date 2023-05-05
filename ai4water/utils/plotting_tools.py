
from typing import Union

from ai4water.backend import easy_mpl as ep
from ai4water.backend import os, np, pd, plt

# TODO
# rank histogram, reliability diagram, ROC curve


class Plots(object):
    # TODO initialte this class with at least path

    def __init__(self, config, path=None, model=None):

        self.path = path or os.path.join(os.getcwd(), "results")
        self.ml_model = model
        self.config = config

    @property
    def training_data(self, *args, **kwargs):
        raise AttributeError

    @property
    def validation_data(self, *args, **kwargs):
        raise AttributeError

    @property
    def test_data(self, *args, **kwargs):
        raise AttributeError

    @property
    def in_cols(self):
        return self.config['input_features']

    @property
    def out_cols(self):
        return self.config['output_features']

    def _imshow_3d(self,
                   activation,
                   lyr_name,
                   xticklabels=None,
                   save=True,
                   where='activations',
                   xlabel=None):

        act_2d = []
        for i in range(activation.shape[2]):
            act_2d.append(activation[:, :, i])
        activation_2d = np.concatenate(act_2d, axis=1)
        self._imshow(activation_2d,
                     lyr_name + " Activations (3d of {})".format(activation.shape),
                     save,
                     lyr_name,
                     where=where,
                     xticklabels=xticklabels,
                     xlabel=xlabel)
        return

    def _imshow(self,
                img,
                label: str = '',
                save=True,
                fname=None,
                interpolation: str = 'none',
                where='',
                rnn_args=None,
                cmap=None,
                show=False,
                **kwargs):

        assert np.ndim(img) == 2, "can not plot {} with shape {} and ndim {}".format(label, img.shape, np.ndim(img))

        im = ep.imshow(img,
                          aspect="auto",
                          interpolation=interpolation,
                          cmap=cmap,
                          ax_kws=dict(
                              xlabel=kwargs.get('xlabel', 'inputs'),
                              title=label),
                          show=False)

        axis = im.axes

        if rnn_args is not None:
            assert isinstance(rnn_args, dict)

            rnn_dim = int(img.shape[1] / rnn_args['n_gates'])
            [plt.axvline(rnn_dim * gate_idx - .5, linewidth=0.8, color='k')
             for gate_idx in range(1, rnn_args['n_gates'])]

            kwargs['xlabel'] = rnn_args['gate_names_str']
            if "RECURRENT" in label.upper():
                plt.ylabel("Hidden Units")
            else:
                plt.ylabel("Channel Units")
        else:
            axis.set_ylabel('Examples' if 'weight' not in label.lower() else '')
            xlabels = kwargs.get('xticklabels', None)
            if xlabels is not None:
                if len(xlabels) < 30:
                    axis.set_xticklabels(xlabels, rotation=90)

        plt.colorbar(im)
        self.save_or_show(save, fname, where=where, show=show)

        return

    def plot1d(self,
               array,
               label: str = '',
               show=False,
               fname=None,
               rnn_args=None,
               where='',
               **kwargs):

        plt.close('all')

        ax = ep.plot(array,
                ax_kws=dict(title=label, xlabel="Examples"),
                show=False,
                **kwargs)

        if rnn_args is not None:
            assert isinstance(rnn_args, dict)

            rnn_dim = int(array.shape[0] / rnn_args['n_gates'])
            [plt.axvline(rnn_dim * gate_idx - .5, linewidth=0.5, color='k')
             for gate_idx in range(1, rnn_args['n_gates'])]
            plt.xlabel(rnn_args['gate_names_str'])

        self.save_or_show(save=True, fname=fname, where=where, show=show)

        return ax

    def save_or_show(self, *args, **kwargs):

        return save_or_show(self.path, *args, **kwargs)

    def plot2d_act_for_a_sample(self, activations, sample=0, save: bool = False, name: str = None):
        from ai4water.utils.visualizations import init_subplots

        fig, axis = init_subplots(height=8)
        # for idx, ax in enumerate(axis):
        im = axis.imshow(activations[sample, :, :].transpose(), aspect='auto')
        axis.set_xlabel('lookback')
        axis.set_ylabel('inputs')
        axis.set_title('Activations of all inputs at different lookbacks for sample ' + str(sample))
        fig.colorbar(im)
        self.save_or_show(save=save, fname=name + '_' + str(sample), where='path')

        return

    def plot1d_act_for_a_sample(self, activations, sample=0, save=False, name=None):
        _, axis = plt.subplots()

        for idx in range(self.lookback-1):
            axis.plot(activations[sample, idx, :].transpose(), label='lookback '+str(idx))
        axis.set_xlabel('inputs')
        axis.set_ylabel('activation weight')
        axis.set_title('Activations at different lookbacks for all inputs for sample ' + str(sample))
        self.save_or_show(save=save, fname=name + '_' + str(sample), where='path')

        return

    def plot_train_data(self, how='3d', save=True,  **kwargs):

        x,  _ = self.training_data(**kwargs)
        self.plot_model_input_data(x, how=how, save=save, which='training')

        return

    def plot_val_data(self, how='3d', save=True,  **kwargs):

        x, y = self.validation_data(**kwargs)
        self.plot_model_input_data(x, how=how, save=save, which='validation')

        return

    def plot_test_data(self, how='3d', save=True,  **kwargs):

        x, _ = self.test_data(**kwargs)
        self.plot_model_input_data(x, how=how, save=save, which='test')

        return

    def plot_model_input_data(self,
                              in_data: Union[list, np.ndarray],
                              how: str,
                              save: bool,
                              which: str = 'training'
                              ) -> None:

        assert how in ['3d', 'hist']

        if not isinstance(in_data, list):
            if isinstance(in_data, dict):
                in_data = list(in_data.values())
            else:
                assert isinstance(in_data, np.ndarray)
                in_data = [in_data]

        for idx, inputs in enumerate(in_data):
            if np.ndim(inputs) == 3:
                if how.upper() == "3D":
                    self._imshow_3d(inputs, which + '_data_' + str(idx), save=save, where='data')
            elif np.ndim(inputs) == 2:
                if how.upper() == "3D":
                    self._imshow(inputs, save=save, fname=which + '_data_' + str(idx), where='data')
                else:
                    self.plot_histogram(inputs,
                                        save=save,
                                        fname=which+'_data_' + str(idx),
                                        features=self.in_cols,
                                        where='data')
            else:
                print(f'skipping shape is {inputs.shape}')
        return

    def plot_histogram(self,
                       data: np.ndarray,
                       save: bool = True,
                       fname='hist',
                       features=None,
                       where='data'
                       ):

        assert data.ndim == 2

        data = pd.DataFrame(data, columns=features)
        data.hist(figsize=(12, 12))

        self.save_or_show(save=save, fname=fname+'hist', where=where)
        return


def validate_freq(df, freq):
    assert isinstance(df.index, pd.DatetimeIndex), "index of dataframe must be pandas DatetimeIndex"
    assert freq in ["weekly", "monthly",
                    "yearly"], f"freq must be one of {'weekly', 'monthly', 'yearly'} but it is {freq}"
    return


def _get_nrows_and_ncols(n_subplots, n_rows=None):
    if n_rows is None:
        n_rows = int(np.sqrt(n_subplots))
    n_cols = max(int(n_subplots / n_rows), 1)  # ensure n_cols != 0
    n_rows = int(n_subplots / n_cols)

    while not ((n_subplots / n_cols).is_integer() and
               (n_subplots / n_rows).is_integer()):
        n_cols -= 1
        n_rows = int(n_subplots / n_cols)
    return n_rows, n_cols


def save_or_show(path, save: bool = True, fname=None,
                 where='',
                 dpi=300, bbox_inches='tight', close=False,
                 show=False):

    if save:
        assert isinstance(fname, str)
        if "/" in fname:
            fname = fname.replace("/", "__")
        if ":" in fname:
            fname = fname.replace(":", "__")

        save_dir = os.path.join(path, where)

        if not os.path.exists(save_dir):
            assert os.path.dirname(where) in ['',
                                              'activations',
                                              'weights',
                                              'plots', 'data', 'results'], f"unknown directory: {where}"
            save_dir = os.path.join(path, where)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        fname = os.path.join(save_dir, fname + ".png")

        plt.savefig(fname, dpi=dpi, bbox_inches=bbox_inches)
    if show:
        plt.show()

    elif close:
        plt.close('all')
    return


def to_1d_array(array_like) -> np.ndarray:

    if array_like.__class__.__name__ in ['list', 'tuple', 'Series']:
        return np.array(array_like)

    elif array_like.__class__.__name__ == 'ndarray':
        if array_like.ndim == 1:
            return array_like
        else:
            assert array_like.size == len(array_like), f'cannot convert multidim ' \
                                                       f'array of shape {array_like.shape} to 1d'
            return array_like.reshape(-1, )

    elif array_like.__class__.__name__ == 'DataFrame' and array_like.ndim == 2:
        return array_like.values.reshape(-1,)
    else:
        raise ValueError(f'cannot convert object array {array_like.__class__.__name__}  to 1d ')
