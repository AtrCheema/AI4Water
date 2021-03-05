import os
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from dl4seq.utils.TSErrors import FindErrors


class Intrepretation(object):

    def __init__(self, model=None):
        self.model = model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, x):
        self._model = x

    def plot(self):
        """
        For NBeats, plot seasonality and trend https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/ar.html#Interpret-model
        For TFT, attention, variable importance of static, encoder and decoder, partial dependency
        # https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/stallion.html#Variable-importances

        """

    def interpret_tft(self, outputs:dict, tft_params:dict, reduction: Union[None, str]=None):
        """
        inspired from  `interpret_output` of PyTorchForecasting
        :param outputs: outputs from tft model. It is expected to have following keys and their values as np.ndarrays
            prediction: (num_examples, forecast_len, outs/quantiles)
            attention: (num_examples, forecast_len, num_heads, total_sequence_length)
            static_variables: (num_examples, 1, 7)
            encoder_variables: (batch_size, encoder_length, 1, 13)
            decoder_variables: (batch_size, decoder_steps, 1, 6)
            encoder_lengths: (num_examples,)
            decoder_lengths: (num_examples,)
            groups: (num_examples, num_groups)
            decoder_time_index: (num_examples, forecast_len)
        :param tft_params: parameters which were used to build tft layer
        :param reduction:

        Returns:
            intetrpretation: dict, dictionary of keys with values as np.ndarrays
                attention: (encoder_length,)
                static_variables: (7,)
                encoder_variables: (13,)
                decoder_variables: (6,)
                encoder_length_histogram: (encoder_length+1)
                decoder_length_histogram: (decoder_length,)
        """

        # histogram of decode and encode lengths
        encoder_length_histogram = integer_histogram(outputs["encoder_lengths"], _min=0, _max=tft_params['max_encoder_length'])
        decoder_length_histogram = integer_histogram(
            outputs["decoder_lengths"], _min=1, _max=outputs["decoder_variables"].size(1)
        )

        # mask where decoder and encoder where not applied when averaging variable selection weights
        encoder_variables = outputs["encoder_variables"].squeeze(-2)
        encode_mask = create_mask(encoder_variables.size(1), outputs["encoder_lengths"])
        encoder_variables = encoder_variables.masked_fill(encode_mask.unsqueeze(-1), 0.0).sum(dim=1)
        encoder_variables /= (
            outputs["encoder_lengths"]
            .where(outputs["encoder_lengths"] > 0, np.ones_like(outputs["encoder_lengths"]))
            .unsqueeze(-1)
        )

        decoder_variables = outputs["decoder_variables"].squeeze(-2)
        decode_mask = create_mask(decoder_variables.size(1), outputs["decoder_lengths"])
        decoder_variables = decoder_variables.masked_fill(decode_mask.unsqueeze(-1), 0.0).sum(dim=1)
        decoder_variables /= outputs["decoder_lengths"].unsqueeze(-1)

        if reduction is not None:  # if to average over batches
            assert reduction in ['mean', 'sum']
            encoder_variables = encoder_variables.sum(dim=0)
            decoder_variables = decoder_variables.sum(dim=0)


        interpretation = dict(
            #attention=attention,
            #static_variables=static_variables,
            encoder_variables=encoder_variables,
            decoder_variables=decoder_variables,
            encoder_length_histogram=encoder_length_histogram,
            decoder_length_histogram=decoder_length_histogram,
        )
        return interpretation


def integer_histogram(
    data: np.ndarray, _min: Union[None, int] = None, _max: Union[None, int] = None
) -> np.ndarray:
    """
    Create histogram of integers in predefined range
    Args:
        data: data for which to create histogram
        _min: minimum of histogram, is inferred from data by default
        _max: maximum of histogram, is inferred from data by default
    Returns:
        histogram
    """
    uniques, counts = np.unique(data, return_counts=True)
    if _min is None:
        _min = uniques.min()
    if _max is None:
        _max = uniques.max()
    #hist = np.zeros(_max - _min + 1, dtype=np.long).scatter(
    #    dim=0, index=uniques - _min, src=counts
    #)
    hist = scatter_numpy(self=np.zeros(_max - _min + 1, dtype=np.long),
                         dim=0, index=uniques-_min, src=counts)
    return hist


def create_mask(size: int, lengths: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Create boolean masks of shape len(lenghts) x size.
    An entry at (i, j) is True if lengths[i] > j.
    Args:
        size (int): size of second dimension
        lengths (torch.LongTensor): tensor of lengths
        inverse (bool, optional): If true, boolean mask is inverted. Defaults to False.
    Returns:
        torch.BoolTensor: mask
    """
    if inverse:  # return where values are
        return np.arange(size, ).unsqueeze(0) < lengths.unsqueeze(-1)
    else:  # return where no values are
        return np.arange(size, ).unsqueeze(0) >= lengths.unsqueeze(-1)


def scatter_numpy(self, dim, index, src):
    """
    Writes all values from the Tensor src into self at the indices specified in the index Tensor.
    :param self:
    :param dim: The axis along which to index
    :param index: The indices of elements to scatter
    :param src: The source element(s) to scatter
    :return: self
    """
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    if self.ndim != index.ndim:
        raise ValueError("Index should have the same number of dimensions as output")
    if dim >= self.ndim or dim < -self.ndim:
        raise IndexError("dim is out of range")
    if dim < 0:
        # Not sure why scatter should accept dim < 0, but that is the behavior in PyTorch's scatter
        dim = self.ndim + dim
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = self.shape[:dim] + self.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and output should be the same size")
    if (index >= self.shape[dim]).any() or (index < 0).any():
        raise IndexError("The values of index must be between 0 and (self.shape[dim] -1)")

    def make_slice(arr, _dim, i):
        slc = [slice(None)] * arr.ndim
        slc[_dim] = i
        return slc

    # We use index and dim parameters to create idx
    # idx is in a form that can be used as a NumPy advanced index for scattering of src param. in self
    idx = [[*np.indices(idx_xsection_shape).reshape(index.ndim - 1, -1),
            index[make_slice(index, dim, i)].reshape(1, -1)[0]] for i in range(index.shape[dim])]
    idx = list(np.concatenate(idx, axis=1))
    idx.insert(dim, idx.pop())

    if not np.isscalar(src):
        if index.shape[dim] > src.shape[dim]:
            raise IndexError("Dimension " + str(dim) + "of index can not be bigger than that of src ")
        src_xsection_shape = src.shape[:dim] + src.shape[dim + 1:]
        if idx_xsection_shape != src_xsection_shape:
            raise ValueError("Except for dimension " +
                             str(dim) + ", all dimensions of index and src should be the same size")
        # src_idx is a NumPy advanced index for indexing of elements in the src
        src_idx = list(idx)
        src_idx.pop(dim)
        src_idx.insert(dim, np.repeat(np.arange(index.shape[dim]), np.prod(idx_xsection_shape)))
        self[idx] = src[src_idx]

    else:
        self[idx] = src

    return self


class Visualizations(object):

    def __init__(self, config: dict=None, data=None, path=None, dpi=300):
        self.config = config
        self.data=data
        self.path = path
        self.dpi = dpi

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, x):
        self._config = x

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, x):
        self._data = x

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, x):
        if x is None:
            x = os.getcwd()
        self._path = x

    def save_or_show(self, save: bool = True, fname=None, where='', dpi=300, bbox_inches='tight', close=True):

        if save:
            assert isinstance(fname, str)
            if "/" in fname:
                fname = fname.replace("/", "__")
            if ":" in fname:
                fname = fname.replace(":", "__")

            save_dir = os.path.join(self.path, where)

            if not os.path.exists(save_dir):
                assert os.path.dirname(where) in ['', 'activations', 'weights', 'plots', 'data', 'results'], f"unknown directory: {where}"
                save_dir = os.path.join(self.path, where)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

            fname = os.path.join(save_dir, fname + ".png")

            plt.savefig(fname, dpi=dpi, bbox_inches=bbox_inches)
        else:
            plt.show()

        if close:
            plt.close('all')
        return

    def horizon_plots(self, errors:dict, fname='', save=True):
        plt.close('')
        fig, axis = plt.subplots(len(errors), sharex='all')

        legends = {'r2': "$R^2$", 'rmse': "RMSE", 'nse': "NSE"}
        idx = 0
        for metric_name, val in errors.items():
            ax = axis[idx]
            ax.plot(val, '--o', label=legends.get(metric_name, metric_name))
            ax.legend(fontsize=14)
            if idx>=len(errors)-1: ax.set_xlabel("Horizons", fontsize=14)
            ax.set_ylabel(legends.get(metric_name, metric_name), fontsize=14)
            idx += 1
        self.save_or_show(save=save, fname=fname)
        return

    def plot_results(self, true, predicted:pd.DataFrame, save=True, name=None, where=None):
        """
        # kwargs can be any/all of followings
            # fillstyle:
            # marker:
            # linestyle:
            # markersize:
            # color:
        """

        self.regplot_using_searborn(true, predicted, save=save, name=name, where=where)
        plt.close('all')
        mpl.rcParams.update(mpl.rcParamsDefault)

        fig, axis = plt.subplots()
        set_fig_dim(fig, 12, 8)

        # it is quite possible that when data is datetime indexed, then it is not equalidistant and large amount of graph
        # will have not data in that case lines plot will create a lot of useless interpolating lines where no data is present.
        style = '.' if isinstance(true.index, pd.DatetimeIndex) else '-'

        if np.isnan(true.values).sum() > 0:
            style = '.' # For Nan values we should be using this style otherwise nothing is plotted.

        ms = 4 if style == '.' else 2

        axis.plot(predicted, style, color='r', linestyle='-', marker='', label='Prediction')

        axis.plot(true, style, color='b', marker='o', fillstyle='none',  markersize=ms, label='True')

        axis.legend(loc="best", fontsize=22, markerscale=4)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel("Time", fontsize=18)

        self.save_or_show(save=save, fname=name, close=False, where=where)
        return

    def regplot_using_searborn(self, true, pred, save, name, where='plots', **kwargs):
        # https://seaborn.pydata.org/generated/seaborn.regplot.html
        if any([isinstance(true, _type) for _type in [pd.DataFrame, pd.Series]]):
            true = true.values.reshape(-1,)
        if any([isinstance(pred, _type) for _type in [pd.DataFrame, pd.Series]]):
            pred = pred.values.reshape(-1,)

        plt.close('all')

        s = kwargs.get('s', 20)
        cmap = kwargs.get('cmap', 'winter')  # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        figsize = kwargs.get('figsize', (8, 5.5))

        plt.figure(figsize=figsize)
        points = plt.scatter(true, pred, c=pred, s=s, cmap=cmap)  # set style options

        if kwargs.get('annotate', True):
            plt.annotate(f'$R^{2}$: {round(FindErrors(true, pred).r2(), 3)}', xy=(0.50, 0.95), xycoords='axes fraction',
                     horizontalalignment='right', verticalalignment='top', fontsize=16)

        if kwargs.get('colorbar', False):
            plt.colorbar(points)

        sns.regplot(x=true, y=pred, scatter=False, color=".1")
        plt.xlabel('Observed', fontsize=14)
        plt.ylabel('Predicted', fontsize=14)

        self.save_or_show(save=save, fname=name + "_reg", close=False, where=where)
        return

    def plot_loss(self, history: dict, name="loss_curve"):
        """Considering history is a dictionary of different arrays, possible training and validation loss arrays,
        this method plots those arrays."""

        plt.clf()
        plt.close('all')
        fig = plt.figure()
        plt.style.use('ggplot')
        i = 1

        legends = {
            'mean_absolute_error': 'Mean Absolute Error',
            'mape': 'Mean Absolute Percentage Error',
            'mean_squared_logarithmic_error': 'Mean Squared Logrithmic Error',
            'pbias': "Percent Bias",
            "nse": "Nash-Sutcliff Efficiency",
            "kge": "Kling-Gupta Efficiency"
        }

        sub_plots = {1: {'axis': (1,1), 'width': 10, 'height': 7},
                     2: {'axis': (1, 1), 'width': 10, 'height': 7},
                     3: {'axis': (1, 2), 'wdith': 10, 'height': 10},
                     4: {'axis': (1, 2), 'width': 10, 'height': 10},
                     5: {'axis': (1, 3), 'width': 15, 'height': 10},
                     6: {'axis': (1, 3), 'width': 20, 'height': 20},
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
                axis.plot(epochs, val, color=[0.96707953, 0.46268314, 0.45772886], label= 'Validation ')
                axis.legend()
            else:
                axis = fig.add_subplot(*sub_plots[len(history)]['axis'], i)
                axis.plot(epochs, val, color=[0.13778617, 0.06228198, 0.33547859], label= 'Training ')
                axis.legend()
                axis.set_xlabel("Epochs")
                axis.set_ylabel(legends.get(key, key))
                axis_cache[key] = axis
                i += 1
            axis.set(frame_on=True)

        fig.set_figheight(sub_plots[len(history)]['height'])
        fig.set_figwidth(sub_plots[len(history)]['width'])
        self.save_or_show(fname=name, save=True if name is not None else False)
        return



def set_fig_dim(fig, width, height):
    fig.set_figwidth(width)
    fig.set_figheight(height)
