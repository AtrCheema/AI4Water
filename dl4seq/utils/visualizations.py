import os
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from dl4seq.utils.TSErrors import FindErrors
from dl4seq.utils.utils import _missing_vals
from dl4seq.utils.utils import find_tot_plots, set_fig_dim
from dl4seq.utils.transformations import Transformations

# TODO add Murphy's plot as shown in MLAir
# https://robjhyndman.com/hyndsight/murphy-diagrams/
# competitive skill score plot/ bootstrap skill score plot as in MLAir
# rank histogram and reliability diagram for probabilitic forecasting model.
# show availability plot of data

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

    def tft_attention_components(self, model=None, **train_data_args):
        """
        Gets attention components of tft layer from dl4seq's Model.
        model: a dl4seq's Model instance.
        train_data_args: keyword arguments which will passed to train_data method to fetch processed input data"""
        if model is None:
            model = self.model

        x, _, _ = model.train_data(**train_data_args)
        attention_components = {}

        for k, v in model.TemporalFusionTransformer_attentions.items():
            if v is not None:
                temp_model = tf.keras.Model(inputs=model._model.inputs,
                                            outputs=v)
                attention_components[k] = temp_model.predict(x=x, verbose=1, steps=1)
        return attention_components

    def interpret_tft(self, outputs:dict, tft_params:dict, reduction: Union[None, str]=None):
        """
        inspired from  `interpret_output` of PyTorchForecasting
        :param outputs: outputs from tft model. It is expected to have following keys and their values as np.ndarrays
            prediction: (num_examples, forecast_len, outs/quantiles)
            attention: (num_examples, forecast_len, num_heads, total_sequence_length)
            static_variable_selection_weights: (num_examples, 1, num_static_inputs)
            encoder_variable_selection_weights: (batch_size, encoder_length, 1, num_enocder_inputs)
            decoder_variable_selection_weights: (batch_size, decoder_steps, 1, num_decoder_inputs)
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
        num_examples = outputs['predictions'].shape[0]
        encoder_lengths = np.full(num_examples, tft_params['num_encoder_steps'])
        decoder_lengths = np.full(num_examples, tft_params['total_time_steps'] - tft_params['num_encoder_steps'])

        # histogram of decode and encode lengths
        encoder_length_histogram = integer_histogram(encoder_lengths, _min=0, _max=tft_params['num_encoder_steps'])
        decoder_length_histogram = integer_histogram(
            decoder_lengths, _min=1, _max=decoder_lengths.shape[1]
        )

        # mask where decoder and encoder where not applied when averaging variable selection weights
        encoder_variables = outputs["encoder_variables"].squeeze(-2)
        encode_mask = create_mask(encoder_variables.shape[1], encoder_lengths)
        encoder_variables = encoder_variables.masked_fill(encode_mask.unsqueeze(-1), 0.0).sum(dim=1)
        encoder_variables /= (
            outputs["encoder_lengths"]
            .where(encoder_lengths > 0, np.ones_like(encoder_lengths))
            .unsqueeze(-1)
        )

        decoder_variables = decoder_lengths.squeeze(-2)
        decode_mask = create_mask(decoder_variables.size(1), decoder_lengths)
        decoder_variables = decoder_variables.masked_fill(decode_mask.unsqueeze(-1), 0.0).sum(dim=1)
        decoder_variables /= decoder_lengths.unsqueeze(-1)

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

    def __init__(self, data=None, config: dict=None, path=None, dpi=300, in_cols=None, out_cols=None):
        self.config = config
        self.data=data
        self.path = path
        self.dpi = dpi
        self.in_cols = in_cols
        self.out_cols = out_cols

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

    def regplot_using_searborn(self, true, pred, save, name=None, where='plots', **kwargs):
        """
        Following kwargs are allowed:
            figsize: tuple
            colorbar: for plt.colorbar
            cmap: for plt.scatter
            s: for plt.scatter
        >>>from dl4seq.utils import Visualizations
        >>>import numpy as np
        >>>true = np.random.random(100)
        >>>pred = np.random.random(100)
        >>>Visualizations().regplot_using_searborn(a,b,save=False)
        """
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

        self.save_or_show(save=save, fname=f"{name}_reg", close=False, where=where)
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
            "kge": "Kling-Gupta Efficiency",
            "tf_r2": "$R^{2}$"
        }

        sub_plots = {1: {'axis': (1,1), 'width': 9, 'height': 6},
                     2: {'axis': (1, 1), 'width': 9, 'height': 6},
                     3: {'axis': (1, 2), 'wdith': 9, 'height': 6},
                     4: {'axis': (1, 2), 'width': 9, 'height': 6},
                     5: {'axis': (3, 1), 'width': 8, 'height': 12},
                     6: {'axis': (3, 1), 'width': 8, 'height': 12},
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

    def plot_index(self, save=True, **kwargs):
        """plots the datetime index of dataframe"""
        if isinstance(self.data, pd.DataFrame):
            self._plot_index(self.data, save=save, **kwargs)

        elif isinstance(self.data, list):
            for data in self.data:
                if isinstance(data, pd.DataFrame):
                    self._plot_index(data, save=save, **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.values():
                if isinstance(data, pd.DataFrame):
                    self._plot_index(data, save=save, **kwargs)
        return

    def _plot_index(self,
                    index,
                    save=True,
                    fname="index",
                    figsize=(10,5),
                    dpi=200,
                    label_fs=18,
                    title_fs=20,
                    leg_fs=14,
                    leg_ms=4,
                    color='r',
                    ):
        """
        Plots the index of a datafram.
        index: can be pandas dataframe or index itself. if dataframe, its index will be used for plotting
        """
        plt.close('all')
        if isinstance(index, pd.DataFrame):
            index=index.index

        idx = pd.DataFrame(np.ones(len(index)), index=index, columns=['Observations'])
        axis = idx.plot(linestyle='', marker='.', color=color, figsize=figsize)
        axis.legend(fontsize=leg_fs, markerscale=leg_ms)
        axis.set_xlabel(axis.get_xlabel(), fontdict={'fontsize': label_fs})
        axis.set_title("Temporal distribution of Observations", fontsize=title_fs)
        axis.get_yaxis().set_visible(False)
        self.save_or_show(save=save, fname=fname, where='data', dpi=dpi)
        return

    def heatmap(self, cols=None, **kwargs):
        """
        Plots a heatmap.
        :param cols:
        :param kwargs:
        :return:
        Examples:
        >>>vis = Visualizations(data)
        >>>vis.heatmap(save=False)
        """
        if isinstance(self.data, pd.DataFrame):
            self.heatmap_df(self.data, cols=cols, **kwargs)

        elif isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                if isinstance(data, pd.DataFrame):
                    self.heatmap_df(data, cols=cols[idx] if isinstance(cols, list) else None,
                                       fname=f"data_heatmap_{idx}", **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                if isinstance(data, pd.DataFrame):
                    _cols = cols[data_name] if cols is not None else None
                    self.heatmap_df(data, _cols, fname=data_name, **kwargs)
        return

    def heatmap_df(
            self,
            data:pd.DataFrame,
            cols=None,
            spine_color: str = "#EEEEEE",
            save=True,
            title=None,
            title_fs=16,
            fname="",
            **kwargs
    ):
        """
        plots a heat map of a dataframe. Helpful to show where missing values are located in a dataframe.
        :param data:
        :param cols:
        :param spine_color:
        :param save:
        :param title:
        :param title_fs:
        :param fname:
        :return:
        """
        if cols is None:
            cols = data.columns
        _kwargs = {
            "xtick_labels_fs": 12,
            "ytick_labels_fs": 20,
            "figsize": (5 + len(cols)*0.25, 10 + len(cols)*0.1),
        }
        for k in _kwargs.keys():
            if k in kwargs:
                _kwargs[k] = kwargs.pop(k)

        fig, axis = plt.subplots(figsize=_kwargs['figsize'])
        # ax2 - Heatmap
        sns.heatmap(data[cols].isna(), cbar=False, cmap="binary", ax=axis)
        axis.set_yticks(axis.get_yticks()[0::5].astype('int'))

        axis.set_yticklabels(axis.get_yticks(),
                            fontsize=_kwargs['ytick_labels_fs'])
        axis.set_xticklabels(
            axis.get_xticklabels(),
            horizontalalignment="center",
            fontweight="light",
            fontsize=_kwargs['xtick_labels_fs'],
        )
        axis.tick_params(length=1, colors="#111111")
        axis.set_ylabel("Examples", fontsize="24")
        for _, spine in axis.spines.items():
            spine.set_visible(True)
            spine.set_color(spine_color)
        if title is not None:
            axis.set_title(title, fontsize=title_fs)

        return self.save_or_show(save=save, fname=fname+'_heat_map', where='data', dpi=500)

    def plot_missing(self, save:bool=True, cols=None, **kwargs):
        """
        cols: columns to be used.
        save: if False, plot will be shown and not plotted.
        Examples:
        >>>vis = Visualizations(data)
        >>>vis.plot_missing(save=False)
        """
        if isinstance(self.data, pd.DataFrame):
            self.plot_missing_df(self.data, cols=cols, save=save, **kwargs)

        elif isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                _cols = cols[idx] if isinstance(cols, list) else None
                self.plot_missing_df(data, cols=None, fname=str(idx), save=save, **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                if isinstance(data, pd.DataFrame):
                    _cols = cols[data_name] if cols else None
                    self.plot_missing_df(data, cols=_cols, fname=data_name, save=save, **kwargs)
        return

    def plot_missing_df(self,
                        data:pd.DataFrame,
                        cols=None,
                        fname:str='',
                        save:bool=True,
                        **kwargs):
        """
        kwargs:
            xtick_labels_fs
            figsize
        """
        if cols is None:
            cols = data.columns
        data = data[cols]
        # Identify missing values
        mv_total, mv_rows, mv_cols, _, mv_cols_ratio = _missing_vals(data).values()

        _kwargs = {
            "xtick_labels_fs": 12,
            "ytick_labels_fs": 20,
            "figsize": (5 + len(cols)*0.25, 10 + len(cols)*0.1),
        }
        for k in _kwargs.keys():
            if k in kwargs:
                _kwargs[k] = kwargs.pop(k)
        if mv_total < 6:
            print("No missing values found in the dataset.")
        else:
            # Create figure and axes
            plt.close('all')
            fig = plt.figure(figsize=_kwargs['figsize'])
            gs = fig.add_gridspec(nrows=1, ncols=1, left=0.1, wspace=0.05)
            ax1 = fig.add_subplot(gs[:1, :5])

            # ax1 - Barplot
            ax1 = sns.barplot(x=list(data.columns), y=np.round(mv_cols_ratio * 100, 2), ax=ax1, **kwargs)
            ax1.set(frame_on=True, xlim=(-0.5, len(mv_cols) - 0.5))
            ax1.set_ylim(0, np.max(mv_cols_ratio) * 100)
            ax1.grid(linestyle=":", linewidth=1)
            ax1.set_yticklabels(ax1.get_yticks(),
                                fontsize="18")
            ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            ax1.set_ylabel("Missing Percentage", fontsize=_kwargs['ytick_labels_fs'])
            ax1.set_xticklabels(
                ax1.get_xticklabels(),
                horizontalalignment="center",
                fontweight="light",
                rotation=90,
                fontsize=_kwargs['xtick_labels_fs'],
            )
            ax1.tick_params(axis="y", colors="#111111", length=1)

            # annotate missing values on top of the bars
            for rect, label in zip(ax1.patches, mv_cols):
                height = rect.get_height()
                ax1.text(
                    0.1 + rect.get_x() + rect.get_width() / 2,
                    height + height*0.02,
                    label,
                    ha="center",
                    va="bottom",
                    rotation="90",
                    alpha=0.5,
                    fontsize="11",
                )
            self.save_or_show(save=save, fname=fname+'_missing_vals', where='data', dpi=500)
        return

    def plot_histograms(self, save=True, cols=None, **kwargs):
        """Plots distribution of data as histogram.
        kwargs: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.hist.html
        """
        if isinstance(self.data, pd.DataFrame):
            self.plot_his_df(self.data, save=save, cols=cols, **kwargs)

        elif isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                self.plot_his_df(data, prefix=str(idx), cols=cols, save=save, **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                if isinstance(data, pd.DataFrame):
                    self.plot_his_df(data, prefix=data_name, save=save, **kwargs)
        return

    def plot_his_df(self, data:pd.DataFrame, prefix='', cols=None, save=True, bins=100, figsize=(20, 14), **kwargs):
        if cols is None:
            cols = data.columns
        data[cols].hist(bins=bins, figsize=figsize, **kwargs)
        self.save_or_show(fname=f"hist_{prefix}", save=save, where='data')
        return

    def feature_feature_corr(self, cols=None, remove_targets=True, save=True, **kwargs):
        """
        >>>from dl4seq.utils.visualizations import Visualizations
        >>>vis = Visualizations(data)
        >>>vis.feature_feature_corr(save=False)
        """
        if cols is None:
            cols = self.in_cols if remove_targets else self.in_cols + self.out_cols
            if isinstance(cols, dict):
                cols = None

        if isinstance(self.data, pd.DataFrame):
            self.feature_feature_corr_df(self.data, cols, save=save, **kwargs)

        elif isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                if isinstance(data, pd.DataFrame):
                    self.feature_feature_corr_df(data, cols[idx] if cols is not None else None,
                                               prefix=str(idx), save=save, **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                if isinstance(data, pd.DataFrame):
                    self.feature_feature_corr_df(data, cols, prefix=data_name, save=save, **kwargs)
        return

    def feature_feature_corr_df(self,
                              data,
                              cols=None,
                              prefix='',
                              save=True,
                              split=None,
                              threshold=0,
                              method='pearson',
                              **kwargs):
        """
        split : Optional[str], optional
        Type of split to be performed {None, "pos", "neg", "high", "low"}, by default None
        method : str, optional
                 {"pearson", "spearman", "kendall"}, by default "pearson"

        kwargs
            * vmax: float, default is calculated from the given correlation \
                coefficients.
                Value between -1 or vmin <= vmax <= 1, limits the range of the cbar.
            * vmin: float, default is calculated from the given correlation \
                coefficients.
                Value between -1 <= vmin <= 1 or vmax, limits the range of the cbar.
        To plot positive correlation only:
        feature_feature_corr_df(model.data, list(model.data.columns), split="pos")
        To plot negative correlation only
        feature_feature_corr_df(model.data, list(model.data.columns), split="neg")
        """
        plt.close('all')

        if cols is None:
            cols = data.columns

        corr = data[cols].corr(method=method)

        if split == "pos":
            corr = corr.where((corr >= threshold) & (corr > 0))
        elif split == "neg":
            corr = corr.where((corr <= threshold) & (corr < 0))

        mask = np.zeros_like(corr, dtype=np.bool)

        vmax = np.round(np.nanmax(corr.where(~mask)) - 0.05, 2)
        vmin = np.round(np.nanmin(corr.where(~mask)) + 0.05, 2)
        # width x height
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (5 + len(cols)*0.25, 9 + len(cols)*0.1)))

        _kwargs = dict()
        _kwargs['annot'] = kwargs.get('annot', True if len(cols) <= 20 else False)
        _kwargs['cmap'] = kwargs.get('cmap', "BrBG")
        _kwargs['vmax'] = kwargs.get('vmax', vmax)
        _kwargs['vmin'] = kwargs.get('vmin', vmin)
        _kwargs['linewidths'] = kwargs.get('linewidths', 0.5)
        _kwargs['annot_kws'] = kwargs.get('annot_kws', {"size": 10})
        _kwargs['cbar_kws'] = kwargs.get('cbar_kws', {"shrink": 0.95, "aspect": 30})

        ax = sns.heatmap(corr, center=0, fmt=".2f", ax=ax, **_kwargs)
        ax.set(frame_on=True)
        self.save_or_show(save, fname=f"{split if split else ''}_feature_corr_{prefix}", where="data")
        return

    def grouped_scatter(self, inputs=True, outputs=True, cols=None, save=True, max_subplots=8, **kwargs):

        fname = "scatter_plot_"

        if cols is None:
            cols = []

            if inputs:
                cols += self.in_cols
                fname += "inputs_"
            if outputs:
                fname += "outptuts_"
        else:
            assert isinstance(cols, list)

        if isinstance(self.data, pd.DataFrame):
            self.grouped_scatter_plot_df(self.data[cols], max_subplots, save, **kwargs)

        elif isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                if isinstance(data, pd.DataFrame):
                    _cols = cols + [self.out_cols[idx]] if outputs else cols
                    self.grouped_scatter_plot_df(data[_cols], max_subplots, save=save, prefix=str(idx), **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                if isinstance(data, pd.DataFrame):
                    self.grouped_scatter_plot_df(data, max_subplots, save=save, prefix=data_name,
                                                 **kwargs)
        return

    def grouped_scatter_plot_df(self, data:pd.DataFrame, max_subplots:int=10, save=True, prefix='', **kwargs):
        """
        max_subplots: int, it can be set to large number to show all the scatter plots on one axis.
        """
        data = data.copy()
        if data.shape[1] <= max_subplots:
            self._grouped_scatter_plot(data, save=save,name=f'grouped_scatter_{prefix}',  **kwargs)
        else:
            tot_plots = find_tot_plots(data.shape[1], max_subplots)
            for i in range(len(tot_plots) - 1):
                st, en = tot_plots[i], tot_plots[i + 1]
                sub_df = data.iloc[:, st:en]
                self._grouped_scatter_plot(sub_df, name=f'grouped_scatter_{prefix}_{st}_{en}', **kwargs)
        return

    def _grouped_scatter_plot(self, df, save=True, name='grouped_scatter', **kwargs):
        plt.close('all')
        sns.set()
        sns.pairplot(df, size=2.5, **kwargs)
        self.save_or_show(fname=name, save=save, where='data')
        return

    def plot_pcs(self, num_pcs=None, save=True, save_as_csv=False, figsize=(12, 8), **kwargs):
        """Plots principle components.
        kwargs will go to sns.pairplot."""
        if isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                self._plot_pcs(data[self.in_cols],
                               num_pcs, save=save, prefix=str(idx), save_as_csv=save_as_csv,
                               hue=self.out_cols[idx], figsize=figsize, **kwargs)

        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                self._plot_pcs(data[self.in_cols], num_pcs, save=save, prefix=data_name, save_as_csv=save_as_csv,
                               hue=self.out_cols,
                               figsize=figsize, **kwargs)
        else:
            self._plot_pcs(self.data[self.in_cols], num_pcs, save=save, save_as_csv=save_as_csv, hue=self.out_cols,
                           figsize=figsize, **kwargs)
        return

    def _plot_pcs(self, data, num_pcs, save=True, prefix='', save_as_csv=False, hue=None, figsize=(12,8), **kwargs):

        if num_pcs is None:
            num_pcs = int(data.shape[1]/2)

        #df_pca = data[self.in_cols]
        #pca = PCA(n_components=num_pcs).fit(df_pca)
        #df_pca = pd.DataFrame(pca.transform(df_pca))

        transformer = Transformations(data=data, method='pca', n_components=num_pcs, replace_nans=True)
        df_pca = transformer.transform()

        pcs = ['pc' + str(i + 1) for i in range(num_pcs)]
        df_pca.columns = pcs

        if hue is not None:
            if isinstance(hue, list):
                hue = hue[0]
            if hue in data:
                df_pca[hue] = data[hue]

                if df_pca[hue].isna().sum() > 0: # output columns contains nans, so don't use it as hue.
                    hue = None
            else: # ignore it
                hue = None

        if save_as_csv:
            df_pca.to_csv(os.path.join(self.path, f"data\\first_{num_pcs}_pcs_{prefix}"))

        plt.close('all')
        plt.figure(figsize=figsize)
        sns.pairplot(data=df_pca, vars=pcs, hue=hue, **kwargs)
        self.save_or_show(fname=f"first_{num_pcs}_pcs_{prefix}", save=save, where='data')
        return

    def plot_data(self, save=True, freq=None, cols=None, max_subplots=10, **kwargs):
        """
        :param save:
        :param max_subplots: int, number of subplots within one plot. Each feature will be shown in a separate subplot.
        :param kwargs: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html
        :param freq: str, one of 'daily', 'weekly', 'monthly', 'yearly', determines interval of plot of data. It is
                     valid for only time-series data.
        :param cols: columns in self.data to plot
        :return:

        ------
        examples:

        >>>model.plot_data(subplots=True, figsize=(12, 14), sharex=True)
        >>>model.plot_data(freq='monthly', subplots=True, figsize=(12, 14), sharex=True)
        """
        # TODO, this method should be available from `model` as well
        if isinstance(self.data, pd.DataFrame):
            self.plot_df(self.data, cols=cols, save=save, freq=freq, max_subplots=max_subplots, **kwargs)

        if isinstance(self.data, list):
            for idx, data in enumerate(self.data):
                self.plot_df(data, cols=cols[idx] if isinstance(cols, list) else None,
                             save=save, freq=freq, prefix=str(idx), max_subplots=max_subplots, **kwargs)
        elif isinstance(self.data, dict):
            for data_name, data in self.data.items():
                if isinstance(data, pd.DataFrame):
                    self.plot_df(data, cols=cols, prefix=data_name, save=save, freq=freq, max_subplots=max_subplots,
                                 **kwargs)
        return

    def plot_df(self, df, cols=None, save=True, freq=None, max_subplots=10,
                prefix='',
                leg_kws=None,
                label_kws=None,
                tick_kws=None,
                **kwargs):
        """Plots each columns of dataframe and saves it if `save` is True.
         max_subplots: determines how many sub_plots are to be plotted within one plot. If dataframe contains columns
         greater than max_subplots, a separate plot will be generated for remaining columns.
         """
        assert isinstance(df, pd.DataFrame)
        if leg_kws is None:
            leg_kws = {'fontsize': 14}
        if label_kws is None:
            label_kws = {'fontsize': 14}
        if tick_kws is None:
            tick_kws = {'axis':"both", 'which':'major', 'labelsize':12}

        if cols is None:
            cols = list(df.columns)
        df = df[cols]

        if df.shape[1] <= max_subplots:

            if freq is None:
                kwargs = plot_style(df, **kwargs)
                axis = df.plot(**kwargs)
                if isinstance(axis, np.ndarray):
                    for ax in axis:
                        set_axis_paras(ax, leg_kws, label_kws, tick_kws)
                else:
                    set_axis_paras(axis, leg_kws, label_kws, tick_kws)

                self.save_or_show(save=save, fname=f"input_{prefix}",  where='data')
            else:
                self.plot_df_with_freq(df, freq, save, **kwargs)
        else:
            tot_plots = find_tot_plots(df.shape[1], max_subplots)

            for i in range(len(tot_plots) - 1):
                st, en = tot_plots[i], tot_plots[i + 1]
                sub_df = df.iloc[:, st:en]

                if freq is None:
                    kwargs = plot_style(sub_df, **kwargs)
                    axis = sub_df.plot(**kwargs)
                    for ax in axis:
                        ax.legend(**leg_kws)
                        ax.set_ylabel(ax.get_ylabel(), **label_kws)
                        ax.set_xlabel(ax.get_xlabel(), **label_kws)
                        ax.tick_params(**tick_kws)
                    self.save_or_show(save=save, fname=f'input_{prefix}_{st}_{en}', where='data')
                else:
                    self.plot_df_with_freq(sub_df, freq, save, prefix=f'{prefix}_{st}_{en}', **kwargs)
        return

    def plot_df_with_freq(self, df:pd.DataFrame, freq:str, save:bool=True, prefix:str='', **kwargs):
        """Plots a dataframe which has data as time-series and its index is pd.DatetimeIndex"""

        validate_freq(df, freq)

        st_year = df.index[0].year
        en_year = df.index[-1].year

        for yr in range(st_year, en_year + 1):

            _df = df[df.index.year == yr]

            if freq == 'yearly':
                kwargs = plot_style(_df, **kwargs)
                _df.plot(**kwargs)
                self.save_or_show(save=save, fname=f'input_{prefix}_{str(yr)}', where='data')

            elif freq == 'monthly':
                st_mon = _df.index[0].month
                en_mon = _df.index[-1].month

                for mon in range(st_mon, en_mon+1):

                    __df = _df[_df.index.month == mon]
                    kwargs = plot_style(__df, **kwargs)
                    __df.plot(**kwargs)
                    self.save_or_show(save=save, fname=f'input_{prefix}_{str(yr)} _{str(mon)}', where='data/monthly')

            elif freq == 'weekly':
                st_week = _df.index[0].isocalendar()[1]
                en_week = _df.index[-1].isocalendar()[1]

                for week in range(st_week, en_week+1):
                    __df = _df[_df.index.week == week]
                    kwargs = plot_style(__df, **kwargs)
                    __df.plot(**kwargs)
                    self.save_or_show(save=save, fname=f'input_{prefix}_{str(yr)} _{str(week)}', where='data/weely')
        return


def set_axis_paras(axis, leg_kws, label_kws, tick_kws):
    axis.legend(**leg_kws)
    axis.set_ylabel(axis.get_ylabel(), **label_kws)
    axis.set_xlabel(axis.get_xlabel(), **label_kws)
    axis.tick_params(**tick_kws)
    return


def plot_style(df:pd.DataFrame, **kwargs):
    if 'style' not in kwargs and df.isna().sum().sum() > 0:
        kwargs['style'] = ['.' for _ in range(df.shape[1])]
    return kwargs


def validate_freq(df, freq):
    assert isinstance(df.index, pd.DatetimeIndex), "index of dataframe must be pandas DatetimeIndex"
    assert freq in ["weekly", "monthly",
                    "yearly"], f"freq must be one of {'weekly', 'monthly', 'yearly'} but it is {freq}"
    return