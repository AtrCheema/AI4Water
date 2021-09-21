
import os
import json
import warnings
from typing import Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from ai4water.backend import xgboost, tf
from ai4water.utils.visualizations import Plot
from ai4water.utils.plotting_tools import draw_bar_sns


class Interpret(Plot):

    def __init__(self, model):
        """Interprets the ai4water Model."""
        self.model = model

        super().__init__(model.path)

        if self.model.category.upper() == "DL":

            if hasattr(model, 'interpret') and not model.__class__.__name__ == "Model":
                model.interpret()
            else:

                if hasattr(model, 'TemporalFusionTransformer_attentions'):
                    atten_components = self.tft_attention_components()

        elif self.model.category == 'ML':
            use_xgb = False
            if self.model._model.__class__.__name__ == "XGBRegressor":
                use_xgb = True
            self.plot_feature_importance(use_xgb = use_xgb)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, x):
        self._model = x

    def plot(self):
        """
        Currently does nothing.

        For NBeats, plot seasonality and trend https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/ar.html#Interpret-model
        For TFT, attention, variable importance of static, encoder and decoder, partial dependency
        # https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/stallion.html#Variable-importances

        """

    def feature_importance(self):
        if self.model.category.upper() == "ML":

            estimator = self.model._model

            if not is_fitted(estimator):
                print(f"the model {estimator} is not fitted yet so not feature importance")
                return

            model_name = list(self.model.config['model'].keys())[0]
            if model_name.upper() in ["SVC", "SVR"]:
                if estimator.kernel == "linear":
                    # https://stackoverflow.com/questions/41592661/determining-the-most-contributing-features-for-svm-classifier-in-sklearn
                    return estimator.coef_
            elif hasattr(estimator, "feature_importances_"):
                return estimator.feature_importances_

    def f_importances_svm(self, coef, names, save):

        plt.close('all')
        mpl.rcParams.update(mpl.rcParamsDefault)
        classes = coef.shape[0]
        features = coef.shape[1]
        _, axis = plt.subplots(classes, sharex='all')
        axis = axis if hasattr(axis, "__len__") else [axis]

        for idx, ax in enumerate(axis):
            # colors = ['red' if c < 0 else 'blue' for c in self._model.coef_[idx]]
            ax.bar(range(features), self._model.coef_[idx], 0.4)

        plt.xticks(ticks=range(features), labels=self.model.in_cols, rotation=90, fontsize=12)
        self.save_or_show(save=save, fname=f"{list(self.model.config['model'].keys())[0]}_feature_importance")
        return

    def plot_feature_importance(self,
                                importance=None,
                                save=True,
                                use_xgb=False,
                                max_num_features=20,
                                figsize=None,
                                **kwargs):

        figsize = figsize or (8, 8)

        if importance is None:
            importance = self.feature_importance()

        if self.model.category == "ML":
            model_name = list(self.model.config['model'].keys())[0]
            if model_name.upper() in ["SVC", "SVR"]:
                if self.model._model.kernel == "linear":
                    return self.f_importances_svm(importance, self.model.in_cols, save=save)
                else:
                    warnings.warn(f"for {self.model._model.kernel} kernels of {model_name}, feature "
                                  f"importance can not be plotted.")
                return

        if isinstance(importance, np.ndarray):
            assert importance.ndim <= 2

        if importance is None:
            return

        use_prev = self.model.config['use_predicted_output']
        all_cols = self.model.config['input_features'] if use_prev else self.model.config['input_features'] + \
                                                                                self.model.config['output_features']
        imp_sort = np.sort(importance)[::-1]
        all_cols = np.array(all_cols)
        all_cols = all_cols[np.argsort(importance)[::-1]]

        # save the whole importance before truncating it
        fname = os.path.join(self.model.path, 'feature_importance.csv')
        pd.DataFrame(imp_sort, index=all_cols, columns=['importance_sorted']).to_csv(fname)

        imp = np.concatenate([imp_sort[0:max_num_features], [imp_sort[max_num_features:].sum()]])
        all_cols = list(all_cols[0:max_num_features]) + [f'rest_{len(all_cols) - max_num_features}']

        if use_xgb:
            self._feature_importance_xgb(max_num_features=max_num_features, save=save)
        else:
            plt.close('all')
            _, axis = plt.subplots(figsize=figsize)
            draw_bar_sns(axis, y=all_cols, x=imp, title="Feature importance", xlabel_fs=12)
            self.save_or_show(save, fname="feature_importance.png")
        return

    def _feature_importance_xgb(self, save=True, max_num_features=None, **kwargs):
        # https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.plot_importance
        if xgboost is None:
            warnings.warn("install xgboost to plot plot_importance using xgboost", UserWarning)
        else:
            booster = self._model.model.get_booster()
            plt.close('all')
            # global feature importance with xgboost comes with different types
            xgboost.plot_importance(booster, max_num_features=max_num_features)
            self.save_or_show(save, fname="feature_importance_weight.png")
            plt.close('all')
            xgboost.plot_importance(booster, importance_type="cover",
                                    max_num_features=max_num_features, **kwargs)
            self.save_or_show(save, fname="feature_importance_type_cover.png")
            plt.close('all')
            xgboost.plot_importance(booster, importance_type="gain",
                                    max_num_features=max_num_features, **kwargs)
            self.save_or_show(save, fname="feature_importance_type_gain.png")

        return

    def tft_attention_components(self, model=None, data_type='training') -> dict:
        """
        Gets attention components of tft layer from ai4water's Model.
        Arguments:
        model : a ai4water's Model instance.
        train_data_args : keyword arguments which will passed to `training_data`
        method to fetch processed input data

        returns:
            dictionary containing attention components of tft as numpy arrays. Following four attention
            components are present in the dictionary
            decoder_self_attn: (attention_heads, ?, total_time_steps, 22)
            static_variable_selection_weights:
            encoder_variable_selection_weights: (?, encoder_steps, input_features)
            decoder_variable_selection_weights: (?, decoder_steps, input_features)
        """
        if model is None:
            model = self.model

        x, _, = getattr(model, f'{data_type}_data')()
        attention_components = {}

        for k, v in model.TemporalFusionTransformer_attentions.items():
            if v is not None:
                temp_model = tf.keras.Model(inputs=model._model.inputs,
                                            outputs=v)
                attention_components[k] = temp_model.predict(x=x, verbose=1, steps=1)
        return attention_components

    def interpret_tft(self, outputs: dict, tft_params: dict, reduction: Union[None, str] = None):
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


def is_fitted(estimator):

    if hasattr(estimator, 'is_fitted'):  # for CATBoost
        return estimator.is_fitted

    attrs = [v for v in vars(estimator)
             if v.endswith("_") and not v.startswith("__")]

    if not attrs:
        return False

    return True
