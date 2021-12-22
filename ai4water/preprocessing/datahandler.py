import os
import copy
import json
import warnings
from typing import Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, LeaveOneOut, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder

from ai4water.utils.utils import prepare_data, jsonize, to_datetime_index
from ai4water.datasets import all_datasets
from ai4water.preprocessing.transformations import Transformation
from ai4water.preprocessing.imputation import Imputation
import ai4water.datasets as datasets
from ai4water.utils.utils import print_something

cmap_cv = plt.cm.coolwarm

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

# todo
# detrend


class AttributeContainer(object):

    def __init__(self):

        self._from_h5 = False
        self.is_multi_source = False
        self.source_is_list = False
        self.source_is_dict = False


class DataHandler(AttributeContainer):
    """
    Using the data source provided by the user, this class divides the data into
    training, validation and test set. It handles all operations around data for
    data preparation. The core idea on which DataHandler is based is of `local`
    and `global` attributes of `data` sources. However, this local and global
    concept comes into play only when multiple data sources are used.

    Methods
    ------------
    - training_data: returns training data
    - validation_data: returns validation data
    - test_data: returns test data
    - from_disk:
    - KFold_splits: creates splits using `KFold` of sklearn
    - LeaveOneOut_splits: creates splits using `LeaveOneOut` of sklearn
    - TimeSeriesSplit_splits: creates splits using `TimeSeriesSplit` of sklearn

    """
    def __init__(
            self,
            data,
            input_features: Union[list, dict, str, None] = None,
            output_features: Union[list, dict, str, None] = None,
            dataset_args: dict = None,

            val_fraction: float = 0.2,
            test_fraction: float = 0.2,

            input_step: int = 1,
            lookback: int = 1,
            forecast_len: int = 1,
            forecast_step: int = 0,
            known_future_inputs: bool = False,
            allow_input_nans: bool = False,

            train_data: Union[str, list] = None,
            val_data: Union[str, list, np.ndarray, None] = None,
            intervals=None,
            transformation: Union[str, list, dict] = None,
            shuffle: bool = True,
            allow_nan_labels: int = 0,
            nan_filler: dict = None,
            batch_size: int = 32,
            drop_remainder: bool = False,
            teacher_forcing: bool = False,
            seed: int = 313,
            save: bool = False,
            verbosity: int = 1,
            mode=None,
            category=None,
    ):
        """

        Arguments:
            data :
                source from which to make the data. It can be one of the following:

                - pandas dataframe: each columns is a feature and each row is an example
                - xarray dataset: it can be xarray dataset or it
                - list of pandas dataframes :
                - dictionary of pandas dataframes :
                - path like: if the path is the path of a file, then this file can
                    be a csv/xlsx/nc/npz/mat/parquet/feather file. The .nc file
                    will be read using xarray to load datasets. If the path refers
                    to a directory, it is supposed that each file in the directory refers to one example.
                - ai4water dataset : any of dataset name from ai4water.datasets
            input_features Union[list, dict, str, None]:
                features to use as input. If `data` is pandas dataframe
                then this is list of column names from `data` to be used as input.
            output_features Union[list, dict, str, None]:
                features to use as output. When `data` is dataframe
                then it is list of column names from `data` to be used as output.
                If `data` is `dict`, then it must be consistent with `data`.
                Default is None,which means the last column of data will be
                used as output. In case of multi-class classification, the output
                column is not supposed to be one-hot-encoded rather in the form
                of [0,1,2,0,1,2,1,2,0] for 3 classes. One-hot-encoding is done
                inside the model.
            dataset_args dict:
                additional arguments for AI4Water's datasets
            val_fraction float:
                The fraction of the training data to be used for validation.
                Set to 0.0 if no validation data is to be used.
            test_fraction float:
                Fraction of the complete data to be used for test
                purpose. Must be greater than 0.0. This is also the hold-out data.
            input_step int:
                step size to keep in input data.
            lookback int:
                The number of lookback steps. The term lookback has been
                adopted from Francois Chollet's "deep learning with keras" book.
                It means how many historical time-steps of data, we want to feed
                to model at time-step to predict next value. This value must be
                one for any non timeseries forecasting related problems.
            forecast_len int:
                how many future values/horizons we want to predict.
            forecast_step int:
                how many steps ahead we want to predict. default is
                 0 which means nowcasting.
            known_future_inputs bool:
            allow_input_nans bool:
                don't know why it exists todo
            train_data Union[str, list]:
                Determines sampling strategy of training data. Possible
                values are

                    - `random`
                    - list of indices to be used
                    - `None` means the trainign data is chosen based upon val_fraction
                    and `test_fraction`. In this case, the first x fraction of data is
                is used for training where $x = 1 - (val_fraction + test_fraction)$.

            val_data Union[str, list, np.ndarray, None]:
                Data to be used for validation. If you want to use same data for
                 validation and test purpose, then set this argument to 'same'. This
                 can also be indices to be used for selecting validation data.
            intervals :
                tuple of tuples where each tuple consits of two integers, marking
                the start and end of interval. An interval here means indices
                from the data. Only rows within those indices will be used when preparing
                data/batches for NN. This is handy when our input data
                contains chunks of missing values or when we don't want to consider several
                rows in input data during data_preparation.
                For further usage see `examples/using_intervals`
            transformation Union[str, list, dict]:
                type of transformation to be applied.
                The transformation can be any transformation name from
                ai4water.utils.transformations.py. The user can specify more than
                one transformation. Moreover, the user can also determine which
                transformation to be applied on which input feature. Default is 'minmax'.
                To apply a single transformation on all the data
                ```python
                transformation = 'minmax'
                ```
                To apply different transformations on different input and output features
                ```python
                transformation = [{'method': 'minmax', 'features': ['input1', 'input2']},
                                {'method': 'zscore', 'features': ['input3', 'output']}
                                ]
                ```
                Here `input1`, `input2`, `input3` and `outptu` are the columns in the
                `data`.
            shuffle bool:
            allow_nan_labels bool:
                whether to allow examples with nan labels or not.
                if it is > 0, and if target values contain Nans, those examples
                will not be ignored and will be used as it is.
                In such a case a customized training and evaluation
                step is performed where the loss is not calculated for predictions
                corresponding to nan observations. Thus this option can be useful
                when we are predicting more than 1 target and some of the examples
                have some of their labels missing. In such a scenario, if we set this
                option to >0, we don't need to ignore those samples at all during data
                preparation. This option should be set to > 0 only when using tensorflow
                for deep learning models. if == 1, then if an example has label [nan, 1]
                it will not be removed while the example with label [nan, nan]
                will be ignored/removed. If ==2, both examples (mentioned before) will be
                considered/will not be removed. This means for multi-outputs, we can end
                up having examples whose all labels are nans. if the number of outputs
                are just one. Then this must be set to 2 in order to use samples with nan labels.
            nan_filler dict:
                Determines how to deal with missing values in the data.
                The default value is None, which will raise error if missing/nan values
                are encountered in the input data. The user can however specify a
                dictionary whose key must be either `fillna` or `interpolate` the value
                of this dictionary should be the keyword arguments will be forwarded
                to pandas .fillna() or .iterpolate() method. For example, to do
                forward filling, the user can do as following
                ```python
                >>>{'fillna': {'method': 'ffill'}}
                ```
                For details about fillna keyword options
                 [see](https://pandas.pydata.org/pandas-docs/version/0.22.0/generated/pandas.DataFrame.fillna.html)
                For `interpolate`, the user can specify  the type of interpolation
                for example
                ```python
                >>>{'interpolate': {'method': 'spline', 'order': 2}}
                ```
                will perform spline interpolation with 2nd order.
                For other possible options/keyword arguments for interpolate
                [see](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html)
                The filling or interpolation is done columnwise, however, the user
                can specify how to do for each column by providing the above mentioned
                arguments as dictionary or list. The sklearn based imputation methods
                can also be used in a similar fashion. For KNN
                ```python
                >>>{'KNNImputer': {'n_neighbors': 3}}
                ```
                or for iterative imputation
                ```python
                >>>{'IterativeImputer': {'n_nearest_features': 2}}
                ```
                To pass additional arguments one can make use of `imputer_args`
                keyword argument
                ```python
                >>>{'method': 'KNNImputer', 'features': ['b'], 'imputer_args': {'n_neighbors': 4}},
                ```
                For more on sklearn based imputation methods
                [see](https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html#sphx-glr-auto-examples-impute-plot-missing-values-py)
            batch_size int:
                size of one batch. Only relevent if `drop_remainder` is True.
            drop_remainder bool:
                whether to drop the remainder if len(data) % batch_size != 0 or not?
            teacher_forcing bool:
                whether to return previous output/target/ground
                truth or not. This is useful when the user wants to feed output
                at t-1 as input at timestep t. For details about this technique
                see [this article](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/)
            seed int:
                random seed for reproducibility
            save bool:
                whether to save the data in an h5 file or not.

        Note: If `indices` are given for `train_data` or `val_data` then these
            indices do not correspond to indices of original data but rather
            indices of 'available examples'. For example if lookback is 10, indices
            will shift backwards by 10, because we have to ignore first 9 rows.

        Example:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from ai4water.preprocessing import DataHandler
            >>> data = pd.DataFrame(np.random.randint(0, 1000, (50, 2)), columns=['input', 'output'])
            >>> data_handler = DataHandler(data=data, lookback=5)
            >>> x,y = data_handler.training_data()

        # Note
        The word 'index' is not allowed as column name, input_features or output_features
        """
        super().__init__()

        self.config = make_config(input_features=input_features,
                                  output_features=output_features,
                                  dataset_args=dataset_args or {},
                                  val_fraction=val_fraction,
                                  test_fraction=test_fraction,
                                  input_step=input_step,
                                  lookback=lookback,
                                  forecast_len=forecast_len,
                                  forecast_step=forecast_step,
                                  known_future_inputs=known_future_inputs,
                                  allow_input_nans=allow_input_nans,  # todo why this is even allowed
                                  train_data=train_data,
                                  val_data=val_data,
                                  intervals=intervals,
                                  transformation=transformation,
                                  shuffle=False,  # todo
                                  allow_nan_labels=allow_nan_labels,
                                  nan_filler=nan_filler,
                                  batch_size=batch_size,
                                  drop_remainder=drop_remainder,
                                  seed=seed,
                                  category=category,
                                  )
        self.data = self._process_source(data, input_features, output_features)
        self.verbosity = verbosity
        self.teacher_forcing = teacher_forcing
        self.mode = mode

        self.scalers = {}
        self.indexes = {}

        if save:
            self._to_disk()

    def __getattr__(self, item):
        if item in ['lookback', 'input_step', 'transformation', 'forecast_step',
                    'forecast_len',  # todo, can it be local?
                    'known_future_inputs', 'allow_nan_labels', 'allow_input_nans']:
            if self.source_is_df:
                return self.config[item]

            elif self.source_is_list:
                attr = self.config[item]

                if not isinstance(attr, list):
                    attr = [attr for _ in range(len(self.data))]

                assert len(attr) == len(self.data)
                return attr

            elif self.source_is_dict:
                attr = self.config[item]
                if not isinstance(attr, dict):
                    attr = {key: attr for key in self.data.keys()}
                assert len(attr) == len(self.data), f"There are {len(attr)} values for {item} while" \
                                                    f" {len(self.data)} values for data"
                return attr

            else:
                raise NotImplementedError(f"Unknown data type {self.data.__class__.__name__}")
        else:
            # Default behaviour
            raise AttributeError(f"DataLoader does not have an attribute {item}")

    @property
    def classes(self):
        _classes = []
        if self.mode == 'classification':
            if self.num_outs == 1:  # for binary/multiclass
                array = self.data[self.output_features].values
                _classes = np.unique(array[~np.isnan(array)])
            else:  # for one-hot encoded
                _classes = self.output_features

        return _classes

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def is_binary(self) -> bool:
        """Returns True if the porblem is binary classification"""
        _default = False
        if self.mode == 'classification':
            if self.num_outs == 1:
                array = self.data[self.output_features].values
                unique_vals = np.unique(array[~np.isnan(array)])
                if len(unique_vals) == 2:
                    _default = True
            else:
                pass  # todo, check when output columns are one-hot encoded

        return _default

    @property
    def is_multiclass(self) -> bool:
        """Returns True if the porblem is multiclass classification"""
        _default = False
        if self.mode == 'classification':
            if self.num_outs == 1:
                array = self.data[self.output_features].values
                unique_vals = np.unique(array[~np.isnan(array)])
                if len(unique_vals) > 2:
                    _default = True
            else:
                pass  # todo, check when output columns are one-hot encoded

        return _default

    @property
    def is_multilabel(self) -> bool:
        """Returns True if the porblem is multilabel classification"""
        _default = False
        if self.mode == 'classification':
            if self.num_outs > 1:
                _default = True

        return _default

    @property
    def _to_categorical(self):
        # whether we have to convert y into one-hot encoded form
        _defualt = False

        if self.is_binary or self.is_multiclass:
            if self.num_outs == 1:
                _defualt = True
        if self.is_binary and self.category == 'ML':  # todo, don't do for ML algorithms
            _defualt = False

        return _defualt

    @property
    def teacher_forcing(self):
        return self._teacher_forcing

    @teacher_forcing.setter
    def teacher_forcing(self, x):
        self._teacher_forcing = x

    @property
    def category(self):
        return self.config['category']

    @property
    def batch_dim(self):

        if self.source_is_df:
            batch_dim = batch_dim_from_lookback(self.lookback)
        elif self.source_is_list:
            if isinstance(self.lookback, int):
                batch_dim = [batch_dim_from_lookback(self.lookback) for _ in range(len(self.data))]
            elif isinstance(self.lookback, list):
                batch_dim = [batch_dim_from_lookback(lb) for lb in self.lookback]
            else:
                raise NotImplementedError
        elif self.source_is_dict:
            if isinstance(self.lookback, int):
                batch_dim = {k: batch_dim_from_lookback(self.lookback) for k, v in zip(self.data.keys(),
                                                                                       range(len(self.data)))}
            elif isinstance(self.lookback, dict):
                batch_dim = {k: batch_dim_from_lookback(lb) for k, lb in self.lookback.items()}
            else:
                raise NotImplementedError(f"incompatible lookback {self.lookback} with data "
                                          f"definition {self.data.__class__.__name__}.")
        else:
            raise NotImplementedError

        return batch_dim

    @property
    def is_multiinput(self):
        if len(self.num_outs) > 1:
            return True
        return False

    @property
    def input_features(self):
        _inputs = self.config['input_features']

        if isinstance(self.data, list):
            assert isinstance(_inputs, list)
        elif isinstance(self.data, dict):
            assert isinstance(_inputs, dict), f'input_features are of type {_inputs.__class__.__name__}'
        elif _inputs is None and self.data is not None:
            assert isinstance(self.data, pd.DataFrame)
            _inputs = self.data.columns[0:-1].to_list()

        return _inputs

    @property
    def output_features(self):
        _outputs = self.config['output_features']

        if isinstance(self.data, list):
            assert isinstance(_outputs, list)

        elif isinstance(self.data, dict):
            assert isinstance(_outputs, dict), f'data is of type dict while output_features are ' \
                                               f'of type {_outputs.__class__.__name__}'
            for k in self.data.keys():
                if k not in _outputs:
                    _outputs[k] = []

        elif _outputs is None and self.data is not None:
            assert isinstance(self.data, pd.DataFrame)
            _outputs = [col for col in self.data.columns if col not in self.input_features]

        return _outputs

    @property
    def is_multioutput(self):
        if isinstance(self.data, list) or isinstance(self.data, dict):
            return True
        return False

    @property
    def input_sources(self):
        return

    @property
    def output_sources(self):
        return

    @property
    def equal_io_sources(self):

        return

    @property
    def any_3d_source(self):
        return

    @property
    def all_2d_sources(self):
        return

    @property
    def all_3d_sources(self):
        return

    @property
    def source_is_df(self):
        if isinstance(self.data, pd.DataFrame):
            return True
        return False

    def len(self):
        """Returns number of examples available where each example refers to an
        input-output pair."""

        if isinstance(self.data, pd.DataFrame):
            # the total number of examples will be less than
            _len = len(self.data) - (self.lookback - 1)

        elif isinstance(self.data, list):
            _len = 0
            for s in self.data:
                _len += len(s)
        elif isinstance(self.data, dict):
            _len = 0
            for k, v in self.data.items():
                _len += len(v)
        else:
            raise NotImplementedError
        return _len

    def _process_source(self, data, input_features, output_features):

        if isinstance(data, str):
            _source = self._get_source_from_str(data, input_features, output_features)
            if isinstance(_source, str) and _source.endswith('.h5'):
                self._from_h5 = True
            self.num_sources = 1

        elif isinstance(data, pd.DataFrame):
            if input_features is None and output_features is not None:
                if isinstance(output_features, str):
                    output_features = [output_features]
                assert isinstance(output_features, list)
                input_features = [col for col in data.columns if col not in output_features]
                # since we have inferred the input_features, they should be put back into config
                self.config['input_features'] = input_features
            _source = data
            self.is_multi_source = False
            self.num_sources = 1

        elif isinstance(data, np.ndarray):
            assert data.ndim == 2, f"input data as numpy array must be 2 dimension, found {data.ndim} dimensions"
            # if output_features is not defined, consider 1 output and name it as 'output'
            if output_features is None:
                output_features = ['outout']
                self.config['output_features'] = output_features  # we should put it in config as well
            elif isinstance(output_features, str):
                output_features = [output_features]
            else:
                assert isinstance(output_features, list)

            if input_features is None:  # define dummy names for input_features
                input_features = [f'input_{i}' for i in range(data.shape[1]-len(output_features))]
                self.config['input_features'] = input_features
            _source = pd.DataFrame(data, columns=input_features + output_features)

        elif data.__class__.__name__ == "Dataset":
            _source = data
            self.num_sources = 1

        elif isinstance(data, list):
            _source = []
            for s in data:
                _source.append(self._process_one_source(s))
            self.is_multi_source = True
            self.source_is_list = True
            self.num_sources = len(data)

        elif isinstance(data, dict):
            _source = {}
            for s_name, s_val in data.items():
                _source[s_name] = self._process_one_source(s_val)
            self.is_multi_source = True
            self.source_is_dict = True
            self.num_sources = len(data)

        elif data is None:
            return data

        else:
            assert data is not None
            raise ValueError(f"unregnizable source of data of type {data.__class__.__name__} given")

        _source = self.impute(_source)

        return _source

    def add_noice(self):
        return

    @property
    def val_data(self):
        if self.source_is_df or self.source_is_list or self.source_is_dict:
            return self.config['val_data']
        else:
            raise NotImplementedError

    def impute(self, data):
        """Imputes the missing values in the data using `Imputation` module"""
        if self.config['nan_filler'] is not None:

            if isinstance(data, pd.DataFrame):

                _source = self._impute_one_source(data, self.config['nan_filler'])

            else:
                raise NotImplementedError
        else:
            _source = data

        return _source

    def _impute_one_source(self, data, impute_config):

        if isinstance(impute_config, str):
            method, impute_args = impute_config, {}
            data = Imputation(data, method=method, **impute_args)()

        elif isinstance(impute_config, dict):
            data = Imputation(data, **impute_config)()

        elif isinstance(impute_config, list):
            for imp_conf in impute_config:
                data = Imputation(data, **imp_conf)()

        else:
            raise NotImplementedError(f'{impute_config.__class__.__name__}')

        return data

    def tot_obs_for_one_df(self):
        if self.source_is_df:
            tot_obs = tot_obs_for_one_df(self.data, self.allow_nan_labels, self.output_features, self.lookback,
                                         self.input_step,
                                         self.num_outs,
                                         self.forecast_step,
                                         self.forecast_len,
                                         self.config['intervals'],
                                         self.input_features,
                                         )
        elif self.source_is_list:
            tot_obs = []
            for idx in range(len(self.data)):
                _tot_obs = tot_obs_for_one_df(self.data[idx],
                                              self.allow_nan_labels[idx],
                                              self.output_features[idx], self.lookback[idx],
                                              self.input_step[idx], self.num_outs[idx],
                                              self.forecast_step[idx],
                                              self.forecast_len[idx],
                                              self.config['intervals'],
                                              self.input_features[idx]
                                              )
                tot_obs.append(_tot_obs)

        elif self.source_is_dict:
            tot_obs = {}
            for src_name in self.data.keys():
                _tot_obs = tot_obs_for_one_df(self.data[src_name],
                                              self.allow_nan_labels[src_name],
                                              self.output_features[src_name],
                                              self.lookback[src_name],
                                              self.input_step[src_name],
                                              self.num_outs[src_name],
                                              self.forecast_step[src_name],
                                              self.forecast_len[src_name],
                                              self.config['intervals'],
                                              self.input_features[src_name]
                                              )
                tot_obs[src_name] = _tot_obs
        else:
            raise NotImplementedError

        return tot_obs

    def get_indices(self):
        """If the data is to be divded into train/test based upon indices,
        here we create train_indices and test_indices.
        """
        if self.source_is_df or self.source_is_list or self.source_is_dict:
            indices = self.config['train_data']
            if isinstance(indices, str):

                # if isinstance(indices, str):
                assert indices == 'random'

                tot_obs = self.tot_obs_for_one_df()

                # tot_obs can be dictionary because lookback is local
                # but tot_obs must be same for all sources
                if isinstance(tot_obs, dict):
                    tot_obs = np.unique(list(tot_obs.values())).item()

                total_indices = np.arange(tot_obs)
                if self.config['test_fraction'] > 0.0:
                    train_indices, test_indices = train_test_split(total_indices,
                                                                   test_size=self.config['test_fraction'],
                                                                   random_state=self.config['seed'])
                else:  # all the indices belong to training
                    train_indices, test_indices = total_indices, None

            elif indices is None:
                train_indices, test_indices = None, None
            else:
                assert isinstance(np.array(indices), np.ndarray)
                tot_obs = self.tot_obs_for_one_df()
                if isinstance(tot_obs, dict):
                    tot_obs = np.unique(list(tot_obs.values())).item()
                tot_indices = np.arange(tot_obs)
                train_indices = np.array(indices)
                test_indices = np.delete(tot_indices, train_indices)

        else:
            raise NotImplementedError(f'Indices can not be found for source of type {self.data.__class__.__name__}')

        setattr(self, 'train_indices', train_indices)
        setattr(self, 'test_indices', test_indices)

        return train_indices, test_indices

    def get_train_args(self):
        """train_data key in config can consist of following keys st, en, indices"""
        indices = self.config['train_data']

        if indices is not None:
            if isinstance(indices, str):
                assert indices == 'random', f'invalid value of indices {indices}'
            else:
                assert isinstance(np.array(indices), np.ndarray), f'invalid value of indices {indices}'

        train_indices, _ = self.get_indices()

        return train_indices

    @property
    def num_ins(self):
        if self.source_is_df or self.data is None:
            return len(self.input_features)
        elif self.source_is_list or self.data is None:
            return [len(in_feat) for in_feat in self.input_features]
        elif self.source_is_dict or self.data is None:
            return {k: len(in_feat) for k, in_feat in self.input_features.items()}
        else:
            raise NotImplementedError

    @property
    def num_outs(self):
        if self.source_is_df:
            return len(self.output_features)
        elif self.source_is_list:
            return [len(out_feat) for out_feat in self.output_features]
        elif self.source_is_dict:
            return {k: len(out_feat) for k, out_feat in self.output_features.items()}
        elif self.data.__class__.__name__ == "NoneType":
            return None
        else:
            raise NotImplementedError(f"Can not determine output features for data "
                                      f"of type {self.data.__class__.__name__}")

    def KFold_splits(self, n_splits=5):
        """returns an iterator for kfold cross validation.

        The iterator yields two tuples of training and test x,y pairs.
        The iterator on every iteration returns following
        `(train_x, train_y), (test_x, test_y)`
        Note: only `training_data` and `validation_data` are used to make kfolds.

        Example:
            >>> data = pd.DataFrame(np.random.randint(0, 10, (20, 3)), columns=['a', 'b', 'c'])
            >>> data_handler = DataHandler(data=data, config={'lookback': 1})
            >>> kfold_splits = data_handler.KFold_splits()
            >>> for (train_x, train_y), (test_x, test_y) in kfold_splits:
            ...     print(train_x, train_y, test_x, test_y)

        """
        x, y = self._get_xy()

        spliter = self._get_kfold_splitter(x=x, n_splits=n_splits)

        for tr_idx, test_idx in spliter:

            yield (x[tr_idx], y[tr_idx]), (x[test_idx], y[test_idx])

    def LeaveOneOut_splits(self):
        """Yields leave one out splits
        The iterator on every iteration returns following
        `(train_x, train_y), (test_x, test_y)`"""
        x, y = self._get_xy()

        kf = LeaveOneOut()

        for tr_idx, test_idx in kf.split(x):

            yield (x[tr_idx], y[tr_idx]), (x[test_idx], y[test_idx])

    def TimeSeriesSplit_splits(self, n_splits=5, **kwargs):
        """returns an iterator for TimeSeriesSplit.
        The iterator on every iteration returns following
        `(train_x, train_y), (test_x, test_y)`
        """
        x, y = self._get_xy()

        tscv = TimeSeriesSplit(n_splits=n_splits, **kwargs)

        for tr_idx, test_idx in tscv.split(x):

            yield (x[tr_idx], y[tr_idx]), (x[test_idx], y[test_idx])

    def _get_kfold_splitter(self, x, n_splits=5):

        kf = KFold(n_splits=n_splits, random_state=self.config['seed'] if self.config['shuffle'] else None,
                   shuffle=self.config['shuffle'])
        spliter = kf.split(x)

        return spliter

    def plot_KFold_splits(self, n_splits=5, show=True, **kwargs):
        """Plots the indices of kfold splits"""
        x, y = self._get_xy()

        spliter = self._get_kfold_splitter(x=x, n_splits=n_splits)

        self._plot_splits(spliter, x, title="KFoldCV", show=show, **kwargs)

        return

    def plot_LeaveOneOut_splits(self, show=True, **kwargs):
        """Plots the indices obtained from LeaveOneOut strategy"""
        x, y = self._get_xy()

        spliter = LeaveOneOut().split(x)

        self._plot_splits(spliter=spliter, x=x, title="LeaveOneOutCV", show=show, **kwargs)

        return

    def plot_TimeSeriesSplit_splits(self, n_splits=5, show=True, **kwargs):
        """Plots the indices obtained from TimeSeriesSplit strategy"""

        x, y = self._get_xy()

        spliter = TimeSeriesSplit(n_splits=n_splits, **kwargs).split(x)

        self._plot_splits(spliter=spliter, x=x, title="TimeSeriesCV", show=show, **kwargs)

        return

    def _plot_splits(self, spliter, x, show=True,  **kwargs):

        splits = list(spliter)

        figsize = kwargs.get('figsize', (10, 8))
        legend_fs = kwargs.get('legend_fs', 20)
        legend_pos = kwargs.get('legend_pos', (1.02, 0.8))
        title = kwargs.get("title", "CV")

        plt.close('all')
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        for ii, split in enumerate(splits):
            indices = np.array([np.nan] * len(x))
            indices[split[0]] = 1
            indices[split[1]] = 0

            ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                       c=indices, marker='_', lw=10, cmap="coolwarm",
                       vmin=-.2, vmax=1.2)

        yticklabels = list(range(len(splits)))

        ax.set(yticks=np.arange(len(splits)) + .5, yticklabels=yticklabels, xlabel='Sample index',
               ylabel="CV iteration")
        ax.set_title(title, fontsize=20)

        ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))], ['Training set', 'Test set'],
                  loc=legend_pos, fontsize=legend_fs)

        if show:
            plt.show()
        return

    def _get_xy(self):
        if self.teacher_forcing:
            tr_x, _, tr_y = self.training_data()
            val_x, _, val_y = self.validation_data()
        else:
            tr_x, tr_y = self.training_data()
            val_x, val_y = self.validation_data()

        def check_if_none(X, Y):

            if X is None:
                shape = list(tr_x.shape)
                shape[0] = 0
                X = np.zeros(tuple(shape))
            if Y is None:
                shape = list(tr_y.shape)
                shape[0] = 0
                Y = np.zeros(tuple(shape))

            return X, Y

        if self.source_is_df:
            val_x, val_y = check_if_none(val_x, val_y)

            x = np.concatenate([tr_x, val_x])
            y = np.concatenate([tr_y, val_y])

        else:
            raise NotImplementedError
        return x, y

    def make_val_frac_zero(self):

        # It is good that the user knows  explicitly that either of val_fraction or test_fraction is used so
        # one of them must be set to 0.0
        if self.config['val_fraction'] > 0.0:
            warnings.warn(f"Setting val_fraction from {self.config['val_fraction']} to 0.0")
            self.config['val_fraction'] = 0.0
        return

    def _indexify_y(self, src: pd.DataFrame, out_features: list):

        # this function is only called when data is a list/dictionary

        src = src.copy()  # make a copy so that df in original data is not altered

        # since there are multiple sources/dfs, we need to keep track that y's in each
        # data are same i.e. y[10] of data[0] is same as y[10] of data[1]
        # but this should only be done if nans in y are ignored
        src['dummy_id'] = np.arange(len(src), dtype=np.int32)

        out_features = out_features + ['dummy_id']  # original should not be changed.
        # todo, all sources should have same length

        return src, out_features

    def _make_data_for_one_src(self,
                               key,
                               indices,
                               shuffle,
                               identifier=None,
                               deindexify=False
                               ):
        """Makes the data for each source."""

        data = self.data if identifier is None else self.data[identifier]
        output_features = self.output_features if identifier is None else self.output_features[identifier]
        if self.source_is_list and all([flag == 0 for flag in self.allow_nan_labels]):
            data, output_features = self._indexify_y(data, output_features)

        data_maker = MakeData(
            input_features=self.input_features if identifier is None else self.input_features[identifier],
            output_features=output_features,
            lookback=self.lookback if identifier is None else self.lookback[identifier],
            input_step=self.input_step if identifier is None else self.input_step[identifier],
            forecast_step=self.forecast_step if identifier is None else self.forecast_step[identifier],
            forecast_len=self.forecast_len if identifier is None else self.forecast_len[identifier],
            known_future_inputs=self.known_future_inputs if identifier is None else self.known_future_inputs[identifier],
            batch_dim=self.batch_dim if identifier is None else self.batch_dim[identifier],
            allow_input_nans=self.allow_input_nans if identifier is None else self.allow_input_nans[identifier],
            allow_nan_labels=self.allow_nan_labels if identifier is None else self.allow_nan_labels[identifier],
            verbosity=self.verbosity
                              )

        data, scalers = data_maker.transform(
            data=data,
            transformation=self.transformation if identifier is None else self.transformation[identifier],
            key=key
        )
        self.scalers.update(scalers)

        # numpy arrays are not indexed and is supposed that the whole array is use as input
        if not isinstance(data, np.ndarray):
            data = data_maker.indexify(data, key)

        x, prev_y, y = data_maker(
            data,
            shuffle=shuffle,
            intervals=self.config['intervals'],
            indices=indices
        )

        if x is not None and deindexify and not isinstance(data, np.ndarray):
            # if x.shape[0] >0:
            x, self.indexes[key] = data_maker.deindexify(x, key)

        # if 'dummy_id' in data_maker.output_features:
        # x, prev_y, y = deindexify_y(x, prev_y, y, np.argmax(self.lookback).item())

        return x, prev_y, y, data_maker

    def _post_process_train_args(self, x, prev_y, y):

        if self.val_data == 'same':
            self.make_val_frac_zero()

        if self.config['val_fraction'] + self.config['test_fraction'] > 0.0:
            if self.train_indices is None:
                train_frac = 1.0 -  self.config['test_fraction']
                train_idx_en = int(round(train_frac * len(x)))
                x = x[0: train_idx_en, ...]
                prev_y = prev_y[0: train_idx_en, ...]
                y = y[0: train_idx_en, ...]

            if self.config['val_fraction'] > 0.0:
                train_frac = 1.0 - self.config['val_fraction']
                train_idx_en = int(round(train_frac * len(x)))

                x = x[0: train_idx_en, ...]
                prev_y = prev_y[0: train_idx_en, ...]
                y = y[0: train_idx_en, ...]

        return x, prev_y, y

    def training_data(self, key: str = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Renders the training data.
        Returns:
            a tuple of nd arrays containing x and y data. If teacher forcing is True, then the tuple
            consists of three values i.e. x, prev_y and y.
        """
        if self._from_h5:

            return load_data_from_hdf5('training_data', self.data)

        train_indices = self.get_train_args()

        if self.source_is_df:

            x, prev_y, y, data_maker = self._make_data_for_one_src(
                key,
                train_indices,
                False,
                deindexify=False
            )

            x, prev_y, y = self._post_process_train_args(x, prev_y, y)

            x, prev_y, y = self.check_for_batch_size(x, prev_y, y)

            if not isinstance(self.data, np.ndarray):
                x, self.indexes[key] = data_maker.deindexify(x, key)

        elif self.is_multi_source:

            if self.source_is_list:
                x, prev_y, y = [], [], []
                for idx, src in enumerate(self.data):

                    _key = f'{key}_{idx}'
                    _x, _prev_y, _y, data_maker = self._make_data_for_one_src(
                        f'{key}_{idx}',
                        train_indices,
                        False,
                        identifier=idx,
                        deindexify=False,
                    )

                    _x, _prev_y, _y = self._post_process_train_args(_x, _prev_y, _y)

                    _x, _prev_y, _y = self.check_for_batch_size(_x, _prev_y, _y)

                    if not isinstance(self.data[idx], np.ndarray):  # todo, one source is indexified and other is not?
                        _x, self.indexes[_key] = data_maker.deindexify(_x, f'{key}_{idx}')

                    x.append(_x)
                    prev_y.append(_prev_y)
                    y.append(_y)

                if 'dummy_id' in data_maker.output_features:
                    x, prev_y, y = deindexify_y(x, prev_y, y, np.argmax(self.lookback).item())

            elif self.source_is_dict:

                x, prev_y, y = {}, {},{}

                for src_name, src in self.data.items():

                    _key = f'{key}_{src_name}'

                    _x, _prev_y, _y, data_maker = self._make_data_for_one_src(
                        f'{key}_{src_name}',
                        train_indices,
                        False,
                        identifier=src_name,
                        deindexify=False
                    )

                    _x, _prev_y, _y = self._post_process_train_args(
                        _x,
                        _prev_y,
                        _y
                    )

                    _x, _prev_y, _y = self.check_for_batch_size(_x, _prev_y, _y)

                    # todo, one source may be indexified and other is not?
                    if not isinstance(self.data[src_name], np.ndarray):
                        _x, self.indexes[_key] = data_maker.deindexify(_x, f'{key}_{src_name}')

                    x[src_name], prev_y[src_name], y[src_name] = _x, _prev_y, _y

                    # if 'dummy_id' in data_maker.output_features:
                    #    x, prev_y, y = deindexify_y(x, prev_y, y, max(self.lookback, key=self.lookback.get))
            else:
                raise NotImplementedError

        elif isinstance(self.data, dict) or isinstance(self.data, list):
            raise NotImplementedError
        else:
            raise NotImplementedError

        prev_y = filter_zero_sized_arrays(prev_y)
        y = filter_zero_sized_arrays(y)
        if self.mode == 'classification':
            y = check_for_classification(y, self._to_categorical)

        if self.teacher_forcing:
            return return_x_yy(x, prev_y, y, "Training", self.verbosity)

        return return_xy(x, y, "Training", self.verbosity)

    def _make_val_data_from_src(self,
                                indices,
                                key,
                                tot_obs,
                                identifier=None
                                ):

        x, prev_y, y = None, None, None
        run = True
        split1 = False
        split2 = False
        if self.val_data == "same":
            self.make_val_frac_zero()
            assert self.config['test_fraction'] > 0.0, f"test_fraction should be > 0.0. " \
                                                       f"It is {self.config['test_fraction']}"
            if indices is None:
                split1 = True
        elif np.array(self.val_data).shape.__len__() > 0:
            indices = np.array(self.val_data)
        elif self.config['val_fraction'] == 0.0:
            run = False
        else:
            if indices is None:
                indices = self.get_train_args()
                split2 = True

        if run:
            x, prev_y, y, data_maker = self._make_data_for_one_src(
                key,
                indices,
                False,
                identifier=identifier
            )

        if split1:

            # if indices is None:
            # we have to remove training data from it which is the first %x percent
            train_frac = 1.0 - self.config['test_fraction']
            test_frac = self.config['test_fraction']
            train_idx_en = int(round(train_frac * len(x)))
            test_idx_st = train_idx_en + int(round(train_frac + test_frac * len(x)))

            x = x[train_idx_en: test_idx_st, ...]
            prev_y = prev_y[train_idx_en: test_idx_st, ...]
            y = y[train_idx_en: test_idx_st, ...]

        elif indices is None and self.config['val_fraction'] == 0.0:  # no validation data
            x, prev_y, y = None, None, None

        elif split2:

            if indices is None:
                train_frac = 1.0 - self.config['test_fraction']
                train_idx_en = int(round(train_frac * len(x)))
                x = x[0: train_idx_en, ...]
                prev_y = prev_y[0: train_idx_en, ...]
                y = y[0: train_idx_en, ...]
            else:
                train_idx_en = len(x)

            if self.config['val_fraction'] > 0.0:
                train_frac = 1.0 - self.config['val_fraction']
                train_idx_st = int(round(train_frac * len(x)))

                x = x[train_idx_st:train_idx_en, ...]
                prev_y = prev_y[train_idx_st:train_idx_en, ...]
                y = y[train_idx_st:train_idx_en, ...]

        if x is not None:
            if x.shape[0] == 0:  # instead of returning arrays with 0 in first dimension, return None
                x, prev_y, y = None, None, None
            else:  # np.ndarray is not indexified
                if identifier and isinstance(self.data[identifier], np.ndarray):
                    pass
                else:
                    x, prev_y, y = self.check_for_batch_size(x, prev_y, y)
                    x, self.indexes[key] = data_maker.deindexify(x, key)

        return x, prev_y, y

    def validation_data(self, key='val', **kwargs
                        ) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
        """Returns the validation data"""
        if self._from_h5:

            return load_data_from_hdf5('validation_data', self.data)

        if getattr(self, 'val_dataset', None).__class__.__name__ in ['BatchDataset', 'TorchDataset']:
            return self.val_dataset

        test_indices = None
        if self.config['val_data'] == 'same':
            _, test_indices = self.get_indices()

        if self.source_is_df:

            x, prev_y, y = self._make_val_data_from_src(
                test_indices,
                key,
                self.tot_obs_for_one_df()
            )
        elif self.source_is_list:
            x, prev_y, y = [], [], []
            for idx, src in enumerate(self.data):

                output_features = self.output_features[idx]
                if all([flag == 0 for flag in self.allow_nan_labels]):
                    src, output_features = self._indexify_y(src, output_features)

                _x, _prev_y, _y = self._make_val_data_from_src(
                    test_indices,
                    f'{key}_{idx}',
                    self.tot_obs_for_one_df()[idx],
                    identifier=idx
                )
                x.append(_x)
                prev_y.append(_prev_y)
                if _y.size > 0:
                    y.append(_y)

            if 'dummy_id' in output_features:  # todo why here as well
                x, prev_y, y = deindexify_y(x, prev_y, y, np.argmax(self.lookback).item())

        elif self.source_is_dict:

            x, prev_y, y = {}, {}, {}

            for src_name, src in self.data.items():

                _x, _prev_y, _y = self._make_val_data_from_src(
                    test_indices,
                    f'{key}_{src_name}',
                    self.tot_obs_for_one_df()[src_name],
                    identifier=src_name
                )
                x[src_name] = _x
                prev_y[src_name] = _prev_y
                y[src_name] = _y

        elif self.data.__class__.__name__ == "NoneType":
            return None, None
        else:
            raise NotImplementedError(f"Can not calculate validation data for data "
                                      f"of type {self.data.__class__.__name__}")

        prev_y = filter_zero_sized_arrays(prev_y)
        y = filter_zero_sized_arrays(y)

        if self.mode == 'classification':
            y = check_for_classification(y, self._to_categorical)

        if self.teacher_forcing:
            return return_x_yy(x, prev_y, y, "Validation", self.verbosity)

        return return_xy(x, y, "Validation", self.verbosity)

    def test_data(self, key='test', data_keys=None, **kwargs
                  ) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
        """Returns the test_data"""
        # user may have defined its own data by overwriting training_data/validation_data
        # and `val_data` is same as test data, thus avoid the situation if the user
        # has not overwritten test_data method.

        if self._from_h5:

            return load_data_from_hdf5('test_data', self.data)

        if self.config['val_data'] == "same" and self.data is None:
            return self.validation_data(key=key, data_keys=data_keys, **kwargs)

        if self.data.__class__.__name__ == "NoneType":
            return None, None

        _, test_indices = self.get_indices()

        if self.val_data == "same":
            return self.validation_data(key=key)

        if self.source_is_df:
            x, prev_y, y = self.test_data_from_one_src(
                key,
                test_indices,
                self.tot_obs_for_one_df()
            )

        elif self.source_is_list:
            x, prev_y, y = [], [], []

            for idx, src in enumerate(self.data):

                output_features = self.output_features[idx]
                if all([flag == 0 for flag in self.allow_nan_labels]):
                    src, output_features = self._indexify_y(src, output_features)

                _x, _prev_y, _y = self.test_data_from_one_src(
                    f'{key}_{idx}',
                    test_indices,
                    self.tot_obs_for_one_df()[idx],
                    identifier=idx
                )
                x.append(_x)
                prev_y.append(_prev_y)

                if _y.size > 0:
                    y.append(_y)

            if 'dummy_id' in output_features:
                x, prev_y, y = deindexify_y(x, prev_y, y, np.argmax(self.lookback).item())

        elif self.source_is_dict:

            x, prev_y, y = {}, {}, {}

            for src_name, src in self.data.items():

                x[src_name], prev_y[src_name], y[src_name] = self.test_data_from_one_src(
                    f'{key}_{src_name}',
                    test_indices,
                    self.tot_obs_for_one_df()[src_name],
                    identifier=src_name,
                )

        else:
            raise NotImplementedError

        prev_y = filter_zero_sized_arrays(prev_y)
        y = filter_zero_sized_arrays(y)
        if self.mode == 'classification':
            y = check_for_classification(y, self._to_categorical)

        if self.teacher_forcing:
            return return_x_yy(x, prev_y, y, "Test", self.verbosity)

        return return_xy(x, y, "Test", self.verbosity)

    def test_data_from_one_src(self,
                               key,
                               test_indices,
                               tot_obs,
                               identifier=None
                               ):

        x, prev_y, y, data_maker = self._make_data_for_one_src(
            key,
            test_indices,
            False,
            identifier=identifier
        )

        if test_indices is None:
            # we need to divide the data into train/val/test based upon given fractions.
            train_frac = 1.0 - self.config['test_fraction']
            # val_frac = self.config['val_fraction']
            train_idx_en = int(round(train_frac * len(x)))
            # val_idx = train_idx + int(round(train_frac + val_frac * tot_obs))

            x = x[train_idx_en:, ...]
            prev_y = prev_y[train_idx_en:, ...]
            y = y[train_idx_en:, ...]

        if x is not None:
            if x.shape[0] == 0:
                x, prev_y, y = None, None, None
            else:
                if identifier and isinstance(self.data[identifier], np.ndarray):
                    pass
                else:
                    x, prev_y, y = self.check_for_batch_size(x, prev_y, y)
                    x, self.indexes[key] = data_maker.deindexify(x, key)

        return x, prev_y, y

    def deindexify(self, data, key):

        if self.source_is_df:
            if key not in self.indexes:
                raise ValueError(f"key `{key}` not found. Available keys are {list(self.indexes.keys())}")
            index = self.indexes[key]

        elif self.source_is_list:
            index = None
            for idx, src in enumerate(data):
                _key = f'{key}_{idx}'

                if _key not in self.indexes:
                    raise ValueError(f"key `{_key}` not found. Available keys are {list(self.indexes.keys())}")

                index = self.indexes[_key]

        elif self.source_is_dict:
            index = None
            for src_name, src in data.items():
                _key = f'{key}_{src_name}'

                if _key not in self.indexes:
                    raise ValueError(f"key `{_key}` not found. Available keys are {list(self.indexes.keys())}")

                index = self.indexes[_key]
        else:
            raise ValueError

        return data, index

    def transform(self):
        return

    def inverse_transform(self, data, key):

        transformation = self.transformation
        if self.source_is_df:

            data = self._inv_transform_one_src(data, key, transformation)

        elif self.source_is_list:
            assert isinstance(data, list)
            _data = []
            for idx, src in enumerate(data):
                __data = self._inv_transform_one_src(src, f'{key}_{idx}', transformation[idx])
                _data.append(__data)
            data = _data

        elif self.source_is_dict:
            assert isinstance(data, dict)
            _data = {}
            for src_name, src in data.items():
                _data[src_name] = self._inv_transform_one_src(src, f'{key}_{src_name}', transformation[src_name])
            data = _data

        else:
            raise NotImplementedError

        return data

    def _inv_transform_one_src(self, data, key, transformation):

        if transformation is not None:
            if isinstance(transformation, str):

                if key not in self.scalers:
                    raise ValueError(f"""
                    key `{key}` for inverse transformation not found. Available keys are {list(self.scalers.keys())}""")

                scaler = self.scalers[key]
                scaler, shape, _key = scaler['scaler'], scaler['shape'], scaler['key']
                original_shape = data.shape

                data, dummy_features = conform_shape(data, shape)  # get data to transform
                transformed_data = scaler.inverse_transform(data)
                data = transformed_data[:, dummy_features:]  # remove the dummy data
                data = data.reshape(original_shape)

            elif isinstance(transformation, list):
                assert data.__class__.__name__ in ['DataFrame', 'Series']
                for idx, trans in reversed(list(enumerate(transformation))):  # idx and trans both in reverse form
                    if trans['method'] is not None:
                        features = trans['features']
                        # if any of the feature in data was transformed
                        if any([True if f in data else False for f in features]):
                            orig_cols = data.columns  # copy teh columns in the original df
                            scaler = self.scalers[f'{key}_{trans["method"]}_{idx}']
                            scaler, shape, _key = scaler['scaler'], scaler['shape'], scaler['key']
                            data, dummy_features = conform_shape(data, shape, features)  # get data to transform

                            transformed_data = Transformation(**trans)(data, what='inverse', scaler=scaler)
                            data = transformed_data[orig_cols]  # remove the dummy data

            elif isinstance(transformation, dict):
                assert data.__class__.__name__ in ['DataFrame', 'Series'], f'data is of type {data.__class__.__name__}'
                if any([True if f in data else False for f in transformation['features']]):
                    orig_cols = data.columns
                    scaler = self.scalers[key]
                    scaler, shape, _key = scaler['scaler'], scaler['shape'], scaler['key']
                    data, dummy_features = conform_shape(data, shape, features=transformation['features'])
                    transformed_data = Transformation(**transformation)(data, what='inverse', scaler=scaler)
                    data = transformed_data[orig_cols]  # remove the dummy data

        return data

    def check_nans(self):
        return

    @classmethod
    def from_h5(cls, path):
        """Creates an instance of DataLoader from .h5 class."""
        f = h5py.File(path, mode='r')

        config = {}
        for k, v in f.attrs.items():
            if isinstance(v, bytes):
                v = decode(v)
            config[k] = v

        cls._from_h5 = True
        f.close()
        return cls(path, **config)

    def _to_disk(self, path=None):
        path = path or os.path.join(os.getcwd(), "results")
        filepath = os.path.join(path, "data.h5")

        f = h5py.File(filepath, mode='w')

        for k,v in self.config.items():
            if isinstance(v, (dict, list, tuple)):
                f.attrs[k] = json.dumps(
                    v, default=jsonize).encode('utf8')

            elif v is not None:
                f.attrs[k] = v

        x, prev_y, y = self.training_data()
        # save in disk
        self._save_data_to_hdf5('training_data', x, prev_y, y, f)

        x, prev_y, y = self.validation_data()
        self._save_data_to_hdf5('validation_data', x, prev_y, y, f)

        x, prev_y, y = self.validation_data()
        self._save_data_to_hdf5('test_data', x, prev_y, y, f)

        f.close()
        return

    def _get_source_from_str(self, data, input_features, output_features):
        if isinstance(output_features, str):
            output_features = [output_features]

        # dir path/file path/ ai4water dataset name
        if data.endswith('.h5'):
            _source = data
        if data.endswith('.csv'):
            _source = pd.read_csv(data)
            if _source.columns[0] in ['index', 'time', 'date']:
                _source.index = pd.to_datetime(_source.pop('index'))

        elif data.endswith('.xlsx') or data.endswith('xlx'):
            _source = pd.read_excel(data)
            if _source.columns[0] in ['index', 'time', 'date']:
                _source.index = pd.to_datetime(_source.pop('index'))

        elif data.endswith('.parquet'):
            _source = pd.read_parquet(data)

        elif data.endswith('.feather'):
            _source = pd.read_feather(data)
            if _source.columns[0] in ['index', 'time', 'date']:
                _source.index = pd.to_datetime(_source.pop('index'))

        # netcdf file
        elif data.endswith('.nc'):
            import xarray as xr
            _source = xr.open_dataset(data)
            _source = _source.to_dataframe()

        elif data.endswith('npz'):
            data = np.load(data)
            assert len(data) == 1
            d = []
            for k, v in data.items():
                d.append(v)

            data: np.ndarray = d[0]
            _source = pd.DataFrame(data, columns=input_features + output_features)

        # matlab's mat file
        elif data.endswith('.mat'):
            import scipy
            mat = scipy.io.loadmat(data)
            data: np.ndarray = mat['data']
            _source = pd.DataFrame(data, columns=input_features + output_features)

        elif os.path.isfile(data):
            assert os.path.exists(data)
            assert len(os.listdir(data)) > 1
            # read from directory
            raise NotImplementedError
        elif data in all_datasets:
            _source = self._get_data_from_ai4w_datasets(data)
        else:
            raise ValueError(f"unregnizable source of data given {data}")

        return _source

    def _process_one_source(self, data):
        if isinstance(data, str):
            _source = self._get_source_from_str(data)
        elif isinstance(data, pd.DataFrame):
            _source = data
        elif isinstance(data, np.ndarray):
            _source = data
        # elif data.__class.__.__name__ == "Dataset":
        #     _source = data

        else:
            raise ValueError(f"unregnizable source of data of type {data.__class__.__name__}")
        return _source

    def _get_data_from_ai4w_datasets(self, data):

        Dataset = getattr(datasets, data)

        dataset = Dataset()
        dataset_args = self.config['dataset_args']
        if dataset_args is None:
            dataset_args = {}

        # if self.config['input_features'] is not None:

        dynamic_features = self.config['input_features'] + self.config['output_features']

        data = dataset.fetch(dynamic_features = dynamic_features,
                             **dataset_args
                             )

        data = data.to_dataframe(['time', 'dynamic_features']).unstack()

        data.columns = [a[1] for a in data.columns.to_flat_index()]

        return data

    def _save_data_to_hdf5(self, data_type, x, prev_y, y, f):
        """Saves one data_type in h5py. data_type is string indicating whether
        it is training, validation or test data."""

        if x is not None:
            model_weights_group = f.create_group(data_type)

            if self.source_is_list:
                idx = 0
                for xi, prevyi, yi in zip(x, prev_y, y):
                    save_in_a_group(xi, prevyi, yi, model_weights_group, prefix=idx)
                    idx += 1
            elif self.source_is_dict:
                for src_name in self.data.keys():
                    save_in_a_group(x.get(src_name, None),
                                    prev_y.get(src_name, None),
                                    y.get(src_name, None),
                                    model_weights_group, prefix=src_name)
            else:
                save_in_a_group(x, prev_y, y, model_weights_group)
        return

    def check_for_batch_size(self, x, prev_y=None, y=None):

        if self.config['drop_remainder']:

            if isinstance(x, list):
                _x = x[0]
            else:
                _x = x
            assert isinstance(_x, np.ndarray)
            remainder = len(_x) % self.config['batch_size']

            if remainder:
                if isinstance(x, list):
                    _x = []
                    for val in x:
                        _x.append(val[0:-remainder])
                    x = _x
                else:
                    x = x[0:-remainder]

                if prev_y is not None:
                    prev_y = prev_y[0:-remainder]
                if y is not None:
                    y = y[0:-remainder]

        return x, prev_y, y


class MakeData(object):

    def __init__(self,
                 input_features,
                 output_features,
                 lookback,
                 input_step,
                 forecast_step,
                 forecast_len,
                 known_future_inputs,
                 batch_dim="3D",
                 allow_input_nans=False,
                 allow_nan_labels=0,
                 verbosity=1,
                 ):

        self.input_features = copy_features(input_features)
        self.output_features = copy_features(output_features)
        self.allow_nan_labels = allow_nan_labels
        self.lookback = lookback
        self.input_step = input_step
        self.forecast_step = forecast_step
        self.forecast_len = forecast_len
        self.allow_input_nans = allow_input_nans
        self.verbosity = verbosity
        self.batch_dim = batch_dim
        self.known_future_inputs = known_future_inputs
        self.nans_removed_4m_st = 0

        self.scalers = {}
        self.indexes = {}
        self.index_types = {}  # saves the information whether an index was datetime index or not?

    def check_nans(self, data, input_x, input_y, label_y, outs, lookback):
        """Checks whether anns are present or not and checks shapes of arrays being prepared."""
        if isinstance(data, pd.DataFrame):
            nans = data[self.output_features].isna()
            nans = nans.sum().sum()
            data = data.values
        else:
            nans = np.isnan(data[:, -outs:])  # df[self.out_cols].isna().sum()
            nans = int(nans.sum())
        if nans > 0:
            if self.allow_nan_labels == 2:
                if self.verbosity > 0: print("\n{} Allowing NANs in predictions {}\n".format(10 * '*', 10 * '*'))
            elif self.allow_nan_labels == 1:
                if self.verbosity>0: print("\n{} Ignoring examples whose all labels are NaNs {}\n".format(10 * '*', 10 * '*'))
                idx = ~np.array([all([np.isnan(x) for x in label_y[i]]) for i in range(len(label_y))])
                input_x = input_x[idx]
                input_y = input_y[idx]
                label_y = label_y[idx]
                if int(np.isnan(data[:, -outs:][0:lookback]).sum() / outs) >= lookback:
                    self.nans_removed_4m_st = -9999
            else:
                if self.verbosity > 0:
                    print('\n{} Removing Examples with nan in labels  {}\n'.format(10 * '*', 10 * '*'))
                if outs == 1:
                    # find out how many nans were present from start of data until lookback, these nans will be removed
                    self.nans_removed_4m_st = np.isnan(data[:, -outs:][0:lookback]).sum()
                # find out such labels where 'y' has at least one nan
                nan_idx = np.array([np.any(i) for i in np.isnan(label_y)])
                non_nan_idx = np.invert(nan_idx)
                label_y = label_y[non_nan_idx]
                input_x = input_x[non_nan_idx]
                input_y = input_y[non_nan_idx]

                assert np.isnan(label_y).sum() < 1, "label still contains {} nans".format(np.isnan(label_y).sum())

        assert input_x.shape[0] == input_y.shape[0] == label_y.shape[0], "shapes are not same"

        if not self.allow_input_nans:
            assert np.isnan(input_x).sum() == 0, "input still contains {} nans".format(np.isnan(input_x).sum())

        return input_x, input_y, label_y

    def transform(self, data, transformation, key='5'):

        # it is better to make a copy here because all the operations on data happen after this.
        data = data.copy()
        scalers = {}
        if transformation:

            if isinstance(transformation, dict):
                data, scaler = Transformation(**transformation)(data, 'transformation', return_key=True)
                scalers[key] = scaler

            # we want to apply multiple transformations
            elif isinstance(transformation, list):
                for idx, trans in enumerate(transformation):
                    if trans['method'] is not None:
                        data, scaler = Transformation(**trans)(data, 'transformation', return_key=True)
                        scalers[f'{key}_{trans["method"]}_{idx}'] = scaler
            else:
                assert isinstance(transformation, str)
                data, scaler = Transformation(method=transformation)(data, 'transformation', return_key=True)
                scalers[key] = scaler

        self.scalers.update(scalers)
        return data, scalers

    def indexify(self, data: pd.DataFrame, key):

        dummy_index = False
        # for dataframes
        if isinstance(data.index, pd.DatetimeIndex):
            index = list(map(int, np.array(data.index.strftime('%Y%m%d%H%M'))))  # datetime index
            self.index_types[key] = 'dt'
            original_index = index
        else:
            try:
                index = list(map(int, np.array(data.index)))
                self.index_types[key] = 'int'
                original_index = index
            except ValueError:  # index may not be convertible to integer, it may be string values
                dummy_index = np.arange(len(data), dtype=np.int64).tolist()
                original_index = pd.Series(data.index, index=dummy_index)
                index = dummy_index
                self.index_types[key] = 'str'
                self.indexes[key] = {'dummy': dummy_index, 'original': original_index}
        # pandas will add the 'datetime' column as first column. This columns will only be used to keep
        # track of indices of train and test data.
        data.insert(0, 'index', index)

        self.input_features = ['index'] + self.input_features
        # setattr(self, 'input_features', ['index'] + self.input_features)
        self.indexes[key] = {'index': index, 'dummy_index': dummy_index, 'original': original_index}
        return data

    def deindexify(self, data, key):

        if isinstance(data, np.ndarray):
            _data, _index = self.deindexify_nparray(data, key)
        elif isinstance(data, list):
            _data, _index = [], []
            for d in data:
                data_, index_ = self.deindexify_nparray(d, key)
                _data.append(data_)
                _index.append(index_)
        else:
            raise NotImplementedError

        if self.indexes[key]['dummy_index']:
            _index = self.indexes[key]['original'].loc[_index].values.tolist()

        if self.index_types[key] == 'dt':
            _index = to_datetime_index(_index)
        return _data, _index

    def get_batches(self, data, num_ins, num_outs):

        if self.batch_dim == "2D":
            return self.get_2d_batches(data, num_ins, num_outs)

        else:
            return self.check_nans(data, *prepare_data(data,
                                                       num_outputs=num_outs,
                                                       lookback_steps=self.lookback,
                                                       input_steps=self.input_step,
                                                       forecast_step=self.forecast_step,
                                                       forecast_len=self.forecast_len,
                                                       known_future_inputs=self.known_future_inputs),
                                   num_outs, self.lookback)

    def get_2d_batches(self, data, ins, outs):
        if not isinstance(data, np.ndarray):
            if isinstance(data, pd.DataFrame):
                data = data.values
            else:
                raise TypeError(f"unknown data type {data.__class__.__name__} for data ")

        if outs > 0:
            input_x, input_y, label_y = data[:, 0:ins], data[:, -outs:], data[:, -outs:]
        else:
            input_x, input_y, label_y = data[:, 0:ins], np.random.random((len(data), outs)), np.random.random((len(data), outs))

        assert self.lookback == 1, """lookback should be one for MLP/Dense layer based model, but it is {}
            """.format(self.lookback)

        return self.check_nans(data, input_x, input_y, np.expand_dims(label_y, axis=2), outs, self.lookback)

    def __call__(
            self,
            data,
            st=0,
            en=None,
            indices=None,
            intervals=None,
            shuffle=False
    ):

        num_ins = len(self.input_features)
        num_outs = len(self.output_features)

        if st is not None:
            assert isinstance(st, int), "starting point must be integer."
        if indices is not None:
            assert isinstance(np.array(indices), np.ndarray), "indices must be array like"
            if en is not None or st != 0:
                raise ValueError(f'When using indices, st and en can not be used. while st:{st}, and en:{en}')
        if en is None:
            en = data.shape[0]

        if isinstance(data, pd.DataFrame):
            data = data[self.input_features + self.output_features].copy()
            df = data
        else:
            num_ins = data.shape[-1]
            num_outs = 0
            data = data.copy()
            df = data

        if intervals is None:
            df = df[st:en]
            x, prev_y, y = self.get_batches(df, num_ins, num_outs)

            if indices is not None:
                # if indices are given then this should be done after `get_batches` method
                x = x[indices]
                prev_y = prev_y[indices]
                y = y[indices]
        else:
            xs, prev_ys, ys = [], [], []
            for _st, _en in intervals:
                df1 = data[_st:_en]

                if df1.shape[0] > 0:

                    x, prev_y, y = self.get_batches(df1.values, num_ins, num_outs)

                    xs.append(x)
                    prev_ys.append(prev_y)
                    ys.append(y)

            if indices is None:
                x = np.vstack(xs)[st:en]
                prev_y = np.vstack(prev_ys)[st:en]
                y = np.vstack(ys)[st:en]
            else:
                x = np.vstack(xs)[indices]
                prev_y = np.vstack(prev_ys)[indices]
                y = np.vstack(ys)[indices]

        if shuffle:
            raise NotImplementedError

        if 'index' in data:
            data.pop('index')

        return x, prev_y, y

    def deindexify_nparray(self, data, key):
        if data.ndim == 3:
            _data, index = data[..., 1:].astype(np.float32), data[:, -1, 0]
        elif data.ndim == 2:
            _data, index = data[..., 1:].astype(np.float32), data[:, 0]
        elif data.ndim == 4:
            _data, index = data[..., 1:].astype(np.float32), data[:, -1, -1, 0]
        elif data.ndim == 5:
            _data, index = data[..., 1:].astype(np.float32), data[:, -1, -1, -1, 0]
        else:
            raise NotImplementedError

        if self.index_types[key] != 'str':
            index = np.array(index, dtype=np.int64)
        return _data, index


class SiteDistributedDataHandler(object):

    """Prepares data for mulpltiple sites/locations.

    This class is useful to prepare data when we have data of different
    sites/locations. A `site` here refers to a specific context i.e. data
    of one site is different from another site. In such a case, training
    and test spearation can be done in one of following two ways

    1) We may wish to use data of some sites for training and data of some
        other sites for validation and/or test. In such a case,  the user must
        provide the names of sites as `training_sites`, `validation_sites` and
        `test_sites`.

    2) We may wish to use a fraction of data from each site for training
        validation and test. In such a case, do not provide any argument
        for `training_sites`, `validation_sites` and `test_sites`, rather
        define the validation and test fraction for each site using the
        keyword arguments.

    Note: The first two axis of the output data are swapped unless `swap_axes`
        is set to `False`.
    That means for 3d batches the x will have the shape:
        `(num_examples, num_sites, lookback, input_features)`
    And the `y` will have shape:
        `(num_examples, num_sites, num_outs, forecast_len)`
    See Example for more illustration

    Methods
    -----------------
    - training_data
    - validation_data
    - test_data


    Examples:
        >>> examples = 50
        >>> data = np.arange(int(examples * 3), dtype=np.int32).reshape(-1, examples).transpose()
        >>> df = pd.DataFrame(data, columns=['a', 'b', 'c'],
        ...                index=pd.date_range('20110101', periods=examples, freq='D'))
        >>> config = {'input_features': ['a', 'b'],
        ...      'output_features': ['c'],
        ...      'lookback': 4}
        >>> data = {'0': df, '1': df, '2': df, '3': df}
        >>> configs = {'0': config, '1': config, '2': config, '3': config}

        >>> dh = SiteDistributedDataHandler(data, configs)
        >>> train_x, train_y = dh.training_data()
        >>> val_x, val_y = dh.validation_data()
        >>> test_x, test_y = dh.test_data()

        >>> dh = SiteDistributedDataHandler(data, configs, training_sites=['0', '1'], validation_sites=['2'], test_sites=['3'])
        >>> train_x, train_y = dh.training_data()
        >>> val_x, val_y = dh.validation_data()
        >>> test_x, test_y = dh.test_data()

        A slightly more complicated example where data of each site consits of 2
        sources

        >>> examples = 40
        >>> data = np.arange(int(examples * 4), dtype=np.int32).reshape(-1, examples).transpose()
        >>> cont_df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd'],
        ...                            index=pd.date_range('20110101', periods=examples, freq='D'))
        >>> static_df = pd.DataFrame(np.array([[5],[6], [7]]).repeat(examples, axis=1).transpose(),
        ...                   columns=['len', 'dep', 'width'],
        ...                   index=pd.date_range('20110101', periods=examples, freq='D'))

        >>> config = {'input_features': {'cont_data': ['a', 'b', 'c'], 'static_data': ['len', 'dep', 'width']},
        ...          'output_features': {'cont_data': ['d']},
        ...          'lookback': {'cont_data': 4, 'static_data':1}}

        >>> data = {'cont_data': cont_df, 'static_data': static_df}
        >>> datas = {'0': data, '1': data, '2': data, '3': data, '4': data, '5': data, '6': data}
        >>> configs = {'0': config, '1': config, '2': config, '3': config, '4': config, '5': config, '6': config}

        >>> dh = SiteDistributedDataHandler(datas, configs)
        >>> train_x, train_y = dh.training_data()
        >>> val_x, val_y = dh.validation_data()
        >>> test_x, test_y = dh.test_data()

        >>> dh = SiteDistributedDataHandler(datas, configs, training_sites=['0', '1', '2'],
        ...                                validation_sites=['3', '4'], test_sites=['5', '6'])
        >>> train_x, train_y = dh.training_data()
        >>> val_x, val_y = dh.validation_data()
        >>> test_x, test_y = dh.test_data()
    """
    def __init__(
            self,
            data: dict,
            config: dict,
            training_sites: list = None,
            validation_sites: list = None,
            test_sites: list = None,
            swap_axes: bool = True,
            allow_variable_len: bool = False,
            verbosity: int = 1
    ):
        """
        Initiates data

        Arguments:
            data dict:
                Must be a dictionary of data for each site.
            config dict:
                Must be a dictionary of keyword arguments for each site.
                The keys of `data` and `config` must be same.
            training_sites Union[list, None]:
                List of names of sites to be used as training.
                If `None`, data from all sites will be used to extract training
                data based upon keyword arguments of corresponding site.
            validation_sites Union[list, None]:
                List of names of sites to be used as validation.
            test_sites Union[list, None]:
                List of names of sites to be used as test.
            swap_axes bool: If True, the returned x data will have shape
                `(num_examples, num_sites, lookback, input_features)` otherwise
                the returned x values will have shape
                `(num_sites, num_examples, lookback, input_features)`.
            allow_variable_len bool: If the number of examples for different sites differ
                from each other, then this argument can be set to `True` to avoid
                `ValueError` from numpy. If the data of different sites results
                in different number of examples and this argument is False, then
                numpy will raise `ValueError` because arrays of different lengths
                can not be stacked together. If this argument is set to `True`,
                then the returned `x` will not be a numpy array but it will be
                a dictionary of numpy arrays where each key correspond to one
                `site`. Note that setting this argument to `True` will render
                `swap_axes` redundent.

        """
        assert isinstance(data, dict), f'data must be of type dict but it is of type {data.__class__.__name__}'
        self.data = data

        new_config = self.process_config(config)

        if training_sites is not None:
            assert all([isinstance(obj, list) for obj in [training_sites, validation_sites, test_sites]])

            for site in [training_sites, validation_sites, test_sites]:
                for s in site:
                    assert s in data

        self.config = new_config
        self.training_sites = training_sites
        self.validation_sites = validation_sites
        self.test_sites = test_sites
        self.swap_axes = swap_axes
        self.allow_variable_len = allow_variable_len
        self.dhs = {}
        self.verbosity = verbosity

    def process_config(self, config):
        assert isinstance(config, dict)
        if len(config) > 1:
            for k in self.data.keys():
                assert k in config, f'{k} present in data but not available in config'
            new_config = config.copy()
        else:
            new_config = {}
            for k in self.data.keys():
                new_config[k] = config
        return new_config

    def training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the x,y pairs for training data"""
        x, y = [], []

        training_sites = self.training_sites
        if training_sites is None:
            training_sites = self.data.keys()

        for site in training_sites:

            src, conf = self.data[site], self.config[site]

            conf['verbosity'] = 0
            if self.training_sites is not None:
                conf['test_fraction'] = 0.0
                conf['val_fraction'] = 0.0

            dh = DataHandler(src, **conf)
            self.dhs[site] = dh  # save in memory
            _x, _y = dh.training_data()

            x.append(_x)
            y.append(_y)

        x = self.stack(x, training_sites)
        y = self.stack(y, training_sites)

        if self.verbosity:
            print(f"{'*' * 5} Training data {'*' * 5}")
            print_something(x, 'x')
            print_something(y, 'y')

        return x, y

    def validation_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the x,y pairs for the validation data"""
        x, y = [], []

        validation_sites = self.validation_sites
        if validation_sites is None:
            validation_sites = self.data.keys()

        for site in validation_sites:

            src, conf = self.data[site], self.config[site]
            conf['verbosity'] = 0
            if self.validation_sites is not None:
                # we have sites, so all the data from these sites is used as validation
                conf['test_fraction'] = 0.0
                conf['val_fraction'] = 0.0

                dh = DataHandler(src, **conf)
                self.dhs[site] = dh  # save in memory
                _x, _y = dh.training_data()
            else:
                dh = DataHandler(src, **conf)
                self.dhs[site] = dh  # save in memory
                _x, _y = dh.validation_data()

            x.append(_x)
            y.append(_y)

        x = self.stack(x, validation_sites)
        y = self.stack(y, validation_sites)

        if self.verbosity:
            print(f"{'*' * 5} Validation data {'*' * 5}")
            print_something(x, 'x')
            print_something(y, 'y')

        return x, y

    def test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the x,y pairs for the test data"""
        x, y = [], []

        test_sites = self.test_sites
        if test_sites is None:
            test_sites = self.data.keys()

        for site in test_sites:

            src, conf = self.data[site], self.config[site]

            if self.test_sites is not None:
                # we have sites, so all the data from these sites is used as test
                conf['test_fraction'] = 0.0
                conf['val_fraction'] = 0.0
                conf['verbosity'] = 0

                dh = DataHandler(src, **conf)
                self.dhs[site] = dh  # save in memory
                _x, _y = dh.training_data()
            else:
                conf['verbosity'] = 0
                dh = DataHandler(src, **conf)
                _x, _y = dh.test_data()

            x.append(_x)
            y.append(_y)

        x = self.stack(x, test_sites)
        y = self.stack(y, test_sites)

        if self.verbosity:
            print(f"{'*' * 5} Test data {'*' * 5}")
            print_something(x, 'x')
            print_something(y, 'y')

        return x, y

    def stack(self, data: list, site_names):
        if isinstance(data[0], np.ndarray):

            if len(set([len(i) for i in data])) == 1:  # all arrays in data are of equal length
                data = np.stack(data)  # (num_sites, num_examples, lookback, num_features)
                if self.swap_axes:
                    data = data.swapaxes(0, 1)  # (num_examples, num_sites, lookback, num_features)

            elif self.allow_variable_len:
                data = {k: v for k, v in zip(site_names, data)}

            else:
                raise NotImplementedError(f"number of examples are {[len(i) for i in data]}. "
                                          f"set `allow_variable_len` to `True`")

        elif isinstance(data[0], dict):

            if self.allow_variable_len:
                raise NotImplementedError  # todo

            new_data = {k: None for k in data[0].keys()}
            for src_name in new_data.keys():
                temp = []
                for src in data:

                    temp.append(src[src_name])

                if self.swap_axes:
                    new_data[src_name] = np.stack(temp).swapaxes(0, 1)
                else:
                    new_data[src_name] = np.stack(temp)

            data = new_data

        else:
            raise ValueError

        return data


class MultiLocDataHandler(object):

    def __init__(self):
        pass

    def training_data(self, data, **kwargs):

        dh = DataHandler(data=data, val_fraction=0.0, test_fraction=0.0, save=False, verbosity=0,
                         **kwargs)
        setattr(self, 'train_dh', dh)

        return dh.training_data()

    def validation_data(self, data, **kwargs) -> Tuple[np.ndarray, np.ndarray]:

        dh = DataHandler(data=data, val_fraction=0.0, test_fraction=0.0, save=False, verbosity=0,
                         **kwargs)
        setattr(self, 'val_dh', dh)

        return dh.training_data()

    def test_data(self, data, **kwargs):

        dh = DataHandler(data=data, val_fraction=0.0, test_fraction=0.0, save=False, verbosity=0,
                         **kwargs)
        setattr(self, 'test_dh', dh)

        return dh.training_data()


def decode(json_string):
    return json.loads(json_string, object_hook=_decode_helper)


def _decode_helper(obj):
    """A decoding helper that is TF-object aware."""
    if isinstance(obj, dict) and 'class_name' in obj:

        if obj['class_name'] == '__tuple__':
            return tuple(_decode_helper(i) for i in obj['items'])
        elif obj['class_name'] == '__ellipsis__':
            return Ellipsis
    return obj


def load_data_from_hdf5(data_type, data):

    f = h5py.File(data, mode='r')

    weight_names = ['x', 'prev_y', 'y']

    g = f[data_type]
    weight_values = (np.asarray(g[weight_name]) for weight_name in weight_names)

    f.close()

    return weight_values


def save_in_a_group(x, prev_y, y, group_name, prefix=None):

    container = {}
    if x is not None:
        key = f'{prefix}_x' if prefix else 'x'
        container[key] = x

    if prev_y is not None:
        key = f'{prefix}_prev_y' if prefix else 'prev_y'
        container[key] = prev_y

    if y is not None:
        key = f'{prefix}_y' if prefix else 'y'
        container[key] = y

    for name, val in container.items():

        param_dset = group_name.create_dataset(name, val.shape, dtype=val.dtype)
        if not val.shape:
            # scalar
            param_dset[()] = val
        else:
            param_dset[:] = val
    return


def make_config(**kwargs):

    return kwargs


def copy_features(features):
    if isinstance(features, str):
        _features = copy.deepcopy(features)
    elif isinstance(features, list) or isinstance(features, pd.Index):
        _features = []
        for f in features:
            _features.append(copy.deepcopy(f))
    else:
        raise NotImplementedError
    return _features


def conform_shape(data, shape, features=None):
    # if the difference is of only 1 dim, we resolve it
    if data.ndim > len(shape):
        data = np.squeeze(data, axis=-1)
    elif data.ndim < len(shape):
        data = np.expand_dims(data, axis=-1)

    assert data.ndim == len(shape), f"original data had {len(shape)} wihle the new data has {data.ndim} dimensions"

    dummy_features = shape[-1] - data.shape[-1]  # how manu dummy features we have to add to match the shape

    if data.__class__.__name__ in ['DataFrame', 'Series']:
        # we know what features must be in data, so put them in data one by one if they do not exist in data already
        if features:
            for f in features:
                if f not in data:
                    data[f] = np.random.random(len(data))
        # identify how many features to be added by shape information
        elif dummy_features > 0:
            dummy_data = pd.DataFrame(np.random.random((len(data), dummy_features)))
            data = pd.concat([dummy_data, data], axis=1)
    else:
        dummy_data = np.random.random((len(data), dummy_features))
        data = np.concatenate([dummy_data, data], axis=1)

    return data, dummy_features


def consider_intervals(data, intervals):
    _source = data
    if intervals is not None:
        if isinstance(data, pd.DataFrame):
            try:  # if indices in intervals are of same type as that of index
                # -1 so that .loc and .iloc give same results, however this is not possible
                # with DatetimeIndex
                if isinstance(data.index, pd.DatetimeIndex):
                    _source = pd.concat([data.loc[st:en] for st, en in intervals])
                else:
                    _source = pd.concat([data.loc[st:en - 1] for st, en in intervals])
            except TypeError:  # assuming indices in intervals are integers
                _source = pd.concat([data.iloc[st:en] for st, en in intervals])

    return _source


def tot_obs_for_one_df(data, allow_nan_labels, output_features, lookback, input_step,
                       num_outs, forecast_step, forecast_len, intervals, input_features:list):

    data = consider_intervals(data, intervals)

    max_tot_obs = 0
    if not allow_nan_labels and intervals is None:
        _data = data[input_features + output_features] if isinstance(data, pd.DataFrame) else data
        x, _, _ = prepare_data(_data,
                               lookback, num_outputs=num_outs, input_steps=input_step,
                               forecast_step=forecast_step,
                               forecast_len=forecast_len, mask=np.nan)
        max_tot_obs = len(x)

    # we need to ignore some values at the start
    more = (lookback * input_step) - 1

    if isinstance(data, np.ndarray):
        return len(data) - more

    # todo, why not when allow_nan_labels>0?
    if forecast_step > 0:
        more += forecast_step

    if forecast_len > 1:
        more += forecast_len

    if intervals is None: intervals = [()]

    more *= len(intervals)

    if allow_nan_labels == 2:
        tot_obs = data.shape[0] - more

    elif allow_nan_labels == 1:
        label_y = data[output_features].values
        idx = ~np.array([all([np.isnan(x) for x in label_y[i]]) for i in range(len(label_y))])
        tot_obs = np.sum(idx) - more
    else:

        if num_outs == 1:
            tot_obs = data.shape[0] - int(data[output_features].isna().sum()) - more
            tot_obs = max(tot_obs, max_tot_obs)

        else:
            # count by droping all the rows when nans occur in output features
            tot_obs = len(data.dropna(subset=output_features))
            tot_obs -= more

    return tot_obs


def batch_dim_from_lookback(lookback):
    default = "3D"
    if lookback == 1:
        default = "2D"
    return default


def deindexify_y(x: list, prev_y: list, y:list, based_upon: int):

    indices_to_keep = []
    for e in y[based_upon]:
        indices_to_keep.append(int(e[-1]))

    if isinstance(x, list):
        return deindexify_lists(x, prev_y, y, indices_to_keep)
    else:
        return deindexify_dicts(x, prev_y, y, indices_to_keep)


def deindexify_lists(x, prev_y, y, indices_to_keep):

    _x, _prevy, _y = [], [], []
    # for x,y of each source
    for xi, prevyi, yi in zip(x, prev_y, y):
        __x, __prevy, __y = [], [], []

        # for individual examples of one source, check that if that example is to included or not
        for _xi, _prevyi, _yi in zip(xi, prevyi, yi):
            if int(_yi[-1]) in indices_to_keep:
                __x.append(_xi)
                __prevy.append(_prevyi)
                __y.append(_yi[0:-1])  # don't consider the last value, that was dummy_index

        _x.append(np.stack(__x))
        _prevy.append(np.stack(__prevy))
        _y.append(np.stack(__y))

    return _x, _prevy, _y


def deindexify_dicts(x: dict, prev_y: dict, y: dict, indices_to_keep):

    _x, _prevy, _y = {}, {}, {}

    for (key, xi), (key, prevyi), (key, yi) in zip(x.items(), prev_y.items(), y.items()):

        __x, __prevy, __y = [], [], []

        # for individual examples of one source, check that if that example is to included or not
        for _xi, _prevyi, _yi in zip(xi, prevyi, yi):
            if int(_yi[-1]) in indices_to_keep:
                __x.append(_xi)
                __prevy.append(_prevyi)
                __y.append(_yi[0:-1])  # don't consider the last value, that was dummy_index

        _x[key] = np.stack(__x)
        _prevy[key] = np.stack(__prevy)
        _y[key] = np.stack(__y)

    return _x, _prevy, _y


def filter_zero_sized_arrays(array):
    if isinstance(array, list):
        new_array = []
        for a in array:
            if a.size > 0:
                new_array.append(a)
    elif isinstance(array, dict):
        new_array = {}
        for k, v in array.items():
            if v.size > 0:
                new_array[k] = v
    else:
        new_array = array
    return new_array


def check_for_classification(label: np.ndarray, to_categorical):

    assert isinstance(label, np.ndarray), f"""
                            classification problem for label of type {label.__class__.__name__} not implemented yet"""

    # for clsasification, it should be 2d
    label = label.reshape(-1, label.shape[1])
    if to_categorical:
        assert label.shape[1] == 1
        label = OneHotEncoder(sparse=False).fit_transform(label)
    # else:   # mutlti_label/binary problem
    #     # todo, is only binary_crossentropy is binary/multi_label problem?
    #     pass #assert self.loss_name() in ['binary_crossentropy']
    return label


def return_x_yy(x, prev_y, y, initial, verbosity):

    if verbosity > 0:
        print(f"{'*' * 5} {initial} data {'*' * 5}")
        print_something(x, "input_x")
        print_something(prev_y, "prev_y")
        print_something(y, "target")
    return x, prev_y, y


def return_xy(x, y, initial, verbosity):

    if verbosity > 0:
        print(f"{'*' * 5} {initial} {'*' * 5}")
        print_something(x, "input_x")
        print_something(y, "target")

    return x, y
