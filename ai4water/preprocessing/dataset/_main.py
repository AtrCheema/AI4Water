
import json
import inspect
import warnings
from typing import Union
from copy import copy, deepcopy

import ai4water.datasets as datasets
from ai4water.datasets import all_datasets
from ai4water.utils.utils import TrainTestSplit
from ai4water.utils.plotting_tools import Plots
from ai4water.preprocessing.imputation import Imputation
from ai4water.utils.utils import prepare_data, jsonize, to_datetime_index, print_something
from ai4water.backend import np, pd, plt, os, mpl, sklearn, h5py

from .utils import check_for_classification
from .utils import consider_intervals, decode
from .utils import load_data_from_hdf5

train_test_split = sklearn.model_selection.train_test_split
KFold = sklearn.model_selection.KFold
LeaveOneOut = sklearn.model_selection.LeaveOneOut
TimeSeriesSplit = sklearn.model_selection.TimeSeriesSplit
ShuffleSplit = sklearn.model_selection.ShuffleSplit

Patch = mpl.patches.Patch
cmap_cv = plt.cm.coolwarm


class _DataSet(Plots):

    def __init__(self, config, path=os.getcwd()):

        Plots.__init__(self, config=config, path=path)

    def training_data(self):
        raise NotImplementedError

    def validation_data(self):
        raise NotImplementedError

    def test_data(self):
        raise NotImplementedError

    def KFold_splits(self, n_splits=5):
        raise NotImplementedError

    def LeaveOneOut_splits(self):
        raise NotImplementedError

    def TimeSeriesSplit_splits(self, n_splits=5):
        raise NotImplementedError

    @classmethod
    def from_h5(cls, h5_file: str):
        raise NotImplementedError

    def to_disk(self, path: str):
        raise NotImplementedError

    def return_xy(self, x, y, initial):

        if self.mode == "classification" and self.is_binary:
            if len(y) == y.size:
                y = y.reshape(-1, 1)

        if self.verbosity > 0:
            print(f"{'*' * 5} {initial} {'*' * 5}")
            print_something(x, "input_x")
            print_something(y, "target")

        return x, y

    def return_x_yy(self, x, prev_y, y, initial):

        if self.verbosity > 0:
            print(f"{'*' * 5} {initial} data {'*' * 5}")
            print_something(x, "input_x")
            print_something(prev_y, "prev_y")
            print_something(y, "target")
        return x, prev_y, y


class DataSet(_DataSet):
    """The purpose of DataSet is to convert unprepared/raw data into prepared data.
    A prepared data consists of x,y pairs where x is inputs and y is outputs. There
    are >1 examples in a DataSet. Both inputs and outputs consists of same number
    of examples. An example consists of one input, output pair which can be given
    to a supervised machine learning algorithm for training. For tabular data, the
    number of examples does not necessarily match number of rows. The number of
    examples depend upon multiple factors such as presence of intervals, how
    nans are handled and the arguments related to time series data preparation
    which are listed in detail in prepare_data function.

    DataSet class can accept the raw, unprepared data in a variety of formats such
    as .csv, .xlsx, .parquet, .mat, .n5 etc. For details see this. The DataSet
    class can save the prepared data into an hdf5 file which can susequently be
    used to load the data and save the time.

    Methods
    ------------
    - training_data: returns training data
    - validation_data: returns validation data
    - test_data: returns test data
    - from_h5:
    - to_disk
    - KFold_splits: creates splits using `KFold` of sklearn
    - LeaveOneOut_splits: creates splits using `LeaveOneOut` of sklearn
    - TimeSeriesSplit_splits: creates splits using `TimeSeriesSplit` of sklearn
    - total_exs

    """

    def __init__(
            self,
            data,
            input_features: Union[str, list] = None,
            output_features: Union[str, list] = None,
            dataset_args: dict = None,

            ts_args: dict = None,

            split_random: bool = False,
            train_fraction: float = 0.7,
            val_fraction: float = 0.2,
            indices: dict = None,

            intervals=None,
            shuffle: bool = True,
            allow_nan_labels: int = 0,
            nan_filler: dict = None,
            batch_size: int = 32,
            drop_remainder: bool = False,
            teacher_forcing: bool = False,
            allow_input_nans: bool = False,

            seed: int = 313,
            verbosity: int = 1,
            mode: str = None,
            category: str = None,
            save: bool = False
    ):
        """
        Initializes the DataSet class

        Parameters
        ----------
            data :
                source from which to make the data. It can be one of the following:

                - pandas dataframe: each columns is a feature and each row is an example
                - numpy array
                - xarray dataset: it can be xarray dataset
                - path like: if the path is the path of a file, then this file can
                    be a csv/xlsx/nc/npz/mat/parquet/feather file. The .nc file
                    will be read using xarray to load datasets. If the path refers
                    to a directory, it is supposed that each file in the directory refers to one example.
                - ai4water dataset : name of any of dataset name from ai4water.datasets
                - name of .h5 file

            input_features : Union[list, dict, str, None]
                features to use as input. If `data` is pandas dataframe
                then this is list of column names from `data` to be used as input.
            output_features : Union[list, dict, str, None]
                features to use as output. When `data` is dataframe
                then it is list of column names from `data` to be used as output.
                If `data` is `dict`, then it must be consistent with `data`.
                Default is None,which means the last column of data will be
                used as output. In case of multi-class classification, the output
                column is not supposed to be one-hot-encoded rather in the form
                of [0,1,2,0,1,2,1,2,0] for 3 classes. One-hot-encoding is done
                inside the model.
            dataset_args : dict
                additional arguments for AI4Water's [datasets][ai4water.datasets]
            ts_args : dict, optional
                This argument should only be used if the data is time series data.
                It must be a dictionary which is then passed to :py:func:`ai4water.utils.prepare_data`
                for data preparation. Possible keys in dictionay are:
                    - lookback
                    - forecast_len
                    - forecast_step
                    - input_steps
            split_random : bool, optional
                whether to split the data into training and test randomly or not.
            train_fraction : float
                Fraction of the complete data to be used for training
                purpose. Must be greater than 0.0.
            val_fraction : float
                The fraction of the training data to be used for validation.
                Set to 0.0 if no validation data is to be used.
            indices : dict, optional
                A dictionary with two possible keys, 'training', 'validation'.
                It determines the indices to be used to select training, validation
                and test data. If indices are given for training, then train_fraction
                must not be given. If indices are given for validation, then indices
                for training must also be given and  val_fraction must not be given.
                Therefore, the possible keys in indices dictionary are follwoing
                    - ``training``
                    - ``training`` and ``validation``
            intervals :
                tuple of tuples where each tuple consits of two integers, marking
                the start and end of interval. An interval here means indices
                from the data. Only rows within those indices will be used when preparing
                data/batches for NN. This is handy when our input data
                contains chunks of missing values or when we don't want to consider several
                rows in input data during data_preparation.
                For further usage see `examples/using_intervals`
            shuffle : bool
                whether to shuffle the samples or not
            allow_nan_labels : bool
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
            nan_filler : dict
                This argument determines the imputation technique used to fill the nans in
                the data. The imputation is actually performed by :py:class:`ai4water.preprocessing.Imputation`
                class. Therefore this argument determines the interaction with `Imputation` class.
                The default value is None, which will raise error if missing/nan values
                are encountered in the input data. The user can however specify a
                dictionary whose one key must be `method`. The value of 'method'
                key can be `fillna` or `interpolate`.  For example, to do forward
                filling, the user can do as following

                >>> {'method': 'fillna', 'imputer_args': {'method': 'ffill'}}

                For details about fillna keyword options see fillna_

                For `interpolate`, the user can specify  the type of interpolation
                for example

                >>> {'method': 'interpolate', 'imputer_args': {'method': 'spline', 'order': 2}}

                will perform spline interpolation with 2nd order.
                For other possible options/keyword arguments for interpolate_
                [see]()
                The filling or interpolation is done columnwise, however, the user
                can specify how to do for each column by providing the above mentioned
                arguments as dictionary or list. The sklearn based imputation methods
                can also be used in a similar fashion. For KNN

                >>> {'method': 'KNNImputer', 'imputer_args': {'n_neighbors': 3}}

                or for iterative imputation

                >>> {'method': 'IterativeImputer', 'imputer_args': {'n_nearest_features': 2}}

                To pass additional arguments one can make use of `imputer_args`
                keyword argument

                >>> {'method': 'KNNImputer', 'features': ['b'], 'imputer_args': {'n_neighbors': 4}},

                For more on sklearn based imputation methods see this blog_
            batch_size : int
                size of one batch. Only relevent if `drop_remainder` is True.
            drop_remainder : bool
                whether to drop the remainder if len(data) % batch_size != 0 or not?
            teacher_forcing : bool
                whether to return previous output/target/ground
                truth or not. This is useful when the user wants to feed output
                at t-1 as input at timestep t. For details about this technique
                see this article_
            allow_input_nans : bool, optional
                If False, the examples containing nans in inputs will be removed.
                Setting this to True will result in feeding nan containing data
                to your algorithm unless nans are filled with `nan_filler`.
            seed : int
                random seed for reproducibility
            verbosity : int
            mode : str
                either ``regression`` or ``classification``
            category : str
            save : bool
                whether to save the data in an h5 file or not.


        Example
        -------
            >>> import pandas as pd
            >>> import numpy as np
            >>> from ai4water.preprocessing import DataSet
            >>> data_ = pd.DataFrame(np.random.randint(0, 1000, (50, 2)), columns=['input', 'output'])
            >>> data_set = DataSet(data=data_, ts_args={'lookback':5})
            >>> x,y = data_set.training_data()

        .. _fillna:
            https://pandas.pydata.org/pandas-docs/version/0.22.0/generated/pandas.DataFrame.fillna.html
        .. _article:
            https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/
        .. _interpolate:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
        .. _blog:
            https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html#sphx-glr-auto-examples-impute-plot-missing-values-py

        Note
        ----
        The word 'index' is not allowed as column name, input_features or output_features
        """

        indices = indices or {}

        if indices:
            assert split_random is False, "indices cannot be used with split_random"

        if 'training' in indices:
            assert train_fraction == 0.7, f"""
            You can not set training data using both indices and train_fraction.
            Use either indices or train_fraction."""

        if 'validation' in indices:
            assert val_fraction == 0.2, f"""
                You can not set validation data using both indices and val_fraction. 
                Use either indices or val_fraction."""
            assert 'training' in indices, f"""
            when defining validation data using indices, training data must also be 
            defined using indices."""

        assert val_fraction < 1.0, f"""
            val_fraction must be less than 1.0 but it is {val_fraction}.
            """

        self.dataset_args = dataset_args

        self.config = {
            'input_features': input_features,
            'output_features': output_features
        }
        self.nan_filler = nan_filler

        self.data = self._process_data(
            data,
            input_features,
            output_features)

        self.ts_args = ts_args
        self.split_random = split_random
        self.indices = indices
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.intervals = intervals
        self.allow_nan_labels = allow_nan_labels
        self.teacher_forcing = teacher_forcing
        self.drop_remainder = drop_remainder
        self.allow_input_nans = allow_input_nans

        self.verbosity = verbosity
        self.seed = seed
        self.mode = mode
        self.category = category
        self.save = save

        self.scalers = {}
        self.indexes = {}
        self.index_types = {}

        self._input_features = copy(input_features)

        if save and h5py:
            self.to_disk()

        _DataSet.__init__(self, config=self.config, path=os.getcwd())

    def init_paras(self) -> dict:
        """Returns the initializing parameters of this class"""
        signature = inspect.signature(self.__init__)

        init_paras = {}
        for para in signature.parameters.values():
            init_paras[para.name] = getattr(self, para.name)

        return init_paras

    @property
    def ts_args(self):
        return self._ts_args

    @ts_args.setter
    def ts_args(self, _ts_args: dict = None):
        default_args = {'input_steps': 1,
                        'lookback': 1,
                        'forecast_len': 1,
                        'forecast_step': 0,
                        'known_future_inputs': False
                        }

        if _ts_args:
            default_args.update(_ts_args)

        self._ts_args = default_args

    @property
    def lookback(self):
        return self.ts_args['lookback']

    @property
    def classes(self):
        _classes = []
        if self.mode == 'classification':
            if self.num_outs == 1:  # for binary/multiclass
                array = self.data[self._output_features].values
                _classes = np.unique(array[~np.isnan(array)])
            else:  # for one-hot encoded
                _classes = self._output_features

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
                array = self.data[self._output_features].values
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
                array = self.data[self._output_features].values
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
        # it seems sklearn can accept one-hot-encoded targets but xgb, lgbm and catboost can't
        # but since since sklearn can also accept non-one-hot-encoded targets for multiclass
        # let's not one-hot-encode for all ML algos
        if self.category == 'ML':
            _defualt = False

        return _defualt

    @property
    def teacher_forcing(self):
        return self._teacher_forcing

    @teacher_forcing.setter
    def teacher_forcing(self, x):
        self._teacher_forcing = x

    @property
    def input_features(self):
        _inputs = self.config['input_features']

        if _inputs is None and self.data is not None:
            assert isinstance(self.data, pd.DataFrame)
            _inputs = self.data.columns[0:-1].to_list()

        return _inputs

    @property
    def output_features(self):
        """for external use"""
        _outputs = self.config['output_features']

        if _outputs is None and self.data is not None:
            # assert isinstance(self.data, pd.DataFrame)
            if self.data.ndim == 2:
                _outputs = [col for col in self.data.columns if col not in self.input_features]
            else:
                _outputs = []  # todo
        return _outputs

    @property
    def _output_features(self):
        """for internal use"""
        _outputs = deepcopy(self.config['output_features'])

        if isinstance(self.data, list):
            assert isinstance(_outputs, list)

        elif isinstance(self.data, dict):
            assert isinstance(_outputs, dict), f"""
            data is of type dict while output_features are
            of type {_outputs.__class__.__name__}"""
            for k in self.data.keys():
                if k not in _outputs:
                    _outputs[k] = []

        elif _outputs is None and self.data is not None:
            assert isinstance(self.data, pd.DataFrame)
            _outputs = [col for col in self.data.columns if col not in self.input_features]

        return _outputs

    @property
    def num_ins(self):
        return len(self.input_features)

    @property
    def num_outs(self):
        return len(self.output_features)

    @property
    def batch_dim(self):

        default = "3D"
        if self.ts_args['lookback'] == 1:
            default = "2D"

        return default

    def _process_data(self,
                      data,
                      input_features,
                      output_features
                      ):

        if isinstance(data, str):
            _source = self._get_data_from_str(data, input_features, output_features)
            if isinstance(_source, str) and _source.endswith('.h5'):
                self._from_h5 = True

        elif isinstance(data, pd.DataFrame):
            _source = self._get_data_from_df(data, input_features, output_features)

        elif isinstance(data, np.ndarray):
            _source = self._get_data_from_ndarray(data, input_features, output_features)

        elif data.__class__.__name__ == "Dataset":
            _source = data

        elif isinstance(data, list):
            raise ValueError(f"""
            data is given as a list. For such cases either use DataSetUnion
            or DataSetPipeline insteadd of DataSet class""")

        elif isinstance(data, dict):
            raise ValueError(f"""
            data is given as a dictionary. For such cases either use DataSetUnion
            or DataSetPipeline insteadd of DataSet class""")

        elif data is None:
            return data

        else:
            assert data is not None
            raise ValueError(f"""
            unregnizable source of data of type {data.__class__.__name__} given
            """)

        _source = self.impute(_source)

        return _source

    def _get_data_from_ndarray(self, data, input_features, output_features):
        if data.ndim == 2:
            # if output_features is not defined, consider 1 output and name it
            # as 'output'
            if output_features is None:
                output_features = ['outout']
                self.config['output_features'] = output_features  # we should put it in config as well
            elif isinstance(output_features, str):
                output_features = [output_features]
            else:
                assert isinstance(output_features, list)

            if input_features is None:  # define dummy names for input_features
                input_features = [f'input_{i}' for i in range(data.shape[1] - len(output_features))]
                self.config['input_features'] = input_features

            return pd.DataFrame(data, columns=input_features + output_features)
        else:
            return data

    def _get_data_from_df(self, data, input_features, output_features):

        if input_features is None and output_features is not None:
            if isinstance(output_features, str):
                output_features = [output_features]
            assert isinstance(output_features, list)
            input_features = [col for col in data.columns if col not in output_features]
            # since we have inferred the input_features, they should be put
            # back into config
            self.config['input_features'] = input_features

        return data

    def _get_data_from_str(self, data, input_features, output_features):
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
            _source = data
        elif os.path.isdir(data):
            assert len(os.listdir(data)) > 1
            # read from directory
            raise NotImplementedError
        elif data in all_datasets:
            _source = self._get_data_from_ai4w_datasets(data)
        else:
            raise ValueError(f"unregnizable source of data given {data}")

        return _source

    def _get_data_from_ai4w_datasets(self, data):

        Dataset = getattr(datasets, data)

        dataset = Dataset()
        dataset_args = self.dataset_args
        if dataset_args is None:
            dataset_args = {}

        # if self.config['input_features'] is not None:

        dynamic_features = self.input_features + self.output_features

        data = dataset.fetch(dynamic_features=dynamic_features,
                             **dataset_args)

        data = data.to_dataframe(['time', 'dynamic_features']).unstack()

        data.columns = [a[1] for a in data.columns.to_flat_index()]

        return data

    def impute(self, data):
        """Imputes the missing values in the data using `Imputation` module"""
        if self.nan_filler is not None:

            if isinstance(data, pd.DataFrame):

                _source = self._impute(data, self.nan_filler)

            else:
                raise NotImplementedError
        else:
            _source = data

        return _source

    def _impute(self, data, impute_config):

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

    def get_indices(self):
        """If the data is to be divded into train/test based upon indices,
        here we create train_indices and test_indices. The train_indices
        contain indices for both training and validation data.
        """

        tot_obs = self.total_exs(**self.ts_args)

        all_indices = np.arange(tot_obs)

        if len(self.indices) == 0:
            if self.train_fraction < 1.0:
                if self.split_random:
                    train_indices, test_indices = train_test_split(
                        all_indices,
                        train_size=self.train_fraction,
                        random_state=self.seed
                    )
                else:
                    train_indices, test_indices = self._get_indices_by_seq_split(
                        all_indices,
                        self.train_fraction)
            else:  # no test data
                train_indices, test_indices = all_indices, []
        else:
            _train_indices = self.indices.get('training', None)
            _val_indices = self.indices.get('validation', None)
            _test_indices = self.indices.get('test', None)

            if _train_indices is not None:
                if _val_indices is None:
                    # even if val_fraction is > 0.0, we will separate validation
                    # data from training later
                    _val_indices = np.array([])  # no validation set
                else:
                    assert isinstance(np.array(_val_indices), np.ndarray)
                    _val_indices = np.array(_val_indices)

                overlap = np.intersect1d(_train_indices, _val_indices)
                assert len(overlap) == 0, f"""
                    Training and validation indices must be mutually exclusive.
                    They contain {len(overlap)} overlaping values."""
                train_indices = np.sort(np.hstack([_train_indices, _val_indices]))

                if _test_indices is None:
                    # get test_indices by subtracting train_indices from all indices
                    test_indices = [ind for ind in all_indices if ind not in train_indices]
                    # _val_indices = np.array([])

            else:  # todo
                train_indices = []

        setattr(self, 'train_indices', train_indices)
        setattr(self, 'test_indices', test_indices)

        return np.array(train_indices).astype("int32"), np.array(test_indices).astype("int32")

    def _get_indices_by_seq_split(
            self,
            all_indices: Union[list, np.ndarray],
            train_fraction):
        """ sequential train/test split"""

        train_indices = all_indices[0:int(train_fraction * len(all_indices))]
        test_indices = all_indices[int(train_fraction * len(all_indices)):]
        return train_indices, test_indices

    def _training_data(self, key="_training", **kwargs):
        """training data including validation data"""

        train_indices, test_indices = self.get_indices()

        if 'validation' in self.indices:
            # when validation indices are given, we first prepare
            # complete data which contains training, validation and test data
            # TODO this is agains function definition
            indices = np.sort(np.hstack([train_indices, test_indices]))
        else:
            indices = train_indices

        data = self.data.copy()

        # numpy arrays are not indexed and is supposed that the whole array is
        # use as input
        if not isinstance(data, np.ndarray):
            data = self.indexify(data, key)

        # get x,_y, y
        x, prev_y, y = self._make_data(
            data,
            intervals=self.intervals,
            indices=indices,
            **kwargs)

        if not isinstance(self.data, np.ndarray):
            x, self.indexes[key] = self.deindexify(x, key)

        if self.mode == 'classification':
            y = check_for_classification(y, self._to_categorical)

        return x, prev_y, y

    def training_data(self, key="train", **kwargs):
        """training data excluding validation data"""

        if getattr(self, '_from_h5', False):
            return load_data_from_hdf5('training_data', self.data)

        x, prev_y, y = self._training_data(key=key, **kwargs)
        if self.val_fraction > 0.0:
            # when no output is generated, corresponding index will not be saved
            idx = self.indexes.get(key, np.arange(len(x)))  # index also needs to be split
            x, prev_y, y, idx = self._train_val_split(x, prev_y, y, idx, 'training')

            # if drop remainder, we need to
            x, prev_y, y = self.check_for_batch_size(x, prev_y, y)

            self.indexes[key] = idx[0:len(x)]

        if self.teacher_forcing:
            return self.return_x_yy(x, prev_y, y, "Training")

        return self.return_xy(x, y, "Training")

    def validation_data(self, key="val", **kwargs):
        """validation data"""

        if getattr(self, '_from_h5', False):
            return load_data_from_hdf5('validation_data', self.data)

        x, prev_y, y = self._training_data(key=key, **kwargs)

        if self.val_fraction > 0.0:
            idx = self.indexes.get(key, np.arange(len(x)))
            x, prev_y, y, idx = self._train_val_split(x, prev_y, y, idx, 'validation')

            x, prev_y, y = self.check_for_batch_size(x, prev_y, y)

            self.indexes[key] = idx[0:len(x)]
        else:
            x, prev_y, y = np.empty(0), np.empty(0), np.empty(0)

        if self.teacher_forcing:
            return self.return_x_yy(x, prev_y, y, "Validation")

        return self.return_xy(x, y, "Validation")

    def _train_val_split(self, x, prev_y, y, idx, return_type):
        """split x,y,idx,prev_y into training and validation data"""

        if self.split_random:
            # split x,y randomly
            splitter = TrainTestSplit(test_fraction=self.val_fraction, seed=self.seed)
            train_x, val_x, train_y, val_y = splitter.split_by_random(x, y)
            splitter = TrainTestSplit(test_fraction=self.val_fraction, seed=self.seed)
            train_idx, val_idx, train_prev_y, val_prev_y = splitter.split_by_random(
                idx, prev_y)

        elif 'validation' in self.indices:
            # separate indices were provided for validation data
            # it must be remembered that x,y now contains training+validation+test data
            # but based upon indices, we will choose either training or validation data
            val_indices = self.indices['validation']
            _train_indices, _ = self.get_indices()
            train_indices = [i for i in _train_indices if i not in val_indices]
            splitter = TrainTestSplit(train_indices=train_indices, test_indices=val_indices)
            train_x, val_x, train_y, val_y = splitter.split_by_indices(
                x, y
            )
            splitter = TrainTestSplit(train_indices=train_indices, test_indices=val_indices)
            train_idx, val_idx, train_prev_y, val_prev_y = splitter.split_by_indices(
                idx, prev_y)
        else:
            # split x,y sequentially
            splitter = TrainTestSplit(test_fraction=self.val_fraction)
            train_x, val_x, train_y, val_y = splitter.split_by_slicing(x, y)
            splitter = TrainTestSplit(test_fraction=self.val_fraction)
            train_idx, val_idx, train_prev_y, val_prev_y = splitter.split_by_slicing(idx, prev_y)

        if return_type == "training":
            return train_x, train_prev_y, train_y, train_idx

        return val_x, val_prev_y, val_y, val_idx

    def test_data(self, key="test", **kwargs):
        """test data"""
        if getattr(self, '_from_h5', False):
            return load_data_from_hdf5('test_data', self.data)

        if self.train_fraction < 1.0:

            data = self.data.copy()

            # numpy arrays are not indexed and is supposed that the whole array
            # is use as input
            if not isinstance(data, np.ndarray):
                data = self.indexify(data, key)

            _, test_indices = self.get_indices()

            if len(test_indices) > 0:  # it is possible that training and validation
                # indices cover whole data
                # get x,_y, y
                x, prev_y, y = self._make_data(
                    data,
                    intervals=self.intervals,
                    indices=test_indices,
                    **kwargs)

                x, prev_y, y = self.check_for_batch_size(x, prev_y, y)

                if not isinstance(self.data, np.ndarray):
                    x, self.indexes[key] = self.deindexify(x, key)

                if self.mode == 'classification':
                    y = check_for_classification(y, self._to_categorical)
            else:
                x, prev_y, y = np.empty(0), np.empty(0), np.empty(0)
        else:
            x, prev_y, y = np.empty(0), np.empty(0), np.empty(0)

        if self.teacher_forcing:
            return self.return_x_yy(x, prev_y, y, "Test")

        return self.return_xy(x, y, "Test")

    def check_for_batch_size(self, x, prev_y=None, y=None):

        if self.drop_remainder:

            assert isinstance(x, np.ndarray)
            remainder = len(x) % self.batch_size

            if remainder:

                x = x[0:-remainder]

                if prev_y is not None:
                    prev_y = prev_y[0:-remainder]
                if y is not None:
                    y = y[0:-remainder]

        return x, prev_y, y

    def check_nans(self, data, input_x, input_y, label_y):
        """Checks whether anns are present or not and checks shapes of arrays
        being prepared."""
        if isinstance(data, pd.DataFrame):
            nans = data[self.output_features].isna()
            nans = nans.sum().sum()
            data = data.values
        else:
            nans = np.isnan(data[:, -self.num_outs:])
            # df[self.out_cols].isna().sum()
            nans = int(nans.sum())
        if nans > 0:
            if self.allow_nan_labels == 2:
                if self.verbosity > 0: print("""
                \n{} Allowing NANs in predictions {}\n""".format(10 * '*', 10 * '*'))
            elif self.allow_nan_labels == 1:
                if self.verbosity > 0: print("""
                \n{} Ignoring examples whose all labels are NaNs {}\n
                """.format(10 * '*', 10 * '*'))
                idx = ~np.array([all([np.isnan(x) for x in label_y[i]]) for i in range(len(label_y))])
                input_x = input_x[idx]
                input_y = input_y[idx]
                label_y = label_y[idx]
                if int(np.isnan(data[:, -self.num_outs:][0:self.lookback]).sum() / self.num_outs) >= self.lookback:
                    self.nans_removed_4m_st = -9999
            else:
                if self.verbosity > 0:
                    print('\n{} Removing Examples with nan in labels  {}\n'.format(10 * '*', 10 * '*'))
                if self.num_outs == 1:
                    # find out how many nans were present from start of data until
                    # lookback, these nans will be removed
                    self.nans_removed_4m_st = np.isnan(data[:, -self.num_outs:][0:self.lookback]).sum()
                # find out such labels where 'y' has at least one nan
                nan_idx = np.array([np.any(i) for i in np.isnan(label_y)])
                non_nan_idx = np.invert(nan_idx)
                label_y = label_y[non_nan_idx]
                input_x = input_x[non_nan_idx]
                input_y = input_y[non_nan_idx]

                assert np.isnan(label_y).sum() < 1, """
                label still contains {} nans""".format(np.isnan(label_y).sum())

        assert input_x.shape[0] == input_y.shape[0] == label_y.shape[0], """
        shapes are not same"""

        if not self.allow_input_nans:
            assert np.isnan(input_x).sum() == 0, """input still contains {} nans
            """.format(np.isnan(input_x).sum())

        return input_x, input_y, label_y

    def indexify(self, data: pd.DataFrame, key):

        data = data.copy()
        dummy_index = False
        # for dataframes
        if isinstance(data.index, pd.DatetimeIndex):
            index = list(map(int, np.array(data.index.strftime('%Y%m%d%H%M'))))
            # datetime index
            self.index_types[key] = 'dt'
            original_index = pd.Series(index, index=index)
        else:
            try:
                index = list(map(int, np.array(data.index)))
                self.index_types[key] = 'int'
                original_index = pd.Series(index, index=index)
            except ValueError:  # index may not be convertible to integer, it may be
                # string values
                dummy_index = np.arange(len(data), dtype=np.int64)
                original_index = pd.Series(data.index, index=dummy_index)
                index = dummy_index
                self.index_types[key] = 'str'
                self.indexes[key] = {'dummy': dummy_index,
                                     'original': original_index}
        # pandas will add the 'datetime' column as first column.
        # This columns will only be used to keep
        # track of indices of train and test data.
        data.insert(0, 'index', index)

        self._input_features = ['index'] + self.input_features
        # setattr(self, 'input_features', ['index'] + self.input_features)
        self.indexes[key] = {'index': index, 'dummy_index': dummy_index,
                             'original': original_index}
        return data

    def deindexify(self, data: np.ndarray, key):

        _data, _index = self.deindexify_nparray(data, key)

        if self.indexes[key].get('dummy_index', None) is not None:
            _index = self.indexes[key]['original'].loc[_index].values

        if self.index_types[key] == 'dt':
            _index = to_datetime_index(_index)
        return _data, _index

    def get_batches(self, data):

        if self.batch_dim == "2D":
            return self.get_2d_batches(data)

        else:
            return self.check_nans(data, *prepare_data(data,
                                                       num_outputs=self.num_outs,
                                                       **self.ts_args))

    def get_2d_batches(self, data):
        # need to count num_ins based upon _input_features as it consider index
        num_ins = len(self._input_features)

        if not isinstance(data, np.ndarray):
            if isinstance(data, pd.DataFrame):
                data = data.values
            else:
                raise TypeError(f"unknown data type {data.__class__.__name__} for data ")

        if self.num_outs > 0:
            input_x = data[:, 0:num_ins]
            input_y, label_y = data[:, -self.num_outs:], data[:, -self.num_outs:]
        else:
            dummy_input_y = np.random.random((len(data), self.num_outs))
            dummy_y = np.random.random((len(data), self.num_outs))
            input_x, input_y, label_y = data[:, 0:num_ins], dummy_input_y, dummy_y

        assert self.lookback == 1, """
        lookback should be one for MLP/Dense layer based model, but it is {}
            """.format(self.lookback)

        return self.check_nans(data, input_x, input_y, np.expand_dims(label_y, axis=2))

    def _make_data(self, data, indices=None, intervals=None, shuffle=False):

        # if indices is not None:
        #    indices = np.array(indices).astype("int32")
        # assert isinstance(np.array(indices), np.ndarray), "indices must be array like"

        if isinstance(data, pd.DataFrame):
            data = data[self._input_features + self.output_features].copy()
            df = data
        else:
            data = data.copy()
            df = data

        if intervals is None:
            x, prev_y, y = self.get_batches(df)

            if indices is not None:
                # if indices are given then this should be done after `get_batches`
                # method
                x = x[indices]
                prev_y = prev_y[indices]
                y = y[indices]
        else:
            xs, prev_ys, ys = [], [], []
            for _st, _en in intervals:
                df1 = data[_st:_en]

                if df1.shape[0] > 0:
                    x, prev_y, y = self.get_batches(df1.values)

                    xs.append(x)
                    prev_ys.append(prev_y)
                    ys.append(y)

            if indices is None:
                x = np.vstack(xs)
                prev_y = np.vstack(prev_ys)
                y = np.vstack(ys)
            else:
                x = np.vstack(xs)[indices]
                prev_y = np.vstack(prev_ys)[indices]
                y = np.vstack(ys)[indices]

        if shuffle:
            raise NotImplementedError

        if isinstance(data, pd.DataFrame) and 'index' in data:
            data.pop('index')

        if self.ts_args['forecast_len'] == 1 and len(self.output_features) > 0:
            y = y.reshape(-1, len(self.output_features))

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

    def total_exs(self,
                  lookback,
                  forecast_step=0, forecast_len=1,
                  **ts_args
                  ):

        intervals = self.intervals
        input_steps = self.ts_args['input_steps']

        data = consider_intervals(self.data, intervals)

        num_outs = len(self.output_features) if self.output_features is not None else None

        max_tot_obs = 0
        if not self.allow_nan_labels and intervals is None:
            _data = data[self.input_features + self.output_features] if isinstance(data, pd.DataFrame) else data
            x, _, _ = prepare_data(_data,
                                   lookback, num_outputs=num_outs,
                                   forecast_step=forecast_step,
                                   forecast_len=forecast_len, mask=np.nan, **ts_args)
            max_tot_obs = len(x)

        # we need to ignore some values at the start
        more = (lookback * input_steps) - 1

        if isinstance(data, np.ndarray):
            return len(data) - more

        # todo, why not when allow_nan_labels>0?
        if forecast_step > 0:
            more += forecast_step

        if forecast_len > 1:
            more += forecast_len

        if intervals is None: intervals = [()]

        more *= len(intervals)

        if self.allow_nan_labels == 2:
            tot_obs = data.shape[0] - more

        elif self.allow_nan_labels == 1:
            label_y = data[self.output_features].values
            idx = ~np.array([all([np.isnan(x) for x in label_y[i]]) for i in range(len(label_y))])
            tot_obs = np.sum(idx) - more
        else:

            if num_outs == 1:
                tot_obs = data.shape[0] - int(data[self.output_features].isna().sum()) - more
                tot_obs = max(tot_obs, max_tot_obs)

            else:
                # count by droping all the rows when nans occur in output features
                tot_obs = len(data.dropna(subset=self.output_features))
                tot_obs -= more

        return tot_obs

    def KFold_splits(self, n_splits=5):
        """returns an iterator for kfold cross validation.

        The iterator yields two tuples of training and test x,y pairs.
        The iterator on every iteration returns following
        `(train_x, train_y), (test_x, test_y)`
        Note: only `training_data` and `validation_data` are used to make kfolds.

        Example
        ---------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from ai4water.preprocessing import DataSet
        >>> data = pd.DataFrame(np.random.randint(0, 10, (20, 3)), columns=['a', 'b', 'c'])
        >>> data_set = DataSet(data=data)
        >>> kfold_splits = data_set.KFold_splits()
        >>> for (train_x, train_y), (test_x, test_y) in kfold_splits:
        ...     print(train_x, train_y, test_x, test_y)

        """
        if self.teacher_forcing:
            warnings.warn("Ignoring prev_y")
        x, _, y = self._training_data()

        kf = KFold(n_splits=n_splits,
                   random_state=self.seed if self.shuffle else None,
                   shuffle=self.shuffle)

        spliter = kf.split(x)

        for tr_idx, test_idx in spliter:
            yield (x[tr_idx], y[tr_idx]), (x[test_idx], y[test_idx])

    def LeaveOneOut_splits(self):
        """Yields leave one out splits
        The iterator on every iteration returns following
        `(train_x, train_y), (test_x, test_y)`"""
        if self.teacher_forcing:
            warnings.warn("Ignoring prev_y")
        x, _, y = self._training_data()

        kf = LeaveOneOut()

        for tr_idx, test_idx in kf.split(x):
            yield (x[tr_idx], y[tr_idx]), (x[test_idx], y[test_idx])

    def ShuffleSplit_splits(self, **kwargs):
        """Yields ShuffleSplit splits
        The iterator on every iteration returns following
        `(train_x, train_y), (test_x, test_y)`"""
        if self.teacher_forcing:
            warnings.warn("Ignoring prev_y")
        x, _, y = self._training_data()

        sf = ShuffleSplit(**kwargs)

        for tr_idx, test_idx in sf.split(x):
            yield (x[tr_idx], y[tr_idx]), (x[test_idx], y[test_idx])

    def TimeSeriesSplit_splits(self, n_splits=5, **kwargs):
        """returns an iterator for TimeSeriesSplit.
        The iterator on every iteration returns following
        `(train_x, train_y), (test_x, test_y)`
        """
        if self.teacher_forcing:
            warnings.warn("Ignoring prev_y")
        x, _, y = self._training_data()

        tscv = TimeSeriesSplit(n_splits=n_splits, **kwargs)

        for tr_idx, test_idx in tscv.split(x):
            yield (x[tr_idx], y[tr_idx]), (x[test_idx], y[test_idx])

    def plot_KFold_splits(self, n_splits=5, show=True, **kwargs):
        """Plots the indices of kfold splits"""
        if self.teacher_forcing:
            warnings.warn("Ignoring prev_y")
        x, _, y = self._training_data()

        kf = KFold(n_splits=n_splits,
                   random_state=self.seed if self.shuffle else None,
                   shuffle=self.shuffle)

        spliter = kf.split(x)

        self._plot_splits(spliter, x, title="KFoldCV", show=show, **kwargs)

        return

    def plot_LeaveOneOut_splits(self, show=True, **kwargs):
        """Plots the indices obtained from LeaveOneOut strategy"""
        if self.teacher_forcing:
            warnings.warn("Ignoring prev_y")
        x, _, y = self._training_data()

        spliter = LeaveOneOut().split(x)

        self._plot_splits(spliter=spliter,
                          x=x,
                          title="LeaveOneOutCV",
                          show=show,
                          **kwargs)

        return

    def plot_TimeSeriesSplit_splits(self, n_splits=5, show=True, **kwargs):
        """Plots the indices obtained from TimeSeriesSplit strategy"""

        if self.teacher_forcing:
            warnings.warn("Ignoring prev_y")

        x, _, y = self._training_data()

        spliter = TimeSeriesSplit(n_splits=n_splits, **kwargs).split(x)

        self._plot_splits(spliter=spliter,
                          x=x,
                          title="TimeSeriesCV",
                          show=show,
                          **kwargs)
        return

    def _plot_splits(self, spliter, x, show=True, **kwargs):

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

        ax.set(yticks=np.arange(len(splits)) + .5, yticklabels=yticklabels)

        ax.set_xlabel("Sample Index", fontsize=18)
        ax.set_ylabel("CV iteration", fontsize=18)
        ax.set_title(title, fontsize=20)

        ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))],
                  ['Training', 'Test'],
                  loc=legend_pos, fontsize=legend_fs)

        if show:
            plt.tight_layout()
            plt.show()
        return

    def to_disk(self, path: str = None):
        import h5py

        path = path or os.getcwd()
        filepath = os.path.join(path, "data.h5")

        f = h5py.File(filepath, mode='w')

        for k, v in self.init_paras().items():
            if isinstance(v, (dict, list, tuple, float, int, str)):
                f.attrs[k] = json.dumps(
                    v, default=jsonize).encode('utf8')

            elif v is not None and k != 'data':
                f.attrs[k] = v

        if self.teacher_forcing:
            x, prev_y, y = self.training_data()
            val_x, val_prev_y, val_y = self.validation_data()
            test_x, test_prev_y, test_y = self.test_data()
        else:
            prev_y, val_prev_y, test_prev_y = np.empty(0), np.empty(0), np.empty(0)
            x, y = self.training_data()
            val_x, val_y = self.validation_data()
            test_x, test_y = self.test_data()

        # save in disk
        self._save_data_to_hdf5('training_data', x, prev_y, y, f)

        self._save_data_to_hdf5('validation_data', val_x, val_prev_y, val_y, f)

        self._save_data_to_hdf5('test_data', test_x, test_prev_y, test_y, f)

        f.close()
        return

    def _save_data_to_hdf5(self, data_type, x, prev_y, y, f):
        """Saves one data_type in h5py. data_type is string indicating whether
        it is training, validation or test data."""

        assert x is not None
        group_name = f.create_group(data_type)

        container = {}
        container['x'] = x

        if self.teacher_forcing:
            container['prev_y'] = prev_y

        container['y'] = y

        for name, val in container.items():

            param_dset = group_name.create_dataset(name, val.shape, dtype=val.dtype)
            if not val.shape:
                # scalar
                param_dset[()] = val
            else:
                param_dset[:] = val
        return

    @classmethod
    def from_h5(cls, path):
        """Creates an instance of DataSet from .h5 file."""
        import h5py

        f = h5py.File(path, mode='r')

        config = {}
        for k, v in f.attrs.items():
            if isinstance(v, str) or isinstance(v, bytes):
                v = decode(v)
            config[k] = v

        cls._from_h5 = True
        f.close()

        # the data is already being loaded from h5 file so no need to save it again
        # upon initialization of class
        config['save'] = False
        return cls(path, **config)
