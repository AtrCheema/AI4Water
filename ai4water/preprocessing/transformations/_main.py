import warnings
from typing import Union

import numpy as np
import pandas as pd

from ai4water.utils.utils import dateandtime_now, jsonize, deepcopy_dict_without_clone

from ._transformations import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from ._transformations import LogScaler, Log10Scaler, Log2Scaler, TanScaler, SqrtScaler, CumsumScaler
from ._transformations import FunctionTransformer, RobustScaler, MaxAbsScaler
from ._transformations import Center
from .utils import InvalidTransformation, TransformerNotFittedError, SP_METHODS


# TODO add logistic, tanh and more scalers.
# which transformation to use? Some related articles/posts
# https://scikit-learn.org/stable/modules/preprocessing.html
# http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html
# https://data.library.virginia.edu/interpreting-log-transformations-in-a-linear-model/


class TransformationsContainer(object):

    def __init__(self):
        self.scalers = {}
        self.transforming_straight = True
        # self.nan_indices = None
        self.index = None


INITIATED_TRANSFORMERS = {
    'log': LogScaler(),
    'log2': Log2Scaler(),
    'log10': Log10Scaler(),
    'sqrt': SqrtScaler()
}

class Transformation(TransformationsContainer):
    """
    Applies transformation to tabular data. It is also possible to apply transformation
    on some selected features/columns of data. This class also performs some optional
    pre-processing on data before applying transformation on it.
    Any new transforming methods should define two methods one starting with
    `transform_with_` and `inverse_transofrm_with_`


    Currently following methods are available for transformation and inverse transformation

    Transformation methods

        - `minmax`
        - `maxabs`
        - `robust`
        - `power` same as yeo-johnson
        - `yeo-johnson`
        - `box-cox`
        - `zscore`    also known as standard scalers
        - `scale`    division by standard deviation
        - 'center'   by subtracting mean
        - `quantile`
        - `log`      natural logrithmic
        - `log10`    log with base 10
        - `log2`  log with base 2
        - `sqrt` square root
        - `tan`      tangent
        - `cumsum`   cummulative sum

    To transform a datafrmae using any of the above methods use

    Examples:
        >>> transformer = Transformation(method='zscore')
        >>> transformer.fit_transform(data=[1,2,3,5])

        or

        >>> transformer = Transformation(method='minmax')
        >>> normalized_df = transformer.fit_transform_with_minmax(data=pd.DataFrame([1,2,3]))

        >>> transformer = Transformation(method='minmax')
        >>> normalized_df, scaler_dict = transformer(data=pd.DataFrame([1,2,3]))

        or using one liner

        >>> normalized_df = Transformation(method='minmax',
        ...                       features=['a'])(data=pd.DataFrame([[1,2],[3,4], [5,6]],
        ...                                       columns=['a', 'b']))

    where `method` can be any of the above mentioned methods.

    Note
    ------
     `tan` and `cumsum` do not return original data upon inverse transformation.

    .. _google:
        https://developers.google.com/machine-learning/data-prep/transform/normalization
    """

    available_transformers = {
        "minmax": MinMaxScaler,
        "zscore": StandardScaler,
        "center": Center,
        "scale": StandardScaler,
        "robust": RobustScaler,
        "maxabs": MaxAbsScaler,
        "power": PowerTransformer,
        "yeo-johnson": PowerTransformer,
        "box-cox": PowerTransformer,
        "quantile": QuantileTransformer,
        "log": LogScaler,
        "log10": Log10Scaler,
        "log2": Log2Scaler,
        "sqrt": SqrtScaler,
        "tan": TanScaler,
        "cumsum": CumsumScaler
    }

    def __init__(self,
                 method: str = 'minmax',
                 features: list = None,
                 replace_zeros: bool = False,
                 replace_zeros_with: Union[str, int, float] = 1,
                 treat_negatives: bool = False,
                 **kwargs
                 ):
        """
        Arguments:
            method : method by which to transform and consequencly inversely
                transform the data. default is 'minmax'. see `Transformations.available_transformers`
                for full list.
            features : string or list of strings. Only applicable if `data` is
                dataframe. It defines the columns on which we want to apply transformation.
                The remaining columns will remain same/unchanged.
            replace_zeros : If true, then setting this argument to True will replace
                the zero values in data with some fixed value `replace_zeros_with`
                before transformation. The zero values will be put back at their
                places after transformation so this replacement/implacement is
                done only to avoid error during transformation for example during Box-Cox.
            replace_zeros_with : if replace_zeros is True, then this value will be used
                to replace zeros in dataframe before doing transformation. You can
                define the method with which to replace nans for exaple by setting
                this argument to 'mean' will replace zeros with 'mean' of the
                array/column which contains zeros. Allowed string values are
                'mean', 'max', 'min'. see https://stats.stackexchange.com/a/222237/338323
            treat_negatives:
                If true, and if data contains negative values, then the absolute
                values of these negative values will be considered for transformation.
                For inverse transformation, the -ve sign is removed, to return the
                original data. This option is necessary for log, sqrt and box-cox
                transformations with -ve values in data.
            kwargs : any arguments which are to be provided to transformer on
                INTIALIZATION and not during transform or inverse transform

        Example:
            >>> from ai4water.preprocessing.transformations import Transformation
            >>> from ai4water.datasets import busan_beach
            >>> df = busan_beach()
            >>> inputs = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm']
            >>> transformer = Transformation(method='minmax', features=['sal_psu', 'air_temp_c'])
            >>> new_data = transformer.fit_transform(df[inputs])

            Following shows how to apply log transformation on an array containing zeros
            by making use of the argument `replace_zeros`. The zeros in the input array
            will be replaced internally but will be inserted back afterwards.
            >>> from ai4water.preprocessing.transformations import Transformation
            >>> transformer = Transformation(method='log', replace_zeros=True)
            >>> transformed_data = transformer.fit_transform([1,2,3,0.0, 5, np.nan, 7])
            ... [0.0, 0.6931, 1.0986, 0.0, 1.609, None, 1.9459]
            >>> original_data = transformer.inverse_transform(data=transformed_data)

        """
        super().__init__()

        if method not in self.available_transformers.keys():
            raise InvalidTransformation(method, list(self.available_transformers.keys()))

        self.method = method
        self.replace_zeros = replace_zeros
        self.replace_zeros_with = replace_zeros_with
        self.treat_negatives = treat_negatives

        self.features = features

        self.kwargs = kwargs
        self.transformed_features = None

    def __call__(self, data, what="fit_transform", return_key=False, **kwargs):
        """
        Calls the `fit_transform` and `inverse_transform` methods.
        """
        if what.startswith("fit"):
            self.transforming_straight = True

            return self.fit_transform(data, return_key=return_key, **kwargs)

        elif what.startswith("inv"):
            self.transforming_straight = False
            return self.inverse_transform(data, **kwargs)

        else:
            raise ValueError(f"The class Transformation can not be called with keyword argument 'what'={what}")

    def __getattr__(self, item):
        """
        Gets the attributes from underlying transformation modules.
        """
        if item.startswith('_'):
            return self.__getattribute__(item)
        elif item.startswith("fit_transform_with"):
            transformer = item.split('_')[-1]
            if transformer.lower() in list(self.available_transformers.keys()):
                self.method = transformer
                return self.fit_transform_with_sklearn

        elif item.startswith("inverse_transform_with"):
            transformer = item.split('_')[-1]
            if transformer.lower() in list(self.available_transformers.keys()):
                self.method = transformer
                return self.inverse_transform_with_sklearn
        else:
            raise AttributeError(f'Transformation has no attribute {item}')

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, x):
        if isinstance(x, pd.DataFrame):
            self._data = x
        else:
            assert isinstance(x, np.ndarray)
            xdf = pd.DataFrame(x, columns=['data'+str(i) for i in range(x.shape[1])])
            self._data = xdf

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, x):
        if x is not None:
            assert len(x) == len(set(x)), f"duplicated features are not allowed. Features are: {x}"
        self._features = x

    @property
    def transformed_features(self):
        return self._transformed_features

    @transformed_features.setter
    def transformed_features(self, x):
        self._transformed_features = x

    @property
    def num_features(self):
        return len(self.features)

    def get_scaler(self):

        return self.available_transformers[self.method.lower()]

    def pre_process_data(self, data):
        """Makes sure that data is dataframe and optionally replaces nans"""
        data = to_dataframe(data)

        # save the index if not already saved so that can be used later
        if self.index is None:
            self.index = data.index

        indices = {}
        if self.replace_zeros and self.transforming_straight:
            # instead of saving indices with column names, using column indices
            # because df.iloc[row_idx, col_idx] is better than df[col_name].iloc[row_idx]
            for col_idx, col in enumerate(data.columns):
                # find index containing 0s in corrent column of dataframe
                i = data.index[data[col] == 0.0]
                if len(i) > 0:
                    indices[col_idx] = i.values
                    if self.replace_zeros_with in ['mean', 'max', 'min']:
                        replace_with = float(getattr(np, 'nan' + self.replace_zeros_with)(data[col]))
                    else:
                        replace_with = self.replace_zeros_with
                    data.iloc[indices[col_idx], col_idx] = get_val(data[col], replace_with)

            #if self.zero_indices is None:
        self.zero_indices_ = indices

        indices = {}
        if self.treat_negatives:
            for col_idx, col in enumerate(data.columns):
                # find index containing negatives in corrent column of dataframe
                i = data.index[data[col] < 0.0]
                if len(i) > 0:
                    indices[col_idx] = i.values
                    # turn -ve values into positives
                    data[col] = data[col].abs()

        self.negative_indices_ = indices

        return data

    def post_process_data(self, data):
        """If nans/zeros were replaced with some value, put nans/zeros back."""
        data = data.copy()
        if self.replace_zeros:
            if hasattr(self, 'zero_indices_'):
                for col, idx in self.zero_indices_.items():
                    data.iloc[idx, col] = 0.0

        if self.treat_negatives:
            if hasattr(self, 'negative_indices_'):
                for col, idx in self.negative_indices_.items():
                    # invert the sign of those values which were originally -ve
                    data.iloc[idx, col] = -data.iloc[idx, col]
        return data

    def fit_transform_with_sklearn(
            self,
            data:Union[pd.DataFrame, np.ndarray],
            return_key=False,
            **kwargs):

        original_data = data.copy()
        to_transform = self.get_features(data)  # TODO, shouldn't kwargs go here as input?

        if self.method.lower() in ["log", "log10", "log2"]:

            if (to_transform.values < 0).any():
                raise InvalidValueError(self.method, "negative")

        _kwargs = {}
        if self.method == "scale":
            _kwargs['with_mean'] = False
        elif self.method in ['power', 'yeo-johnson', 'box-cox']:
            # a = np.array([87.52, 89.41, 89.4, 89.23, 89.92], dtype=np.float32).reshape(-1,1)
            # power transformers sometimes overflow with small data which causes inf error
            to_transform = to_transform.astype("float64")
            if self.method == "box-cox":
                _kwargs['method'] = "box-cox"

        for k,v in self.kwargs.items():
            if k in _kwargs:
                _kwargs.pop(k)

        scaler = self.get_scaler()(**_kwargs, **self.kwargs)

        data = scaler.fit_transform(to_transform, **kwargs)

        data = pd.DataFrame(data, columns=to_transform.columns)

        scaler = self.serialize_scaler(scaler, to_transform)

        data = self.maybe_insert_features(original_data, data)

        data = self.post_process_data(data)

        self.tr_data = data
        if return_key:
            return data, scaler
        return data

    def inverse_transform_with_sklearn(self,
                                       data,
                                       postprocess=True,
                                       without_fit=False,
                                       **kwargs):

        self.transforming_straight = False

        scaler = self.get_scaler_from_dict(**kwargs)

        original_data = data.copy()
        to_transform = self.get_features(data)

        if without_fit:
            data = scaler.inverse_transform_without_fit(to_transform)
        else:
            data = scaler.inverse_transform(to_transform)

        data = pd.DataFrame(data, columns=to_transform.columns)

        data = self.maybe_insert_features(original_data, data)

        if postprocess:
            data = self.post_process_data(data)

        return data

    def fit_transform(self, data, return_key=False, **kwargs):
        """
        Transforms the data

        Arguments:
            data : a dataframe or numpy ndarray or array like. The transformed or inversely
                transformed value will have the same type as data and will have
                the same index as data (in case data is dataframe). The shape of
                `data` is supposed to be (num_examples, num_features).
            return_key : whether to return the scaler or not. If True, then a
                tuple is returned which consists of transformed data and scaler itself.
            kwargs :
        """
        self.transforming_straight = True

        data = self.pre_process_data(data.copy())

        if self.features is None:
            self.features = list(data.columns)

        setattr(self, 'initial_shape_', data.shape)

        return getattr(self, "fit_transform_with_" + self.method)(data, return_key=return_key, **kwargs)

    def inverse_transform(self,
                          data,
                          postprocess=True,
                          without_fit=False,
                          **kwargs):
        """
        Inverse transforms the data.

        Parameters
        ---------
            data:
            postprocess : bool
            without_fit : bool
            kwargs : any of the folliwng keyword arguments

            - data: data on which to apply inverse transformation
            - key : key to fetch scaler
            - scaler : scaler to use for inverse transformation. If not given, then
                the available scaler is used.
        """
        self.transforming_straight = False

        # during transform, we convert to df even when input is list or np array
        # which inserts columns/features into data.
        data = to_dataframe(data)

        if 'key' in kwargs or 'scaler' in kwargs:
            pass
        elif len(self.scalers) == 1:
            kwargs['scaler'] = list(self.scalers.values())[0]
        elif self.method in SP_METHODS:
            kwargs['scaler'] = INITIATED_TRANSFORMERS[self.method]
            without_fit = True
        else:
            raise TransformerNotFittedError()

        if self.treat_negatives and hasattr(self, "negative_indices_"):
            for col, idx in self.negative_indices_.items():
                data.iloc[idx, col] = -data.iloc[idx, col]

        return getattr(self, "inverse_transform_with_" + self.method.lower())(
            data,
            postprocess=postprocess,
            without_fit=without_fit,
            **kwargs)

    def get_features(self, data) -> pd.DataFrame:

        if self.features is None:
            return data
        else:
            assert isinstance(self.features, list)
            return data[self.features]

    def serialize_scaler(self, scaler, to_transform):
        key = self.method + str(dateandtime_now())
        serialized_scaler = {
            "scaler": scaler,
            "shape": to_transform.shape,
            "key": key
        }
        self.scalers[key] = scaler

        return serialized_scaler

    def get_scaler_from_dict(self, **kwargs):
        if 'scaler' in kwargs:
            scaler = kwargs['scaler']
        elif 'key' in kwargs:
            scaler = self.scalers[kwargs['key']]
        else:
            raise TransformerNotFittedError()
        return scaler

    def maybe_insert_features(self, original_df, trans_df):

        if self.index is not None:
            # the data on which to perform inverse transformation has different
            # length so can't insert index, therefore make sure that negatives and
            # zeros were not treated beccause their indexes do not remain same now
            if len(trans_df) != len(self.index):
                assert len(self.zero_indices_) == 0
                assert len(self.negative_indices_) == 0
            else:
                trans_df.index = self.index

        num_features = len(original_df.columns)
        if len(trans_df.columns) != num_features:
            df = pd.DataFrame(index=self.index)
            for col in original_df.columns:  # features:
                if col in trans_df.columns:
                    _df = trans_df[col]
                else:
                    _df = original_df[col]

                df = pd.concat([df, _df], axis=1)
        else:
            df = trans_df

        assert df.shape == original_df.shape, f"shape changed from {original_df.shape} to {df.shape}"
        return df

    def config(self)->dict:
        """returns a dictionary which can be used to reconstruct `Transformation`
        class using `from_config`.
        Returns:
            a dictionary
        """
        assert len(self.scalers) > 0, f"Transformation is not fitted yet"

        assert len(self.scalers) == 1
        scaler = list(self.scalers.values())[0].config()

        return {
            "scaler": {self.method: scaler},
            "shape": self.initial_shape_,
            "method": self.method,
            "features": self.features,
            "replace_zeros": self.replace_zeros,
            "replace_zeros_with": self.replace_zeros_with,
            "treat_negatives": self.treat_negatives,
            "kwargs": self.kwargs,
            "negative_indices_": jsonize(self.negative_indices_),
            "zero_indices_": jsonize(self.zero_indices_)
        }


    @classmethod
    def from_config(
            cls,
            config:dict
    )-> "Transformation":
        """constructs the `Transformation` class from `config` which has
        already been fitted/transformed.
        Arguments:
            config:
                a dicionary which is the output of `config()` method.
        Returns:
            an instance of `Transformation` class.
        """
        config = deepcopy_dict_without_clone(config)

        shape = config.pop('shape')
        scaler = config.pop('scaler')
        negative_indices_ = config.pop("negative_indices_")
        zero_indices_ = config.pop("zero_indices_")

        assert len(scaler) == 1
        scaler_name = list(scaler.keys())[0]
        scaler_config = list(scaler.values())[0]

        transformer = cls(**config)

        # initiate the scalers
        sc_initiated = transformer.available_transformers[scaler_name].from_config(scaler_config)


        transformer.scalers = {scaler_name: sc_initiated}
        transformer.negative_indices_ = negative_indices_
        transformer.zero_indices_ = zero_indices_
        transformer.initial_shape_ = shape

        return transformer


def get_val(df: pd.DataFrame, method):

    if isinstance(method, str):
        if method.lower() == "mean":
            return df.mean()
        elif method.lower() == "max":
            return df.max()
        elif method.lower() == "min":
            return df.min()
    elif isinstance(method, int) or isinstance(method, float):
        return method
    else:
        raise ValueError(f"unknown method {method} to replace nan vlaues")


class InvalidValueError(Exception):
    def __init__(self, method, reason):
        self.method = method
        self.reason = reason

    def remedy(self):
        if self.reason == "NaN":
            return "Try setting 'replace_nans' to True"
        elif self.reason == "zero":
            return "Try setting 'replace_zeros' to True"
        elif self.reason == "negative":
            return "Try setting 'treat_negatives' to True"

    def __str__(self):
        return (f"""
Input data contains {self.reason} values so {self.method} transformation
can not be applied.
{self.remedy()}
""")


def to_dataframe(data)->pd.DataFrame:

    if isinstance(data, pd.DataFrame):
        data = data
    else:
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        assert isinstance(data, np.ndarray)
        data = pd.DataFrame(data, columns=['data' + str(i) for i in range(data.shape[1])])
    return data
