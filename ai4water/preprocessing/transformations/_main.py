
from typing import Union

from ai4water.backend import np, pd
from ai4water.utils.utils import dateandtime_now, deepcopy_dict_without_clone

from ._transformations import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from ._transformations import LogScaler, Log10Scaler, Log2Scaler, TanScaler, SqrtScaler, CumsumScaler
from ._transformations import FunctionTransformer, RobustScaler, MaxAbsScaler
from ._transformations import Center
from .utils import InvalidTransformation, TransformerNotFittedError, SP_METHODS


# TODO add logistic, tanh and more transformers.
# which transformation to use? Some related articles/posts
# https://scikit-learn.org/stable/modules/preprocessing.html
# http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html
# https://data.library.virginia.edu/interpreting-log-transformations-in-a-linear-model/


class TransformationsContainer(object):

    def __init__(self):
        self.transformer_ = None
        self.transforming_straight = True
        self.index = None


INITIATED_TRANSFORMERS = {
    'log': LogScaler(),
    'log2': Log2Scaler(),
    'log10': Log10Scaler(),
    'sqrt': SqrtScaler()
}

class _Processor(object):

    def __init__(self,
                 replace_zeros,
                 replace_zeros_with,
                 treat_negatives,
                 features=None
                 ):
        self.replace_zeros = replace_zeros
        self.replace_zeros_with = replace_zeros_with
        self.treat_negatives = treat_negatives
        self.features = features

        self.index = None

    def preprocess(self, data, transforming_straight=True):
        """Makes sure that data is dataframe and optionally replaces nans"""
        data = to_dataframe(data)

        # save the index if not already saved so that can be used later
        if self.index is None:
            self.index = data.index

        columns = self.features or data.columns

        indices = {}
        if self.replace_zeros and transforming_straight:
            # instead of saving indices with column names, using column indices
            # because df.iloc[row_idx, col_idx] is better than df[col_name].iloc[row_idx]
            for col_idx, col in enumerate(columns):
                # find index containing 0s in corrent column of dataframe
                i = data.index[data[col] == 0.0]
                if len(i) > 0:
                    indices[col_idx] = i.values
                    if self.replace_zeros_with in ['mean', 'max', 'min']:
                        replace_with = float(getattr(np, 'nan' + self.replace_zeros_with)(data[col]))
                    else:
                        replace_with = self.replace_zeros_with
                    data.loc[indices[col_idx], col] = get_val(data[col], replace_with)

            #if self.zero_indices is None:
        self.zero_indices_ = indices

        indices = {}
        if self.treat_negatives:
            for col_idx, col in enumerate(columns):
                # find index containing negatives in corrent column of dataframe
                i = data.index[data[col] < 0.0]
                if len(i) > 0:
                    indices[col_idx] = i.values
                    # turn -ve values into positives
                    data[col] = data[col].abs()

        self.negative_indices_ = indices

        return data

    def postprocess(self, data):
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
                    for _idx in idx:
                        data.iat[_idx, col] = -data.iat[_idx, col]
        return data

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
        -  `quantile_normal` quantile with normal distribution as target
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
        >>> normalized_df = transformer.fit_transform(data=pd.DataFrame([1,2,3]))

        >>> transformer = Transformation(method='log', replace_zeros=True)
        >>> trans_df, proc = transformer.fit_transform(data=pd.DataFrame([1,0,2,3]),
        >>>                                                 return_proc=True)
        >>> detransfomred_df = transformer.inverse_transform(trans_df, postprocessor=proc)

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
        "quantile_normal": QuantileTransformer,
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
                'mean', 'max', 'min'. see_
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

        .. _see:
            https://stats.stackexchange.com/a/222237/338323

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

        if self.transformer_ is None:  # self.transformer_ can be set during from_config
            _kwargs = {}
            if self.method == "scale":
                _kwargs['with_mean'] = False

            elif self.method == "box-cox":
                _kwargs['method'] = "box-cox"

            elif self.method == "quantile_normal":
                _kwargs["output_distribution"] = "normal"

            for k,v in self.kwargs.items():
                if k in _kwargs:
                    _kwargs.pop(k)

            transformer = self.get_transformer()(**_kwargs, **kwargs)
            self.transformer_ = transformer

    def __call__(self, data, what="fit_transform", return_proc=False, **kwargs):
        """
        Calls the `fit_transform` and `inverse_transform` methods.
        """
        if what.startswith("fit"):
            self.transforming_straight = True

            return self.fit_transform(data, return_proc=return_proc, **kwargs)

        elif what.startswith("inv"):
            self.transforming_straight = False
            return self.inverse_transform(data, **kwargs)

        else:
            raise ValueError(f"The class Transformation can not be called with keyword argument 'what'={what}")

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

    def get_transformer(self):

        return self.available_transformers[self.method.lower()]

    def _preprocess(self, data):
        self.transforming_straight = True

        proc = _Processor(self.replace_zeros,
                          self.replace_zeros_with,
                          self.treat_negatives,
                          features=self.features
                          )

        data = proc.preprocess(data.copy())

        if self.features is None:
            self.features = list(data.columns)

        setattr(self, 'initial_shape_', data.shape)

        to_transform = self.get_features(data)

        if self.method.lower() in ["log", "log10", "log2"]:

            if (to_transform.values < 0).any():
                raise InvalidValueError(self.method, "negative")

        return to_transform, proc

    def fit(self, data, **kwargs):
        """fits the data according the transformation methods."""

        to_transform, proc = self._preprocess(data)

        if self.method in ['power', 'yeo-johnson', 'box-cox']:
            # a = np.array([87.52, 89.41, 89.4, 89.23, 89.92], dtype=np.float32).reshape(-1,1)
            # power transformers sometimes overflow with small data which causes inf error
            to_transform = to_transform.astype("float64")

        return self.transformer_.fit(to_transform.values, **kwargs)

    def transform(self, data, return_proc=False, **kwargs):
        """transforms the data according to fitted transformers."""

        original_data = to_dataframe(data.copy())

        to_transform, proc = self._preprocess(data)

        if self.method in ['power', 'yeo-johnson', 'box-cox']:
            # a = np.array([87.52, 89.41, 89.4, 89.23, 89.92], dtype=np.float32).reshape(-1,1)
            # power transformers sometimes overflow with small data which causes inf error
            to_transform = to_transform.astype("float64")

        data = self.transformer_.transform(to_transform.values, **kwargs)

        return self._postprocess(data, to_transform, original_data, proc, return_proc)

    def fit_transform(self, data, return_proc=False, **kwargs):
        """
        Transforms the data

        Arguments:
            data : a dataframe or numpy ndarray or array like. The transformed or inversely
                transformed value will have the same type as data and will have
                the same index as data (in case data is dataframe). The shape of
                `data` is supposed to be (num_examples, num_features).
            return_proc : whether to return the processer or not. If True, then a
                tuple is returned which consists of transformed data and second is the preprocessor.
            kwargs :
        """
        original_data = to_dataframe(data.copy())

        to_transform, proc = self._preprocess(data)

        data = self.transformer_.fit_transform(to_transform.values, **kwargs)

        return self._postprocess(data, to_transform, original_data, proc, return_proc)

    def _postprocess(self, data, to_transform, original_data, proc, return_proc):

        data = pd.DataFrame(data, columns=to_transform.columns)

        data = self.maybe_insert_features(original_data, data)

        data = proc.postprocess(data)

        if return_proc:
            return data, proc
        return data

    def inverse_transform(self,
                          data,
                          postprocessor:_Processor=None,
                          without_fit=False,
                          **kwargs):
        """
        Inverse transforms the data.

        Parameters
        ---------
            data:
            postprocessor :
            without_fit : bool
            kwargs : any of the folliwng keyword arguments

            - data: data on which to apply inverse transformation
            - key : key to fetch transformer
            - transformer : transformer to use for inverse transformation. If not given, then
                the available transformer is used.
        """
        self.transforming_straight = False

        # during transform, we convert to df even when input is list or np array
        # which inserts columns/features into data.
        data = to_dataframe(data)

        if self.treat_negatives and hasattr(postprocessor, "negative_indices_"):
            for col, idx in postprocessor.negative_indices_.items():
                data.iloc[idx, col] = -data.iloc[idx, col]

        if 'transformer' in kwargs:
            transformer = kwargs['transformer']
        elif self.transformer_ is not None:
            transformer = self.transformer_
        elif self.method in SP_METHODS:
            transformer = INITIATED_TRANSFORMERS[self.method]
            without_fit = True
        else:
            raise TransformerNotFittedError()

        if self.treat_negatives and hasattr(self, "negative_indices_"):
            for col, idx in self.negative_indices_.items():
                data.iloc[idx, col] = -data.iloc[idx, col]

        self.transforming_straight = False

        original_data = data.copy()
        to_transform = self.get_features(data)

        if without_fit:
            data = transformer.inverse_transform_without_fit(to_transform)
        else:
            data = transformer.inverse_transform(to_transform.values)

        data = pd.DataFrame(data, columns=to_transform.columns)

        data = self.maybe_insert_features(original_data, data)

        if postprocessor is not None:
            data = postprocessor.postprocess(data)

        return data

    def get_features(self, data) -> pd.DataFrame:

        if self.features is None:
            return data
        else:
            assert isinstance(self.features, list)
            return data[self.features]

    def serialize_transformer(self, transformer):
        key = self.method + str(dateandtime_now())
        serialized_transformer = {
            "transformer": transformer,
            "key": key
        }
        self.transformer_ = transformer

        return serialized_transformer

    def get_transformer_from_dict(self, **kwargs):
        if 'transformer' in kwargs:
            transformer = kwargs['transformer']
        else:
            raise TransformerNotFittedError()
        return transformer

    def maybe_insert_features(self, original_df, trans_df):

        trans_df.index = original_df.index

        num_features = len(original_df.columns)
        if len(trans_df.columns) != num_features:
            df = pd.DataFrame(index=original_df.index)
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
        assert self.transformer_ is not None, f"Transformation is not fitted yet"

        return {
            "transformer": {self.method: self.transformer_.config()},
            "shape": self.initial_shape_,
            "method": self.method,
            "features": self.features,
            "replace_zeros": self.replace_zeros,
            "replace_zeros_with": self.replace_zeros_with,
            "treat_negatives": self.treat_negatives,
            "kwargs": self.kwargs,
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
        transformer = config.pop('transformer')

        assert len(transformer) == 1
        transformer_name = list(transformer.keys())[0]
        transformer_config = list(transformer.values())[0]

        if 'kwargs' in config:
            kwargs = config.pop('kwargs')
        transformer = cls(**config, **kwargs)

        # initiate the transformer
        tr_initiated = transformer.available_transformers[transformer_name].from_config(transformer_config)

        transformer.transformer_ = tr_initiated
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
        data = pd.DataFrame(data, #columns=['data' + str(i) for i in range(data.shape[1])]
                            )
    return data
