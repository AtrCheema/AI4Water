
from typing import List

from scipy.special import boxcox
from scipy.special import expit

from ai4water.backend import np, sklearn
from ai4water.utils.utils import jsonize

SKMinMaxScaler = sklearn.preprocessing.MinMaxScaler
SKStandardScaler = sklearn.preprocessing.StandardScaler
SKRobustScaler = sklearn.preprocessing.RobustScaler
SKPowerTransformer = sklearn.preprocessing.PowerTransformer
SKQuantileTransformer = sklearn.preprocessing.QuantileTransformer
SKFunctionTransformer = sklearn.preprocessing.FunctionTransformer
SKMaxAbsScaler = sklearn.preprocessing.MaxAbsScaler
check_is_fitted = sklearn.utils.validation.check_is_fitted

# todo
# inverse hyperbolic transformation: effective with many zeros

class ScalerWithConfig(object):
    """Extends the sklearn's scalers in such a way that they can be
    saved to a json file an d loaded from a json file

    Methods
    --------
        - config
        - form_config
    """

    @property
    def config_paras(self) -> list:
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict):
        """Build the scaler/transformer from config

        Arguments:
            config : dictionary of parameters which can be used to build transformer/scaler.

        Returns :
            An instance of scaler/transformer
        """
        scaler = cls(**config['params'])
        setattr(scaler, '_config', config['config'])
        setattr(scaler, '_from_config', True)
        for attr, attr_val in config['config'].items():
            setattr(scaler, attr, attr_val)
        return scaler

    def config(self) -> dict:
        """Returns all the parameters in scaler/transformer in a dictionary"""
        if self.__class__.__name__ == 'MyFunctionTransformer':
            pass
        else:
            check_is_fitted(self)

        _config = {}
        for attr in self.config_paras:
            _config[attr] = getattr(self, attr)

        return {"params": self.get_params(),
                "config": _config}


class MinMaxScaler(SKMinMaxScaler, ScalerWithConfig):

    @property
    def config_paras(self):
        return ['scale_', 'min_', 'n_samples_seen_', 'data_min_', 'data_max_', 'data_range_']


class StandardScaler(SKStandardScaler, ScalerWithConfig):

    @property
    def config_paras(self):
        return ['scale_', 'n_samples_seen_', 'mean_', 'var_', 'n_features_in_']


class RobustScaler(SKRobustScaler, ScalerWithConfig):

    @property
    def config_paras(self):
        return ['scale_', 'center_']


class PowerTransformer(SKPowerTransformer, ScalerWithConfig):
    """This transformation enhances scikit-learn's PowerTransformer by allowing
    the user to define `lambdas` parameter for each input feature. The default
    behaviour of this transformer is same as that of scikit-learn's.
    """
    def __init__(self, method='yeo-johnson', *,
                 rescale=False,
                 pre_center:bool = False,
                 standardize=True,
                 copy=True,
                 lambdas=None):
        """
        lambdas: float or 1d array like for each feature. If not given, it is
            calculated from scipy.stats.boxcox(X, lmbda=None). Only available
            if method is box-cox.
        pre_center:
            center the data before applying power transformation. see github [1] for more discussion
        rescale:
        For complete documentation see scikit-learn's documentation [2]

        .. [2]
            https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html

        .. [1]
            https://github.com/scikit-learn/scikit-learn/issues/14959
        """
        if lambdas is not None:
            if isinstance(lambdas, float):
                lambdas = np.array([lambdas])
            lambdas = np.array(lambdas)
            # if given, lambdas must be a 1d array
            assert lambdas.size == len(lambdas)
            lambdas = lambdas.reshape(-1,)
            assert method != "yeo-johnson"

        self.lambdas = lambdas
        self.rescale = rescale
        self.pre_center = pre_center

        super(PowerTransformer, self).__init__(method=method,
                                               standardize=standardize,
                                               copy=copy)

    @property
    def config_paras(self):
        return ['lambdas_', 'scaler_to_standardize_',
                'pre_center_config_', 'rescaler_config_', 'n_features_in_']

    @classmethod
    def from_config(cls, config: dict):
        """Build the scaler/transformer from config

        Arguments:
            config : dictionary of parameters which can be used to build transformer/scaler.

        Returns :
            An instance of scaler/transformer
        """
        scaler = cls(**config['params'])
        setattr(scaler, '_config', config['config'])
        setattr(scaler, '_from_config', True)

        _scaler_config = config['config'].pop('scaler_to_standardize_')
        setattr(scaler, '_scaler', StandardScaler.from_config(_scaler_config))

        rescaler = config['config'].pop('rescaler_config_')
        if rescaler:
            setattr(scaler, 'rescaler_', MinMaxScaler.from_config(rescaler))
        else:
            setattr(scaler, 'rescaler_', None)

        pre_standardizer = config['config'].pop('pre_center_config_')
        if pre_standardizer:
            setattr(scaler, 'pre_centerer_', Center.from_config(pre_standardizer))
        else:
            setattr(scaler, 'pre_centerer_', None)

        for attr, attr_val in config['config'].items():
            setattr(scaler, attr, attr_val)

        if isinstance(scaler.lambdas_, float):
            scaler.lambdas_ = [scaler.lambdas_]
        return scaler

    def _fit(self, X, y=None, force_transform=False):
        """copying from sklearn because we want to use our own StandardScaler
        which can be serialzied. and optionally with user provided with lambda
        parameter."""
        X = self._check_input(X, in_fit=True, check_positive=True,
                              check_method=True)

        if not self.copy and not force_transform:  # if call from fit()
            X = X.copy()  # force copy so that fit does not change X inplace

        X = self._maybe_rescale(X, force_transform)

        X = self._maybe_precenter(X, force_transform)

        optim_function = {'box-cox': self._box_cox_optimize,
                          'yeo-johnson': self._yeo_johnson_optimize
                          }[self.method]
        if self.lambdas is None:
            with np.errstate(invalid='ignore'):  # hide NaN warnings
                self.lambdas_ = np.array([optim_function(col) for col in X.T])
        else:  # take user defined lambdas
            self.lambdas_ = self.lambdas

        if self.standardize or force_transform:
            transform_function = {'box-cox': boxcox,
                                  'yeo-johnson': self._yeo_johnson_transform
                                  }[self.method]
            for i, lmbda in enumerate(self.lambdas_):
                with np.errstate(invalid='ignore'):  # hide NaN warnings
                    X[:, i] = transform_function(X[:, i], lmbda)

        setattr(self, 'scaler_to_standardize_', None)
        if self.standardize:
            self._scaler = StandardScaler(copy=False)
            if force_transform:
                X = self._scaler.fit_transform(X)
            else:
                self._scaler.fit(X)

            setattr(self, 'scaler_to_standardize_', self._scaler.config())

        return X

    def _maybe_rescale(self, X, force_transform):
        self.rescaler_config_ = None
        if self.rescale:
            rescaler = MinMaxScaler()
            self.rescaler_ = rescaler
            if force_transform:
                X = rescaler.fit_transform(X)
            else:
                X = rescaler.fit(X)

            self.rescaler_config_ = rescaler.config()
        return X

    def _maybe_precenter(self, X, force_transform=False):
        self.pre_center_config_ = None
        if self.pre_center:
            pre_centerer = Center()
            self.pre_centerer_ = pre_centerer
            if force_transform:
                X = pre_centerer.fit_transform(X)
            else:
                X = pre_centerer.fit(X)

            self.pre_center_config_ = pre_centerer.config()
        return X

    def inverse_transform(self, X):
        """Apply the inverse power transformation using the fitted lambdas.

        The inverse of the Box-Cox transformation is given by::

            if lambda_ == 0:
                X = exp(X_trans)
            else:
                X = (X_trans * lambda_ + 1) ** (1 / lambda_)

        The inverse of the Yeo-Johnson transformation is given by::

            if X >= 0 and lambda_ == 0:
                X = exp(X_trans) - 1
            elif X >= 0 and lambda_ != 0:
                X = (X_trans * lambda_ + 1) ** (1 / lambda_) - 1
            elif X < 0 and lambda_ != 2:
                X = 1 - (-(2 - lambda_) * X_trans + 1) ** (1 / (2 - lambda_))
            elif X < 0 and lambda_ == 2:
                X = 1 - exp(-X_trans)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The transformed data.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            The original data.
        """
        X = super(PowerTransformer, self).inverse_transform(X)

        if self.pre_center:
            X = self.pre_centerer_.inverse_transform(X)

        if self.rescale:
            X = self.rescaler_.inverse_transform(X)

        return X

class QuantileTransformer(SKQuantileTransformer, ScalerWithConfig):

    @property
    def config_paras(self):
        return ['n_quantiles_', 'references_', 'quantiles_']

    @classmethod
    def from_config(cls, config: dict):
        """Build the scaler/transformer from config

        Arguments:
            config : dictionary of parameters which can be used to build transformer/scaler.

        Returns :
            An instance of scaler/transformer
        """
        scaler = cls(**config['params'])
        setattr(scaler, '_config', config['config'])
        setattr(scaler, '_from_config', True)

        scaler.n_quantiles_ = config['config']['n_quantiles_']
        scaler.references_ = np.array(config['config']['references_'])
        quantiles_ = np.array(config['config']['quantiles_'])
        # make sure it is 2d
        quantiles_ = quantiles_.reshape(len(quantiles_), -1)
        scaler.quantiles_ = quantiles_
        return scaler


class MaxAbsScaler(SKMaxAbsScaler, ScalerWithConfig):

    @property
    def config_paras(self):
        return ['scale_', 'n_samples_seen_', 'max_abs_']


class Center(ScalerWithConfig):

    def __init__(
            self,
            feature_dim="2d",
            axis=0
    ):
        self.feature_dim = feature_dim
        self.axis = axis

    def fit(self, x:np.ndarray):
        dim = x.ndim

        mean = np.nanmean(x, axis=self.axis)

        setattr(self, 'mean_', mean)
        setattr(self, 'data_dim_', dim)
        return x

    def transform(self, x):

        return x - self.mean_

    def fit_transform(self, x:np.ndarray)->np.ndarray:

        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x:np.ndarray)->np.ndarray:

        assert x.ndim == self.data_dim_
        return x + self.mean_

    @property
    def config_paras(self):
        return ['data_dim_', 'mean_']

    def get_params(self):
        return {'feature_dim': self.feature_dim, 'axis': self.axis}

class Closures(ScalerWithConfig):

    def __init__(
            self,
            force_closure:bool = False,
            treat_negative:bool = False
    ):

        """
        force_closure: bool
            if ture, and input data is not a closure, it will be converted
            into closure by dividing with the sum of input data
        """
        self.force_closure = force_closure
        self.treat_negative = treat_negative

    def _check_array(self, x):

        self.sum_, self.min_ = None, None

        if len(x) == x.size:
            x = x.reshape(-1,)

        if (x<0).sum() > 0:
            if self.treat_negative:
                self.min_ = np.min(x)
                x = x + self.min_
            else:
                ValueError(f"x contains {(x[x<0]).sum()} -ve values")

        if not np.allclose(x.sum(), 1.0):
            if self.force_closure:
                self.sum_ = np.sum(x)
                x = x / self.sum_
            else:
                raise ValueError(f"x is not a closure with sum of {round(x.sum(), 5)}")
        return x

    def _check_array_inv(self, x):

        if len(x) == x.size:
            x = x.reshape(-1,)

        if self.force_closure:
            x = x * self.sum_

        if self.treat_negative:
            x = x - self.min_

        return x

    def transform(self, x):
        raise NotImplementedError

    def fit(self, x):
        return x

    def fit_transform(self, x):
        return self.transform(x)

    @property
    def config_paras(self)->List[str]:
        return ['sum_', 'min_']

    def get_params(self)->dict:
        return {
            'force_closure': self.force_closure,
            'treat_negative': self.treat_negative,
        }


class ALR(Closures):
    """
    Additive log ratio transformation

    Examples
    ---------
    >>> from easy_mpl import hist
    >>> from ai4water.datasets import busan_beach
    >>> data = busan_beach()

    >>> alr_tr = ALR(True, True)
    >>> x = data.iloc[:, 0].values
    >>> x_ = alr_tr.fit_transform(x)
    >>> _x = alr_tr.inverse_transform(x_)
    >>> np.allclose(_x, x)
    True
    >>> hist([x, x_], hist_kws={"bins": 100}, share_axes=False,
    ...   labels=["Original", "Transformed"])
    """
    def transform(self, x):

        denominator_idx = 0
        x = self._check_array(x)
        self.x0_ = x[denominator_idx]

        if x.ndim == 2:
            mat_t = x.T
            numerator_idx = list(range(0, mat_t.shape[0]))
            del numerator_idx[denominator_idx]
            x = np.log(mat_t[numerator_idx, :] / mat_t[denominator_idx, :]).T

        elif x.ndim == 1:
            numerator_idx = list(range(0, x.shape[0]))
            del numerator_idx[denominator_idx]
            x = np.log(x[numerator_idx] / x[denominator_idx])
            x = np.roll(np.append(x, self.x0_), shift=1)

        else:
            raise ValueError("mat must be either 1D or 2D")
        return x

    def inverse_transform(self, x):

        denominator_idx = 0
        x = np.array(x)

        if x.ndim == 2:
            mat_idx = np.insert(x, denominator_idx,
                                np.repeat(0, x.shape[0]), axis=1)
            comp = np.zeros(mat_idx.shape)
            comp[:, denominator_idx] = 1 / (np.exp(x).sum(axis=1) + 1)
            numerator_idx = list(range(0, comp.shape[1]))
            del numerator_idx[denominator_idx]
            for i in numerator_idx:
                comp[:, i] = comp[:, denominator_idx] * np.exp(mat_idx[:, i])
        elif x.ndim == 1:
            mat_idx = np.insert(x, denominator_idx, 0, axis=0)
            comp = np.zeros(mat_idx.shape)
            comp[denominator_idx] = 1 / (np.exp(x).sum(axis=0) + 1)
            numerator_idx = list(range(0, comp.shape[0]))
            del numerator_idx[denominator_idx]
            for i in numerator_idx:
                comp[i] = comp[denominator_idx] * np.exp(mat_idx[i])
        else:
            raise ValueError("mat must be either 1D or 2D")

        x = self._check_array_inv(x)

        return x


class CLR(Closures):
    """centre log ratio transformation

    Examples
    ---------
    >>> from easy_mpl import hist
    >>> from ai4water.datasets import busan_beach
    >>> data = busan_beach()

    >>> clr_tr = CLR(True, True)
    >>> x = data.iloc[:, 0].values
    >>> x_ = clr_tr.fit_transform(x)
    >>> _x = clr_tr.inverse_transform(x_)
    >>> np.allclose(_x, x)
    True
    >>> hist([x, x_], hist_kws={"bins": 100}, share_axes=False,
    ...   labels=["Original", "Transformed"])
    """

    def transform(
            self,
            x:np.ndarray
    )->np.ndarray:

        x = self._check_array(x)

        lmat = np.log(x)
        gm = lmat.mean(axis=-1, keepdims=True)
        return (lmat - gm).squeeze()


    def inverse_transform(self, x:np.ndarray)->np.ndarray:
        emat = np.exp(x)
        x = closure(emat, out=emat)

        x = self._check_array_inv(x)

        return x


class FuncTransformer(ScalerWithConfig):

    """function transformer. Transforms the array element wise."""
    @property
    def func(self):
        raise NotImplementedError

    @property
    def inv_func(self):
        raise NotImplementedError

    def fit(self, X, y=None):
        return self

    def fit_transform(self, x:np.ndarray)->np.ndarray:
        return self.transform(x)

    def transform(self, x:np.ndarray)-> np.ndarray:

        setattr(self, 'data_dim_', np.ndim(x))
        return self.func(x)

    def inverse_transform_without_fit(self, x):
        return self._inverse_transform(x, False)

    def _inverse_transform(self, x, check_dim=True):
        return self.inv_func(x)

    def inverse_transform(self, x):
        return self._inverse_transform(x)

    @property
    def config_paras(self):
        return ['data_dim_']

    def get_params(self):
        return {}


class SqrtScaler(FuncTransformer):

    @property
    def func(self):
        return np.sqrt

    @property
    def inv_func(self):
        return np.square


class LogScaler(FuncTransformer):

    @property
    def func(self):
        return np.log

    @property
    def inv_func(self):
        return np.exp


class Log2Scaler(FuncTransformer):

    @property
    def func(self):
        return np.log2

    @property
    def inv_func(self):
        return lambda x: np.power(2, x)


class Log10Scaler(FuncTransformer):

    @property
    def func(self):
        return np.log10

    @property
    def inv_func(self):
        return lambda x: np.power(10, x)


class TanScaler(FuncTransformer):

    @property
    def func(self):
        return np.tan

    @property
    def inv_func(self):
        return np.tanh


class LogisticSigmoidTransformer(FuncTransformer):
    """logistic sigmoid transformer.
    Note that inverse transform of logistic sigmoid does not return
    original array.
    """
    @property
    def func(self):
        return expit

    @property
    def inv_func(self):
        raise ValueError("inverse transform of sigmoid can not be computed")


class HyperbolicTangentTransformer(FuncTransformer):
    """Hyperbolic tangent"""
    @property
    def func(self):
        return np.tanh

    @property
    def inv_func(self):
        raise ValueError("inverse transform of tanh can not be computed")


class CumsumScaler(FuncTransformer):

    def __init__(
            self,
            feature_dim: str = "2d"
    ):
        """
        Arguments:
            feature_dim:
                whether the features are 2 dimensional or 1 dimensional. Only
                relevant if the `x` to `fit_transform` is 3D. In such as case
                if feature_dim is `1D`, it will be considered that the x consists
                of following shape (num_examples, time_steps, num_features)

        """
        assert feature_dim in ("1d", "2d")
        self.feature_dim = feature_dim

    def fit_transform(self, x:np.ndarray) -> np.ndarray:
        self.data_dim_ = np.ndim(x)

        dim = np.ndim(x)

        if dim == 3 and self.feature_dim == "1d":
            _x = np.full(x.shape, np.nan)
            for time_step in range(x.shape[1]):
                _x[:, time_step] = self.func(x[:, time_step], axis=0)
        else:
            _x = np.cumsum(x, axis=0)

        return _x

    def inverse_transform(self, x):

        dim = x.ndim
        assert dim == self.data_dim_, f"dimension of data changed from {self.data_dim_} to {dim}"

        if dim == 3 and self.feature_dim == "1d":
            _x = np.full(x.shape, np.nan)
            for time_step in range(x.shape[1]):
                _x[:, time_step] = np.diff(x[:, time_step], axis=0, append=0)

        elif 2 <= dim < 4:
            _x = np.diff(x, axis=0, append=0)
        else:
            raise  ValueError(f" dimension {dim} not allowed")

        return _x


class FunctionTransformer(SKFunctionTransformer):
    """Serializing a custom func/inverse_func is difficult. Therefore
    we expect the func/inverse_func to be either numpy function or
    the code as a string.

    Methods
    -------
    from_config

    Attributes
    ----------
    inverse_func_ser

    Example
    -------
        >>> array = np.random.randint(1, 100, (20, 2))
        >>> transformer = FunctionTransformer(func=np.log2,
        >>>                inverse_func="lambda _x: 2**_x", validate=True)
        >>> t_array = transformer.fit_transform(array)
        >>> transformer.config()
        >>> new_transformer = FunctionTransformer.from_config(transformer.config())
        >>> original_array = new_transformer.inverse_transform(t_array)

    """
    def __init__(self, func=None, inverse_func=None, validate=False,
                 accept_sparse=False, check_inverse=True, kw_args=None,
                 inv_kw_args=None):

        # if inverse_func is string, we save a serialized version of it in memory
        # to save it in config later.
        self.inverse_func_ser = inverse_func

        super().__init__(func=func,
                         inverse_func=inverse_func,
                         validate=validate,
                         accept_sparse=accept_sparse,
                         check_inverse=check_inverse,
                         kw_args=kw_args,
                         inv_kw_args=inv_kw_args)

    @property
    def inverse_func(self):
        return self._inverse_func

    @inverse_func.setter
    def inverse_func(self, func):
        self._inverse_func = self.deserialize_func(func)

    @property
    def inverse_func_ser(self):
        return self._inverse_func_ser

    @inverse_func_ser.setter
    def inverse_func_ser(self, func):
        self._inverse_func_ser = self.serialize_func(func)

    @classmethod
    def from_config(cls, config: dict):
        """Build the estimator from config file"""

        func = cls.deserialize_func(config.pop('func'))

        # do not deserialize inverse_func here, it will be done in init method
        scaler = cls(func=func, inverse_func=config.pop('inverse_func'), **cls.deserialize(**config))

        setattr(scaler, '_from_config', True)

        return scaler

    @staticmethod
    def deserialize_func(func):
        if func is not None:
            if isinstance(func, str):
                if func in np.__dict__:
                    func = getattr(np, func)
                else:
                    func = eval(func)
            elif isinstance(func, np.ufunc):  # np.log2
                func = func
            elif func.__name__ in np.__dict__:  # np.diff
                func = func
            else:
                raise ValueError(f"{func}")

        return func

    def config(self) -> dict:
        """Returns all the parameters in scaler in a dictionary"""

        params = self.get_params()
        _config = dict()
        _config['func'] = self.serialize_func(self.func)
        _config['inverse_func'] = self.inverse_func_ser
        _config['kw_args'] = jsonize(self.kw_args)
        _config['inv_kw_args'] = jsonize(self.inv_kw_args)

        for k, v in params.items():
            if k not in _config:
                _config.update({k: v})

        return _config

    @staticmethod
    def deserialize(**kwargs):
        _kwargs = {}
        for k, v in kwargs.items():
            if v == "None":
                v = None
            _kwargs[k] = v

        return _kwargs

    @staticmethod
    def serialize_func(func):

        if type(func) == np.ufunc:
            func = func.__name__

        elif func.__class__.__name__ == "function" and func.__module__ == "numpy":
            func = func.__name__

        elif func is not None:
            if isinstance(func, str):
                func = f"""{func}"""
            else:
                raise ValueError(f"{func} is not serializable")

        return func


class ParetoTransformer(ScalerWithConfig):
    """
    Similar to zscore/StandardScaler, but instead of dividing by standard
    deviation, it devides by square root of standard deviation [11]_ and [12]_.

    The standard score of a sample `x` is calculated as:
    :: math
        z = (x - u) / sqrt(s)

    """
    def __init__(
            self,
            feature_dim="2d",
            axis=0
    ):
        self.feature_dim = feature_dim
        self.axis = axis

    def _reset(self):
        for arg in ['mean_', 'var_', 'scale_', 'data_dim_']:
            setattr(self, arg, None)
        return

    def fit(self, X, y=None):
        self._reset()
        self.data_dim_ = np.ndim(X)

        self.mean_ = np.nanmean(X, axis=self.axis)
        self.scale_ = np.sqrt(np.nanvar(X, axis=self.axis))
        self.var_ = np.nanvar(X, axis=self.axis)

        return self

    def transform(self, X, y=None):

        assert np.ndim(X) == self.data_dim_

        X = X - self.mean_

        return X / np.sqrt(self.scale_)

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)

    def inverse_transform(self, X):
        X = X * np.sqrt(self.scale_)
        return X + self.mean_

    @property
    def config_paras(self):
        return ['data_dim_', 'mean_', 'var_', 'scale_']

    def get_params(self):
        return {'feature_dim': self.feature_dim, "axis": self.axis}


class VastTransformer(ParetoTransformer):
    """
    Variable Stability Scaling following the works of Nicholson et al., 2003 [11]_
    and van der Berg et al., 2006 [12]_ .

    The standard score of a sample `x` is calculated as:
    :: math
        z = (x - u) / s * u/s

    .. [11] https://doi.org/10.1016/S0003-2670(03)00094-1
    .. [12] https://doi.org/10.1186/1471-2164-7-142

    """

    def transform(self, X, y=None):

        assert np.ndim(X) == self.data_dim_

        X = X - self.mean_
        X = X / self.scale_
        # coefficient of variation
        cv = np.divide(self.mean_ , self.scale_)
        return X * cv

    def inverse_transform(self, X, y=None):
        cv = np.divide(self.mean_ , self.scale_)
        X = X / cv
        X = X * self.scale_
        X = X + self.mean_
        return X


class MmadTransformer(ScalerWithConfig):
    """
    Median and median absolute deviation following Jain et al., 2005[13]_ and
    Singh and Singh 2020 [14]_.

    The standard score of a sample `x` is calculated as:
    :: math
        z = (x - median) / MAD

    .. [13] https://doi.org/10.1016/j.patcog.2005.01.012
    .. [14] https://doi.org/10.1016/j.asoc.2019.105524
    """
    def __init__(
            self,
            feature_dim="1d",
            axis=0
    ):
        self.feature_dim = feature_dim
        self.axis = axis

    def get_params(self):
        return {'feature_dim': self.feature_dim, "axis": self.axis}

    def _reset(self):
        for arg in ['med_', 'mad_', 'data_dim_']:
            setattr(self, arg, None)
        return

    def fit(self, X, y=None):
        """fits the data i.e. calculates median and MAD of the data.
        These parameters will be used during transform.
        """
        self._reset()
        self.data_dim_ = np.ndim(X)
        self.med_ = np.nanmedian(X, axis=self.axis)

        self.mad_ = np.nanmedian(np.absolute(X - self.med_), axis=self.axis)

        return self

    def transform(self, X, y=None):
        """transforms the data i.e. changes the data using the parameters calculated
        during ``fit``.
        """
        assert np.ndim(X) == self.data_dim_
        X = X - self.med_
        return X / self.mad_

    def fit_transform(self, X, y=None):
        """First calls fit and then calls transform."""
        return self.fit(X, y=y).transform(X)

    def inverse_transform(self, X):
        """inverse transforms the X i.e. brings the X to original scale by using
        the parameters calculated during ``fit``."""
        X = X * self.mad_
        return X + self.med_

    @property
    def config_paras(self):
        return ['data_dim_', 'med_', 'mad_']


def closure(mat, out=None):
    mat = np.atleast_2d(mat)
    if out is not None:
        out = np.atleast_2d(out)
    if np.any(mat < 0):
        raise ValueError("Cannot have negative proportions")
    if mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    norm = mat.sum(axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("Input matrix cannot have rows with all zeros")
    return np.divide(mat, norm, out=out).squeeze()
