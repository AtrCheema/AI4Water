import math
import warnings

from scipy.special import boxcox

from sklearn.preprocessing import MinMaxScaler as SKMinMaxScaler
from sklearn.preprocessing import StandardScaler as SKStandardScaler
from sklearn.preprocessing import RobustScaler as SKRobustScaler
from sklearn.preprocessing import PowerTransformer as SKPowerTransformer
from sklearn.preprocessing import QuantileTransformer as SKQuantileTransformer
from sklearn.preprocessing import FunctionTransformer as SKFunctionTransformer
from sklearn.preprocessing import MaxAbsScaler as SKMaxAbsScaler
from sklearn.utils.validation import check_is_fitted

import numpy as np

from ai4water.utils.utils import jsonize


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
        return ['scale_', 'n_samples_seen_', 'mean_', 'var_']


class RobustScaler(SKRobustScaler, ScalerWithConfig):

    @property
    def config_paras(self):
        return ['scale_', 'center_']


class PowerTransformer(SKPowerTransformer, ScalerWithConfig):
    """This transformation enhances scikit-learn's PowerTransformer by allowing
    the user to define `lambdas` parameter for each input feature. The default
    behaviour of this transformer is same as that of scikit-learn's.
    """
    def __init__(self, method='yeo-johnson', *, standardize=True, copy=True,
                 lambdas=None):
        """
        lambdas: float or 1d array like for each feature. If not given, it is
            calculated from scipy.stats.boxcox(X, lmbda=None). Only available
            if method is box-cox.
        For complete documentation see scikit-learn's documentation_

        .. documentation:
            https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html
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

        super(PowerTransformer, self).__init__(method=method,
                                               standardize=standardize,
                                               copy=copy)

    @property
    def config_paras(self):
        return ['lambdas_', 'scaler_to_standardize_']

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
            feature_dim="2d"
    ):
        self.feature_dim = feature_dim

    def fit(self, x:np.ndarray):
        dim = x.ndim
        assert dim == 2

        mean = np.mean(x, axis=0)

        setattr(self, 'mean_', mean)
        setattr(self, 'data_dim_', dim)
        return

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
        return {'feature_dim': self.feature_dim}


class CLR(object):

    """centre log ratio transformation"""
    @staticmethod
    def fit_transform(x:np.ndarray)->np.ndarray:
        assert x.sum() == 1.0
        lmat = np.log(x)
        gm = lmat.mean(axis=-1, keepdims=True)
        return (lmat - gm).squeeze()

    @staticmethod
    def inverse_transform(x:np.ndarray)->np.ndarray:
        emat = np.exp(x)
        return closure(emat, out=emat)


class FuncTransformer(ScalerWithConfig):

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

    @property
    def func(self):
        raise NotImplementedError

    @property
    def inv_func(self):
        raise NotImplementedError

    def fit(self):
        return

    def _get_dim(self, x:np.ndarray):
        dim = x.ndim
        setattr(self, 'data_dim_', dim)

        if dim > 3:
            raise ValueError(f" dimension {dim} not allowed")
        return dim

    def fit_transform(self, x:np.ndarray)->np.ndarray:
        return self.transform(x)

    def transform(self, x:np.ndarray)-> np.ndarray:

        dim = self._get_dim(x)

        if dim == 3 and self.feature_dim == "1d":
            _x = np.full(x.shape, np.nan)
            for time_step in range(x.shape[1]):
                _x[:, time_step] = self.func(x[:, time_step])
        else:
            _x = self.func(x)

        return _x

    def inverse_transform_without_fit(self, x):
        return self._inverse_transform(x, False)

    def _inverse_transform(self, x, check_dim=True):
        dim = x.ndim
        #if check_dim:
        #assert dim == self.data_dim_, f"dimension of data changed from {self.data_dim_} to {dim}"

        if dim == 3 and self.feature_dim == "1d":
            _x = np.full(x.shape, np.nan)
            for time_step in range(x.shape[1]):
                _x[:, time_step] = self.inv_func(x[:, time_step])

        elif 2 <= dim < 4:
            _x = self.inv_func(x)
        else:
            raise  ValueError(f" dimension {dim} not allowed")

        return _x

    def inverse_transform(self, x):
        return self._inverse_transform(x)

    @property
    def config_paras(self):
        return ['data_dim_']

    def get_params(self):
        return {'feature_dim': self.feature_dim}


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


class CumsumScaler(FuncTransformer):

    def fit_transform(self, x:np.ndarray) -> np.ndarray:

        dim = self._get_dim(x)

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