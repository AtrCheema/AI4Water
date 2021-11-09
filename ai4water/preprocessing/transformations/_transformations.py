

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
    def config_paras(self)->list:
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config:dict):
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

    def config(self)->dict:
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

    @property
    def config_paras(self):
        return ['lambdas_']


class QuantileTransformer(SKQuantileTransformer, ScalerWithConfig):

    @property
    def config_paras(self):
        return ['n_quantiles_', 'references_']


class MaxAbsScaler(SKMaxAbsScaler, ScalerWithConfig):

    @property
    def config_paras(self):
        return ['scale_', 'n_samples_seen_', 'max_abs_']


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
    ```python
    >>> array = np.random.randint(1, 100, (20, 2))
    >>> transformer = FunctionTransformer(func=np.log2, inverse_func="lambda _x: 2**_x", validate=True)
    >>> t_array = transformer.fit_transform(array)
    >>> transformer.config()
    >>> new_transformer = FunctionTransformer.from_config(transformer.config())
    >>> original_array = new_transformer.inverse_transform(t_array)
    ```
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
    def from_config(cls, config:dict):
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

    def config(self)->dict:
        """Returns all the parameters in scaler in a dictionary"""

        params = self.get_params()
        _config = dict()
        _config['func'] = self.serialize_func(self.func)
        _config['inverse_func'] = self.inverse_func_ser
        _config['kw_args'] = jsonize(self.kw_args)
        _config['inv_kw_args'] = jsonize(self.inv_kw_args)

        for k,v in params.items():
            if k not in _config:
                _config.update({k:v})

        return _config

    @staticmethod
    def deserialize(**kwargs):
        _kwargs = {}
        for k,v in kwargs.items():
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
