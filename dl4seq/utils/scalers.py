import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer,\
    QuantileTransformer, FunctionTransformer
import numpy as np

from .utils import dateandtime_now

class scaler_container(object):

    scalers = {}

class Scalers(scaler_container):
    """
    Any new transforming methods should define two methods one starting with "transform_with_" and "inverse_transofrm_with_"
    https://developers.google.com/machine-learning/data-prep/transform/normalization

    Currently following methods are available
      minmax
      maxabs
      robust
      power
      zscore,
      quantile,
      log
    """

    def __init__(self,
                 data: pd.DataFrame,
                 method:str = 'minmax',
                 features=None):

        self.data = data
        self.method = method
        self.features = features
        self.initial_shape = data.shape

    def __call__(self, what="normalize", **kwargs):

        if what.upper().startswith("NORM"):

            return self.transform(**kwargs)

        elif what.upper().startswith("DENORM"):

            return self.inverse_transform(**kwargs)

        else:
            raise ValueError

    def __getattr__(self, item):

        if item.startswith('_'):
            return self.__getattribute__(item)
        elif item.startswith("transform_with"):
            transformer = item.split('_')[2]
            if transformer.lower() in ["minmax", "zscore", "maxabs", "robust", "power", "quantile", "log"]:
                self.method = transformer
                return self.transform_with_sklearn
        elif item.startswith("inverse_transform_with"):
            transformer = item.split('_')[3]
            if transformer.lower() in ["minmax", "zscore", "maxabs", "robust", "power", "quantile", "log"]:
                self.method = transformer
                return self.inverse_transform_with_sklearn

    def get_scaler(self):
        scalers = {
            "minmax": MinMaxScaler,
            "zscore": StandardScaler,
            "robust": RobustScaler,
            "maxabs": MaxAbsScaler,
            "power": PowerTransformer,
            "quantile": QuantileTransformer
        }

        return scalers[self.method.lower()]

    def transform_with_sklearn(self, **kwargs):

        if self.method.lower() == "log":
            scaler = FunctionTransformer(func=np.log, inverse_func=np.exp, validate=True)
        else:
            scaler = self.get_scaler()()  # TODO there should be a way to provide initializing arguments

        to_transform = self.get_features()

        data = scaler.fit_transform(to_transform, **kwargs)

        data = pd.DataFrame(data, columns=to_transform.columns)

        scaler = self.serialize_scaler(scaler, to_transform)

        data = self.maybe_insert_features(data)

        return data, scaler

    def inverse_transform_with_sklearn(self, **kwargs):

        scaler = self.get_scaler_from_dict(**kwargs)

        to_transform = self.get_features(**kwargs)
        data = scaler.inverse_transform(to_transform)
        data = pd.DataFrame(data, columns=to_transform.columns)

        data = self.maybe_insert_features(data)

        return data

    def transform(self, **kwargs):

        return getattr(self, "transform_with_" + self.method.lower())(**kwargs)

    def inverse_transform(self, **kwargs):

        return getattr(self, "inverse_transform_with_" + self.method.lower())(**kwargs)

    def get_features(self, **kwargs) -> pd.DataFrame:
        # use the provided data if given otherwise use self.data
        data = kwargs['data'] if 'data' in kwargs else self.data

        if self.features is None:
            return data
        else:
            assert isinstance(self.features, list)
            return data[self.features]

    def serialize_scaler(self, scaler, to_transform):
        key = 'minmax_' + str(dateandtime_now())
        serialized_scaler = {
            "scaler": scaler,
            "shape": to_transform.shape,
            "key": key
        }
        self.scalers[key] = serialized_scaler

        return serialized_scaler

    def get_scaler_from_dict(self, **kwargs):
        if 'scaler' in kwargs:
            scaler = kwargs['scaler']
        elif 'key' in kwargs:
            scaler = self.scalers[kwargs['key']]['scaler']
        else:
            raise ValueError("provide scaler of key")
        return scaler

    def maybe_insert_features(self, trans_df):

        if len(trans_df.columns) != len(self.data.columns):
            df = pd.DataFrame()
            for col in self.data.columns:
                if col in trans_df.columns:
                    _df = trans_df[col]
                else:
                    _df = self.data[col]

                df = pd.concat([df, _df], axis=1)
        else:
            df = trans_df

        assert df.shape == self.initial_shape
        return df
