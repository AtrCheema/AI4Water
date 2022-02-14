
from typing import Union, List, Dict

import numpy as np
import pandas as pd

from ai4water.utils.utils import jsonize, deepcopy_dict_without_clone
from ai4water.preprocessing.transformations import Transformation
from .utils import TransformerNotFittedError, SP_METHODS


class Transformations(object):
    """
    While the [Transformation][ai4water.preprocessing.transformations.Transformation]
    class is useful to apply a single transformation to a single data source, this
    class is helpful to apply multple transformations to a single data or multiple
    transformations to multiple data. This class is especially designed to be applied
    as part of `model` inside the `fit`, `predict` or `evaluate` methods. The
    `fit_transform` method should be applied before feeding the data to the
    algorithm and `inverse_transform` method should be called after algorithm has
    worked with data.

    Examples
    --------
        >>> import numpy as np
        >>> from ai4water.preprocessing.transformations import Transformations
        >>> x = np.arange(50).reshape(25, 2)
        >>> transformer = Transformations(['a', 'b'], config=['minmax', 'zscore'])
        >>> x_ = transformer.fit_transform(x)
        >>> _x = transformer.inverse_transform(x_)
        ...
        ... # Apply multiple transformations on multiple arrays which are passed as list
        >>> transformer = Transformations([['a', 'b'], ['a', 'b']],
        ...                              config=['minmax', 'zscore'])
        >>> x1 = np.arange(50).reshape(25, 2)
        >>> x2 = np.arange(50, 100).reshape(25, 2)
        >>> x1_transformed = transformer.fit_transform([x1, x2])
        >>> _x1 = transformer.inverse_transform(x1_transformed)
        ...
        ... # We can also do more complicated stuff as following
        >>> transformer = Transformations({'x1': ['a', 'b'], 'x2': ['a', 'b']},
        ...        config={'x1': ['minmax', 'zscore'],
        ...                'x2': [{'method': 'log', 'features': ['a', 'b']},
        ...                       {'method': 'robust', 'features': ['a', 'b']}]
        ...                                      })
        >>> x1 = np.arange(20).reshape(10, 2)
        >>> x2 = np.arange(100, 120).reshape(10, 2)
        >>> x = {'x1': x1, 'x2': x2}
        >>> x_transformed = transformer.fit_transform(x)
        >>> _x = transformer.inverse_transform(x_transformed)

        In above example we apply `minmax` and `zscore` transformations on x1
        and `log` and `robust` transformations on x2 array
    """
    def __init__(
            self,
            feature_names: Union[list, dict],
            config: Union[str, list, dict] = None,
    ):
        """
        Arguments:
            feature_names:
                names of features in data
            config:
                Determines the type of transformation to be applied on data.
                It can be one of the following types

                - `string` when you want to apply single transformation
                ```python
                >>> config='minmax'
                ```
                - `dict`: to pass additional arguments to the [Transformation][ai4water.preprocessing.Transformation]
                   class
                ```python
                >>> config = {"method": 'log', 'treat_negatives': True, 'features': ['features']}
                ```
                - `list` when we want to apply multiple transformations
                ```python
                >>> ['minmax', 'zscore']
                ```
                or
                ```python
                >>> [{"method": 'log', 'treat_negatives': True, 'features': ['features']},
                >>>  {'method': 'sqrt', 'treat_negatives': True}]
                ```

        """
        self.names = feature_names
        self.t_config = config
        self.without_fit = False

    def _fetch_transformation(self, data):
        config = self.t_config

        if isinstance(data, list):
            if isinstance(config, str):
                config = [config for _ in range(len(data))]
        elif isinstance(data, dict):
            if isinstance(config, str):
                config = {k:config for k in data.keys()}

        return config

    def _check_features(self):

        if self.is_numpy_:
            assert isinstance(self.names, list), f"""
            feature_names are of type {type(self.names)}"""

        elif self.is_list_:
            for idx, n in enumerate(self.names):
                assert isinstance(n, list), f"""
                feature_names for {idx} source is {type(n)}. It should be list"""

        elif self.is_dict_:
            assert isinstance(self.names, dict), f"""
            feature_names are of type {type(self.names)}"""
            for src_name, n in self.names.items():
                assert n.__class__.__name__ in ["ListWrapper", 'list']

        return

    def fit_transform(self, data:Union[np.ndarray, List, Dict]):
        """Transforms the data according the the `config`.

        Arguments:
            data:
                The data on which to apply transformations. It can be one of following

                - a (2d or 3d) numpy array
                - a list of numpy arrays
                - a dictionary of numpy arrays
        Returns:
            The transformed data which has same type and dimensions as the input data
        """
        setattr(self, 'is_numpy_', False)
        setattr(self, 'is_list_', False)
        setattr(self, 'is_dict_', False)
        setattr(self, 'scalers_', {})

        if self.t_config is None:  # if no transformation then just return the data as it is
            return data

        orignal_data_type = data.__class__.__name__

        if isinstance(data, np.ndarray):
            setattr(self, 'is_numpy_', True)
        elif isinstance(data, list):
            setattr(self, 'is_list_', True)
        elif isinstance(data, dict):
            setattr(self, 'is_dict_', True)
        else:
            raise ValueError(f"invalid data of type {data.__class__.__name__}")

        # first check that data matches config
        self._check_features()

        # then apply transformation
        data = self._fit_transform(data)

        # now pack it in original form
        assert data.__class__.__name__ == orignal_data_type, f"""
        type changed from {orignal_data_type} to {data.__class__.__name__}
        """

        #self._assert_same_dim(self, orignal_data, data)

        return data

    def _transform_2d(self, data, columns, transformation=None, key="5"):
        """performs transformation on single data 2D source"""
        # it is better to make a copy here because all the operations on data happen after this.
        data = data.copy()
        scalers = {}
        if transformation:

            if isinstance(transformation, dict):
                transformer = Transformation(**transformation)
                data = transformer.fit_transform(pd.DataFrame(data, columns=columns))
                scalers[key] = transformer.config()

            # we want to apply multiple transformations
            elif isinstance(transformation, list):
                for idx, trans in enumerate(transformation):

                    if isinstance(trans, str):
                        transformer = Transformation(method=trans)
                        data = transformer.fit_transform(pd.DataFrame(data, columns=columns))
                        scalers[f'{key}_{trans}_{idx}'] = transformer.config()

                    elif trans['method'] is not None:
                        transformer = Transformation(**trans)
                        data = transformer.fit_transform(pd.DataFrame(data, columns=columns))
                        scalers[f'{key}_{trans["method"]}_{idx}'] = transformer.config()
            else:
                assert isinstance(transformation, str)
                transformer = Transformation(method=transformation)
                data = transformer.fit_transform(pd.DataFrame(data, columns=columns))
                scalers[key] = transformer.config()

            data = data.values

        self.scalers_.update(scalers)

        return data

    def __fit_transform(self, data, feature_names, transformation=None, key="5"):
        """performs transformation on single data source
        In case of 3d array, the shape is supposed to be following
        (num_examples, time_steps, num_features)
        Therefore, each time_step is extracted and transfomred individually
        for example with time_steps of 2, two 2d arrays will be extracted and
        transformed individually
        (num_examples, 0,num_features), (num_examples, 1, num_features)
        """
        if data.ndim == 3:
            _data = np.full(data.shape, np.nan)
            for time_step in range(data.shape[1]):
                _data[:, time_step] = self._transform_2d(data[:, time_step],
                                                        feature_names,
                                                        transformation,
                                                        key=f"{key}_{time_step}")
        else:
            _data = self._transform_2d(data, feature_names, transformation, key=key)

        return _data

    def _fit_transform(self, data, key="5"):
        """performs transformation on every data source in data"""
        transformation = self._fetch_transformation(data)
        if self.is_numpy_:
            _data = self.__fit_transform(data, self.names, transformation, key)

        elif self.is_list_:
            _data = []
            for idx, array in enumerate(data):
                _data.append(self.__fit_transform(array,
                                                  self.names[idx],
                                                  transformation[idx],
                                                  key=f"{key}_{idx}")
                             )
        else:
            _data = {}
            for src_name, array in data.items():
                _data[src_name] = self.__fit_transform(array,
                                                       self.names[src_name],
                                                       transformation[src_name],
                                                       f"{key}_{src_name}")
        return _data

    def inverse_transform(self, data, postprocess=True):
        """inverse transforms data where data can be dictionary, list or numpy
        array.

        Arguments:
            data:
                the data which is to be inverse transformed. The output of
                `fit_transform` method.
            postprocess : bool

        Returns:
            The original data which was given to `fit_transform` method.
        """
        if not hasattr(self, 'scalers_'):
            raise ValueError(f"Transformations class has not been fitted yet")
        return self._inverse_transform(data, postprocess=postprocess)

    def inverse_transform_without_fit(self, data, postprocess=True)->np.ndarray:
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        assert isinstance(self.names, list)
        assert data.shape[-1] == len(self.names)

        data = pd.DataFrame(data, columns=self.names)

        kwargs = {}
        if isinstance(self.t_config, str):
            kwargs['method'] = self.t_config
        elif isinstance(self.t_config, dict):
            kwargs = self.t_config
        elif isinstance(self.t_config, list):
            assert len(self.t_config) == 1
            t_config = self.t_config[0]
            if isinstance(t_config, str):
                kwargs['method'] = t_config
            elif isinstance(t_config, dict):
                kwargs = t_config
            else:
                raise ValueError(f"invalid type of t_config {t_config.__class__.__name__}")
        else:
            raise ValueError(f"invalid type of t_config {self.t_config.__class__.__name__}")

        transformer = Transformation(**kwargs)
        transformed_data = transformer.inverse_transform(data=data, postprocess=postprocess)

        return transformed_data.values

    def _inverse_transform(self, data, key="5", postprocess=True):

        transformation = self._fetch_transformation(data)

        if self.is_numpy_:
            data = self.__inverse_transform(data,
                                            self.names,
                                            transformation,
                                            key,
                                            postprocess=postprocess)

        elif self.is_list_:
            assert isinstance(data, list)
            _data = []
            for idx, src in enumerate(data):
                __data = self.__inverse_transform(src,
                                                 self.names[idx],
                                                 transformation[idx],
                                                 f'{key}_{idx}',
                                                  postprocess=postprocess)
                _data.append(__data)
            data = _data

        elif self.is_dict_:
            assert isinstance(data, dict)
            _data = {}
            for src_name, src in data.items():
                _data[src_name] = self.__inverse_transform(src,
                                                          self.names[src_name],
                                                          transformation[src_name],
                                                          f'{key}_{src_name}',
                                                           postprocess=postprocess)
            data = _data

        return data

    def __inverse_transform(self,
                            data,
                            feature_names,
                            transformation, key="5",
                            postprocess=True):
        """inverse transforms one data source which may 2d or 3d nd array"""
        if data.ndim == 3:
            _data = np.full(data.shape, np.nan)
            for time_step in range(data.shape[1]):
                _data[:, time_step] = self._inverse_transform_2d(
                    data[:, time_step],
                    columns=feature_names,
                    transformation=transformation,
                    key=f"{key}_{time_step}",
                    postprocess=postprocess)
        else:
            _data = self._inverse_transform_2d(data,
                                               feature_names,
                                               key,
                                               transformation,
                                               postprocess=postprocess)

        return _data

    def _inverse_transform_2d(self,
                              data,
                              columns,
                              key,
                              transformation,
                              postprocess=True)->np.ndarray:
        """inverse transforms one 2d array"""
        data = pd.DataFrame(data.copy(), columns=columns)

        if transformation is not None:
            if isinstance(transformation, str):

                if key not in self.scalers_:
                    raise ValueError(f"""
                    key `{key}` for inverse transformation not found. Available keys are {list(self.scalers_.keys())}""")

                scaler = self.scalers_[key]
                scaler, shape = scaler, scaler['shape']
                original_shape = data.shape

                transformer = Transformation.from_config(scaler)
                transformed_data = transformer.inverse_transform(data, postprocess=postprocess)
                data = transformed_data

            elif isinstance(transformation, list):
                # idx and trans both in reverse form
                for idx, trans in reversed(list(enumerate(transformation))):
                    if isinstance(trans, str):
                        scaler = self.scalers_[f'{key}_{trans}_{idx}']
                        scaler, shape = scaler, scaler['shape']
                        transformer = Transformation.from_config(scaler)
                        data = transformer.inverse_transform(data=data, postprocess=postprocess)

                    elif trans['method'] is not None:
                        features = trans.get('features', columns)
                        # if any of the feature in data was transformed
                        if any([True if f in data else False for f in features]):
                            orig_cols = data.columns  # copy teh columns in the original df
                            scaler = self.scalers_[f'{key}_{trans["method"]}_{idx}']
                            scaler, shape = scaler, scaler['shape']
                            data, dummy_features = conform_shape(data, shape, features)  # get data to transform

                            transformer = Transformation.from_config(scaler)
                            transformed_data = transformer.inverse_transform(data=data,
                                                                             postprocess=postprocess)
                            data = transformed_data[orig_cols]  # remove the dummy data

            elif isinstance(transformation, dict):

                features = transformation.get('features', columns)
                if any([True if f in data else False for f in features]):
                    orig_cols = data.columns
                    scaler = self.scalers_[key]
                    scaler, shape = scaler, scaler['shape']
                    data, dummy_features = conform_shape(data, shape, features=features)

                    transformer = Transformation.from_config(scaler)
                    transformed_data = transformer.inverse_transform(data=data, postprocess=postprocess)
                    data = transformed_data[orig_cols]  # remove the dummy data

        if data.__class__.__name__ == "DataFrame":
            data = data.values  # there is no need to return DataFrame

        return data

    def config(self)->dict:
        """returns a python dictionary which can be used to construct this class
        in fitted form i.e as if the fit_transform method has already been applied.
        Returns:
            a dictionary from which `Transformations` class can be constructed
        """
        return {
            'scalers_': jsonize(self.scalers_),
            "feature_names": self.names,
            "config": self.t_config,
            "is_numpy_": self.is_numpy_,
            "is_dict_": self.is_dict_,
            "is_list_": self.is_list_,
        }

    @classmethod
    def from_config(cls, config:dict)->"Transformations":
        """constructs the Transformations class which may has already been fitted.
        """
        config = deepcopy_dict_without_clone(config)

        transformer = cls(config.pop('feature_names'), config.pop('config'))

        for attr_name, attr_val in config.items():
            setattr(cls, attr_name, attr_val)

        return transformer


def conform_shape(data, shape, features=None):
    # if the difference is of only 1 dim, we resolve it
    if data.ndim > len(shape):
        data = np.squeeze(data, axis=-1)
    elif data.ndim < len(shape):
        data = np.expand_dims(data, axis=-1)

    assert data.ndim == len(shape), f"""original data had {len(shape)} wihle the 
    new data has {data.ndim} dimensions"""

    # how manu dummy features we have to add to match the shape
    dummy_features = shape[-1] - data.shape[-1]

    if data.__class__.__name__ in ['DataFrame', 'Series']:
        # we know what features must be in data, so put them in data one by one
        # if they do not exist in data already
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
