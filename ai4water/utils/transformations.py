from typing import Union
import warnings

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer,\
    QuantileTransformer, FunctionTransformer
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA, FastICA, SparsePCA
try:
    from PyEMD import EMD, EEMD
except ModuleNotFoundError:
    EMD, EEMD = None, None

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ai4water.utils.utils import dateandtime_now

# TODO add logistic, tanh and more scalers.
# rpca
# tSNE
# UMAP
# PaCMAP
# which transformation to use? Some related articles/posts
# https://scikit-learn.org/stable/modules/preprocessing.html
# http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html
# https://data.library.virginia.edu/interpreting-log-transformations-in-a-linear-model/


class EmdTransformer(object):
    """Empirical Mode Decomposition"""
    def __init__(self, ensemble=False, **kwargs):
        self.ensemble = ensemble
        if ensemble:
            self.emd_obj = EEMD(**kwargs)
        else:
            self.emd_obj = EMD(**kwargs)

    def fit_transform(self, data, **kwargs):

        if isinstance(data, pd.DataFrame):
            data = data.values
        else:
            assert isinstance(data, np.ndarray)

        assert len(data.shape) == 2

        imfs = []
        for col in range(data.shape[1]):

            if self.ensemble:
                IMFs = self.emd_obj.eemd(data[:, col], **kwargs)
            else:
                IMFs = self.emd_obj.emd(data[:, col], **kwargs)

            imfs.append(IMFs.T)

        return np.concatenate(imfs, axis=1)

    def inverse_transform(self, **kwargs):
        raise NotImplementedError


class scaler_container(object):

    def __init__(self):
        self.scalers = {}
        self.transforming_straight = True
        self.zero_indices = None
        self.nan_indices = None
        self.index = None


class Transformations(scaler_container):
    """
    Applies transformation to tabular data.
    Any new transforming methods should define two methods one starting with
   `transform_with_` and `inverse_transofrm_with_`
    https://developers.google.com/machine-learning/data-prep/transform/normalization

    Currently following methods are available for transformation and inverse transformation

        - minmax :
        - maxabs :
        - robust :
        - power :
        - zscore :   also known as standard scalers
        - quantile :
        - log :     natural logrithmic
        - log10 :   log with base 10
        - log2 : log with base 2
        - tan :     tangent
        - cumsum :  cummulative sum
        - pca :     principle component analysis
        - kpca :    kernel component analysis
        - ipca :    incremental principle component analysis
        - fastica : fast incremental component analysis

    Following methods have only transformations and not inverse transformations.
    They can be used for feature creation.

        emd :    empirical mode decomposition
        eemd :   ensemble empirical mode decomposition

    To transform a datafrmae using any of the above methods use

    ```python
    >>>scaler = Transformations(data=[1,2,3,5], method='zscore')
    >>>scaler.transform()
    ```

    or

    ```python
    >>>scaler = Transformations(data=pd.DataFrame([1,2,3]))
    >>>normalized_df, scaler_dict = scaler.transform_with_minmax(return_key=True)
    ```

    or

    ```python
    >>>scaler = Transformations(data=pd.DataFrame([1,2,3]), method='minmax')
    >>>normalized_df, scaler_dict = scaler()
    ```

    or using one liner

    ```python
    >>>normalized_df, scaler = Transformations(data=pd.DataFrame([[1,2,3],[4,5,6]], columns=['a', 'b']),
    ...                                      method='log', features=['a'])('transform')
    ```

    where `method` can be any of the above mentioned methods.

    Note: `tan` and `cumsum` do not return original data upon inverse transformation.
        Same holds true for methods which causes change in dimension
    """

    available_transformers = {
        "minmax": MinMaxScaler,
        "zscore": StandardScaler,
        "robust": RobustScaler,
        "maxabs": MaxAbsScaler,
        "power": PowerTransformer,
        "quantile": QuantileTransformer,
        "pca": PCA,
        "kpca": KernelPCA,
        "ipca": IncrementalPCA,
        "fastica": FastICA,
        "sparsepca": SparsePCA,
        "emd": EmdTransformer,
        "eemd": EmdTransformer,
    }

    dim_expand_methods = ['emd', 'eemd']
    dim_red_methods = ["pca", "kpca", "ipca", "fastica", "sparsepca"]  # dimensionality reduction methods
    mod_dim_methods = dim_red_methods + dim_expand_methods

    def __init__(self,
                 data: Union[pd.DataFrame, np.ndarray, list],
                 method: str = 'minmax',
                 features: list=None,
                 replace_nans: bool = False,
                 replace_with: Union[str, int, float] = 'mean',
                 replace_zeros: bool = False,
                 replace_zeros_with: Union[str, int, float] = 'mean',
                 **kwargs
                 ):
        """
        Arguments:
            data : a dataframe or numpy ndarray or array like. The transformed or inversely
                transformed value will have the same type as data and will have
                the same index as data (in case data is dataframe).
            method : method by which to transform and consequencly inversely
                transform the data. default is 'minmax'. see `Transformations.available_transformers`
                for full list.
            features : string or list of strings. Only applicable if `data` is
                dataframe. It defines the columns on which we want to apply transformation.
                The remaining columns will remain same/unchanged.
            replace_nans : If true, then will replace the nan values in data with
                some fixed value `replace_with` before transformation. The nan
                values will be put back at their places after transformation so
                this replacement is done only to avoid error during transformation.
                However, the process of putting the nans back does not happen when
                the `method` results in dimention change, such as for PCA etc.
            replace_with : if replace_nans is True, then this value will be used
                to replace nans in dataframe before doing transformation. You can
                define the method with which to replace nans for exaple by setting
                this argument to 'mean' will replace nans with 'mean' of the
                array/column which contains nans. Allowed string values are
                'mean', 'max', 'man'.
            replace_zeros : same as replace_nans but for zeros in the data.
            replace_zeros_with : same as `replace_with` for for zeros in the data.
            kwargs : any arguments which are to be provided to transformer on
                INTIALIZATION and not during transform or inverse transform e.g.
                `n_components` for pca.

        Example
        ---------
        ```python
        >>>from ai4water.utils.transformations import Transformations
        >>>from ai4water.utils.datasets import load_u1
        >>>data = load_u1()
        >>>inputs = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
        >>>transformer = Transformations(data=data[inputs], method='minmax', features=['x1', 'x2'])
        >>>new_data = transformer.transform()
        ```

        Following shows how to apply log transformation on an array containing zeros
        by making use of the argument `replace_zeros`. The zeros in the input array
        will be replaced internally but will be inserted back afterwards.

        ```python
        >>>from ai4water.utils.transformations import Transformations
        >>>transformer = Transformations([1,2,3,0.0, 5, np.nan, 7], method='log', replace_nans=True, replace_zeros=True)
        >>>transformed_data = transformer.transform()
        ... [0.0, 0.6931, 1.0986, 0.0, 1.609, None, 1.9459]
        >>>original_data = transformer.inverse_transform(data=transformed_data)
        ```

        """
        super().__init__()

        self.method = method
        self.replace_nans=replace_nans
        self.replace_with=replace_with
        self.replace_zeros=replace_zeros
        self.replace_zeros_with=replace_zeros_with
        data = self.pre_process_data(data.copy())
        self.data = data

        self.features = features
        self.initial_shape = data.shape
        self.kwargs = kwargs
        self.transformed_features = None

    def __call__(self, what="transform", return_key=False, **kwargs):
        """
        Calls the `transform` and `inverse_transform` methods.
        """
        if what.upper().startswith("TRANS"):
            self.transforming_straight = True

            return self.transform(return_key=return_key, **kwargs)

        elif what.upper().startswith("INV"):
            self.transforming_straight = False
            return self.inverse_transform(**kwargs)

        else:
            raise ValueError(f"The class Transformation can not be called with keyword argument 'what'={what}")

    def __getattr__(self, item):
        """
        Gets the attributes from underlying transformation modules.
        """
        if item.startswith('_'):
            return self.__getattribute__(item)
        elif item.startswith("transform_with"):
            transformer = item.split('_')[2]
            if transformer.lower() in list(self.available_transformers.keys()) + ["log", "tan", "cumsum", "log10", "log2"]:
                self.method = transformer
                return self.transform_with_sklearn
        elif item.startswith("inverse_transform_with"):
            transformer = item.split('_')[3]
            if transformer.lower() in list(self.available_transformers.keys()) + ["log", "tan", "cumsum", "log10", "log2"]:
                self.method = transformer
                return self.inverse_transform_with_sklearn
        else:
            raise AttributeError(f'Transformations has not attribute {item}')

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
        if x is None:
            x = list(self.data.columns)
        assert len(x) == len(set(x)), "duplicated features are not allowed"
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

    @property
    def change_dim(self):
        if self.method.lower() in self.mod_dim_methods:
            return True
        else:
            return False

    def get_scaler(self):

        return self.available_transformers[self.method.lower()]

    def pre_process_data(self, data):
        """Makes sure that data is dataframe and optionally replaces nans"""
        if isinstance(data, pd.DataFrame):
            data = data
        else:
            data = np.array(data)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            assert isinstance(data, np.ndarray)
            data = pd.DataFrame(data, columns=['data'+str(i) for i in range(data.shape[1])])

        # save the index if not already saved so that can be used later
        if self.index is None:
            self.index = data.index

        if self.replace_nans:
            indices = {}

            for col in data.columns:

                # find index of nan values in current column of data
                # https://stackoverflow.com/questions/14016247/find-integer-index-of-rows-with-nan-in-pandas-dataframe
                i = data[col].index[data[col].apply(np.isnan)]
                if len(i) > 0:
                    indices[col] = i.values
                    # replace nans with values
                    if self.replace_with in ['mean', 'max', 'min']:
                        replace_with = float(getattr(np, 'nan'+self.replace_with)(data[col]))
                    else:
                        replace_with = self.replace_with
                    data[col][indices[col]] = get_val(data[col], replace_with)

            # because pre_processing is implemented 2 times, we don't want to overwrite nan_indices
            if self.nan_indices is None: self.nan_indices = indices

            if len(indices) > 0:
                if self.method.lower() == "cumsum":
                    warnings.warn("Warning: nan values found and they may cause problem")

        if self.replace_zeros and self.transforming_straight:
            indices = {}
            for col in data.columns:
                # find index containing 0s in corrent column of dataframe
                i = data.index[data[col] == 0.0]
                if len(i) > 0:
                    indices[col] = i.values
                    if self.replace_zeros_with in ['mean', 'max', 'min']:
                        replace_with = float(getattr(np, 'nan' + self.replace_zeros_with)(data[col]))
                    else:
                        replace_with = self.replace_zeros_with
                    data[col][indices[col]] = get_val(data[col], replace_with)

            if self.zero_indices is None: self.zero_indices = indices

        # if self.replace_negatives:
        #     indices = {}
        #     for col in data.columns:
        #         # find index containing negatives in corrent column of dataframe
        #         i = data.index[data[col] < 0.0]
        #         if len(i) > 0:
        #             indices[col] = i.values
        #             if self.replace_negatives_with in ['mean', 'max', 'min']:
        #                 replace_with = float(getattr(np, 'nan' + self.replace_negatives_with)(data[col]))
        #             else:
        #                 replace_with = self.replace_negatives_with
        #             data[col][indices[col]] = get_val(data[col], replace_with)
        #
        #     if self.negative_indices is None: self.negative_indices = indices

        return data

    def post_process_data(self, data):
        """If nans/zeros were replaced with some value, put nans/zeros back."""
        if self.method not in self.dim_red_methods:
            if self.replace_nans:
                if hasattr(self, 'nan_indices'):

                    for col, idx in self.nan_indices.items():
                        data[col][idx] = np.nan

            if self.replace_zeros:
                if hasattr(self, 'zero_indices'):
                    for col, idx in self.zero_indices.items():
                        data[col][idx] = 0.0

            # if self.replace_negatives:
            #     if hasattr(self, 'negative_indices'):
            #         for col, idx in self.negative_indices.items():
            #             data[col][idx] = 0.0
        return data

    def transform_with_sklearn(self, return_key=False, **kwargs):

        to_transform = self.get_features()  #TODO, shouldn't kwargs go here as input?

        if self.method.lower() in ["log", "log10", "log2"]:

            if (to_transform.values < 0).any():
                raise InvalidValueError(self.method, "negative")

            if (np.isnan(to_transform.values).sum() > 0).any():
                raise InvalidValueError(self.method, "NaN")

            if 0 in to_transform.values:
                raise InvalidValueError(self.method, "zero")

            if self.method == "log":
                scaler = FunctionTransformer(func=np.log, inverse_func=np.exp, validate=True, check_inverse=True)
            elif self.method == "log2":
                scaler = FunctionTransformer(func=np.log2, inverse_func=lambda x:2**x, validate=True,
                                             check_inverse=True)
            else:   # "log10":
                scaler = FunctionTransformer(func=np.log10, inverse_func=lambda x:10**x, validate=True,
                                             check_inverse=True)
        elif self.method.lower() == "tan":
            scaler = FunctionTransformer(func=np.tan, inverse_func=np.tanh, validate=True, check_inverse=False)
        elif self.method.lower() == "cumsum":
            scaler = FunctionTransformer(func=np.cumsum, inverse_func=np.diff, validate=True, check_inverse=False,
                                         kw_args={"axis": 0}, inv_kw_args={"axis": 0, "append": 0})
        else:
            scaler = self.get_scaler()(**self.kwargs)

        data = scaler.fit_transform(to_transform, **kwargs)

        if self.method.lower() in self.mod_dim_methods:
            features = [self.method.lower() + str(i+1) for i in range(data.shape[1])]
            data = pd.DataFrame(data, columns=features)
            self.transformed_features = features
            self.features = features
        else:
            data = pd.DataFrame(data, columns=to_transform.columns)

        scaler = self.serialize_scaler(scaler, to_transform)

        data = self.maybe_insert_features(data)

        data = self.post_process_data(data)

        self.tr_data = data
        if return_key:
            return data, scaler
        return data

    def inverse_transform_with_sklearn(self, **kwargs):

        self.transforming_straight = False

        scaler = self.get_scaler_from_dict(**kwargs)

        to_transform = self.get_features(**kwargs)

        data = scaler.inverse_transform(to_transform)

        if self.method.lower() in self.mod_dim_methods:
            # now use orignal data columns names, but if the class is being directly called for inverse transform
            # then we don't know what cols were transformed, in that scenariio use dummy col name.
            cols = ['data'+str(i) for i in range(data.shape[1])] if self.transformed_features is None else self.data.columns
            data = pd.DataFrame(data, columns=cols)
        else:
            data = pd.DataFrame(data, columns=to_transform.columns)

        data = self.maybe_insert_features(data)

        data = self.post_process_data(data)

        return data

    def transform(self, return_key=False, **kwargs):
        """
        Transforms the data

        Arguments:
            return_key : whether to return the scaler or not. If True, then a
                tuple is returned which consists of transformed data and scaler itself.
            kwargs :
        """
        self.transforming_straight = True
        return getattr(self, "transform_with_" + self.method.lower())(return_key=return_key, **kwargs)

    def inverse_transform(self, **kwargs):
        """
        Inverse transforms the data.
        Arguments:
            kwargs : any of the folliwng keyword arguments
                data : data on which to apply inverse transformation
                key : key to fetch scaler
                scaler : scaler to use for inverse transformation. If not given, then
                    the available scaler is used.
        """
        self.transforming_straight = False

        if 'key' in kwargs or 'scaler' in kwargs:
            pass
        elif len(self.scalers) ==1:
            kwargs['scaler'] = self.scalers[list(self.scalers.keys())[0]]['scaler']

        return getattr(self, "inverse_transform_with_" + self.method.lower())(**kwargs)

    def get_features(self, **kwargs) -> pd.DataFrame:
        # use the provided data if given otherwise use self.data
        data = kwargs['data'] if 'data' in kwargs else self.data

        if self.replace_nans:
            data = self.pre_process_data(data)

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
        self.scalers[key] = serialized_scaler

        return serialized_scaler

    def get_scaler_from_dict(self, **kwargs):
        if 'scaler' in kwargs:
            scaler = kwargs['scaler']
        elif 'key' in kwargs:
            scaler = self.scalers[kwargs['key']]['scaler']
        else:
            raise ValueError("provide scaler which was used to transform or key to fetch the scaler")
        return scaler

    def maybe_insert_features(self, trans_df):

        if self.index is not None:
            trans_df.index = self.index

        transformed_features = len(self.transformed_features) if self.transformed_features is not None else len(trans_df.columns)
        num_features = len(self.data.columns) if self.method.lower() not in self.mod_dim_methods else transformed_features
        if len(trans_df.columns) != num_features:
            df = pd.DataFrame(index=self.index)
            for col in self.data.columns: #features:
                if col in trans_df.columns:
                    _df = trans_df[col]
                else:
                    _df = self.data[col]

                df = pd.concat([df, _df], axis=1)
        else:
            df = trans_df

        if not self.change_dim:
            assert df.shape == self.initial_shape, f"shape changed from {self.initial_shape} to {df.shape}"
        return df

    def plot_pca(self, target:np.ndarray, pcs:np.ndarray=None, labels:list=None, save=False, dim="3d"):

        if pcs is None:
            pcs = self.tr_data.values
            labels = list(self.tr_data.columns) if labels is None else labels

        if labels is not None:
            assert len(labels) == pcs.shape[1]
        else:
            labels = self.transformed_features

        if dim.upper() == "3D":
            self.plot_pca3d(target, pcs, labels, save)
        elif dim.upper() == "2D":
            self.plot_pca2d(target, pcs, labels, save)
        else:
            raise ValueError

    def plot_pca3d(self, target, pcs, labels, save):

        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        plt.cla()
        for label, name in enumerate(labels):
            ax.text3D(pcs[target == label, 0].mean(),
                      pcs[target == label, 1].mean() + 1.5,
                      pcs[target == label, 2].mean(), name,
                      horizontalalignment='center',
                      bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
        # Reorder the labels to have colors matching the cluster results
        target = np.choose(target, [1, 2, 0]).astype(np.float)

        ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], c=target, cmap=plt.cm.nipy_spectral,
               edgecolor='k')

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

        end_fig(save)
        return

    def plot_pca2d(self, target, pcs, labels, save):

        for i, target_name in zip([0, 1, 2], labels):
            plt.scatter(pcs[target == i, 0], pcs[target == i, 1], alpha=.8, lw=2,
                        label=target_name)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('PCA of IRIS dataset')
        end_fig(save)
        return


def get_val(df:pd.DataFrame, method):

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


def end_fig(save):
    if save is None:
        pass
    elif save:
        plt.savefig('pca')
    else:
        plt.show()
    plt.close('all')
    return


class InvalidValueError(Exception):
    def __init__(self, method, reason):
        self.method = method
        self.reason = reason

    def __str__(self):
        return (f"""
Input data contains {self.reason} values so {self.method} transformation
can not be applied.
""")
