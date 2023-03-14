
"""
tensorflow, torch, numpy, matplotlib, random and other libraries are imported here
once and then used all over ai4water. This file does not import anything from
other files of ai4water.
"""
__all__ = ["np", "os", "plt", "mpl", "pd", "random", "scipy", "stats",
           "easy_mpl", "SeqMetrics",
           "sklearn",
           "xgboost", "catboost", "lightgbm",
           "skopt", "hyperopt", "hp", "optuna",
           "xr", "fiona", "netCDF4",
           "sns", "imageio", "shapefile", "tf", "torch", "keras",
           "requests", "plotly", "h5py", "lime",
           "xgboost_models", "catboost_models", "lightgbm_models", "sklearn_models",
           "get_attributes",
           "wandb", "WandbCallback",
           ]

from types import FunctionType

import os
import random

import easy_mpl
import scipy
from scipy import stats
import SeqMetrics
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import sklearn
except (ModuleNotFoundError, ImportError):
    sklearn = None


def get_attributes(
        aus,
        what: str,
        retain: str = None,
        case_sensitive: bool = False
) -> dict:
    """gets all callable attributes of aus from what and saves them in dictionary
    with their names as keys. If case_sensitive is True, then the all keys are
    capitalized so that calling them becomes case insensitive. It is possible
    that some of the attributes of tf.keras.layers are callable but still not
    a valid `layer`, sor some attributes of tf.keras.losses are callable but
    still not valid losses, in that case the error will be generated from tensorflow.
    We are not catching those error right now.

    Parameters
    ----------
        aus :
            parent module
        what : str
            child module/package
        retain : str, optional (default=None)
            if duplicates of 'what' exist then whether to prefer class or function.
            For example, fastica and FastICA exist in sklearn.decomposition then if retain
            is 'function' then fastica will be kept, if retain is 'class' then FastICA is
            kept. If retain is None, then what comes later will overwrite the previously
            kept object.
        case_sensitive : bool, optional (default=False)
            whether to consider what as case-sensitive or not. In such
            a case, fastica and FastICA will both be saved as separate objects.

    Example
    -------
        >>> get_attributes(tf.keras, 'layers')  # will get all layers from tf.keras.layers

    """

    if retain:
        assert retain in ("class", "function")
    all_attrs = {}
    for obj in dir(getattr(aus, what)):
        attr = getattr(getattr(aus, what), obj)
        if callable(attr) and not obj.startswith('_'):

            if not case_sensitive:
                obj = obj.upper()

            if obj in all_attrs and retain == 'function':
                if isinstance(attr, FunctionType):
                    all_attrs[obj] = attr
            elif obj in all_attrs and retain == 'class':
                if not isinstance(attr, FunctionType):
                    all_attrs[obj] = attr
            else:
                all_attrs[obj] = attr

    return all_attrs


def get_sklearn_models():

    if sklearn is not None:
        # the following line must be executed in order for get_attributes to work, don't know why
        from sklearn.ensemble import RandomForestRegressor
        sk_maj_ver = int(sklearn.__version__.split('.')[0])
        sk_min_ver = int(sklearn.__version__.split('.')[1])
        if sk_maj_ver == 0 and sk_min_ver < 24:
            from sklearn.neural_network import multilayer_perceptron
        else:
            from sklearn.neural_network import MLPClassifier
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.naive_bayes import GaussianNB
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.isotonic import isotonic_regression
        from sklearn.gaussian_process import GaussianProcessRegressor

        from sklearn.experimental import enable_hist_gradient_boosting
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.compose import TransformedTargetRegressor

        skl_models = get_attributes(sklearn, "ensemble", case_sensitive=True)
        skl_models.update(get_attributes(sklearn, "dummy", case_sensitive=True))
        skl_models.update(get_attributes(sklearn, "gaussian_process", case_sensitive=True))
        skl_models.update(get_attributes(sklearn, "compose", case_sensitive=True))
        skl_models.update(get_attributes(sklearn, "linear_model", case_sensitive=True))
        skl_models.update(get_attributes(sklearn, "multioutput", case_sensitive=True))
        skl_models.update(get_attributes(sklearn, "neighbors", case_sensitive=True))
        skl_models.update(get_attributes(sklearn, "neural_network", case_sensitive=True))
        skl_models.update(get_attributes(sklearn, "svm", case_sensitive=True))
        skl_models.update(get_attributes(sklearn, "tree", case_sensitive=True))
        skl_models.update(get_attributes(sklearn, "naive_bayes", case_sensitive=True))
        skl_models.update(get_attributes(sklearn, "kernel_ridge", case_sensitive=True))
        skl_models.update(get_attributes(sklearn, "isotonic", case_sensitive=True))

        from sklearn.calibration import CalibratedClassifierCV
        skl_models.update(get_attributes(sklearn, "calibration", case_sensitive=True))

        from sklearn.semi_supervised import LabelPropagation
        skl_models.update(get_attributes(sklearn, "semi_supervised", case_sensitive=True))

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        skl_models.update(get_attributes(sklearn, "discriminant_analysis", case_sensitive=True))

        skl_models.update({"HistGradientBoostingRegressor": HistGradientBoostingRegressor,
                           "HistGradientBoostingClassifier": HistGradientBoostingClassifier})
    else:
        skl_models = {}

    return skl_models


maj_version = 0
min_version = 0
try:
    from tensorflow import keras
    import tensorflow as tf
    maj_version = int(tf.__version__[0])
    min_version = int(tf.__version__[2])
except ModuleNotFoundError:
    keras = None
    tf = None

try:
    import skopt
except ModuleNotFoundError:
    skopt = None
try:
    import tcn
except ModuleNotFoundError:
    tcn = None

try:
    import torch
except (ModuleNotFoundError, ImportError):
    torch = None

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

try:
    import imageio
except (ModuleNotFoundError, ImportError):
    imageio = None


try:
    import shapefile
except (ModuleNotFoundError, ImportError):
    shapefile = None

catboost_models = {}

try:
    import hyperopt
except (ModuleNotFoundError, ImportError):
    hyperopt = None

if hyperopt is None:
    hp = None
else:
    from hyperopt import hp

try:
    import xarray as xr
except (ModuleNotFoundError, ImportError):
    xr = None

try:
    import fiona
except (ModuleNotFoundError, ImportError):
    fiona = None

try:
    import netCDF4
except (ModuleNotFoundError, ImportError):
    netCDF4 = None

try:
    import requests
except (ModuleNotFoundError, ImportError):
    requests = None

try:
    import optuna
except (ModuleNotFoundError, ImportError):
    optuna = None

try:
    import plotly
except ImportError:
    plotly = None

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

try:
    import lime
except ModuleNotFoundError:
    lime = None

try:
    import catboost
    from catboost import CatBoostClassifier, CatBoostRegressor
    catboost_models.update({"CatBoostClassifier": CatBoostClassifier})
    catboost_models.update({"CatBoostRegressor": CatBoostRegressor})

except ModuleNotFoundError:
    catboost = None


xgboost_models = {}

try:
    import xgboost
    from xgboost import XGBRegressor, XGBClassifier, XGBRFRegressor, XGBRFClassifier
    xgboost_models.update({
        "XGBRegressor": XGBRegressor,
        "XGBClassifier": XGBClassifier,
        "XGBRFRegressor": XGBRFRegressor,
        "XGBRFClassifier": XGBRFClassifier,
    })
except ModuleNotFoundError:
    xgboost = None

lightgbm_models = {}

try:
    import lightgbm
    from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
    lightgbm_models.update({"LGBMClassifier": LGBMClassifier,
                            "LGBMRegressor": LGBMRegressor})
except ModuleNotFoundError:
    lightgbm = None

sklearn_models = get_sklearn_models()

if sklearn is not None:
    from sklearn.experimental import enable_iterative_imputer  # noqa
    imputations = get_attributes(sklearn, 'impute', case_sensitive=True)
else:
    imputations = {}

keras = keras
torch = torch
tf = tf


try:
    import wandb
except ModuleNotFoundError:
    wandb = None

if tf is not None:
    BACKEND = 'tensorflow'
elif torch is not None:
    BACKEND = 'pytorch'
else:
    BACKEND = None
