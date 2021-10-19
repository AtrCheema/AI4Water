__all__ = ["tf", "keras", "torch",
           "xgboost_models", "catboost_models", "lightgbm_models", "sklearn_models",
           "VERSION_INFO"]

import os
import sys
from types import FunctionType

try:
    import sklearn
except ModuleNotFoundError:
    sklearn = None

def get_attributes(
        aus,
        what:str,
        retain:str=None,
        case_sensitive:bool=False
) ->dict:
    """gets all callable attributes of aus from what and saves them in dictionary
    with their names as keys. If case_sensitive is True, then the all keys are
    capitalized so that calling them becomes case insensitive. It is possible
    that some of the attributes of tf.keras.layers are callable but still not
    a valid `layer`, sor some attributes of tf.keras.losses are callable but
    still not valid losses, in that case the error will be generated from tensorflow.
    We are not catching those error right now.

    Arguments:
        aus : parent module
        what : child module/package
        retain : if duplicates of 'what' exist then whether to prefer class or function.
            For example, fastica and FastICA exist in sklearn.decomposition then if retain
            is 'function' then fastica will be kept, if retain is 'class' then FastICA is
            kept. If retain is None, then what comes later will overwrite the previously
            kept object.
        case_sensitive : whether to consider what as case-sensitive or not. In such
            a case, fastica and FastICA will both be saved as separate objects.
    Example
    ---------
    ```python
    >>> get_attributes(tf.keras, 'layers')  # will get all layers from tf.keras.layers
    ```
    """

    if retain:
        assert retain in ("class", "function")
    all_attrs = {}
    for l in dir(getattr(aus, what)):
        attr = getattr(getattr(aus, what), l)
        if callable(attr) and not l.startswith('_'):

            if not case_sensitive:
                l = l.upper()

            if l in all_attrs and retain == 'function':
                if isinstance(attr, FunctionType):
                    all_attrs[l] = attr
            elif l in all_attrs and retain == 'class':
                if not isinstance(attr, FunctionType):
                    all_attrs[l] = attr
            else:
                all_attrs[l] = attr

    return all_attrs

def get_sklearn_models():

    if sklearn is not None:
        # the following line must be executed in order for get_attributes to work, don't know why
        from sklearn.ensemble import RandomForestRegressor
        sk_maj_ver = int(sklearn.__version__.split('.')[0])
        sk_min_ver = int(sklearn.__version__.split('.')[1])
        if sk_maj_ver==0 and sk_min_ver < 24:
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

        skl_models = get_attributes(sklearn, "ensemble")
        skl_models.update(get_attributes(sklearn, "dummy"))
        skl_models.update(get_attributes(sklearn, "gaussian_process"))
        skl_models.update(get_attributes(sklearn, "compose"))
        skl_models.update(get_attributes(sklearn, "linear_model"))
        skl_models.update(get_attributes(sklearn, "multioutput"))
        skl_models.update(get_attributes(sklearn, "neighbors"))
        skl_models.update(get_attributes(sklearn, "neural_network"))
        skl_models.update(get_attributes(sklearn, "svm"))
        skl_models.update(get_attributes(sklearn, "tree"))
        skl_models.update(get_attributes(sklearn, "naive_bayes"))
        skl_models.update(get_attributes(sklearn, "kernel_ridge"))
        skl_models.update(get_attributes(sklearn, "isotonic"))

        skl_models.update({"HISTGRADIENTBOOSTINGREGRESSOR": HistGradientBoostingRegressor,
            "HISTGRADIENTBOOSTINGCLASSIFIER": HistGradientBoostingClassifier})
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
    import tcn
except ModuleNotFoundError:
    tcn = None

try:
    import torch
except Exception:  # there can be many reasons for unavailability(unproper installation) fo pytorch
    torch = None

catboost_models = {}

try:
    import catboost
    from catboost import CatBoostClassifier, CatBoostRegressor
    catboost_models.update({"CATBOOSTCLASSIFIER": CatBoostClassifier})
    catboost_models.update({"CATBOOSTREGRESSOR": CatBoostRegressor})

except ModuleNotFoundError:
    catboost = None


xgboost_models = {}

try:
    import xgboost
    from xgboost import XGBRegressor, XGBClassifier, XGBRFRegressor, XGBRFClassifier
    xgboost_models.update({
        "XGBOOSTREGRESSOR": XGBRegressor,
        "XGBOOSTCLASSIFIER": XGBClassifier,
        "XGBOOSTRFREGRESSOR": XGBRFRegressor,
        "XGBOOSTRFCLASSIFIER": XGBRFClassifier,
    })
except ModuleNotFoundError:
    xgboost = None

lightgbm_models = {}

try:
    import lightgbm
    from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
    lightgbm_models.update({"LGBMCLASSIFIER": LGBMClassifier,
                            "LGBMREGRESSOR": LGBMRegressor})
except ModuleNotFoundError:
    lightgbm = None

sklearn_models = get_sklearn_models()

if sklearn is not None:
    from sklearn.experimental import enable_iterative_imputer  # noqa
    imputations = get_attributes(sklearn, 'impute')
else:
    imputations = {}

keras = keras
torch = torch
tf = tf

def get_version_info():
    # todo, chekc which attributes are not available in different versions
    return {
    'python': sys.version,
    'os': os.name,
    'tensorflow': str(tf.__version__) if tf is not None else None,
    'tf_is_built_with_cuda': tf.test.is_built_with_cuda() if tf is not None else None,
    'is_built_with_gpu_support': tf.test.is_built_with_gpu_support() if tf is not None else None,
    'tf_is_gpu_available': tf.test.is_gpu_available() if tf is not None else None,
    'keras': str(keras.__version__) if keras is not None else None,
    'tcn': str(tcn.__version__) if tcn is not None else None,
    'pytorch': str(torch.__version__) if torch is not None else None,
    'catboost': str(catboost.__version__) if catboost is not None else None,
    'xgboost': str(xgboost.__version__) if xgboost is not None else None,
    'lightgbm': str(lightgbm.__version__) if lightgbm is not None else None,
    'sklearn': str(sklearn.__version__) if sklearn is not None else None,
    'tpot': str(tpot.__version__) if tpot else None,
    'eager_execution': tf.executing_eagerly() if tf is not None else None
}

VERSION_INFO = get_version_info()

if tf is not None:
    BACKEND = 'tensorflow'
elif torch is not None:
    BACKEND = 'pytorch'
else:
    BACKEND = None
