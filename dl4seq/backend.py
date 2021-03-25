__all__ = ["tf", "keras", "torch",
           "xgboost_models", "catboost_models", "lightgbm_models", "sklearn_models",
           "VERSION_INFO"]

from dl4seq.utils.utils import get_attributes

try:
    import sklearn
except ModuleNotFoundError:
    sklearn = None

def get_sklearn_models():

    if sklearn is not None:
        # the following line must be executed in order for get_attributes to work, don't know why
        from sklearn.ensemble import RandomForestRegressor
        if int(sklearn.__version__.split('.')[1]) < 24:
            from sklearn.neural_network import multilayer_perceptron
        else:
            from sklearn.neural_network import MLPClassifier
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.naive_bayes import GaussianNB
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.isotonic import isotonic_regression

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
except ModuleNotFoundError:
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

tpot_models = {}
try:
    import tpot
    from tpot import TPOTRegressor, TPOTClassifier
    tpot_models.update({'TPOTREGRESSOR': TPOTRegressor,
                       'TPOTCLASSIFIER': TPOTClassifier})
except ModuleNotFoundError:
    tpot = None

sklearn_models = get_sklearn_models()

if sklearn is not None:
    from sklearn.experimental import enable_iterative_imputer  # noqa
    imputations = get_attributes(sklearn, 'impute')
else:
    imputations = {}

keras = keras
torch = torch
tf = tf

VERSION_INFO = {
    'tensorflow': str(tf.__version__) if tf is not None else None,
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
