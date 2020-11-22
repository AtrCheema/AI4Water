__all__ = ["tf", "keras", "torch"]

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

keras = keras
torch = torch
tf = tf

VERSION_INFO = {
    'tensorflow_version': str(tf.__version__) if tf is not None else None,
    'keras_version': str(keras.__version__) if keras is not None else None,
    'tcn_version': str(tcn.__version__) if tcn is not None else None,
    'pytorch_version': str(torch.__version__) if torch is not None else None
}
