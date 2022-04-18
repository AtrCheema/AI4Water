import copy
import os
import json
import pprint
import datetime
import warnings
from typing import Union
from shutil import rmtree
from collections import OrderedDict
from typing import Tuple, List
import collections.abc as collections_abc

import scipy
import numpy as np
import pandas as pd
from easy_mpl import imshow
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, variation, gmean, hmean

try:
    import wrapt
except ModuleNotFoundError:
    wrapt = None


MATRIC_TYPES = {
    "r2": "max",
    "nse": "max",
    "r2_score": "max",
    "kge": "max",
    "corr_coeff": "max",
    'accuracy': "max",
    'f1_score': 'max',
    "mse": "min",
    "rmse": "min",
    "mape": "min",
    "nrmse": "min",
}

ERROR_LABELS = {
    'r2': "$R^{2}$",
    'nse': 'NSE',
    'rmse': 'RMSE',
    'mse': 'MSE',
    'msle': 'MSLE',
    'nrmse': 'Normalized RMSE',
    'mape': 'MAPE',
    'r2_score': "$R^{2}$ Score"
}

def reset_seed(seed: Union[int, None], os=None, random=None, np=None, tf=None, torch=None):
    """
    Sets the random seed for a given module if the module is not None
    Arguments:
        seed : Value of seed to set. If None, then it means we don't wan't to set
        the seed.
        os : alias for `os` module of python
        random : alias for `random` module of python
        np : alias for `numpy` module
        tf : alias for `tensorflow` module.
        torch : alias for `pytorch` module.
        """
    if seed:
        if np:
            np.random.seed(seed)

        if random:
            random.seed(seed)

        if os:
            os.environ['PYTHONHASHSEED'] = str(seed)

        if tf:
            if int(tf.__version__.split('.')[0]) == 1:
                tf.compat.v1.random.set_random_seed(seed)
            elif int(tf.__version__.split('.')[0]) > 1:
                tf.random.set_seed(seed)

        if torch:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
    return


def maybe_create_path(prefix=None, path=None):
    if path is None:
        save_dir = dateandtime_now()
        model_dir = os.path.join(os.getcwd(), "results")

        if prefix:
            model_dir = os.path.join(model_dir, prefix)

        save_dir = os.path.join(model_dir, save_dir)
    else:
        save_dir = path

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for _dir in ['weights']:
        if not os.path.exists(os.path.join(save_dir, _dir)):
            os.makedirs(os.path.join(save_dir, _dir))

    return save_dir


def dateandtime_now() -> str:
    """
    Returns the datetime in following format as string
    YYYYMMDD_HHMMSS
    """
    jetzt = datetime.datetime.now()
    dt = ''
    for time in ['year', 'month', 'day', 'hour', 'minute', 'second']:
        _time = str(getattr(jetzt, time))
        if len(_time) < 2:
            _time = '0' + _time
        if time == 'hour':
            _time = '_' + _time
        dt += _time
    return dt


def dict_to_file(path, config=None, errors=None, indices=None, others=None, name=''):

    sort_keys = True
    if errors is not None:
        suffix = dateandtime_now()
        fpath = path + "/errors_" + name + suffix + ".json"
        # maybe some errors are not json serializable.
        for er_name, er_val in errors.items():
            if "int" in er_val.__class__.__name__:
                errors[er_name] = int(er_val)
            elif "float" in er_val.__class__.__name__:
                errors[er_name] = float(er_val)

        data = errors
    elif config is not None:
        fpath = path + "/config.json"
        data = config
        sort_keys = False
    elif indices is not None:
        fpath = path + "/indices.json"
        data = indices
    else:
        assert others is not None
        data = others
        fpath = path

    if 'config' in data:
        if data['config'].get('model', None) is not None:
            model = data['config']['model']
            if 'layers' not in model:  # because ML args which come algorithms may not be of json serializable.
                model = Jsonize(model)()
                data['config']['model'] = model

    with open(fpath, 'w') as fp:
        json.dump(data, fp, sort_keys=sort_keys, indent=4, cls=JsonEncoder)

    return


def check_min_loss(epoch_losses, epoch, msg: str, save_fg: bool, to_save=None):
    epoch_loss_array = epoch_losses[:-1]

    current_epoch_loss = epoch_losses[-1]

    if len(epoch_loss_array) > 0:
        min_loss = np.min(epoch_loss_array)
    else:
        min_loss = current_epoch_loss

    if np.less(current_epoch_loss, min_loss):
        msg = msg + "    {:10.5f} ".format(current_epoch_loss)

        if to_save is not None:
            save_fg = True
    else:
        msg = msg + "              "

    return msg, save_fg


def check_kwargs(**kwargs):

    mode = "ML"
    if kwargs.get('model', None) is not None:
        model = kwargs['model']
        if isinstance(model, dict):
            if 'layers' in model:
                is_custom_model=False
                model_name = None
                mode="DL"
            else:
                assert len(model)==1
                _model = list(model.keys())[0]
                if isinstance(_model, str):
                    model_name = _model
                    is_custom_model = False
                elif hasattr(_model, '__call__'):   # uninitiated class
                    check_attributes(_model, ['fit', 'predict', '__init__'])
                    model_name = _model.__name__
                    is_custom_model = True
                else:  # custom class is already initiated
                    check_attributes(_model, ['fit', 'predict'])
                    is_custom_model = True
                    model_name = _model.__class__.__name__

        # for case when model='randomforestregressor'
        elif isinstance(model, str):
            kwargs['model'] = {model: {}}
            is_custom_model = False
            model_name = model

        elif hasattr(model, '__call__'):  # uninitiated class
            check_attributes(model, ['fit', 'predict', '__init__'])
            model_name = model.__name__
            is_custom_model = True
            kwargs['model'] = {model: {}}

        else:
            check_attributes(model, ['fit', 'predict'])
            is_custom_model = True
            model_name = model.__class__.__name__
            kwargs['model'] = {model: {}}

        if mode=="ML":
            if "batches" not in kwargs:  # for ML, default batches will be 2d unless the user specifies otherwise.
                kwargs["batches"] = "2d"

            if "ts_args" not in kwargs:
                kwargs["ts_args"] = {'lookback': 1,
                                     'forecast_len': 1,
                                     'forecast_step': 0,
                                     'known_future_inputs': False,
                                     'input_steps': 1,
                                     'output_steps': 1}
    else:
        is_custom_model = False
        model_name = None

    if is_custom_model:
        if 'mode' not in kwargs:
            raise ValueError("""your must provide 'mode' keyword either as 
            mode='regression' or mode='classification' for custom models""")

    return kwargs, model_name, is_custom_model


class make_model(object):

    def __init__(self, **kwargs):

        self.config, self.data_config, self.opt_paras, self.orig_model = _make_model(**kwargs)


def process_io(**kwargs):

    input_features = kwargs.get('input_features', None)
    output_features = kwargs.get('output_features', None)

    if isinstance(input_features, str):
        input_features = [input_features]
    if isinstance(output_features, str):
        output_features = [output_features]

    kwargs['input_features'] = input_features
    kwargs['output_features'] = output_features
    return kwargs


def _make_model(**kwargs):
    """
    This functions fills the default arguments needed to run all the models.
    All the input arguments can be overwritten
    by providing their name.
    :return
      nn_config: `dict`, contais parameters to build and train the neural network
        such as `layers`
      data_config: `dict`, contains parameters for data preparation/pre-processing/post-processing etc.
    """

    kwargs = process_io(**kwargs)

    kwargs, model_name, is_custom_model = check_kwargs(**kwargs)

    model = kwargs.get('model', None)
    def_cat = None

    if model is not None:
        if 'layers' in model:
            def_cat = "DL"
            # for DL, the default mode case will be regression
        else:
            def_cat = "ML"

    accept_additional_args = False
    if 'accept_additional_args' in kwargs:
        accept_additional_args = kwargs.pop('accept_additional_args')

    model_args = {

        'model': {'type': dict, 'default': None, 'lower': None, 'upper': None, 'between': None},
        # can be None or any of the method defined in ai4water.utils.transformatinos.py
        'x_transformation': {"type": [str, type(None), dict, list], "default": None, 'lower': None,
                             'upper': None, 'between': None},
        'y_transformation': {"type": [str, type(None), dict, list], "default": None, 'lower': None,
                             'upper': None, 'between': None},
        # for auto-encoders
        'composite':    {'type': bool, 'default': False, 'lower': None, 'upper': None, 'between': None},
        'lr':           {'type': float, 'default': 0.001, 'lower': None, 'upper': None, 'between': None},
        # can be any of valid keras optimizers https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
        'optimizer':    {'type': str, 'default': 'adam', 'lower': None, 'upper': None, 'between': None},
        'loss':         {'type': [str, 'callable'], 'default': 'mse', 'lower': None, 'upper': None, 'between': None},
        'quantiles':    {'type': list, 'default': None, 'lower': None, 'upper': None, 'between': None},
        'epochs':       {'type': int, 'default': 14, 'lower': None, 'upper': None, 'between': None},
        'min_val_loss': {'type': float, 'default': 0.0001, 'lower': None, 'upper': None, 'between': None},
        'patience':     {'type': int, 'default': 100, 'lower': None, 'upper': None, 'between': None},
        'shuffle':      {'type': bool, 'default': True, 'lower': None, 'upper': None, 'between': None},
        # to save the best models using checkpoints
        'save_model':   {'type': bool, 'default': True, 'lower': None, 'upper': None, 'between': None},
        'backend':       {'type': None, 'default': 'tensorflow', 'lower': None, 'upper': None,
                          'between': ['tensorflow', 'pytorch']},
        # buffer_size is only relevant if 'val_data' is same and shuffle is true.
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
        # It is used to shuffle tf.Dataset of training data.
        'buffer_size': {'type': int, 'default': 100, 'lower': None, 'upper': None, 'between': None},
        # comes handy if we want to skip certain batches from last
        'batches_per_epoch': {"type": int, "default": None, 'lower': None, 'upper': None, 'between': None},
        # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        'steps_per_epoch': {"type": int, "default": None, 'lower': None, 'upper': None, 'between': None},
        # can be string or list of strings such as 'mse', 'kge', 'nse', 'pbias'
        'monitor': {"type": [list, type(None), str], "default": None, 'lower': None, 'upper': None, 'between': None},
        # todo, is it  redundant?
        # If the model takes one kind of input_features that is it consists of
        # only 1 Input layer, then the shape of the batches
        # will be inferred from the Input layer but for cases, the model takes more than 1 Input, then there can be two
        # cases, either all the input_features are of same shape or they
        # are not. In second case, we should overwrite `train_paras`
        # method. In former case, define whether the batches are 2d or 3d. 3d means it is for an LSTM and 2d means it is
        # for Dense layer.
        'batches': {"type": str, "default": '3d', 'lower': None, 'upper': None, 'between': ["2d", "3d"]},
        'prefix': {"type": str, "default": None, 'lower': None, 'upper': None, 'between': None},
        'path': {"type": str, "default": None, 'lower': None, 'upper': None, 'between': None},
        'kmodel': {'type': None, "default": None, 'lower': None, 'upper': None, 'between': None},
        'cross_validator': {'default': None, 'between': ['LeaveOneOut', 'kfold']},
        'wandb_config': {'type': dict, 'default': None, 'between': None},
        'val_metric': {'type': str, 'default': None},
        'model_name_': {'default': None},
        'is_custom_model_': {"default": None},
    }

    data_args = {
        # if the shape of last batch is smaller than batch size and if we
        # want to skip this last batch, set following to True.
        # Useful if we have fixed batch size in our model but the number of samples is not fully divisble by batch size
        'drop_remainder': {"type": bool, "default": False, 'lower': None, 'upper': None, 'between': [True, False]},
        'category': {'type': str, 'default': def_cat, 'lower': None, 'upper': None, 'between': ["ML", "DL"]},
        'mode': {'type': str, 'default': None, 'lower': None, 'upper': None,
                 'between': ["regression", "classification"]},
        'batch_size':        {"type": int,   "default": 32, 'lower': None, 'upper': None, 'between': None},
        'split_random': {'type': bool, 'default': False, 'between': [True, False]},
        # fraction of data to be used for validation
        'val_fraction':      {"type": float, "default": 0.2, 'lower': None, 'upper': None, 'between': None},
        # the following argument can be set to 'same' for cases if you want to use same data as validation as well as
        # test data. If it is 'same', then same fraction/amount of data will be used for validation and test.
        # If this is not string and not None, this will overwite `val_fraction`
        'indices':          {"type": dict,  "default": None, 'lower': None, 'upper': None, 'between': ["same", None]},
        # fraction of data to be used for test
        'train_fraction':     {"type": float, "default": 0.7, 'lower': None, 'upper': None, 'between': None},
        # write the data/batches as hdf5 file
        'save':        {"type": bool,  "default": False, 'lower': None, 'upper': None, 'between': None},
        'allow_nan_labels':       {"type": int,  "default": 0, 'lower': 0, 'upper': 2, 'between': None},

        'nan_filler':        {"type": None, "default": None, "lower": None, "upper": None, "between": None},

        # for reproducability
        'seed':              {"type": None, "default": 313, 'lower': None, 'upper': None, 'between': None},
        # input features in data_frame
        'input_features':            {"type": None, "default": None, 'lower': None, 'upper': None, 'between': None},
        # column in dataframe to bse used as output/target
        'output_features':           {"type": None, "default": None, 'lower': None, 'upper': None, 'between': None},
        # tuple of tuples where each tuple consits of two integers, marking the start and end
        # of interval. An interval here
        # means chunk/rows from the input file/dataframe to be skipped when when preparing
        # data/batches for NN. This happens
        # when we have for example some missing values at some time in our data.
        # For further usage see `examples/using_intervals`
        "intervals":         {"type": None, "default": None, 'lower': None, 'upper': None, 'between': None},
        'verbosity':         {"type": int, "default": 1, 'lower': None, 'upper': None, 'between': None},
        'teacher_forcing': {'type': bool, 'default': False, 'lower': None, 'upper': None, 'between': [True, False]},
        'dataset_args': {'type': dict, 'default': {}},
        'ts_args': {"type": dict, "default": {'lookback': 1,
                                              'forecast_len': 1,
                                              'forecast_step': 0,
                                              'known_future_inputs': False,
                                              'input_steps': 1,
                                              'output_steps': 1}}
    }

    model_config = {key: val['default'] for key, val in model_args.items()}

    config = {key: val['default'] for key, val in data_args.items()}

    opt_paras = {}
    # because there are two kinds of hpos which can be optimized
    # some can be in model config and others are in main config
    original_other_conf = {}
    original_mod_conf = {}

    for key, val in kwargs.items():
        arg_name = key.lower()  # todo, why this?

        if val.__class__.__name__ in ['Integer', "Real", "Categorical"]:
            opt_paras[key] = val
            val2 = val
            val = jsonize(val.rvs(1)[0])

            val2.name = key
            original_other_conf[key] = val2

        if key == 'model':
            val, _opt_paras, original_mod_conf = find_opt_paras_from_model_config(val)
            opt_paras.update(_opt_paras)

        if key == 'ts_args':
            val, _opt_paras = find_opt_paras_from_ts_args(val)
            opt_paras.update(_opt_paras)

        if arg_name in model_config:
            update_dict(arg_name, val, model_args, model_config)

        elif arg_name in config:
            update_dict(arg_name, val, data_args, config)

        elif arg_name in ['x_transformer_', 'y_transformer_', 'val_x_transformer_', 'val_y_transformer_']:
            pass

        # config may contain additional user defined args which will not be checked
        elif not accept_additional_args:
            raise ValueError(f"Unknown keyworkd argument '{key}' provided")
        else:
            config[key] = val

    if config['allow_nan_labels'] > 0:
        assert 'layers' in model_config['model'], f"""
The model appears to be deep learning based because 
the argument `model` does not have layers. But you are
allowing nan labels in the targets.
However, `allow_nan_labels` should be > 0 only for deep learning models
"""

    config.update(model_config)

    if isinstance(config['input_features'], dict):
        for data in [config['input_features'], config['output_features']]:
            for k, v in data.items():
                assert isinstance(v, list), f"{k} is of type {v.__class__.__name__} but it must of of type list"

    _data_config = {}
    for key, val in config.items():
        if key in data_args:
            _data_config[key] = val

    config['model_name_'] = model_name
    config['is_custom_model_'] = is_custom_model

    return config, _data_config, opt_paras, {'model': original_mod_conf, 'other': original_other_conf}


def update_dict(key, val, dict_to_lookup, dict_to_update):
    """Updates the dictionary with key, val if the val is of type dtype."""
    dtype = dict_to_lookup[key].get('type', None)
    low = dict_to_lookup[key].get('lower', None)
    up = dict_to_lookup[key].get('upper', None)
    between = dict_to_lookup[key].get('between', None)

    if dtype is not None:
        if isinstance(dtype, list):
            val_type = type(val)
            if 'callable' in dtype:
                if callable(val):
                    pass

            elif val_type not in dtype:
                raise TypeError("{} must be any of the type {} but it is of type {}"
                                .format(key, dtype, val.__class__.__name__))
        elif not isinstance(val, dtype):
            # the default value may be None which will be different than dtype
            if val != dict_to_lookup[key]['default']:
                raise TypeError(f"{key} must be of type {dtype} but it is of type {val.__class__.__name__}")

    if isinstance(val, int) or isinstance(val, float):
        if low is not None:
            if val < low:
                raise ValueError(f"The value '{val}' for '{key}' must be greater than '{low}'")
        if up is not None:
            if val > up:
                raise ValueError(f"The value '{val} for '{key} must be less than '{up}'")

    if isinstance(val, str):
        if between is not None:
            if val not in between:
                raise ValueError(f"Unknown value '{val}' for '{key}'. It must be one of '{between}'")

    dict_to_update[key] = val
    return


def deepcopy_dict_without_clone(d: dict) -> dict:
    """makes deepcopy of a dictionary without cloning it"""
    assert isinstance(d, dict)

    new_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            new_d[k] = deepcopy_dict_without_clone(v)
        elif hasattr(v, '__len__'):
            new_d[k] = v[:]
        else:
            new_d[k] = copy.copy(v)
    return new_d


def find_opt_paras_from_ts_args(ts_args:dict)->tuple:
    opt_paras = {}
    new_ts_args = {'lookback': 15,
                   'forecast_len': 1,
                   'forecast_step': 0,
                   'known_future_inputs': False,
                   'input_steps': 1,
                   'output_steps': 1}
    new_ts_args.update(ts_args)

    for k,v in ts_args.items():

        if v.__class__.__name__ in ['Integer', 'Real', 'Categorical']:
            if v.name is None or v.name.startswith("integer_") or v.name.startswith("real_"):
                v.name = k
            opt_paras[k] = v
            v = v.rvs(1)[0]
            new_ts_args[k] = v

    return new_ts_args, opt_paras

def find_opt_paras_from_model_config(
        config: Union[dict, str, None]
) -> Tuple[Union[dict, None, str], dict, Union[dict, str, None]]:

    opt_paras = {}

    if config is None or isinstance(config, str):
        return config, opt_paras, config

    assert isinstance(config, dict) and len(config) == 1

    if 'layers' in config:
        original_model_config, _ = process_config_dict(deepcopy_dict_without_clone(config['layers']), False)

        # it is a nn based model
        new_lyrs_config, opt_paras = process_config_dict(config['layers'])
        new_model_config = {'layers': new_lyrs_config}

    else:
        # it is a classical ml model
        _ml_config = {}
        ml_config: dict = list(config.values())[0]
        model_name = list(config.keys())[0]

        original_model_config, _ = process_config_dict(copy.deepcopy(config[model_name]), False)

        for k, v in ml_config.items():

            if v.__class__.__name__ in ['Integer', 'Real', 'Categorical']:

                if v.name is None or v.name.startswith("integer_") or v.name.startswith("real_"):
                    v.name = k
                opt_paras[k] = v
                v = v.rvs(1)[0]

            _ml_config[k] = v
        val = _ml_config
        new_model_config = {model_name: val}

    return new_model_config, opt_paras, original_model_config


def process_config_dict(config_dict: dict, update_initial_guess=True):
    """From a dicitonary defining structure of neural networks, this function
    finds out which are hyperparameters from them"""

    assert isinstance(config_dict, dict)

    opt_paras = {}

    def pd(d):
        for k, v in d.items():
            if isinstance(v, dict) and len(v) > 0:
                pd(v)
            elif v.__class__.__name__ in ["Integer", "Real", "Categorical"]:
                if v.name is None or v.name.startswith("integer_") or v.name.startswith("real_"):
                    v.name = k

                if v.name in opt_paras:
                    raise ValueError("Hyperparameters with duplicate name found. A hyperparameter to be "
                                     f"optimized with name '{v.name}' already exists")
                opt_paras[v.name] = v
                if update_initial_guess:
                    x0 = jsonize(v.rvs(1)[0])  # get initial guess
                    d[k] = x0  # inplace change of dictionary
                else:  # we most probably have updated the name, so doing inplace change
                    d[k] = v
        return

    pd(config_dict)
    return config_dict, opt_paras


def update_model_config(config: dict, suggestions):
    """returns the updated config if config contains any parameter from suggestions."""
    cc = copy.deepcopy(config)

    def update(c):
        for k, v in c.items():
            if isinstance(v, dict):
                update(v)
            elif v.__class__.__name__ in ["Integer", "Real", "Categorical"]:
                c[k] = suggestions[v.name]
        return

    update(cc)

    return cc


def to_datetime_index(idx_array, fmt='%Y%m%d%H%M') -> pd.DatetimeIndex:
    """ converts a numpy 1d array into pandas DatetimeIndex type."""

    if not isinstance(idx_array, np.ndarray):
        raise TypeError

    idx = pd.to_datetime(idx_array.astype(str), format=fmt)
    idx.freq = pd.infer_freq(idx)
    return idx


class Jsonize(object):
    """Converts the objects to json compatible format i.e to native python types.
    If the object is sequence then each member of the sequence is checked and
    converted if needed. Same goes for nested sequences like lists of lists
    or list of dictionaries.

    Examples:
    ---------
    >>>import numpy as np
    >>>from ai4water.utils.utils import Jsonize
    >>>a = np.array([2.0])
    >>>b = Jsonize(a)(a)
    >>>type(b)  # int
    """
    # TODO, repeating code in __call__ and stage2
    # TODO, stage2 not considering tuple

    def __init__(self, obj):
        self.obj = obj

    def __call__(self):
        """Serializes one object"""
        if 'int' in self.obj.__class__.__name__:
            return int(self.obj)
        if 'float' in self.obj.__class__.__name__:
            return float(self.obj)

        if isinstance(self.obj, dict):
            return {k: self.stage2(v) for k, v in self.obj.items()}

        if hasattr(self.obj, '__len__') and not isinstance(self.obj, str):
            return [self.stage2(i) for i in self.obj]

        # if obj is a python 'type'
        if type(self.obj).__name__ == type.__name__:
            return self.obj.__name__

        if isinstance(self.obj, collections_abc.Mapping):
            return dict(self.obj)

        if self.obj is Ellipsis:
            return {'class_name': '__ellipsis__'}

        if wrapt and isinstance(self.obj, wrapt.ObjectProxy):
            return self.obj.__wrapped__

        return str(self.obj)

    def stage2(self, obj):
        """Serializes one object"""
        if any([isinstance(obj, _type) for _type in [bool, set, type(None)]]) or callable(obj):
            return obj

        if 'int' in obj.__class__.__name__:
            return int(obj)

        if 'float' in obj.__class__.__name__:
            return float(obj)

        # tensorflow tensor shape
        if obj.__class__.__name__ == 'TensorShape':
            return obj.as_list()

        if isinstance(obj, dict):  # iterate over obj until it is a dictionary
            return {k: self.stage2(v) for k, v in obj.items()}

        if hasattr(obj, '__len__') and not isinstance(obj, str):
            if len(obj) > 1:  # it is a list like with length greater than 1
                return [self.stage2(i) for i in obj]
            elif isinstance(obj, list) and len(obj) > 0:  # for cases like obj is [np.array([1.0])] -> [1.0]
                return [self.stage2(obj[0])]
            elif len(obj) == 1:  # for cases like obj is np.array([1.0])
                if isinstance(obj, list) or isinstance(obj, tuple):
                    return obj  # for cases like (1, ) or [1,]
                return self.stage2(obj[0])
            else:  # when object is []
                return obj

        # if obj is a python 'type'
        if type(obj).__name__ == type.__name__:
            return obj.__name__

        if obj is Ellipsis:
            return {'class_name': '__ellipsis__'}

        if wrapt and isinstance(obj, wrapt.ObjectProxy):
            return obj.__wrapped__

        # last solution, it must be of of string type
        return str(obj)


def jsonize(obj):
    """functional interface to `Jsonize` class"""
    return Jsonize(obj)()


def make_hpo_results(opt_dir, metric_name='val_loss') -> dict:
    """Looks in opt_dir and saves the min val_loss with the folder name"""
    results = {}
    for folder in os.listdir(opt_dir):
        fname = os.path.join(os.path.join(opt_dir, folder), 'losses.csv')

        if os.path.exists(fname):
            df = pd.read_csv(fname)

            if 'val_loss' in df:
                min_val_loss = round(float(np.nanmin(df[metric_name])), 6)
                results[min_val_loss] = {'folder': os.path.basename(folder)}
    return results


def find_best_weight(w_path: str, best: str = "min", ext: str = ".hdf5", epoch_identifier: int = None):
    """Given weights in w_path, find the best weight.
    if epoch_identifier is given, it will be given priority to find best_weights
    The file_names are supposed in following format FileName_Epoch_Error.ext

    Note: if we are monitoring more than two metrics whose desired behaviour
        is opposite to each other then this method does not work as desired. However
        this can be avoided by specifying `epoch_identifier`.
    """
    assert best in ['min', 'max']
    all_weights = os.listdir(w_path)

    if len(all_weights) == 1:
        return all_weights[0]

    losses = {}
    for w in all_weights:
        wname = w.split(ext)[0]
        try:
            val_loss = str(float(wname.split('_')[2]))  # converting to float so that trailing 0 is removed
        except ValueError as e:
            raise ValueError(f"while trying to find best weight in {w_path} with {best} and"
                             f" {ext} and {epoch_identifier}"
                             f" encountered following error \n{e}")
        losses[val_loss] = {'loss': wname.split('_')[2], 'epoch': wname.split('_')[1]}

    best_weight = None
    if epoch_identifier:
        for v in losses.values():
            if str(epoch_identifier) in v['epoch']:
                best_weight = f"weights_{v['epoch']}_{v['loss']}.hdf5"
                break

    else:
        loss_array = np.array([float(l) for l in losses.keys()])
        if len(loss_array) == 0:
            return None
        best_loss = getattr(np, best)(loss_array)
        best_weight = f"weights_{losses[str(best_loss)]['epoch']}_{losses[str(best_loss)]['loss']}.hdf5"

    return best_weight


def remove_all_but_best_weights(w_path, best: str = "min", ext: str = ".hdf5"):
    """removes all the weights from a folder except the best weigtht"""
    best_weights = None
    if os.path.exists(w_path):
        # remove all but best weight
        all_weights = os.listdir(w_path)
        best_weights = find_best_weight(w_path, best=best, ext=ext)
        ws_to_del = [w for w in all_weights if w != best_weights]
        for w in ws_to_del:
            os.remove(os.path.join(w_path, w))

    return best_weights


def clear_weights(opt_dir, results: dict = None, keep=3, rename=True, write=True):
    """Optimization will save weights of all the trained models, not all of them
    are useful. Here removing weights of all except top 3. The number of models
    whose weights to be retained can be set by `keep` para.
    """
    fname = 'sorted.json'
    if results is None:
        results = make_hpo_results(opt_dir)
        fname = 'sorted_folders.json'

    od = OrderedDict(sorted(results.items()))

    idx = 0
    best_results = {}

    for v in od.values():
        if 'folder' in v:
            folder = v['folder']
            _path = os.path.join(opt_dir, folder)
            w_path = os.path.join(_path, 'weights')

            if idx > keep-1:
                if os.path.exists(w_path):
                    rmtree(w_path)
            else:
                best_weights = remove_all_but_best_weights(w_path)
                best_results[folder] = {'path': _path, 'weights': best_weights}

            idx += 1

    if rename:
        # append ranking of models to folder_names
        idx = 0
        for v in od.values():
            if 'folder' in v:
                folder = v['folder']
                old_path = os.path.join(opt_dir, folder)
                new_path = os.path.join(opt_dir, str(idx+1) + "_" + folder)
                os.rename(old_path, new_path)

                if folder in best_results:
                    best_results[folder] = {'path': new_path, 'weights': best_results[folder]}

                idx += 1

    od = {k: Jsonize(v)() for k, v in od.items()}

    if write:
        sorted_fname = os.path.join(opt_dir, fname)
        with open(sorted_fname, 'w') as sfp:
            json.dump(od, sfp, sort_keys=True, indent=True)

    return best_results


class TrainTestSplit(object):
    """train_test_split of sklearn can not be used for list of arrays so here
    we go
    """
    def __init__(
            self,
            test_fraction: float = 0.3,
            seed : int = None,
            train_indices: Union[list, np.ndarray] = None,
            test_indices: Union[list, np.ndarray] = None
    ):
        """
            test_fraction:
                test fraction. Must be greater than 0. and less than 1.
            seed:
                random seed for reproducibility
        """
        self.test_fraction = test_fraction
        self.state = np.random.RandomState(seed=seed)
        self.train_indices = train_indices
        self.test_indices = test_indices

    def split_by_slicing(
            self,
            x: Union[list, np.ndarray, pd.Series, pd.DataFrame, List[np.ndarray]],
            y: Union[list, np.ndarray, pd.Series, pd.DataFrame, List[np.ndarray]],
    ):
        """splits the x and y by slicing which is defined by `test_fraction`
        Arguments:
            x:
                arrays to split

                - array like such as list, numpy array or pandas dataframe/series
                - list of array like objects
            y:
                array like

                - array like such as list, numpy array or pandas dataframe/series
                - list of array like objects
        """
        def split_arrays(array):

            if isinstance(array, list):
                # x is list of arrays
                # assert that all arrays are of equal length
                assert len(set([len(_array) for _array in array])) == 1, f"arrays are of not same length"
                split_at = int(array[0].shape[0] * (1. - self.test_fraction))
            else:
                split_at = int(len(array) * (1. - self.test_fraction))

            train, test = (self.slice_arrays(array, 0, split_at), self.slice_arrays(array, split_at))

            return train, test

        train_x, test_x = split_arrays(x)
        train_y, test_y = split_arrays(y)

        return train_x, test_x, train_y, test_y

    def split_by_random(
            self,
            x: Union[list, np.ndarray, pd.Series, pd.DataFrame, List[np.ndarray]],
            y: Union[list, np.ndarray, pd.Series, pd.DataFrame, List[np.ndarray]],
    ):
        """
        splits the x and y by random splitting.
        Arguments:
            x:
                arrays to split

                - array like such as list, numpy array or pandas dataframe/series
                - list of array like objects
            y:
                array like

                - array like such as list, numpy array or pandas dataframe/series
                - list of array like objects

        """
        if isinstance(x, list):
            indices = np.arange(len(x[0]))
        else:
            indices = np.arange(len(x))

        indices = self.state.permutation(indices)

        split_at = int(len(indices) * (1. - self.test_fraction))
        train_indices, test_indices = (self.slice_arrays(indices, 0, split_at), self.slice_arrays(indices, split_at))

        train_x = self.slice_with_indices(x, train_indices)
        train_y = self.slice_with_indices(y, train_indices)

        test_x = self.slice_with_indices(x, test_indices)
        test_y = self.slice_with_indices(y, test_indices)

        return train_x, test_x, train_y, test_y

    def split_by_indices(
            self,
            x: Union[list, np.ndarray, pd.Series, pd.DataFrame, List[np.ndarray]],
            y: Union[list, np.ndarray, pd.Series, pd.DataFrame, List[np.ndarray]],
    ):
        """splits the x and y by user defined `train_indices` and `test_indices`"""

        return self.slice_with_indices(x, self.train_indices), \
               self.slice_with_indices(x, self.test_indices), \
               self.slice_with_indices(y, self.train_indices), \
               self.slice_with_indices(y, self.test_indices)

    @staticmethod
    def slice_with_indices(array, indices):
        if isinstance(array, list):
            _data = []

            for d in array:
                assert isinstance(d, np.ndarray)
                _data.append(d[indices])
        else:
            assert isinstance(array, np.ndarray) or isinstance(array, pd.DatetimeIndex)
            _data = array[indices]
        return _data

    @staticmethod
    def slice_arrays(arrays, start, stop=None):
        if isinstance(arrays, list):
            return [array[start:stop] for array in arrays]
        elif hasattr(arrays, 'shape'):
            return arrays[start:stop]

    def KFold_splits(
            self,
            x,
            y,
            n_splits,
            shuffle=True,
            random_state=None
    ):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, random_state=random_state,  shuffle=shuffle)
        spliter = kf.split(x[0] if isinstance(x, list) else x)

        for tr_idx, test_idx in spliter:

            if isinstance(x, list):
                train_x = [xarray[tr_idx] for xarray in x]
                test_x = [xarray[test_idx] for xarray in x]
            else:
                train_x = x[tr_idx]
                test_x = x[test_idx]

            if isinstance(y, list):
                train_y = [yarray[tr_idx] for yarray in y]
                test_y = [yarray[test_idx] for yarray in y]
            else:
                train_y = y[tr_idx]
                test_y = y[test_idx]

            yield (train_x, train_y), (test_x, test_y)


def ts_features(data: Union[np.ndarray, pd.DataFrame, pd.Series],
                precision: int = 3,
                name: str = '',
                st: int = 0,
                en: int = None,
                features: Union[list, str] = None
                ) -> dict:
    """
    Extracts features from 1d time series data. Features can be
        * point, one integer or float point value for example mean
        * 1D, 1D array for example sin(data)
        * 2D, 2D array for example wavelent transform
    Arguments:
        Gets all the possible stats about an array like object `data`.
        data: array like
        precision: number of significant figures
        name: str, only for erro or warning messages
        st: str/int, starting index of data to be considered.
        en: str/int, end index of data to be considered.
        features: name/names of features to extract from data.
    # information holding degree
    """
    point_features = {
        'Skew': skew,
        'Kurtosis': kurtosis,
        'Mean': np.nanmean,
        'Geometric Mean': gmean,
        'Standard error of mean': scipy.stats.sem,
        'Median': np.nanmedian,
        'Variance': np.nanvar,
        'Coefficient of Variation': variation,
        'Std': np.nanstd,
        'Non Zeros': np.count_nonzero,
        'Min': np.nanmin,
        'Max': np.nanmax,
        'Sum': np.nansum,
        'Counts': np.size
    }

    point_features_lambda = {
        'Shannon entropy': lambda x: np.round(scipy.stats.entropy(pd.Series(x).value_counts()), precision),
        'Negative counts': lambda x: int(np.sum(x < 0.0)),
        '90th percentile': lambda x: np.round(np.nanpercentile(x, 90), precision),
        '75th percentile': lambda x: np.round(np.nanpercentile(x, 75), precision),
        '50th percentile': lambda x: np.round(np.nanpercentile(x, 50), precision),
        '25th percentile': lambda x: np.round(np.nanpercentile(x, 25), precision),
        '10th percentile': lambda x: np.round(np.nanpercentile(x, 10), precision),
    }

    if not isinstance(data, np.ndarray):
        if hasattr(data, '__len__'):
            data = np.array(data)
        else:
            raise TypeError(f"{name} must be array like but it is of type {data.__class__.__name__}")

    if np.array(data).dtype.type is np.str_:
        warnings.warn(f"{name} contains string values")
        return {}

    if 'int' not in data.dtype.name:
        if 'float' not in data.dtype.name:
            warnings.warn(f"changing the dtype of {name} from {data.dtype.name} to float")
            data = data.astype(np.float64)

    assert data.size == len(data), f"""
data must be 1 dimensional array but it has shape {np.shape(data)}
"""
    data = data[st:en]
    stats = dict()

    if features is None:
        features = list(point_features.keys()) + list(point_features_lambda.keys())
    elif isinstance(features, str):
        features = [features]

    for feat in features:
        if feat in point_features:
            stats[feat] = np.round(point_features[feat](data), precision)
        elif feat in point_features_lambda:
            stats[feat] = point_features_lambda[feat](data)

    if 'Harmonic Mean' in features:
        try:
            stats['Harmonic Mean'] = np.round(hmean(data), precision)
        except ValueError:
            warnings.warn(f"""Unable to calculate Harmonic mean for {name}. Harmonic mean only defined if all
                          elements are greater than or equal to zero""", UserWarning)

    return Jsonize(stats)()


def prepare_data(
        data: np.ndarray,
        lookback: int,
        num_inputs: int = None,
        num_outputs: int = None,
        input_steps: int = 1,
        forecast_step: int = 0,
        forecast_len: int = 1,
        known_future_inputs: bool = False,
        output_steps: int = 1,
        mask: Union[int, float, np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    converts a numpy nd array into a supervised machine learning problem.

    Parameters
    ----------
        data :
            nd numpy array whose first dimension represents the number
            of examples and the second dimension represents the number of features.
            Some of those features will be used as inputs and some will be considered
            as outputs depending upon the values of `num_inputs` and `num_outputs`.
        lookback :
            number of previous steps/values to be used at one step.
        num_inputs :
            default None, number of input features in data. If None,
            it will be calculated as features-outputs. The input data will be all
            from start till num_outputs in second dimension.
        num_outputs :
            number of columns (from last) in data to be used as output.
            If None, it will be caculated as features-inputs.
        input_steps:
            strides/number of steps in input data
        forecast_step :
            must be greater than equal to 0, which t+ith value to
            use as target where i is the horizon. For time series prediction, we
            can say, which horizon to predict.
        forecast_len :
            number of horizons/future values to predict.
        known_future_inputs :
            Only useful if `forecast_len`>1. If True, this
            means, we know and use 'future inputs' while making predictions at t>0
        output_steps :
            step size in outputs. If =2, it means we want to predict
            every second value from the targets
        mask :
            If int, then the examples with these values in
            the output will be skipped. If array then it must be a boolean mask
            indicating which examples to include/exclude. The length of mask should
            be equal to the number of generated examples. The number of generated
            examples is difficult to prognose because it depend upon lookback, input_steps,
            and forecast_step. Thus it is better to provide an integer indicating
            which values in outputs are to be considered as invalid. Default is
            None, which indicates all the generated examples will be returned.

    Returns
    -------
        x : numpy array of shape (examples, lookback, ins) consisting of
        input examples
        prev_y : numpy array consisting of previous outputs
        y : numpy array consisting of target values

    Given following data consisting of input/output pairs

    +--------+--------+---------+---------+----------+
    | input1 | input2 | output1 | output2 | output 3 |
    +========+========+=========+=========+==========+
    |   1    |   11   |   21    |    31   |   41     |
    +--------+--------+---------+---------+----------+
    |   2    |   12   |   22    |    32   |   42     |
    +--------+--------+---------+---------+----------+
    |   3    |   13   |   23    |    33   |   43     |
    +--------+--------+---------+---------+----------+
    |   4    |   14   |   24    |    34   |   44     |
    +--------+--------+---------+---------+----------+
    |   5    |   15   |   25    |    35   |   45     |
    +--------+--------+---------+---------+----------+
    |   6    |   16   |   26    |    36   |   46     |
    +--------+--------+---------+---------+----------+
    |   7    |   17   |   27    |    37   |   47     |
    +--------+--------+---------+---------+----------+

    If we use following 2 time series as input

    +--------+--------+
    | input1 | input2 |
    +========+========+
    |  1     |  11    |
    +--------+--------+
    |     2  |  12    |
    +--------+--------+
    | 3      |  13    |
    +--------+--------+
    | 4      |  14    |
    +--------+--------+
    | 5      |  15    |
    +--------+--------+
    | 6      |  16    |
    +--------+--------+
    | 7      |  17    |
    +--------+--------+

    then  ``num_inputs`` =2, ``lookback`` =7, ``input_steps`` =1

    and if we want to predict

    +---------+---------+----------+
    | output1 | output2 | output 3 |
    +=========+=========+==========+
    |   27    |   37    |   47     |
    +---------+---------+----------+

    then ``num_outputs`` =3, ``forecast_len`` =1,  ``forecast_step`` =0,

    if we want to predict

    +---------+---------+----------+
    | output1 | output2 | output 3 |
    +=========+=========+==========+
    | 28      | 38      | 48       |
    +---------+---------+----------+

    then ``num_outputs`` =3, ``forecast_len`` =1,  ``forecast_step`` =1,

    if we want to predict

    +---------+---------+----------+
    | output1 | output2 | output 3 |
    +=========+=========+==========+
    |  27     |  37     |  47      |
    +---------+---------+----------+
    |  28     |  38     |  48      |
    +---------+---------+----------+

    then ``num_outputs`` =3, ``forecast_len`` =2,  horizon/forecast_step=0,

    if we want to predict

    +---------+---------+----------+
    | output1 | output2 | output 3 |
    +=========+=========+==========+
    |   28    |   38    |   48     |
    +---------+---------+----------+
    |   29    |   39    |   49     |
    +---------+---------+----------+
    |   30    |   40    |   50     |
    +---------+---------+----------+

    then ``num_outputs`` =3, ``forecast_len`` =3,  ``forecast_step`` =1,

    if we want to predict

    +---------+
    | output2 |
    +=========+
    |   38    |
    +---------+
    |   39    |
    +---------+
    |   40    |
    +---------+

    then ``num_outputs`` =1, ``forecast_len`` =3, ``forecast_step`` =0

    if we predict

    +---------+
    | output2 |
    +=========+
    | 39      |
    +---------+

    then ``num_outputs`` =1, ``forecast_len`` =1, ``forecast_step`` =2

    if we predict

    +---------+
    | output2 |
    +=========+
    | 39      |
    +---------+
    | 40      |
    +---------+
    | 41      |
    +---------+

     then ``num_outputs`` =1, ``forecast_len`` =3, ``forecast_step`` =2

    If we use following two time series as input

    +--------+--------+
    |input1  | input2 |
    +========+========+
    |   1    |  11    |
    +--------+--------+
    |   3    |  13    |
    +--------+--------+
    |   5    |  15    |
    +--------+--------+
    |   7    |  17    |
    +--------+--------+

    then   ``num_inputs`` =2, ``lookback`` =4, ``input_steps`` =2

    If the input is

    +--------+--------+
    | input1 | input2 |
    +========+========+
    |    1   |  11    |
    +--------+--------+
    |    2   |  12    |
    +--------+--------+
    |    3   |  13    |
    +--------+--------+
    |    4   |  14    |
    +--------+--------+
    |    5   |  15    |
    +--------+--------+
    |    6   |  16    |
    +--------+--------+
    |   7    |  17    |
    +--------+--------+

    and target/output is

    +---------+---------+----------+
    | output1 | output2 | output 3 |
    +=========+=========+==========+
    |    25   |    35   |    45    |
    +---------+---------+----------+
    |    26   |    36   |    46    |
    +---------+---------+----------+
    |    27   |    37   |    47    |
    +---------+---------+----------+

    This means we make use of ``known future inputs``. This can be achieved using
    following configuration
    num_inputs=2, num_outputs=3, lookback=4, forecast_len=3, forecast_step=1, known_future_inputs=True

    The general shape of output/target/label is
    (examples, num_outputs, forecast_len)

    The general shape of inputs/x is
    (examples, lookback + forecast_len-1, ....num_inputs)


    Examples:
        >>> import numpy as np
        >>> from ai4water.utils.utils import prepare_data
        >>> num_examples = 50
        >>> dataframe = np.arange(int(num_examples*5)).reshape(-1, num_examples).transpose()
        >>> dataframe[0:10]
        array([[  0,  50, 100, 150, 200],
               [  1,  51, 101, 151, 201],
               [  2,  52, 102, 152, 202],
               [  3,  53, 103, 153, 203],
               [  4,  54, 104, 154, 204],
               [  5,  55, 105, 155, 205],
               [  6,  56, 106, 156, 206],
               [  7,  57, 107, 157, 207],
               [  8,  58, 108, 158, 208],
               [  9,  59, 109, 159, 209]])
        >>> x, prevy, y = prepare_data(data, num_outputs=2, lookback=4,
        ...    input_steps=2, forecast_step=2, forecast_len=4)
        >>> x[0]
        array([[  0.,  50., 100.],
              [  2.,  52., 102.],
              [  4.,  54., 104.],
              [  6.,  56., 106.]], dtype=float32)
        >>> y[0]
        array([[158., 159., 160., 161.],
              [208., 209., 210., 211.]], dtype=float32)

        >>> x, prevy, y = prepare_data(data, num_outputs=2, lookback=4,
        ...    forecast_len=3, known_future_inputs=True)
        >>> x[0]
        array([[  0,  50, 100],
               [  1,  51, 101],
               [  2,  52, 102],
               [  3,  53, 103],
               [  4,  54, 104],
               [  5,  55, 105],
               [  6,  56, 106]])       # (7, 3)
        >>> # it is important to note that although lookback=4 but x[0] has shape of 7
        >>> y[0]
        array([[154., 155., 156.],
               [204., 205., 206.]], dtype=float32)  # (2, 3)
    """
    if not isinstance(data, np.ndarray):
        if isinstance(data, pd.DataFrame):
            data = data.values
        else:
            raise TypeError(f"unknown data type for data {data.__class__.__name__}")

    if num_inputs is None and num_outputs is None:
        raise ValueError("""
Either of num_inputs or num_outputs must be provided.
""")

    features = data.shape[1]
    if num_outputs is None:
        num_outputs = features - num_inputs

    if num_inputs is None:
        num_inputs = features - num_outputs

    assert num_inputs + num_outputs == features, f"""
num_inputs {num_inputs} + num_outputs {num_outputs} != total features {features}"""

    if len(data) <= 1:
        raise ValueError(f"Can not create batches from data with shape {data.shape}")

    time_steps = lookback
    if known_future_inputs:
        lookback = lookback + forecast_len
        assert forecast_len > 1, f"""
            known_futre_inputs should be True only when making predictions at multiple 
            horizons i.e. when forecast length/number of horizons to predict is > 1.
            known_future_inputs: {known_future_inputs}
            forecast_len: {forecast_len}"""

    examples = len(data)

    x = []
    prev_y = []
    y = []

    for i in range(examples - lookback * input_steps + 1 - forecast_step - forecast_len + 1):
        stx, enx = i, i + lookback * input_steps
        x_example = data[stx:enx:input_steps, 0:features - num_outputs]

        st, en = i, i + (lookback - 1) * input_steps
        y_data = data[st:en:input_steps, features - num_outputs:]

        sty = (i + time_steps * input_steps) + forecast_step - input_steps
        eny = sty + forecast_len
        target = data[sty:eny, features - num_outputs:]

        x.append(np.array(x_example))
        prev_y.append(np.array(y_data))
        y.append(np.array(target))

    x = np.stack(x)
    prev_y = np.array([np.array(i, dtype=np.float32) for i in prev_y], dtype=np.float32)
    # transpose because we want labels to be of shape (examples, outs, forecast_len)
    y = np.array([np.array(i, dtype=np.float32).T for i in y], dtype=np.float32)

    if mask is not None:
        if isinstance(mask, np.ndarray):
            assert mask.ndim == 1
            assert len(x) == len(mask), f"Number of generated examples are {len(x)} " \
                                        f"but the length of mask is {len(mask)}"
        elif isinstance(mask, float) and np.isnan(mask):
            mask = np.invert(np.isnan(y))
            mask = np.array([all(i.reshape(-1,)) for i in mask])
        else:
            assert isinstance(mask, int), f"""
                    Invalid mask identifier given of type: {mask.__class__.__name__}"""
            mask = y != mask
            mask = np.array([all(i.reshape(-1,)) for i in mask])

        x = x[mask]
        prev_y = prev_y[mask]
        y = y[mask]

    return x, prev_y, y


def find_tot_plots(features, max_subplots):

    tot_plots = np.linspace(0, features, int(features / max_subplots) + 1 if features % max_subplots == 0 else int(
        features / max_subplots) + 2)
    # converting each value to int because linspace can return array containing floats if features is odd
    tot_plots = [int(i) for i in tot_plots]
    return tot_plots



class JsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if 'int' in obj.__class__.__name__:
            return int(obj)
        elif 'float' in obj.__class__.__name__:
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif 'bool' in obj.__class__.__name__:
            return bool(obj)
        elif callable(obj) and hasattr(obj, '__module__'):
            return obj.__module__
        else:
            return super(JsonEncoder, self).default(obj)


def plot_activations_along_inputs(
        data: np.ndarray,
        activations: np.ndarray,
        observations: np.ndarray,
        predictions: np.ndarray,
        in_cols: list,
        out_cols: list,
        lookback: int,
        name: str,
        path: str,
        vmin=None,
        vmax=None,
        show=False
):

    # activation must be of shape (num_examples, lookback, input_features)
    assert activations.shape[1] == lookback
    assert activations.shape[2] == len(in_cols), f'{activations.shape}, {len(in_cols)}'

    # data is of shape (num_examples, input_features)
    assert data.shape[1] == len(in_cols)

    assert len(data) == len(activations)

    for out in range(len(out_cols)):
        pred = predictions[:, out]
        obs = observations[:, out]
        out_name = out_cols[out]

        for idx in range(len(in_cols)):
            plt.close('all')
            fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='all')
            fig.set_figheight(12)

            ax1.plot(data[:, idx], label=in_cols[idx])
            ax1.legend()
            ax1.set_title('activations w.r.t ' + in_cols[idx])
            ax1.set_ylabel(in_cols[idx])

            ax2.plot(pred, label='Prediction')
            ax2.plot(obs, '.', label='Observed')
            ax2.legend()
            ytick_labels = [f"t-{int(i)}" for i in np.linspace(lookback - 1, 0, lookback)]
            axis, im = imshow(activations[:, :, idx].transpose(),
                              vmin=vmin,
                              vmax=vmax,
                              aspect="auto",
                              ax = ax3,
                              xlabel="Examples",
                              ylabel="lookback steps",
                              show=False,
                              yticklabels=ytick_labels)
            fig.colorbar(im, orientation='horizontal', pad=0.2)
            plt.subplots_adjust(wspace=0.005, hspace=0.005)
            _name = f'attn_weights_{out_name}_{name}_'
            plt.savefig(os.path.join(path, _name) + in_cols[idx], dpi=400, bbox_inches='tight')

            if show:
                plt.show()

            plt.close('all')

    return


def print_something(something, prefix=''):
    """prints shape of some python object"""
    if isinstance(something, np.ndarray):
        print(f"{prefix} shape: ", something.shape)
    elif isinstance(something, list):
        print(f"{prefix} shape: ", [thing.shape for thing in something if isinstance(thing, np.ndarray)])
    elif isinstance(something, dict):
        print(f"{prefix} shape: ")
        pprint.pprint({k: v.shape for k, v in something.items()}, width=40)
    elif something is not None:
        print(f"{prefix} shape: ", something.shape)
        print(something)
    else:
        print(something)


def maybe_three_outputs(data, teacher_forcing=False):
    """num_outputs: how many outputs from data we want"""
    if teacher_forcing:
        num_outputs = 3
    else:
        num_outputs = 2

    if num_outputs == 2:
        if len(data) == 2:
            return data[0], data[1]
        elif len(data) == 3:
            return data[0], data[2]
    else:
        if len(data)==3:
            return [data[0], data[1]], data[2]
        # DA, IA-LSTM models return [x,prevy],y even when teacher_forcing is on!
        return data


def get_version_info(
        **kwargs
) -> dict:
    # todo, chekc which attributes are not available in different versions
    import sys
    info = {'python': sys.version, 'os': os.name}
    if kwargs.get('tf', None):
        tf = kwargs['tf']
        info['tf_is_built_with_cuda'] = tf.test.is_built_with_cuda()
        info['is_built_with_gpu_support'] = tf.test.is_built_with_gpu_support()
        info['tf_is_gpu_available'] = tf.test.is_gpu_available()
        info['eager_execution'] = tf.executing_eagerly()

    for k, v in kwargs.items():
        if v is not None:
            info[k] = getattr(v, '__version__', 'NotDefined')

    return info

def check_attributes(model, attributes):
    for method in attributes:
        if not hasattr(model, method):
            raise ValueError(f"your custom class does not have {method}")


def get_nrows_ncols(n_rows, n_subplots)->"tuple[int, int]":

    if n_rows is None:
        n_rows = int(np.sqrt(n_subplots))
    n_cols = max(int(n_subplots / n_rows), 1)  # ensure n_cols != 0
    n_rows = int(n_subplots / n_cols)

    while not ((n_subplots / n_cols).is_integer() and
               (n_subplots / n_rows).is_integer()):
        n_cols -= 1
        n_rows = int(n_subplots / n_cols)
    return n_rows, n_cols
