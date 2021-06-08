import warnings
import os
import json
import datetime
from typing import Union
from shutil import rmtree
from copy import deepcopy
from typing import Any, Dict, Tuple
from collections import OrderedDict

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import skew, kurtosis, variation, gmean, hmean


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

    for _dir in ['activations', 'weights', 'data']:
        if not os.path.exists(os.path.join(save_dir, _dir)):
            os.makedirs(os.path.join(save_dir, _dir))

    return save_dir


def dateandtime_now()->str:
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


def save_config_file(path, config=None, errors=None, indices=None, others=None, name=''):

    sort_keys = True
    if errors is not None:
        suffix = dateandtime_now()
        fpath = path + "/errors_" + name +  suffix + ".json"
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
        if 'model' in data['config']:
            model = data['config']['model']
            if 'layers' not in model:  # because ML args which come algorithms may not be of json serializable.
                model = Jsonize(model)()
                data['config']['model'] = model

    with open(fpath, 'w') as fp:
        json.dump(data, fp, sort_keys=sort_keys, indent=4)

    return


def check_min_loss(epoch_losses, epoch, msg:str, save_fg:bool, to_save=None):
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

    # If learning rate for XGBoost is not provided use same as default for NN
    lr = kwargs.get("lr", 0.001)
    if 'model' in kwargs:
        model = kwargs['model']
        if 'layers' not  in model:
            if list(model.keys())[0].startswith("XGB"):
                if "learning_rate" not in model:
                        kwargs["model"]["learning_rate"] = lr

            if "batches" not in kwargs: # for ML, default batches will be 2d unless the user specifies otherwise.
                kwargs["batches"] = "2d"

            if "lookback" not in kwargs:
                kwargs["lookback"] = 1

    return kwargs


class make_model(object):

    def __init__(self,data, **kwargs):

        self.config = _make_model(data, **kwargs)


def process_io(data, **kwargs):

    inputs = kwargs.get('inputs', None)
    outputs = kwargs.get('outputs', None)

    if isinstance(inputs, str):
        inputs = [inputs]
    if isinstance(outputs, str):
        outputs = [outputs]

    if inputs is None:  # when inputs and outputs are not defined.
        if data is not None:  # when no data is given, inputs and outputs can not be defined.
            assert isinstance(data, pd.DataFrame)
            if outputs is None:
                inputs = list(data.columns)[0:-1]
                outputs = [list(data.columns)[-1]]
            elif isinstance(outputs, list):
                inputs = [col for col in list(data.columns) if col not in outputs]

    kwargs['inputs'] = inputs
    kwargs['outputs'] = outputs
    return kwargs


def _make_model(data, **kwargs):
    """
    This functions fills the default arguments needed to run all the models. All the input arguments can be overwritten
    by providing their name.
    :return
      nn_config: `dict`, contais parameters to build and train the neural network such as `layers`
      data_config: `dict`, contains parameters for data preparation/pre-processing/post-processing etc.
    """
    kwargs = process_io(data, **kwargs)

    default_model = {'layers': {
        "Dense_0": {'config': {'units': 64, 'activation': 'relu'}},
        "Flatten": {"config": {}},
        "Dense_3": {'config':  {'units': 1}},
        "Reshape": {"config": {"target_shape": (1, 1)}}
    }}

    kwargs = check_kwargs(**kwargs)

    def_prob = "regression"  # default problem
    model = kwargs.get('model', default_model)
    if 'layers' in model:
        def_cat = "DL"
        # for DL, the default problem case will be regression
    else:
        if list(model.keys())[0].startswith("CLASS"):
            def_prob = "classification"
        def_cat = "ML"

    if 'loss' in kwargs:
        if callable(kwargs['loss']) and hasattr(kwargs['loss'], 'name'):
            loss_name = kwargs['loss'].name
        else:
            loss_name = kwargs['loss']
        if loss_name in [
            'sparse_categorical_crossentropy',
            'categorical_crossentropy',
            'binary_crossentropy'
        ]:
            def_prob = 'classification'

    model_args = {

        'model': {'type': dict, 'default': default_model, 'lower': None, 'upper': None, 'between': None},
        'composite':    {'type': bool, 'default': False, 'lower': None, 'upper': None, 'between': None},   # for auto-encoders
        'lr':           {'type': float, 'default': 0.001, 'lower': None, 'upper': None, 'between': None},
        'optimizer':    {'type': str, 'default': 'adam', 'lower': None, 'upper': None, 'between': None},  # can be any of valid keras optimizers https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
        'loss':         {'type': [str, 'callable'], 'default': 'mse', 'lower': None, 'upper': None, 'between': None},
        'quantiles':    {'type': list, 'default': None, 'lower': None, 'upper': None, 'between': None},
        'epochs':       {'type': int, 'default': 14, 'lower': None, 'upper': None, 'between': None},
        'min_val_loss': {'type': float, 'default': 0.0001, 'lower': None, 'upper': None, 'between': None},
        'patience':     {'type': int, 'default': 100, 'lower': None, 'upper': None, 'between': None},
        'shuffle':      {'type': bool, 'default': True, 'lower': None, 'upper': None, 'between': None},
        'save_model':   {'type': bool, 'default': True, 'lower': None, 'upper': None, 'between': None},  # to save the best models using checkpoints
        'subsequences': {'type': int, 'default': 3, 'lower': 2, "upper": None, "between": None},  # used for cnn_lst structure
        'harhn_config': {'type': dict, 'default': {'n_conv_lyrs': 3,
                                                  'enc_units': 64,
                                                  'dec_units': 64}, 'lower': None, 'upper': None, 'between': None},
        'nbeats_options': {'type': dict, 'default': {
                                        'backcast_length': 15 if 'lookback' not in kwargs else int(kwargs['lookback']),
                                        'forecast_length': 1,
                                        'stack_types': ('generic', 'generic'),
                                        'nb_blocks_per_stack': 2,
                                        'thetas_dim': (4, 4),
                                        'share_weights_in_stack': True,
                                        'hidden_layer_units': 62
                                    }, 'lower': None, 'upper': None, 'between': None},
        'category':      {'type': str, 'default': def_cat, 'lower': None, 'upper': None, 'between': ["ML", "DL"]},
        'problem':       {'type': str, 'default': def_prob, 'lower': None, 'upper': None, 'between': ["regression", "classification"]}
    }

    data_args = {
        # buffer_size is only relevant if 'val_data' is same and shuffle is true. https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
        # It is used to shuffle tf.Dataset of training data.
        'buffer_size':       {'type': int, 'default': 100, 'lower': None, 'upper': None, 'between': None},
        # how many future values we want to predict
        'forecast_length':   {"type": int, "default": 1, 'lower': 1, 'upper': None, 'between': None},
        # comes handy if we want to skip certain batches from last
        'batches_per_epoch': {"type": int, "default": None, 'lower': None, 'upper': None, 'between': None},
        # if the shape of last batch is smaller than batch size and if we want to skip this last batch, set following to True.
        # Useful if we have fixed batch size in our model but the number of samples is not fully divisble by batch size
        'drop_remainder':    {"type": bool,  "default": False, 'lower': None, 'upper': None, 'between': None},
        # can be None or any of the method defined in AI4Water.utils.transformatinos.py
        'transformation':         {"type": [str, type(None), dict, list],   "default": 'minmax', 'lower': None, 'upper': None, 'between': None},
        # The term lookback has been adopted from Francois Chollet's "deep learning with keras" book. This means how many
        # historical time-steps of data, we want to feed to at time-step to predict next value. This value must be one
        # for any non timeseries forecasting related problems.
        'lookback':          {"type": int,   "default": 15, 'lower': 1, 'upper': None, 'between': None},
        'batch_size':        {"type": int,   "default": 32, 'lower': None, 'upper': None, 'between': None},
        # fraction of data to be used for validation
        'val_fraction':      {"type": float, "default": 0.2, 'lower': None, 'upper': None, 'between': None},
        # the following argument can be set to 'same' for cases if you want to use same data as validation as well as
        # test data. If it is 'same', then same fraction/amount of data will be used for validation and test.
        # If this is not string and not None, this will overwite `val_fraction`
        'val_data':          {"type": None,  "default": None, 'lower': None, 'upper': None, 'between': ["same", None]},
        # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        'steps_per_epoch':   {"type": int,   "default": None, 'lower': None, 'upper': None, 'between': None},
        # fraction of data to be used for test
        'test_fraction':     {"type": float, "default": 0.2, 'lower': None, 'upper': None, 'between': None},
        # write the data/batches as hdf5 file
        'cache_data':        {"type": bool,  "default": False, 'lower': None, 'upper': None, 'between': None},

        'allow_nan_labels':       {"type": int,  "default": 0, 'lower': 0, 'upper': 2, 'between': None},

        'input_nans':        {"type": None, "default": None, "lower": None, "upper": None, "between": None},
        # can be string or list of strings such as 'mse', 'kge', 'nse', 'pbias'
        'metrics':           {"type": list,  "default": ['nse'], 'lower': None, 'upper': None, 'between': None},
        # if true, model will use previous predictions as input
        'use_predicted_output': {"type": bool , "default": True, 'lower': None, 'upper': None, 'between': None},
        # If the model takes one kind of inputs that is it consists of only 1 Input layer, then the shape of the batches
        # will be inferred from the Input layer but for cases, the model takes more than 1 Input, then there can be two
        # cases, either all the inputs are of same shape or they  are not. In second case, we should overwrite `train_paras`
        # method. In former case, define whether the batches are 2d or 3d. 3d means it is for an LSTM and 2d means it is
        # for Dense layer.
        'batches':           {"type": str, "default": '3d', 'lower': None, 'upper': None, 'between': ["2d", "3d"]},
        # for reproducability
        'seed':              {"type": int, "default": 313, 'lower': None, 'upper': None, 'between': None},
        # how many steps ahead we want to predict
        'forecast_step':     {"type": int, "default": 0, 'lower': 0, 'upper': None, 'between': None},
        # step size of input data
        'input_step':        {"type": int, "default": 1, 'lower': 1, 'upper': None, 'between': None},
        # whether to use future input data for multi horizon prediction or not
        'known_future_inputs': {'type': bool, 'default': False, 'lower': None, 'upper': None, 'between': [True, False]},
        # input features in data_frame
        'inputs':            {"type": None, "default": None, 'lower': None, 'upper': None, 'between': None},
        # column in dataframe to bse used as output/target
        'outputs':           {"type": None, "default": None, 'lower': None, 'upper': None, 'between': None},
        # tuple of tuples where each tuple consits of two integers, marking the start and end of interval. An interval here
        # means chunk/rows from the input file/dataframe to be skipped when when preparing data/batches for NN. This happens
        # when we have for example some missing values at some time in our data. For further usage see `examples/using_intervals`
        "intervals":         {"type": None, "default": None, 'lower': None, 'upper': None, 'between': None}
    }

    model_config=  {key:val['default'] for key,val in model_args.items()}

    config = {key:val['default'] for key,val in data_args.items()}

    for key, val in kwargs.items():
        arg_name = key.lower()

        if arg_name in model_config:
            update_dict(arg_name, val, model_args, model_config)

        elif arg_name in config:
            update_dict(arg_name, val, data_args, config)

        # config may contain additional user defined args which will not be checked
        elif not kwargs.get('accept_additional_args', False):
            raise ValueError(f"Unknown keyworkd argument '{key}' provided")
        else:
            config[key] = val

    if config['allow_nan_labels']>0:
        assert 'layers' in model_config['model'], f"""
The model appears to be deep learning based because 
the argument `model` does not have layers. But you are
allowing nan labels in the targets.
However, `allow_nan_labels` should be > 0 only for deep learning models
"""

    config.update(model_config)

    assert type(config['inputs']) == type(config['outputs']), f"""
inputs is of type {config['inputs'].__class__.__name__} but outputs is of type {config['outputs'].__class__.__name__}.
Inputs and outputs must be of same type.
"""

    if isinstance(config['inputs'], dict):
        for data in [config['inputs'], config['outputs']]:
            for k,v in data.items():
                assert isinstance(v, list), f"{k} is of type {v.__class__.__name__} but it must of of type list"
    return config


def update_dict(key, val, dict_to_lookup, dict_to_update):
    """Updates the dictionary with key, val if the val is of type dtype."""
    dtype = dict_to_lookup[key]['type']
    low = dict_to_lookup[key]['lower']
    up = dict_to_lookup[key]['upper']
    between = dict_to_lookup[key]['between']

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
            if val != dict_to_lookup[key]['default']: # the default value may be None which will be different than dtype
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


def get_index(idx_array, fmt='%Y%m%d%H%M'):
    """ converts a numpy 1d array into pandas DatetimeIndex type."""

    if not isinstance(idx_array, np.ndarray):
        raise TypeError

    return pd.to_datetime(idx_array.astype(str), format=fmt)


class Jsonize(object):
    """Converts the objects to basic python types so that they can be written to json files.
    Examples:
    ---------
    >>>import numpy as np
    >>>from AI4Water.utils.utils import Jsonize
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
            return {k:self.stage2(v) for k,v in self.obj.items()}
        if hasattr(self.obj, '__len__') and not isinstance(self.obj, str):
            return [self.stage2(i) for i in self.obj]
        return str(self.obj)

    def stage2(self, obj):
        """Serializes one object"""
        if any([isinstance(obj, _type) for _type in [bool, set, type(None)]]) or callable(obj):
            return obj
        if 'int' in obj.__class__.__name__:
            return int(obj)
        if 'float' in obj.__class__.__name__:
            return float(obj)
        if isinstance(obj, dict):  # iterate over obj until it is a dictionary
            return {k: self.stage2(v) for k, v in obj.items()}

        if hasattr(obj, '__len__') and not isinstance(obj, str):
            if len(obj) > 1:  # it is a list like with length greater than 1
                return [self.stage2(i) for i in obj]
            elif isinstance(obj, list) and len(obj)>0:  # for cases like obj is [np.array([1.0])] -> [1.0]
                return [self.stage2(obj[0])]
            elif len(obj)==1:  # for cases like obj is np.array([1.0])
                if isinstance(obj, list) or isinstance(obj, tuple):
                    return obj  # for cases like (1, ) or [1,]
                return self.stage2(obj[0])
            else: # when object is []
                return obj

        # last solution, it must be of of string type
        return str(obj)


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


def find_best_weight(w_path:str, best:str="min", ext:str=".hdf5"):
    """Given weights in w_path, find the best weight."""
    assert best in ['min', 'max']
    all_weights = os.listdir(w_path)
    losses = {}
    for w in all_weights:
        wname = w.split(ext)[0]
        val_loss = str(float(wname.split('_')[2]))  # converting to float so that trailing 0 is removed
        losses[val_loss] = {'loss': wname.split('_')[2], 'epoch': wname.split('_')[1]}

    loss_array = np.array([float(l) for l in losses.keys()])
    if len(loss_array)==0:
        return None
    best_loss = getattr(np, best)(loss_array)
    best_weight = f"weights_{losses[str(best_loss)]['epoch']}_{losses[str(best_loss)]['loss']}.hdf5"
    return best_weight


def remove_all_but_best_weights(w_path, best:str="min", ext:str=".hdf5"):
    """removes all the weights from a folder except the best weigtht"""
    best_weights =  None
    if os.path.exists(w_path):
        # remove all but best weight
        all_weights = os.listdir(w_path)
        best_weights = find_best_weight(w_path, best=best, ext=ext)
        ws_to_del = [w for w in all_weights if w != best_weights]
        for w in ws_to_del:
            os.remove(os.path.join(w_path, w))

    return best_weights


def clear_weights(opt_dir, results:dict=None, keep=3, rename=True, write=True):
    """ Optimization will save weights of all the trained models, not all of them are useful. Here removing weights
    of all except top 3. The number of models whose weights to be retained can be set by `keep` para."""

    fname = 'sorted.json'
    if results is None:
        results = make_hpo_results(opt_dir)
        fname = 'sorted_folders.json'

    od = OrderedDict(sorted(results.items()))

    idx = 0
    best_results = {}
    for k,v in od.items():
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
        for k,v in od.items():
            if 'folder' in v:
                folder = v['folder']
                old_path = os.path.join(opt_dir, folder)
                new_path = os.path.join(opt_dir, str(idx+1) + "_" + folder)
                os.rename(old_path, new_path)

                if folder in best_results:
                    best_results[folder] = {'path': new_path, 'weights': best_results[folder]}

                idx += 1

    od = {k:Jsonize(v)() for k,v in od.items()}

    if write:
        sorted_fname = os.path.join(opt_dir, fname)
        with open(sorted_fname, 'w') as sfp:
            json.dump(od, sfp, sort_keys=True, indent=True)

    return best_results


def get_attributes(aus, what:str) ->dict:
    """ gets all callable attributes of aus e.g. from tf.keras.what and saves them in dictionary with their names all
    capitalized so that calling them becomes case insensitive. It is possible that some of the attributes of tf.keras.layers
    are callable but still not a valid `layer`, sor some attributes of tf.keras.losses are callable but still not valid
    losses, in that case the error will be generated from tensorflow. We are not catching those error right now."""
    all_attrs = {}
    for l in dir(getattr(aus, what)):
        attr = getattr(getattr(aus, what), l)
        if callable(attr) and not l.startswith('_'):
            all_attrs[l.upper()] = attr

    return all_attrs


def train_val_split(x, y, validation_split):
    if hasattr(x[0], 'shape'):
        # x is list of arrays
        # assert that all arrays are of equal length
        split_at = int(x[0].shape[0] * (1. - validation_split))
    else:
        split_at = int(len(x[0]) * (1. - validation_split))

    x, val_x = (slice_arrays(x, 0, split_at), slice_arrays(x, split_at))
    y, val_y = (slice_arrays(y, 0, split_at), slice_arrays(y, split_at))

    return x, y, val_x, val_y


def slice_arrays(arrays, start, stop=None):
    if isinstance(arrays, list):
        return [array[start:stop] for array in arrays]
    elif hasattr(arrays, 'shape'):
        return arrays[start:stop]


def split_by_indices(x, y, indices):
    """Slices the x and y arrays or lists of arrays by the indices"""

    def split_with_indices(data):
        if isinstance(data, list):
            _data = []

            for d in data:
                assert isinstance(d, np.ndarray)
                _data.append(d[indices])
        else:
            assert isinstance(data, np.ndarray)
            _data = data[indices]
        return _data

    x = split_with_indices(x)
    y = split_with_indices(y)

    return x, y


def ts_features(data: Union[np.ndarray, pd.DataFrame, pd.Series],
          precision:int=3,
          name:str='',
          st: int = 0,
          en: int = None,
          features: Union[list, str] = None
          ) ->dict:
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


def _missing_vals(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Modified after https://github.com/akanz1/klib/blob/main/klib/utils.py#L197
     Gives metrics of missing values in the dataset.
    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame
    Returns
    -------
    Dict[str, float]
        mv_total: float, number of missing values in the entire dataset
        mv_rows: float, number of missing values in each row
        mv_cols: float, number of missing values in each column
        mv_rows_ratio: float, ratio of missing values for each row
        mv_cols_ratio: float, ratio of missing values for each column
    """

    data = pd.DataFrame(data).copy()
    mv_rows = data.isna().sum(axis=1)
    mv_cols = data.isna().sum(axis=0)
    mv_total = data.isna().sum().sum()
    mv_rows_ratio = mv_rows / data.shape[1]
    mv_cols_ratio = mv_cols / data.shape[0]

    return {
        "mv_total": mv_total,
        "mv_rows": mv_rows,
        "mv_cols": mv_cols,
        "mv_rows_ratio": mv_rows_ratio,
        "mv_cols_ratio": mv_cols_ratio,
    }


def prepare_data(
        data: np.ndarray,
        lookback_steps:int,
        num_inputs:int=None,
        num_outputs:int=None,
        input_steps:int=1,
        forecast_step:int=0,
        forecast_len:int=1,
        known_future_inputs:bool=False,
        output_steps=1,
        mask:Union[int, float, np.ndarray]=None
)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    converts a numpy nd array into a supervised machine learning problem.

    Arguments:
        data np.ndarray :
            nd numpy array whose first dimension represents the number
            of examples and the second dimension represents the number of features.
            Some of those features will be used as inputs and some will be considered
            as outputs depending upon the values of `num_inputs` and `num_outputs`.
        lookback_steps int :
            number of previous steps/values to be used at one step.
        num_inputs int :
            default None, number of input features in data. If None,
            it will be calculated as features-outputs. The input data will be all
            from start till num_outputs in second dimension.
        num_outputs int :
            number of columns (from last) in data to be used as output.
            If None, it will be caculated as features-inputs.
        input_steps int :
            strides/number of steps in input data
        forecast_step int :
            must be greater than equal to 0, which t+ith value to
            use as target where i is the horizon. For time series prediction, we
            can say, which horizon to predict.
        forecast_len int :
            number of horizons/future values to predict.
        known_future_inputs bool : Only useful if `forecast_len`>1. If True, this
            means, we know and use 'future inputs' while making predictions at t>0
        output_steps int :
            step size in outputs. If =2, it means we want to predict
            every second value from the targets
        mask int/np.nan/1darray :
            If int, then the examples with these values in
            the output will be skipped. If array then it must be a boolean mask
            indicating which examples to include/exclude. The length of mask should
            be equal to the number of generated examples. The number of generated
            examples is difficult to prognose because it depend upon lookback, input_steps,
            and forecast_step. Thus it is better to provide an integer indicating
            which values in outputs are to be considered as invalid. Default is
            None, which indicates all the generated examples will be returned.

    Returns:
      x np.ndarray: numpy array of shape (examples, lookback, ins) consisting of input examples
      prev_y np.ndarray: numpy array consisting of previous outputs
      y np.ndarray: numpy array consisting of target values

    Given following example consisting of input/output pairs
    input1, input2, output1, output2, output 3
        1,     11,     21,       31,     41
        2,     12,     22,       32,     42
        3,     13,     23,       33,     43
        4,     14,     24,       34,     44
        5,     15,     25,       35,     45
        6,     16,     26,       36,     46
        7,     17,     27,       37,     47

    If we use following 2 time series as input
    1,     11,
    2,     12,
    3,     13,
    4,     14,
    5,     15,
    6,     16,
    7,     17,
                          input_features=2, lookback=7, input_steps=1
    and if we predict
    27, 37, 47     outputs=3, forecast_len=1,  horizon/forecast_step=0,

    if we predict
    28, 38, 48     outputs=3, forecast_length=1,  horizon/forecast_step=1,

    if we predict
    27, 37, 47
    28, 38, 48     outputs=3, forecast_length=2,  horizon/forecast_step=0,

    if we predict
    28, 38, 48
    29, 39, 49   outputs=3, forecast_length=3,  horizon/forecast_step=1,
    30, 40, 50

    if we predict
    38            outputs=1, forecast_length=3, forecast_step=0
    39
    40

    if we predict
    39            outputs=1, forecast_length=1, forecast_step=2

    if we predict
    39            outputs=1, forecast_length=3, forecast_step=2
    40
    41

    If we use following two time series as input
    1,     11,
    3,     13,
    5,     15,
    7,     17,

           then        input_features=2, lookback=4, input_steps=2

    If the input is
    1,     11,
    2,     12,
    3,     13,
    4,     14,
    5,     15,
    6,     16,
    7,     17,

    and target/output is
    25, 35, 45
    26, 36, 46
    27, 37, 47
    This means we make use of 'known future inputs'. This can be achieved using following configuration
    num_inputs=2, num_outputs=3, lookback_steps=4, forecast_len=3, forecast_step=1, known_future_inputs=True

    The general shape of output/target/label is
    (examples, num_outputs, forecast_length)

    The general shape of inputs/x is
    (examples, lookback_steps+forecast_len-1, ....num_inputs)

    ----------
    Example
    ---------
    ```python
    >>>import numpy as np
    >>>from AI4Water.utils.utils import prepare_data
    >>>examples = 50
    >>>data = np.arange(int(examples*5)).reshape(-1,examples).transpose()
    >>>data[0:10]
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
    >>>x, prevy, y = prepare_data(data, num_outputs=2, lookback_steps=4,
    ...    input_steps=2, forecast_step=2, forecast_len=4)
    >>>x[0]
       array([[  0.,  50., 100.],
              [  2.,  52., 102.],
              [  4.,  54., 104.],
              [  6.,  56., 106.]], dtype=float32)
    >>>y[0]
       array([[158., 159., 160., 161.],
              [208., 209., 210., 211.]], dtype=float32)

    >>>x, prevy, y = prepare_data(data, num_outputs=2, lookback_steps=4,
    ...    forecast_len=3, known_future_inputs=True)
    >>>x[0]
        array([[  0,  50, 100],
               [  1,  51, 101],
               [  2,  52, 102],
               [  3,  53, 103],
               [  4,  54, 104],
               [  5,  55, 105],
               [  6,  56, 106]])       # (7, 3)
    >>># it is import to note that although lookback_steps=4 but x[0] has shape of 7
    >>>y[0]

        array([[154., 155., 156.],
               [204., 205., 206.]], dtype=float32)  # (2, 3)
    ```
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

    time_steps = lookback_steps
    if known_future_inputs:
        lookback_steps = lookback_steps + forecast_len
        assert forecast_len>1, f"""
            known_futre_inputs should be True only when making predictions at multiple 
            horizons i.e. when forecast length/number of horizons to predict is > 1.
            known_future_inputs: {known_future_inputs}
            forecast_len: {forecast_len}"""

    examples = len(data)

    x = []
    prev_y = []
    y = []

    for i in range(examples - lookback_steps * input_steps + 1 - forecast_step - forecast_len + 1):
        stx, enx = i, i + lookback_steps * input_steps
        x_example = data[stx:enx:input_steps, 0:features - num_outputs]

        st, en = i, i + (lookback_steps - 1) * input_steps
        y_data = data[st:en:input_steps, features - num_outputs:]

        sty = (i + time_steps * input_steps) + forecast_step - input_steps
        eny = sty + forecast_len
        target = data[sty:eny, features - num_outputs:]

        x.append(np.array(x_example))
        prev_y.append(np.array(y_data))
        y.append(np.array(target))

    x = np.stack(x)
    prev_y = np.array([np.array(i, dtype=np.float32) for i in prev_y], dtype=np.float32)
    # transpose because we want labels to be of shape (examples, outs, forecast_length)
    y = np.array([np.array(i, dtype=np.float32).T for i in y], dtype=np.float32)


    if mask is not None:
        if isinstance(mask, np.ndarray):
            assert mask.ndim == 1
            assert len(x) == len(mask), f"Number of generated examples are {len(x)} but the length of mask is {len(mask)}"
        elif isinstance(mask, float) and np.isnan(mask):
            mask = np.invert(np.isnan(y))
            mask = np.array([all(i.reshape(-1,)) for i in mask])
        else:
            assert isinstance(mask, int), f"""
                    Invalid mask identifier given of type: {mask.__class__.__name__}"""
            mask = y!=mask
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


def init_subplots(width=None, height=None, nrows=1, ncols=1, **kwargs):
    """Initializes the fig for subplots"""
    plt.close('all')
    fig, axis = plt.subplots(nrows=nrows, ncols=ncols, **kwargs)
    if width is not None:
        fig.set_figwidth(width)
    if height is not None:
        fig.set_figheight(height)
    return fig, axis



def process_axis(axis,
                 data: Union[list, np.ndarray, pd.Series, pd.DataFrame],
                 x: Union[list, np.ndarray] = None,  # array to plot as x
                 marker='',
                 fillstyle=None,
                 linestyle='-',
                 c=None,
                 ms=6.0,  # markersize
                 label=None,  # legend
                 leg_pos="best",
                 bbox_to_anchor=None,  # will take priority over leg_pos
                 leg_fs=12,
                 leg_ms=1,  # legend scale
                 ylim=None,  # limit for y axis
                 x_label=None,
                 xl_fs=None,
                 y_label=None,
                 yl_fs=12,  # ylabel font size
                 yl_c='k',  # y label color, if 'same', c will be used else black
                 xtp_ls=None,  # x tick_params labelsize
                 ytp_ls=None,  # x tick_params labelsize
                 xtp_c='k',  # x tick colors if 'same' c will be used else black
                 ytp_c='k',  # y tick colors, if 'same', c will be used else else black
                 log=False,
                 show_xaxis=True,
                 top_spine=True,
                 bottom_spine=True,
                 invert_yaxis=False,
                 max_xticks=None,
                 min_xticks=None,
                 title=None,
                 title_fs=None,  # title fontszie
                 log_nz=False,
                 ):

    """Purpose to act as a middle man between axis.plot/plt.plot.
    Returns:
        axis
        """
    # TODO
    # default fontsizes should be same as used by matplotlib
    # should not complicate plt.plot or axis.plto
    # allow multiple plots on same axis

    if log and log_nz:
        raise ValueError

    use_third = False
    if x is not None:
        if isinstance(x, str):  # the user has not specified x so x is currently plot style.
            style = x
            x = None
            if marker == '.':
                use_third=  True

    if log_nz:
        data = deepcopy(data)
        _data = data.values
        d_nz_idx = np.where(_data > 0.0)
        data_nz = _data[d_nz_idx]
        d_nz_log = np.log(data_nz)
        _data[d_nz_idx] = d_nz_log
        _data = np.where(_data < 0.0, 0.0, _data)
        data = pd.Series(_data, index=data.index)

    if log:
        data = deepcopy(data)
        _data = np.where(data.values < 0.0, 0.0, data.values)
        print(len(_data[np.where(_data < 0.0)]))
        data = pd.Series(_data, index=data.index)

    if x is not None:
        axis.plot(x, data, fillstyle=fillstyle, color=c, marker=marker, linestyle=linestyle, ms=ms, label=label)
    elif use_third:
        axis.plot(data, style, color=c, ms=ms, label=label)
    else:
        axis.plot(data, fillstyle=fillstyle, color=c, marker=marker, linestyle=linestyle, ms=ms, label=label)

    ylc = c
    if yl_c != 'same':
        ylc = 'k'

    _kwargs = {}
    if label is not None:
        if label != "__nolabel__":
            if leg_fs is not None: _kwargs.update({'fontsize': leg_fs})
            if leg_ms is not None: _kwargs.update({'markerscale': leg_ms})
            if bbox_to_anchor is not None:
                _kwargs['bbox_to_anchor'] = bbox_to_anchor
            else:
                _kwargs['loc'] = leg_pos
            axis.legend(**_kwargs)

    if y_label is not None:
        axis.set_ylabel(y_label, fontsize=yl_fs, color=ylc)

    if log:
        axis.set_yscale('log')

    if invert_yaxis:
        axis.set_ylim(axis.get_ylim()[::-1])

    if ylim is not None:
        if not isinstance(ylim, tuple):
            raise TypeError("ylim must be tuple {} provided".format(ylim))
        axis.set_ylim(ylim)

    xtpc = c
    if xtp_c != 'same':
        xtpc = 'k'

    ytpc = c
    if ytp_c != 'same':
        ytpc = 'k'

    _kwargs = {'colors': xtpc}
    if x_label is not None or xtp_ls is not None: # better not change these paras if user has not defined any x_label
        if xtp_ls is not None:
            _kwargs.update({'labelsize': xtp_ls})
        axis.tick_params(axis="x", which='major', **_kwargs)

    _kwargs = {'colors': ytpc}
    if y_label is not None or ytp_ls is not None:
        if ytp_ls is not None:
            _kwargs.update({'labelsize': ytp_ls})
        axis.tick_params(axis="y", which='major', **_kwargs)

    axis.get_xaxis().set_visible(show_xaxis)

    _kwargs = {}
    if x_label is not None:
        if xl_fs is not None: _kwargs.update({'fontsize': xl_fs})
        axis.set_xlabel(x_label, **_kwargs)

    axis.spines['top'].set_visible(top_spine)
    axis.spines['bottom'].set_visible(bottom_spine)

    if min_xticks is not None:
        assert isinstance(min_xticks, int)
        assert isinstance(max_xticks, int)
        loc = mdates.AutoDateLocator(minticks=min_xticks, maxticks=max_xticks)
        axis.xaxis.set_major_locator(loc)
        fmt = mdates.AutoDateFormatter(loc)
        axis.xaxis.set_major_formatter(fmt)

    if title_fs is None:
        title_fs = plt.rcParams['axes.titlesize']

    if title is not None:
        axis.set_title(title, fontsize=title_fs)

    return axis


def plot(*args, show=True, **kwargs):
    """
    One liner plot function. It should not be more complex than axis.plot() or plt.plot()
    yet it must accomplish all in one line what requires multiple lines in matplotlib.
    args and kwargs can be anything which goes into plt.plot() or axis.plot().
    They can also be anything which goes into `process_axis`.
    """
    fig, axis = init_subplots()
    axis = process_axis(axis, *args, **kwargs)
    if kwargs.get('save', False):
        plt.savefig(f"{kwargs.get('name', 'fig.png')}")
    if show:
        plt.show()
    return axis


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
        else:
            return super(JsonEncoder, self).default(obj)
