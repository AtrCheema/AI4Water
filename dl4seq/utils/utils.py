import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import os
from shutil import rmtree
import datetime
import json
import pandas as pd
from skopt.plots import plot_evaluations, plot_objective, plot_convergence
from skopt.utils import dump
from pickle import PicklingError
import matplotlib as mpl
from scipy.stats import skew, kurtosis, variation, gmean, hmean
import scipy
import warnings


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


def dateandtime_now():
    """ returns the datetime in following format
    YYYYMMDD_HHMMSS
    """
    jetzt = datetime.datetime.now()
    dt = str(jetzt.year)
    for time in ['month', 'day', 'hour', 'minute', 'second']:
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

    with open(fpath, 'w') as fp:
        try:
            json.dump(data, fp, sort_keys=sort_keys, indent=4)
        except TypeError:
            json.dump(str(data), fp, sort_keys=sort_keys, indent=4)
    return


def skopt_plots(search_result, pref=os.getcwd()):

    plt.close('all')
    _ = plot_evaluations(search_result)
    plt.savefig(os.path.join(pref , 'evaluations'), dpi=400, bbox_inches='tight')

    plt.close('all')
    _ = plot_objective(search_result)
    plt.savefig(os.path.join(pref , 'objective'), dpi=400, bbox_inches='tight')

    plt.close('all')
    _ = plot_convergence(search_result)
    plt.savefig(os.path.join(pref , 'convergence'), dpi=400, bbox_inches='tight')


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
    lr = kwargs["lr"] if "lr" in kwargs else 0.001
    if "ml_model" in kwargs:
        if kwargs['ml_model'].upper().startswith("XGB"):
            if "ml_model_args" in kwargs:
                if "learning_rate" not in kwargs["ml_model_args"]:
                    kwargs["ml_model_args"]["learning_rate"] = lr

        if "batches" not in kwargs: # for ML, default batches will be 2d unless the user specifies otherwise.
            kwargs["batches"] = "2d"

        if "lookback" not in kwargs:
            kwargs["lookback"] = 1

    return kwargs


class make_model(object):

    def __init__(self, **kwargs):

        data_config, model_config = _make_model(**kwargs)
        self.data = data_config
        self.model = model_config


def _make_model(**kwargs):
    """
    This functions fills the default arguments needed to run all the models. All the input arguments can be overwritten
    by providing their name.
    :return
      nn_config: `dict`, contais parameters to build and train the neural network such as `layers`
      data_config: `dict`, contains parameters for data preparation/pre-processing/post-processing etc.
    """
    kwargs = check_kwargs(**kwargs)

    def_prob = "regression"
    if "ml_model" not in kwargs:
        def_cat = "DL"
        # for DL, the default problem case will be regression
    else:
        if kwargs["ml_model"].upper().startswith("CLASS"):
            def_prob = "classification"
        def_cat = "ML"



    dpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    fname = os.path.join(dpath, "nasdaq100_padding.csv")

    if not os.path.exists(dpath):
        os.makedirs(dpath)

    if not os.path.exists(fname):
        print(f"downloading file to {fname}")
        df = pd.read_csv("https://raw.githubusercontent.com/KurochkinAlexey/DA-RNN/master/nasdaq100_padding.csv")
        df.to_csv(fname)
    df = pd.read_csv(fname)
    in_cols = list(df.columns)
    in_cols.remove("NDX")

    model_args = {
        'enc_config': {'type': dict, 'default': {'n_h': 20,  # length of hidden state m
                                'n_s': 20,  # length of hidden state m
                                'm': 20,  # length of hidden state m
                                'enc_lstm1_act': None,
                                'enc_lstm2_act': None,
                                }, 'lower': None, 'upper': None, 'between': None},
        'dec_config': {'': dict, 'default': {        # arguments for decoder/outputAttention in Dual stage attention
                                            'p': 30,
                                            'n_hde0': 30,
                                            'n_sde0': 30
                                            }, 'lower': None, 'upper': None, 'between': None},
        'layers': {'type': dict, 'default': {
                                "Dense_0": {'config': {'units': 64, 'activation': 'relu'}},
                                "Dense_3": {'config':  {'units': 1}}
                                 }, 'lower': None, 'upper': None, 'between': None},
        'composite':    {'type': bool, 'default': False, 'lower': None, 'upper': None, 'between': None},   # for auto-encoders
        'lr':           {'type': float, 'default': 0.001, 'lower': None, 'upper': None, 'between': None},
        'optimizer':    {'type': str, 'default': 'adam', 'lower': None, 'upper': None, 'between': None},  # can be any of valid keras optimizers https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
        'loss':         {'type': str, 'default': 'mse', 'lower': None, 'upper': None, 'between': None},
        'quantiles':    {'type': list, 'default': None, 'lower': None, 'upper': None, 'between': None},
        'epochs':       {'type': int, 'default': 14, 'lower': None, 'upper': None, 'between': None},
        'min_val_loss': {'type': float, 'default': 0.0001, 'lower': None, 'upper': None, 'between': None},
        'patience':     {'type': int, 'default': 100, 'lower': None, 'upper': None, 'between': None},
        'shuffle':      {'type': bool, 'default': True, 'lower': None, 'upper': None, 'between': None},
        'save_model':   {'type': bool, 'default': True, 'lower': None, 'upper': None, 'between': None},  # to save the best models using checkpoints
        'subsequences': {'type': int, 'default': 3, 'lower': 2, "upper": None, "between": None},  # used for cnn_lst structure
        'HARHN_config': {'type': dict, 'default': {'n_conv_lyrs': 3,
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
        'ml_model':      {'type': str, 'default': None, 'lower': None, 'upper': None, 'between': None},  # name of machine learning model
        'ml_model_args': {'type': dict, 'default':{}, 'lower': None, 'upper': None, 'between': None},  # arguments to instantiate/initiate ML model
        'category':      {'type': str, 'default': def_cat, 'lower': None, 'upper': None, 'between': ["ML", "DL"]},
        'problem':       {'type': str, 'default': def_prob, 'lower': None, 'upper': None, 'between': ["regression", "classification"]}
    }

    data_args = {
        # buffer_size is only relevant if 'val_data' is same and shuffle is true. https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
        'buffer_size':       {'type': int, 'default': 100, 'lower': None, 'upper': None, 'between': None}, # It is used to shuffle tf.Dataset of training data.
        'forecast_length':   {"type": int, "default": 1, 'lower': 1, 'upper': None, 'between': None},   # how many future values we want to predict
        'batches_per_epoch': {"type": int, "default": None, 'lower': None, 'upper': None, 'between': None},  # comes handy if we want to skip certain batches from last
    # if the shape of last batch is smaller than batch size and if we want to skip this last batch, set following to True.
    # Useful if we have fixed batch size in our model but the number of samples is not fully divisble by batch size
        'drop_remainder':    {"type": bool,  "default": False, 'lower': None, 'upper': None, 'between': None},
        'transformation':         {"type": [str, type(None), dict, list],   "default": 'minmax', 'lower': None, 'upper': None, 'between': None},  # can be None or any of the method defined in scalers.py
        # The term lookback has been adopted from Francois Chollet's "deep learning with keras" book. This means how many
        # historical time-steps of data, we want to feed to at time-step to predict next value. This value must be one
        # for any non timeseries forecasting related problems.
        'lookback':          {"type": int,   "default": 15, 'lower': 1, 'upper': None, 'between': None},
        'batch_size':        {"type": int,   "default": 32, 'lower': None, 'upper': None, 'between': None},
        'val_fraction':      {"type": float, "default": 0.2, 'lower': None, 'upper': None, 'between': None}, # fraction of data to be used for validation
        # the following argument can be set to 'same' for cases if you want to use same data as validation as well as
        # test data. If it is 'same', then same fraction/amount of data will be used for validation and test.
        'val_data':          {"type": None,  "default": None, 'lower': None, 'upper': None, 'between': ["same", None]}, # If this is not string and not None, this will overwite `val_fraction`
        'steps_per_epoch':   {"type": int,   "default": None, 'lower': None, 'upper': None, 'between': None},  # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        'test_fraction':     {"type": float, "default": 0.2, 'lower': None, 'upper': None, 'between': None},   # fraction of data to be used for test
        'cache_data':        {"type": bool,  "default": False, 'lower': None, 'upper': None, 'between': None},   # write the data/batches as hdf5 file
        'ignore_nans':       {"type": bool,  "default": False, 'lower': None, 'upper': None, 'between': None},  # if True, and if target values contain Nans, those samples will not be ignored
        'metrics':           {"type": list,  "default": ['nse'], 'lower': None, 'upper': None, 'between': None},  # can be string or list of strings such as 'mse', 'kge', 'nse', 'pbias'
        'use_predicted_output': {"type": bool , "default": True, 'lower': None, 'upper': None, 'between': None},  # if true, model will use previous predictions as input
    # If the model takes one kind of inputs that is it consists of only 1 Input layer, then the shape of the batches
    # will be inferred from this Input layer but for cases,  the model takes more than 1 Input, then there can be two
    # cases, either all the inputs are of same shape or they  are not. In second case, we should overwrite `train_paras`
    # method. In former case, define whether the batches are 2d or 3d. 3d means it is for an LSTM and 2d means it is
    # for Dense layer.
        'batches':           {"type": str, "default": '3d', 'lower': None, 'upper': None, 'between': ["2d", "3d"]},
        'seed':              {"type": int, "default": 313, 'lower': None, 'upper': None, 'between': None},  # for reproducability
        'forecast_step':     {"type": int, "default": 0, 'lower': 0, 'upper': None, 'between': None},  # how many steps ahead we want to predict
        'input_step':        {"type": int, "default": 1, 'lower': 1, 'upper': None, 'between': None},  # step size of input data
        'inputs':            {"type": list, "default": in_cols, 'lower': None, 'upper': None, 'between': None}, # input features in data_frame
        'outputs':           {"type": list, "default": ["NDX"], 'lower': None, 'upper': None, 'between': None}, # column in dataframe to bse used as output/target
    # tuple of tuples where each tuple consits of two integers, marking the start and end of interval. An interval here
    # means chunk/rows from the input file/dataframe to be skipped when when preparing data/batches for NN. This happens
    # when we have for example some missing values at some time in our data. For further usage see `examples/using_intervals`
        "intervals":         {"type": tuple, "default": None, 'lower': None, 'upper': None, 'between': None}
    }

    model_config=  {key:val['default'] for key,val in model_args.items()}

    data_config=  {key:val['default'] for key,val in data_args.items()}

    for key, val in kwargs.items():
        arg_name = key.lower()

        if arg_name in model_config:
            update_dict(arg_name, val, model_args, model_config)

        elif arg_name in data_config:
            update_dict(arg_name, val, data_args, data_config)

        else:
            raise ValueError(f"Unknown keyworkd argument '{key}' provided")

    return data_config, model_config


def update_dict(key, val, dict_to_lookup, dict_to_update):
    """Updates the dictionary with key, val if the val is of type dtype."""
    dtype = dict_to_lookup[key]['type']
    low = dict_to_lookup[key]['lower']
    up = dict_to_lookup[key]['upper']
    between = dict_to_lookup[key]['between']

    if dtype is not None:
        if isinstance(dtype, list):
            val_type = type(val)
            if val_type not in dtype:
                raise TypeError("{} must be any of the type {} but it is of type {}"
                                .format(key, dtype, type(val)))
        elif not isinstance(val, dtype):
            raise TypeError(f"{key} must be of type {dtype} but it is of type {type(val)}")

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


class SerializeSKOptResults(object):
    """
    This class has two functions
      - converts everything in skopt results into python native types so that these results can be saved in readable
        json files.
      - Store as much attributes in in serialized form that a skopt `search_result` object can be generated from it
        which then can be used to regenerate all hyper-parameter optimization related plots.

     skopt_results is a dictionary which contains following keys
     x: list, list of parameters being optimized
     fun: float, final value of objective function
     func_vals: numpy array, of length equal to number of iterations
     x_iters: list of lists, outer list is equal to number of iterations and each inner list is equal number of parameters
              being optimized
     models: list of models, where a model has following attributes
             - noise: str
             - kernel: skopt.learning.gaussian_process.kernels.Sum, it has following 2 attributes
                 - k1: skopt.learning.gaussian_process.kernels.Product, it has following attributes
                     -k1: skopt.learning.gaussian_process.kernels.ConstantKernel
                         - constant_value: flaot
                         - constant_value_bounds: tuple of floats
                     -k2: skopt.learning.gaussian_process.kernels.Matern
                         - length_scale: numpy ndarray
                         - length_scale_bounds: list of floats
                         - nu: float
                 - k2: skopt.learning.gaussian_process.kernels.WhiteKernel
                     - noise_level: float
                     - noise_level_bounds: tuple of floats
             - alpha: float
             - optimizer: str
             - n_restarts_optimizer: int
             - normalize_y: bool
             - copy_X_train: bool
             - random_state: int
             - kernel_: skopt.learning.gaussian_process.kernels.Sum
             - _rng: numpy.random.mtrand.RandomState
             - n_features_in_: int
             - _y_train_mean: np.float64
             - _y_train_std: np.float64
             - X_train_: numpy array
             - y_train_: numpy array
             - log_marginal_likelihood_value_: numpy array
             - L_: numpy array
             - _K_inv: NoneType
             - alpha_: numpy array
             - noise_: np.float64
             - K_inv_: numpy array
             - y_train_std_: np.float64
             - y_train_mean_: np.float64
     space: skopt.space.space.Space, parameter spaces
     random_state: numpy.random.mtrand.RandomState
     specs: dict,  specs of each iteration. It has following keys
         - args: dict, which has following keys
             - func: function
             - dimensions: skopt.space.space.Space
             - base_estimator: skopt.learning.gaussian_process.gpr.GaussianProcessRegressor, which has following attributes
                 - noise: str
                 - kernel: skopt.learning.gaussian_process.kernels.Product
                     - k1: skopt.learning.gaussian_process.kernels.ConstantKernel, which has following attributes
                         - constant_value: flaot
                         - constant_value_bounds: tuple of floats
                     - k2: skopt.learning.gaussian_process.kernels.Matern, which has following attributes
                         - length_scale: numpy ndarray
                         - length_scale_bounds: list of floats
                         - nu: float
                 - alpha: float
                 - optimizer: str
                 - n_restarts_optimizer: int
                 - normalize_y: bool
                 - copy_X_train: bool
                 - random_state: int
             - n_cals: int
             - n_random_starts: NoneType
             - n_initial_points: str
             - initial_point_generator: str
             - acq_func: str
             - acq_optimizer: str
             - x0: list
             - y0: NoneType
             - random_state: numpy.random.mtrand.RandomState
             - verbose: book
             - callback: NoneType
             - n_points: int
             - n_restarts_optimizer: int
             - xi: float
             - kappa: float
             - n_jobs: int
             - model_queue_size: NoneType
         - function: str
     """
    def __init__(self, results:dict):
        self.results = results
        self.iters = len(results['func_vals'])
        self.paras = len(results['x'])
        self.serialized_results = {}

        for key in results.keys():
            self.serialized_results[key] = getattr(self, key)()

    def x(self):

        return self.para_list(self.results['x'])

    def para_list(self, x):
        """Serializes list of parameters"""
        _x = []
        for para in x:
            _x.append(self.obj(para))
        return _x

    def obj(self, obj):
        """Serializes one object"""
        if 'int' in obj.__class__.__name__:
            return int(obj)
        if 'float' in obj.__class__.__name__:
            return float(obj)
        if hasattr(obj, '__len__') and not isinstance(obj, str):
            return [self.obj1(i) for i in obj]
        return str(obj)


    def obj1(self, obj):
        """Serializes one object"""
        if 'int' in obj.__class__.__name__:
            return int(obj)
        if 'float' in obj.__class__.__name__:
            return float(obj)
        if hasattr(obj, '__len__') and not isinstance(obj, str):
            return [self.obj2(i) for i in obj]
        return str(obj)

    @staticmethod
    def obj2(_obj):
        if 'int' in _obj.__class__.__name__:
            return int(_obj)
        if 'float' in _obj.__class__.__name__:
            return float(_obj)
        return str(_obj)

    def x0(self):
        _x0 = []

        __xo = self.results['specs']['args']['x0']

        if __xo is not None:
            for para in __xo:
                if isinstance(para, list):
                    _x0.append(self.para_list(para))
                else:
                    _x0.append(self.obj(para))
        return _x0

    def y0(self):
        __y0 = self.results['specs']['args']['y0']
        if __y0 is None:
            return __y0
        if isinstance(list, __y0):
            _y0 = []
            for y in self.results['specs']['args']['y0']:
                _y0.append(self.obj(y))
            return _y0

        return self.obj(self.results['specs']['args']['y0'])


    def fun(self):
        return float(self.results['fun'])

    def func_vals(self):
        return [float(i) for i in self.results['func_vals']]

    def x_iters(self):
        out_x = []
        for i in range(self.iters):
            x = []
            for para in self.results['x_iters'][i]:
                x.append(self.obj(para))

            out_x.append(x)

        return out_x

    def space(self):
        raum = {}
        for sp in self.results['space'].dimensions:
            if sp.__class__.__name__ == 'Categorical':
                _raum = {k: self.obj(v) for k,v in sp.__dict__.items() if k in ['categories', 'transform_', 'prior', '_name']}
                _raum.update({'type': 'Categorical'})
                raum[sp.name] = _raum

            elif sp.__class__.__name__ == 'Integer':
                _raum = {k: self.obj(v) for k, v in sp.__dict__.items() if
                                       k in ['low', 'transform_', 'prior', '_name', 'high', 'base', 'dtype', 'log_base']}
                _raum.update({'type': 'Integer'})
                raum[sp.name] = _raum

            elif sp.__class__.__name__ == 'Real':
                _raum = {k: self.obj(v) for k, v in sp.__dict__.items() if
                                       k in ['low', 'transform_', 'prior', '_name', 'high', 'base', 'dtype', 'log_base']}
                _raum.update({'type': 'Real'})
                raum[sp.name] = _raum

        return raum

    def random_state(self):
        return str(self.results['random_state'])

    def kernel(self, k):
        """Serializes Kernel"""
        if k.__class__.__name__ == "Product":
            return self.prod_kernel(k)

        if k.__class__.__name__ == "Sum":
            return self.sum_kernel(k)
        # default scenario, just converts it to string
        return str(k)

    def prod_kernel(self, k):
        """Serializes product kernel"""
        kernel = {}
        for _k,v in k.__dict__.items():


            kernel[_k] = self.singleton_kernel(v)

        return {"ProductKernel": kernel}

    def sum_kernel(self, k):
        """Serializes sum kernel"""
        kernel = {}

        for _k,v in k.__dict__.items():

            if v.__class__.__name__ == "Product":
                kernel[_k] = self.prod_kernel(v)
            else:
                kernel[_k] = self.singleton_kernel(v)

        return {"SumKernel": kernel}

    def singleton_kernel(self, k):
        """Serializes Kernels such as  Matern, White, Constant Kernels"""
        return {k:self.obj(v) for k,v in k.__dict__.items()}


    def specs(self):
        _specs = {}

        _specs['function'] = self.results['specs']['function']

        args = {}

        args['func'] = str(self.results['specs']['args']['func'])

        args['dimensions'] = self.space()

        be = self.results['specs']['args']['base_estimator']
        b_e = {k: self.obj(v) for k, v in be.__dict__.items() if
                                       k in ['noise', 'alpha', 'optimizer', 'n_restarts_optimizer', 'normalize_y', 'copy_X_train', 'random_state']}
        b_e['kernel'] = self.kernel(be.kernel)

        args['base_estimator'] = b_e

        for k,v in self.results['specs']['args'].items():
            if k in ['n_cals', 'n_random_starts', 'n_initial_points', 'initial_point_generator', 'acq_func', 'acq_optimizer',
                     'verbose', 'callback', 'n_points', 'n_restarts_optimizer', 'xi', 'kappa', 'n_jobs', 'model_queue_size']:
                args[k] = self.obj(v)

        args['x0'] = self.x0()
        args['y0'] = self.y0()

        _specs['args'] = args
        return _specs

    def models(self):

        mods = []
        for model in self.results['models']:
            mod = {k:self.obj(v) for k,v in model.__dict__.items() if k in  ['noise',
                         'alpha', 'optimizer', 'n_restarts_optimizer', 'normalize_y', 'copy_X_train', 'random_state',
                         '_rng', 'n_features_in', '_y_tain_mean', '_y_train_std', 'X_train', 'y_train', 'log_marginal_likelihood',
                         'L_', 'K_inv', 'alpha', 'noise_', 'K_inv_', 'y_train_std_', 'y_train_mean_']}

            mod['kernel'] = self.kernel(model.kernel)
            mods.append({model.__class__.__name__: mod})

        return mods


def make_hpo_results(opt_dir, metric_name='val_loss') -> dict:
    """Looks in opt_dir and saves the min val_loss with the folder name"""
    results = {}
    for folder in os.listdir(opt_dir):
        fname = os.path.join(os.path.join(opt_dir, folder), 'losses.csv')

        if os.path.exists(fname):
            df = pd.read_csv(fname)

            if 'val_loss' in df:
                min_val_loss = round(float(np.min(df[metric_name])), 6)
                results[min_val_loss] = {'folder': os.path.basename(folder)}
    return results

def clear_weights(opt_dir, results:dict=None, keep=3):
    """ Optimization will save weights of all the trained models, not all of them are useful. Here removing weights
    of all except top 3. The number of models whose weights to be retained can be set by `keep` para."""

    fname = 'sorted.json'
    if results is None:
        results = make_hpo_results(opt_dir)
        fname = 'sorted_folders.json'

    od = OrderedDict(sorted(results.items()))

    idx = 0
    for k,v in od.items():
        if 'folder' in v:
            folder = v['folder']
            _path = os.path.join(opt_dir, folder)
            w_path = os.path.join(_path, 'weights')

            if idx > keep-1:
                if os.path.exists(w_path):
                    rmtree(w_path)

            idx += 1

    # append ranking of models to folder_names
    idx = 0
    for k,v in od.items():
        if 'folder' in v:
            folder = v['folder']
            old_path = os.path.join(opt_dir, folder)
            new_path = os.path.join(opt_dir, str(idx+1) + "_" + folder)
            os.rename(old_path, new_path)

            idx += 1

    sorted_fname = os.path.join(opt_dir, fname)
    with open(sorted_fname, 'w') as sfp:
        json.dump(od, sfp, sort_keys=True, indent=True)
    return


def get_attributes(aus, what:str='losses') ->dict:
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


def post_process_skopt_results(skopt_results, results, opt_path):
    mpl.rcParams.update(mpl.rcParamsDefault)

    skopt_plots(skopt_results, pref=opt_path)

    fname = os.path.join(opt_path, 'gp_parameters')

    sr_res = SerializeSKOptResults(skopt_results)

    if 'folder' in list(results.items())[0]:
        clear_weights(results=results, opt_dir=opt_path)
    else:
        clear_weights(opt_dir=opt_path)

    try:
        dump(skopt_results, os.path.join(opt_path, os.path.basename(opt_path)))
    except PicklingError:
        print("could not pickle results")

    try:
        with open(fname + '.json', 'w') as fp:
            json.dump(sr_res.serialized_results, fp, sort_keys=True, indent=4)
    except TypeError:
        with open(fname + '.json', 'w') as fp:
            json.dump(str(sr_res.serialized_results), fp, sort_keys=True, indent=4)

    return


def stats(feature) ->dict:
    """Gets all the possible stats about an array like object `feature`.
    `feature: array like
    """
    if not isinstance(feature, np.ndarray):
        if hasattr(feature, '__len__'):
            feature = np.array(feature)
        else:
            raise TypeError(f"input must be array like but it is of type {type(feature)}")

    _stats = dict()
    _stats['Skew'] = skew(feature)
    _stats['Kurtosis'] = kurtosis(feature)
    _stats['Mean'] = np.nanmean(feature)
    _stats['Geometric Mean'] = gmean(feature)
    try:
        _stats['Harmonic Mean'] = hmean(feature)
    except ValueError:
        warnings.warn("Harmonic mean only defined if all elements greater than or equal to zero", UserWarning)
    _stats['Standard error of mean'] = scipy.stats.sem(feature)
    _stats['Median'] = np.nanmedian(feature)
    _stats['Variance'] = np.nanvar(feature)
    _stats['Coefficient of Variation'] = variation(feature)
    _stats['Std'] = np.nanstd(feature)
    _stats['Non zeros'] = np.count_nonzero(feature)
    _stats['10 quant'] = np.nanquantile(feature, 0.1)
    _stats['50 quant'] = np.nanquantile(feature, 0.5)
    _stats['90 quant'] = np.nanquantile(feature, 0.9)
    _stats['25 %ile'] = np.nanpercentile(feature, 25)
    _stats['50 %ile'] = np.nanpercentile(feature, 50)
    _stats['75 %ile'] = np.nanpercentile(feature, 75)
    _stats['Min'] = np.nanmin(feature)
    _stats['Max'] = np.nanmax(feature)
    _stats["Negative counts"] = float(np.sum(feature < 0.0))
    _stats["NaN counts"] = np.isnan(feature).sum()
    _stats['Counts'] = len(feature)

    return _stats