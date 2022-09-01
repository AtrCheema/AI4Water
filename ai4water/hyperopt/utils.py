
import json
from itertools import islice
from collections import OrderedDict

try:
    from skopt.plots import plot_evaluations, plot_objective
except ImportError:
    plot_evaluations, plot_objective = None, None

from ai4water.utils.utils import jsonize, clear_weights
from ai4water.backend import os, np, pd, mpl, plt, skopt, easy_mpl
from ai4water.backend import hyperopt as _hyperopt
from ._space import Categorical, Real, Integer

Space = skopt.space.space.Space
Dimension = skopt.space.space.Dimension
dump = skopt.utils.dump

if _hyperopt is not None:
    space_eval = _hyperopt.space_eval
    hp = _hyperopt.hp
    miscs_to_idxs_vals = _hyperopt.base.miscs_to_idxs_vals
else:
    space_eval, hp = None, None
    miscs_to_idxs_vals = None

plot = easy_mpl.plot


def is_choice(space):
    """checks if an hp.space is hp.choice or not"""
    if 'switch' in space.name:
        return True
    return False


def x_iter_for_tpe(trials, param_space: dict, as_list=True):

    assert isinstance(param_space, dict)

    x_iters = []  # todo, remove x_iters, it is just values of iterations
    iterations = {}
    for idx, t in enumerate(trials.trials):

        vals = t['misc']['vals']
        y = t['result']['loss']

        x = get_one_tpe_x_iter(vals, param_space)
        iterations[idx] = {'x': x, 'y': y}

        x_iters.append(x)

    if as_list:
        return [list(d.values()) for d in x_iters]
    return iterations


def get_one_tpe_x_iter(tpe_vals, param_space: dict, sort=True):

    x_iter = {}
    for para, para_val in tpe_vals.items():
        if is_choice(param_space[para]):
            hp_assign = {para: para_val[0]}
            cval = space_eval(param_space[para], hp_assign)
            x_iter[para] = cval
        else:
            x_iter[para] = para_val[0]

    if sort:
        x_iter = sort_x_iters(x_iter, list(param_space.keys()))

    return x_iter


def sort_x_iters(x_iter: dict, original_order: list):
    # the values in x_iter may not be sorted as the parameters provided in original order

    new_x_iter = {}
    for s in original_order:
        new_x_iter[s] = x_iter[s]

    return new_x_iter


def skopt_space_from_hp_spaces(hp_space: dict) -> list:
    """given a dictionary of hp spaces where keys are names, this function
    converts it into skopt space."""
    new_spaces = []

    for k, s in hp_space.items():

        new_spaces.append(skopt_space_from_hp_space(s, k))

    return new_spaces


def skopt_space_from_hp_space(hp_space, prior_name=None):
    """Converts on hp space into a corresponding skopt space."""
    if is_choice(hp_space):
        skopt_space = to_categorical(hp_space, prior_name=prior_name)
    elif any([i in hp_space.__str__() for i in ['loguniform', 'quniform', 'qloguniform', 'uniform']]):
        skopt_space = to_real(hp_space, prior_name=prior_name)
    elif 'randint' in hp_space.__str__():
        skopt_space = to_int(hp_space, prior_name=prior_name)
    else:
        raise NotImplementedError

    return skopt_space


def to_categorical(_space, prior_name=None):
    """converts an hp space into a Dimension object."""
    cats = []
    inferred_name = None
    for arg in _space.pos_args:
        if hasattr(arg, '_obj') and arg.pure:
            cats.append(arg._obj)
        elif arg.name == 'hyperopt_param' and len(arg.pos_args)>0:
            for a in arg.pos_args:
                if a.name == 'literal' and a.pure:
                    inferred_name = a._obj

    prior_name = verify_name(prior_name, inferred_name)

    if len(cats) == 0:
        raise NotImplementedError
    return Categorical(categories=cats, name=prior_name)


def verify_name(prior_name, inferred_name):
    """Verfies that the given/prior name matches with the inferred name.
    """
    if prior_name is None:
        prior_name = inferred_name
    else:
        assert prior_name == inferred_name, f"""given name {prior_name} does not mach with 
                                                inferred name {inferred_name}"""
    return prior_name


def to_int(_space, prior_name=None):
    """converts an hp.randint into a Dimension object"""
    inferred_name = None
    limits = None
    for arg in _space.pos_args:
        if arg.name == 'literal' and len(arg.named_args) == 0:
            inferred_name = arg._obj
        elif len(arg.named_args) == 2 and arg.name == 'randint':
            limits = {}
            for a in arg.named_args:
                limits[a[0]] = a[1]._obj

        elif len(arg.pos_args) == 2 and arg.name == 'randint':
            # high and low are not named args
            _limits = []
            for a in arg.pos_args:
                _limits.append(a._obj)
            limits = {'high': np.max(_limits), 'low': np.min(_limits)}
        else:
            raise NotImplementedError

    prior_name = verify_name(prior_name, inferred_name)

    if limits is None:
        raise NotImplementedError
    return Integer(low=limits['low'], high=limits['high'], name=prior_name)


def to_real(_space, prior_name=None):
    """converts an an hp space to real. """
    inferred_name = None
    limits = None
    prior = None
    allowed_names = ['uniform', 'loguniform', 'quniform', 'loguniform', 'qloguniform']
    for arg in _space.pos_args:
        if len(arg.pos_args) > 0:
            for a in arg.pos_args:
                if a.name == 'literal' and len(arg.named_args) == 0:
                    inferred_name = a._obj
                elif a.name in allowed_names and len(a.named_args) == 2:
                    prior = a.name
                    limits = {}
                    for _a in a.named_args:
                        limits[_a[0]] = _a[1]._obj
                elif a.name in allowed_names and len(a.pos_args) == 2:
                    prior = a.name
                    _limits = []
                    for _a in a.pos_args:
                        _limits.append(_a._obj)
                    limits = {'high': np.max(_limits), 'low': np.min(_limits)}

    prior_name = verify_name(prior_name, inferred_name)

    # prior must be inferred because hp spaces always have prior
    if limits is None or prior is None or 'high' not in limits:
        raise NotImplementedError

    return Real(low=limits['low'], high=limits['high'], prior=prior, name=prior_name)


def loss_histogram(losses,  # array like
                   xlabel='objective_fn',
                   ylabel='Frequency',
                   save=True,
                   fname="histogram.png"):

    plt.hist(losses)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_hyperparameters(
        trials,
        save=True,
        fontsize=10,
        colorize_best=None,
        columns=5,
        arrange_by_loss=False,
        fname='parameters_selection_plot.png'
):
    """Copying from hyperopt because original hyperopt does not allow saving the plot."""

    idxs, vals = miscs_to_idxs_vals(trials.miscs)
    losses = trials.losses()
    finite_losses = [y for y in losses if y not in (None, float("inf"))]
    asrt = np.argsort(finite_losses)
    if colorize_best is not None:
        colorize_thresh = finite_losses[asrt[colorize_best + 1]]
    else:
        # -- set to lower than best (disabled)
        colorize_thresh = finite_losses[asrt[0]] - 1

    loss_min = min(finite_losses)
    loss_max = max(finite_losses)
    print("finite loss range", loss_min, loss_max, colorize_thresh)

    loss_by_tid = dict(zip(trials.tids, losses))

    def color_fn_bw(lossval):
        if lossval in (None, float("inf")):
            return 1, 1, 1
        else:
            t = (lossval - loss_min) / (loss_max - loss_min + 0.0001)
            if lossval < colorize_thresh:
                return 0.0, 1.0 - t, 0.0  # -- red best black worst
            else:
                return t, t, t  # -- white=worst, black=best

    all_labels = list(idxs.keys())
    titles = all_labels
    order = np.argsort(titles)

    C = min(columns, len(all_labels))
    R = int(np.ceil(len(all_labels) / float(C)))

    for plotnum, varnum in enumerate(order):
        label = all_labels[varnum]
        plt.subplot(R, C, plotnum + 1)

        # hide x ticks
        ticks_num, _ = plt.xticks()
        plt.xticks(ticks_num, [""] * len(ticks_num))

        dist_name = label

        if arrange_by_loss:
            x = [loss_by_tid[ii] for ii in idxs[label]]
        else:
            x = idxs[label]
        if "log" in dist_name:
            y = np.log(vals[label])
        else:
            y = vals[label]
        plt.title(titles[varnum], fontsize=fontsize)
        c = list(map(color_fn_bw, [loss_by_tid[ii] for ii in idxs[label]]))
        if len(y):
            plt.scatter(x, y, c=c)
        if "log" in dist_name:
            nums, _ = plt.yticks()
            plt.yticks(nums, ["%.2e" % np.exp(t) for t in nums])

    if save:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def post_process_skopt_results(skopt_results, results, opt_path):

    skopt_plots(skopt_results, pref=opt_path)

    clear_weights(results=results, opt_dir=opt_path)

    return


def save_skopt_results(skopt_results, opt_path):

    fname = os.path.join(opt_path, 'gp_parameters')

    sr_res = SerializeSKOptResults(skopt_results)

    try:
        with open(fname + '.json', 'w') as fp:
            json.dump(sr_res.serialized_results, fp, sort_keys=True, indent=4)
    except TypeError:
        with open(fname + '.json', 'w') as fp:
            json.dump(str(sr_res.serialized_results), fp, sort_keys=True, indent=4)

    return


def _plot_objective(search_results, pref="", threshold=20):
    if len(search_results.x) < threshold:
        if search_results.space.n_dims == 1:
            pass
        else:
            plt.close('all')
            _ = plot_objective(search_results)
            plt.savefig(os.path.join(pref, 'objective'), dpi=400, bbox_inches='tight')
    return


def skopt_plots(search_result, pref=os.getcwd(), threshold=20):

    if len(search_result.x) < threshold:  # it takes forever if parameters are > 20
        plt.close('all')
        _ = plot_evaluations(search_result)
        plt.savefig(os.path.join(pref, 'evaluations'), dpi=400, bbox_inches='tight')

    _plot_objective(search_result, pref=pref, threshold=threshold)

    convergence(search_result.func_vals)
    plt.savefig(os.path.join(pref, 'convergence'), dpi=300, bbox_inches='tight')
    convergence(search_result.func_vals, show_original=True)
    plt.savefig(os.path.join(pref, 'convergence_original'), dpi=300, bbox_inches='tight')
    return


def convergence(func_vals, color=None, show_original=False):

    _, ax = plt.subplots()
    ax.grid()

    n_calls = len(func_vals)
    mins = [np.min(func_vals[:i])
            for i in range(1, n_calls + 1)]

    if show_original:
        data = func_vals
    else:
        data = mins

    plot(data,
         color=color,
         marker=".", markersize=12, lw=2,
         title="Convergence plot",
         xlabel="Number of calls $n$",
         ylabel=r"$\min f(x)$ after $n$ calls",
         show=False,
         ax=ax)

    return

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
    def __init__(self, results: dict):
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
            _x.append(jsonize(para))
        return _x

    def x0(self):
        _x0 = []

        __xo = self.results['specs']['args']['x0']

        if __xo is not None:
            for para in __xo:
                if isinstance(para, list):
                    _x0.append(self.para_list(para))
                else:
                    _x0.append(jsonize(para))
        return _x0

    def y0(self):
        __y0 = self.results['specs']['args']['y0']
        if __y0 is None:
            return __y0
        if isinstance(__y0, list):
            _y0 = []
            for y in self.results['specs']['args']['y0']:
                _y0.append(jsonize(y))
            return _y0

        return jsonize(self.results['specs']['args']['y0'])

    def fun(self):
        return float(self.results['fun'])

    def func_vals(self):
        return [float(i) for i in self.results['func_vals']]

    def x_iters(self):
        out_x = []
        for i in range(self.iters):
            x = []
            for para in self.results['x_iters'][i]:
                x.append(jsonize(para))

            out_x.append(x)

        return out_x

    def space(self):
        raum = {}
        for sp in self.results['space'].dimensions:
            if sp.__class__.__name__ == 'Categorical':
                _raum = {k: jsonize(v) for k, v in sp.__dict__.items() if k in ['categories', 'transform_',
                                                                                  'prior', '_name']}
                _raum.update({'type': 'Categorical'})
                raum[sp.name] = _raum

            elif sp.__class__.__name__ == 'Integer':
                _raum = {k: jsonize(v) for k, v in sp.__dict__.items() if
                                       k in ['low', 'transform_', 'prior', '_name', 'high', 'base',
                                             'dtype', 'log_base']}
                _raum.update({'type': 'Integer'})
                raum[sp.name] = _raum

            elif sp.__class__.__name__ == 'Real':
                _raum = {k: jsonize(v) for k, v in sp.__dict__.items() if
                                       k in ['low', 'transform_', 'prior', '_name', 'high', 'base', 'dtype',
                                             'log_base']}
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
        for _k, v in k.__dict__.items():

            kernel[_k] = self.singleton_kernel(v)

        return {"ProductKernel": kernel}

    def sum_kernel(self, k):
        """Serializes sum kernel"""
        kernel = {}

        for _k, v in k.__dict__.items():

            if v.__class__.__name__ == "Product":
                kernel[_k] = self.prod_kernel(v)
            else:
                kernel[_k] = self.singleton_kernel(v)

        return {"SumKernel": kernel}

    def singleton_kernel(self, k):
        """Serializes Kernels such as  Matern, White, Constant Kernels"""
        return {k: jsonize(v) for k, v in k.__dict__.items()}

    def specs(self):
        _specs = {}

        _specs['function'] = self.results['specs']['function']

        args = {}

        args['func'] = str(self.results['specs']['args']['func'])

        args['dimensions'] = self.space()

        be = self.results['specs']['args']['base_estimator']
        b_e = {k: jsonize(v) for k, v in be.__dict__.items() if
                                       k in ['noise', 'alpha', 'optimizer', 'n_restarts_optimizer', 'normalize_y',
                                             'copy_X_train', 'random_state']}
        b_e['kernel'] = self.kernel(be.kernel)

        args['base_estimator'] = b_e

        for k,v in self.results['specs']['args'].items():
            if k in ['n_cals', 'n_random_starts', 'n_initial_points', 'initial_point_generator', 'acq_func',
                     'acq_optimizer', 'verbose', 'callback', 'n_points', 'n_restarts_optimizer', 'xi', 'kappa',
                     'n_jobs', 'model_queue_size']:
                args[k] = jsonize(v)

        args['x0'] = self.x0()
        args['y0'] = self.y0()

        _specs['args'] = args
        return _specs

    def models(self):

        mods = []
        for model in self.results['models']:
            mod = {k: jsonize(v) for k, v in model.__dict__.items() if k in [
                'noise','alpha', 'optimizer', 'n_restarts_optimizer', 'normalize_y', 'copy_X_train', 'random_state',
                '_rng', 'n_features_in', '_y_tain_mean', '_y_train_std', 'X_train', 'y_train',
                'log_marginal_likelihood', 'L_', 'K_inv', 'alpha', 'noise_', 'K_inv_', 'y_train_std_', 'y_train_mean_']}

            mod['kernel'] = self.kernel(model.kernel)
            mods.append({model.__class__.__name__: mod})

        return mods


def to_skopt_space(x):
    """converts the space x into skopt compatible space"""
    if isinstance(x, list):
        if all([isinstance(s, Dimension) for s in x]):
            _space = Space(x)
        elif len(x) == 1 and isinstance(x[0], tuple):
            if len(x[0]) == 2:
                if 'int' in x[0][0].__class__.__name__:
                    _space = Integer(low=x[0][0], high=x[0][1])
                elif 'float' in x[0][0].__class__.__name__:
                    _space = Integer(low=x[0][0], high=x[0][1])
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        elif all([s.__class__.__name__== "Apply" for s in x]):
            _space = Space([skopt_space_from_hp_space(v) for v in x])
        else:
            raise NotImplementedError
    elif isinstance(x, dict):  # todo, in random, should we build Only Categorical space?
        space_ = []
        for k, v in x.items():
            if isinstance(v, list):
                s = space_from_list(v, k)
            elif isinstance(v, Dimension):
                # it is possible that the user has not specified the name so assign the names
                # because we have keys.
                if v.name is None or v.name.startswith('real_') or v.name.startswith('integer_'):
                    v.name = k
                s = v
            elif v.__class__.__name__== "Apply" or 'rv_frozen' in v.__class__.__name__:
                s = skopt_space_from_hp_space(v, k)
            elif isinstance(v, tuple):
                s = Categorical(v, name=k)
            elif isinstance(v, np.ndarray):
                s = Categorical(v.tolist(), name=k)
            else:
                raise NotImplementedError(f"unknown type {v}, {type(v)}")
            space_.append(s)

        # todo, why converting to Space
        _space = Space(space_) if len(space_) > 0 else None
    elif 'rv_frozen' in x.__class__.__name__ or x.__class__.__name__== "Apply":
        _space = Space([skopt_space_from_hp_space(x)])
    else:
        raise NotImplementedError(f"unknown type {x}, {type(x)}")
    return _space


def scatterplot_matrix_colored(params_names: list,
                               params_values: list,
                               best_accs: list,
                               blur=False,
                               save=True,
                               fname='scatter_plot.png'
                               ):
    """Scatterplot colored according to the Z values of the points.
    https://github.com/guillaume-chevalier/Hyperopt-Keras-CNN-CIFAR-100/blob/Vooban/AnalyzeTestHyperoptResults.ipynb
    """

    nb_params = len(params_values)
    best_accs = np.array(best_accs)
    norm = mpl.colors.Normalize(vmin=best_accs.min(), vmax=best_accs.max())

    fig, ax = plt.subplots(nb_params, nb_params, figsize=(16, 16))  # , facecolor=bg_color, edgecolor=fg_color)

    for i in range(nb_params):
        p1 = params_values[i]
        for j in range(nb_params):
            p2 = params_values[j]

            axes = ax[i, j]
            # Subplot:
            if blur:
                axes.scatter(p2, p1, s=400, alpha=.1,
                             c=best_accs, cmap='viridis', norm=norm)
                axes.scatter(p2, p1, s=200, alpha=.2,
                             c=best_accs, cmap='viridis', norm=norm)
                axes.scatter(p2, p1, s=100, alpha=.3,
                             c=best_accs, cmap='viridis', norm=norm)
            s = axes.scatter(p2, p1, s=15,
                             c=best_accs, cmap='viridis', norm=norm)

            # Labels only on side subplots, for x and y:
            if j == 0:
                axes.set_ylabel(params_names[i], rotation=0)
            else:
                axes.set_yticks([])

            if i == nb_params - 1:
                axes.set_xlabel(params_names[j], rotation=90)
            else:
                axes.set_xticks([])

    fig.subplots_adjust(right=0.82, top=0.95)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(s, cax=cbar_ax)

    plt.suptitle(
        'Mmatrix of tried values in the search space over different params, colored in function of best test accuracy')
    plt.show()


def take(n, iterable):
    """Return first n items of the iterable as a list
    https://stackoverflow.com/questions/7971618/return-first-n-keyvalue-pairs-from-dict
    """
    return list(islice(iterable, n))


def plot_convergences(opt_dir, what='val_loss', show_whole=True, show_min=False,
                      **kwargs):

    plot_dir = os.path.join(opt_dir, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    max_len = kwargs.get('max_len', 500)
    show_top = kwargs.get('show_top', 3)
    ylim_bottom = kwargs.get('ylim_bottom', None)
    ylim_top = kwargs.get('ylim_top', None)
    font_size = kwargs.get('font_size', 16)
    ylabel = kwargs.get('ylabel', 'Validation MSE')
    leg_pos = kwargs.get('leg_pos', 'upper right')
    # style = kwargs.get('style', 'ggplot')

    models = []
    for f in os.listdir(opt_dir):
        if os.path.isdir(os.path.join(opt_dir, f)):
            models.append(f)

    val_loss = pd.DataFrame()
    default = pd.Series(np.full(max_len, 0.0))
    min_val_loss = {}

    for mod in models:
        loss_fname = os.path.join(os.path.join(opt_dir, str(mod) + "\\losses.csv"))
        if os.path.exists(loss_fname):
            losses = pd.read_csv(loss_fname)
            vl = losses[what]

            vl1 = default.add(vl, fill_value=None)
            vl1.name = mod

            min_val_loss[mod] = vl1.min()

            val_loss = pd.concat([val_loss, vl1], axis=1)

    # sort min_val_loss by value
    min_vl_sorted = {k: v for k, v in sorted(min_val_loss.items(), key=lambda item: item[1])}
    top_3 = take(show_top, min_vl_sorted)

    colors = {
        0: 'b',
        1: 'g',
        2: 'orange'
    }

    default = np.full(max_len, np.nan)

    plt.close('all')
    _, axis = plt.subplots()

    for k in min_vl_sorted.keys():

        val = val_loss[k]

        default[np.argmin(val.values)] = val.min()
        if k not in top_3:
            if show_whole:
                axis.plot(val, color='silver', linewidth=0.5,  label='_nolegend_')

            if show_min:
                axis.plot(default, '.', markersize=2.5, color='silver', label='_nolegend_')
        default = np.full(max_len, np.nan)

    for idx, k in enumerate(top_3):
        val = val_loss[k]

        if show_whole:
            axis.plot(val, color=colors[idx], linewidth=1,   label=f'Rank {idx+1}')

        default[np.argmin(val.values)] = val.min()
        if show_min:
            axis.plot(default, 'x', markersize=5.5, color=colors[idx], label='Rank ' + str(idx))

        default = np.full(max_len, np.nan)

    axis.legend(loc=leg_pos, fontsize=font_size)
    axis.set_yscale('log')
    axis.set_ylabel(ylabel, fontsize=font_size)
    axis.set_xlabel('Epochs', fontsize=font_size)

    if ylim_bottom is not None:
        axis.set_ylim(bottom=ylim_bottom)
    if ylim_top is not None:
        axis.set_ylim(top=ylim_top)

    _name = what
    _name += '_whole_loss_' if show_whole else ''
    _name += '_loss_points' if show_min else ''
    fname = os.path.join(plot_dir, _name)

    plt.savefig(fname, dpi=300, bbox_inches='tight')

    return


def to_skopt_as_dict(algorithm:str, backend:str, original_space)->dict:

    if backend == 'hyperopt':
        if original_space.__class__.__name__ == "Apply":
            _space = skopt_space_from_hp_space(original_space)
            _space = {_space.name: _space}
        elif isinstance(original_space, dict):
            _space = OrderedDict()
            for k, v in original_space.items():
                if v.__class__.__name__ == "Apply" or 'rv_frozen' in v.__class__.__name__:
                    _space[k] = skopt_space_from_hp_space(v)
                elif isinstance(v, Dimension):
                    _space[v.name] = v
                else:
                    raise NotImplementedError
        elif isinstance(original_space, list):
            if all([isinstance(s, Dimension) for s in original_space]):
                _space = OrderedDict({s.name: s for s in original_space})
            elif all([s.__class__.__name__== "Apply" for s in original_space]):
                d = [skopt_space_from_hp_space(v) for v in original_space]
                _space = OrderedDict({s.name: s for s in d})
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    elif backend == 'optuna':
        if isinstance(original_space, list):
            if all([isinstance(s, Dimension) for s in original_space]):
                _space = OrderedDict({s.name: s for s in original_space})
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    elif backend == 'skopt':
        sk_space = to_skopt_space(original_space)

        if isinstance(sk_space, Dimension):
            _space = {sk_space.name: sk_space}

        elif all([isinstance(s, Dimension) for s in sk_space]):
            _space = OrderedDict()
            for s in sk_space:
                _space[s.name] = s
        else:
            raise NotImplementedError

    elif backend == 'sklearn':
        if isinstance(original_space, list):
            if all([isinstance(s, Dimension) for s in original_space]):
                _space = OrderedDict({s.name: s for s in original_space})
            else:
                raise NotImplementedError
        elif isinstance(original_space, dict):
            _space = OrderedDict()
            for k, v in original_space.items():
                if isinstance(v, list):
                    s = space_from_list(v, k)
                elif isinstance(v, Dimension):
                    s = v
                elif isinstance(v, tuple) or isinstance(v, list):
                    s = Categorical(v, name=k)
                elif  v.__class__.__name__ == "Apply" or 'rv_frozen' in v.__class__.__name__:
                    if algorithm == 'random':
                        s = Real(v.kwds['loc'], v.kwds['loc'] + v.kwds['scale'], name=k, prior=v.dist.name)
                    else:
                        s = skopt_space_from_hp_space(v)
                else:
                    raise NotImplementedError(f"unknown type {v}, {type(v)}")
                _space[k] = s
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return _space


def space_from_list(v: list, k: str):
    """Returns space of tyep "Dimension"
    """
    if len(v) > 2:
        if isinstance(v[0], int):
            s = Integer(grid=v, name=k)
        elif isinstance(v[0], float):
            s = Real(grid=v, name=k)
        else:
            s = Categorical(v, name=k)
    else:
        if isinstance(v[0], int):
            s = Integer(low=np.min(v), high=np.max(v), name=k)
        elif isinstance(v[0], float):
            s = Real(low=np.min(v), high=np.max(v), name=k)
        elif isinstance(v[0], str):
            s = Categorical(v, name=k)
        else:
            raise NotImplementedError
    return s
