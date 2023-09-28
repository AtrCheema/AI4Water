"""
This file should not import from other files in hyperopt submodule.
"""
import warnings

from ai4water.backend import optuna, skopt, np, sklearn
from ai4water.backend import hyperopt as _hyperopt

class SKOPT:

    Space = None
    forest_minimize = None
    gp_minimize = None
    BayesSearchCV = None
    use_named_args = None
    plot_objective = None
    dump = None

    if skopt:

        if np.__version__>="1.23":
            np.int = int

        if sklearn.__version__ >= "1.2":
            warnings.warn("skopt may not work with sklearn version >= 1.2")

        Space = skopt.space.space.Space
        forest_minimize = skopt.forest_minimize
        gp_minimize = skopt.gp_minimize
        BayesSearchCV = skopt.BayesSearchCV
        use_named_args = skopt.utils.use_named_args
        dump = skopt.utils.dump

        from skopt.plots import plot_objective
        plot_objective = plot_objective


class HYPEROPT:
    hp = None
    Apply = None
    fmin_hyperopt = None
    tpe = None
    Trials = None
    rand = None
    atpe = None
    space_eval = None
    miscs_to_idxs_vals = None

    if _hyperopt:
        hp = _hyperopt.hp
        Apply = _hyperopt.pyll.base.Apply
        fmin_hyperopt = _hyperopt.fmin
        tpe = _hyperopt.tpe
        Trials = _hyperopt.Trials
        rand = _hyperopt.rand
        space_eval = _hyperopt.space_eval
        miscs_to_idxs_vals = _hyperopt.base.miscs_to_idxs_vals

        try:  # atpe is only available in later versions of hyperopt
            atpe = _hyperopt.atpe
        except AttributeError:
            pass


class OPTUNA:

    CategoricalDistribution = None
    UniformDistribution = None
    IntLogUniformDistribution = None
    IntUniformDistribution = None
    LogUniformDistribution = None
    plot_contour = None

    if optuna:
        plot_contour = optuna.visualization.plot_contour
        CategoricalDistribution = optuna.distributions.CategoricalDistribution
        UniformDistribution = optuna.distributions.UniformDistribution
        IntLogUniformDistribution = optuna.distributions.IntLogUniformDistribution
        IntUniformDistribution = optuna.distributions.IntUniformDistribution
        LogUniformDistribution = optuna.distributions.LogUniformDistribution
