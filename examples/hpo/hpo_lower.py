"""
==========================================
hyperparameter optimization using HyperOpt
==========================================

There are two ways of optimization of hyperparameters in AI4Water. The  :py:class:`HyperOpt`
class is the lower level api while :py:meth:`Model.optimize_hyperparameters` is the higher level api.
For using HyperOpt class, the user has to define the objecive function and
hyerparameter space explicitly. Morevoer, the user has to instantiate the HyperOpt
class and call the fit method on it.

This example shows, how to use HyperOpt class for optimization of hyperparameters.
"""

import os
import math

import numpy as np

from skopt.plots import plot_objective
from SeqMetrics import RegressionMetrics

from ai4water.functional import Model
from ai4water.datasets import busan_beach
from ai4water.utils.utils import jsonize, dateandtime_now
from ai4water.hyperopt import HyperOpt, Categorical, Real, Integer

data = busan_beach()

SEP = os.sep

# sphinx_gallery_thumbnail_number = 2

#%%

PREFIX = f"hpo_{dateandtime_now()}"
ITER = 0

##############################################

# Optimizing the hyperparameters usually involves four steps

###########################################
# 1) define objective function
#-------------------------------

def objective_fn(
        prefix=None,
        **suggestions)->float:
    """This function must build, train and evaluate the ML model.
    The output of this function will be minimized by optimization algorithm.
    """
    suggestions = jsonize(suggestions)
    global ITER

    # build model
    _model = Model(model={"CatBoostRegressor": suggestions},
                  prefix=prefix or PREFIX,
                  train_fraction=1.0,
                  split_random=True,
                  verbosity=0,
                  )

    # train model
    _model.fit(data=data)

    # evaluate model
    t, p = _model.predict(data='validation', return_true=True, process_results=False)
    val_score = RegressionMetrics(t, p).r2_score()

    if not math.isfinite(val_score):
        val_score = 1.0

    # since the optimization algorithm solves minimization algorithm
    # we have to subtract r2_score from 1.0
    # if our validation metric is something like mse or rmse,
    # then we don't need to subtract it from 1.0
    val_score = 1.0 - val_score

    ITER += 1

    print(f"{ITER} {val_score}")

    return val_score

###########################################
# 2) define parameter space
#-------------------------------
# the parameter space determines the pool of candidates from which
# hyperparameters will be choosen during optimization

num_samples=10
space = [
# maximum number of trees that can be built
Integer(low=10, high=30, name='iterations', num_samples=num_samples),
# Used for reducing the gradient step.
Real(low=0.09, high=0.3, prior='log', name='learning_rate', num_samples=num_samples),
# Coefficient at the L2 regularization term of the cost function.
Real(low=0.5, high=5.0, name='l2_leaf_reg', num_samples=num_samples),
# arger the value, the smaller the model size.
Real(low=0.1, high=10, name='model_size_reg', num_samples=num_samples),
# percentage of features to use at each split selection, when features are selected over again at random.
Real(low=0.1, high=0.5, name='rsm', num_samples=num_samples),
# number of splits for numerical features
Integer(low=32, high=50, name='border_count', num_samples=num_samples),
# The quantization mode for numerical features.  The quantization mode for numerical features.
Categorical(categories=['Median', 'Uniform', 'UniformAndQuantiles',
                        'MaxLogSum', 'MinEntropy', 'GreedyLogSum'], name='feature_border_type')
]

###########################################
# 3) initial state
#-------------------------------
# this step is optional but it is always better to
# provide a good initial guess to the optimization algorithm
x0 = [10, 0.11, 1.0, 1.0, 0.2, 45, "Uniform"]

#############################################
# 4) run optimization algorithm
#---------------------------------------------

# Now instantiate the HyperOpt class and call .fit on it
# algorithm can be either ``random``, ``grid``, ``bayes``, ``tpe``, ``bayes_rf``
#

optimizer = HyperOpt(
    algorithm="bayes",
    objective_fn=objective_fn,
    param_space=space,
    x0=x0,
    num_iterations=30,
    process_results=False,
    opt_path=f"results{SEP}{PREFIX}",
    verbosity=0,
)

results = optimizer.fit()

###########################################

print(f"optimized parameters are \n{optimizer.best_paras()}")

print(np.min(optimizer.func_vals()))

###########################################
# postprocessing of results
#---------------------------------------------
# save hyperparameters at each iteration

optimizer.save_iterations_as_xy()

#%%
# save convergence plot

optimizer._plot_convergence(save=False)

###########################################

optimizer._plot_parallel_coords(figsize=(14, 8), save=False)

###########################################

optimizer._plot_distributions(save=False)

###########################################

optimizer.plot_importance(save=False)

###########################################

_ = plot_objective(results)

###########################################

optimizer._plot_evaluations(save=False)


###########################################

optimizer._plot_edf(save=False)

###########################################


# If you set ``process_results`` to True above, all of the results are automatically
#saved in the optimization directory.


print(f"All the results are save in {optimizer.opt_path} directory")

