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
from typing import Union

import numpy as np
from skopt.plots import plot_objective
from SeqMetrics import RegressionMetrics

from ai4water.functional import Model
from ai4water.datasets import busan_beach
from ai4water.utils.utils import jsonize, dateandtime_now
from ai4water.hyperopt import HyperOpt, Categorical, Real, Integer

data = busan_beach()

SEP = os.sep
PREFIX = f"hpo_{dateandtime_now()}"
ITER = 0

# sphinx_gallery_thumbnail_number = 2

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
    model = Model(model={"CatBoostRegressor": suggestions},
                  prefix=prefix or PREFIX,
                  train_fraction=1.0,
                  split_random=True,
                  verbosity=0,
                  )

    # train model
    model.fit(data=data)

    # evaluate model
    t, p = model.predict(data='validation', return_true=True, process_results=False)
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
Integer(low=100, high=5000, name='iterations', num_samples=num_samples),
# Used for reducing the gradient step.
Real(low=0.0001, high=0.5, prior='log', name='learning_rate', num_samples=num_samples),
# Coefficient at the L2 regularization term of the cost function.
Real(low=0.5, high=5.0, name='l2_leaf_reg', num_samples=num_samples),
# arger the value, the smaller the model size.
Real(low=0.1, high=10, name='model_size_reg', num_samples=num_samples),
# percentage of features to use at each split selection, when features are selected over again at random.
Real(low=0.1, high=0.95, name='rsm', num_samples=num_samples),
# number of splits for numerical features
Integer(low=32, high=1032, name='border_count', num_samples=num_samples),
# The quantization mode for numerical features.  The quantization mode for numerical features.
Categorical(categories=['Median', 'Uniform', 'UniformAndQuantiles',
                        'MaxLogSum', 'MinEntropy', 'GreedyLogSum'], name='feature_border_type')
]

###########################################
# 3) initial state
#-------------------------------
# this step is optional but it is always better to
# provide a good initial guess to the optimization algorithm
x0 = [200, 0.01, 1.0, 1.0, 0.2, 64, "Uniform"]

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
    num_iterations=15,
    process_results=False,
    opt_path=f"results{SEP}{PREFIX}",
    verbosity=0,
)

results = optimizer.fit()

###########################################

print(f"optimized parameters are \n{optimizer.best_paras()}")

###########################################
# postprocessing of results
#---------------------------------------------

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


###########################################
# Using HyperOpt with Neural Networks
#--------------------------------------

# If your model consists of layers of neural neworks, you can easily modify
# the objective function accordingly. The only change you have to do is to
# get corresponding parameters from ``suggestions`` and use them
# to build the layers of neural network.

###########################################

import tensorflow as tf

print(tf.__version__, np.__version__)

##############################################

PREFIX = f"hpo_nn{dateandtime_now()}"
ITER = 0
num_iterations = 15

# these seeds are randomly generated but we keep track of the seed
# used at each iteration, so that when we rebuilt the model with optimized
# hyperparameters, we get reproducible results
SEEDS = np.random.randint(0, 1000, num_iterations)
# to keep track of seed being used at every optimization iteration
SEEDS_USED = []


def objective_fn(
        prefix=None,
        return_model=False,
        seed:int=None,
        epochs:int=100,
        **suggestions)->Union[float, Model]:
    """This function must build, train and evaluate the ML model.
    The output of this function will be minimized by optimization algorithm.

    Parameters
    ----------
    prefix : str
        prefix to save the results. This argument will only be used after
        the optimization is complete
    return_model : bool
        if True, then objective function will return the built model. This
        argument will only be used after the optimization is complete
    seed : int, optional
        random seed for reproducibility. During optimization, its value will
        be None and we will use the value from SEEDS. After optimization,
        we will again call the objective function but this time with fixed
        seed.
    epochs : int, optional
        the number of epochs for which to train the model
    suggestions : dict
        a dictionary with values of hyperparameters at the iteration when
        this objective function is called. The objective function will be
        called as many times as the number of iterations in optimization
        algorithm.
    """
    suggestions = jsonize(suggestions)
    global ITER

    # build model
    _model = Model(
        model={"layers": {
        "LSTM": {"units": suggestions['units']},
        "Activation": suggestions["activation"],
        "Dense": 1
    }},
        batch_size=suggestions["batch_size"],
        lr=suggestions["lr"],
        prefix=prefix or PREFIX,
        train_fraction=1.0,
        split_random=True,
        epochs=epochs,
        ts_args={"lookback": 14},
        input_features=data.columns.tolist()[0:-1],
        output_features=data.columns.tolist()[-1:],
        verbosity=0)

    # ai4water's Model class does not fix numpy seed
    # below we fix all the seeds including numpy but this seed it itself randomly generated
    if seed is None:
        seed = SEEDS[ITER]
        SEEDS_USED.append(seed)

    _model.seed_everything(seed)

    # train model
    _model.fit(data=data)

    # evaluate model
    t, p = _model.predict(data='validation', return_true=True, process_results=False)
    val_score = RegressionMetrics(t, p).mse()

    # here we are evaluating model with respect to mse, therefore
    # we don't need to subtract it from 1.0
    if not math.isfinite(val_score):
        val_score = 9999

    ITER += 1

    print(f"{ITER} {val_score} {seed}")

    if return_model:
        return _model

    return val_score


# parameter space
param_space = [
    Integer(16, 50, name="units"),
    Categorical(["relu", "elu", "tanh"], name="activation"),
    Real(0.00001, 0.01, name="lr"),
    Categorical([4, 8, 12, 16, 24], name="batch_size")
]

# initial values
x0 = [32, "relu", 0.001, 8]

# initialize the HyperOpt class and call fit method on it
optimizer = HyperOpt(
    algorithm="bayes",
    objective_fn=objective_fn,
    param_space=param_space,
    x0=x0,
    num_iterations=num_iterations,
    process_results=False,
    opt_path=f"results{SEP}{PREFIX}"
)

results = optimizer.fit()

###########################################

best_iteration = list(optimizer.best_xy().keys())[0].split('_')[-1]

seed_on_best_iter = SEEDS_USED[int(best_iteration)]

print(f"optimized parameters are \n{optimizer.best_paras()} at {best_iteration} with seed {seed_on_best_iter}")

##################################################

# we can now again call the objective function with best/optimium parameters


model = objective_fn(prefix=f"{PREFIX}{SEP}best",
                     seed=seed_on_best_iter,
                     return_model=True,
                     epochs=200,
                     **optimizer.best_paras())
