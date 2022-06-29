"""
============================
HyperOpt for neural networks
============================

If your model consists of layers of neural neworks, you can easily modify
the objective function accordingly. The only change you have to do is to
get corresponding parameters from ``suggestions`` and use them
to build the layers of neural network.
"""

import os
import math
from typing import Union

import numpy as np
from SeqMetrics import RegressionMetrics

from ai4water.functional import Model
from ai4water.datasets import busan_beach
from ai4water.models import LSTM
from ai4water.utils.utils import jsonize, dateandtime_now
from ai4water.hyperopt import HyperOpt, Categorical, Real, Integer

data = busan_beach()

SEP = os.sep

import tensorflow as tf

print(tf.__version__, np.__version__)

#%%
#np.random.seed(313)

###########################################

PREFIX = f"hpo_nn_{dateandtime_now()}"
ITER = 0
num_iterations = 30

# these seeds are randomly generated but we keep track of the seed
# used at each iteration, so that when we rebuilt the model with optimized
# hyperparameters, we get reproducible results
SEEDS = np.random.randint(0, 1000, num_iterations)
# to keep track of seed being used at every optimization iteration
SEEDS_USED = []
SUGGESTIONS = {}

# It is always a good practice to monitor more than 1 performance metric,
# even though our objective function will not be based upon these
# performance metrics.
MONITOR = {"mse": [], "nse": [], "r2": [], "pbias": [], "nrmse": []}

###########################################
# 1) define objective function
#-------------------------------

def objective_fn(
        prefix: str = None,
        return_model: bool = False,
        epochs:int = 50,
        verbosity: int = 0,
        predict : bool = False,
        seed=None,
        **suggestions
)->Union[float, Model]:
    """This function must build, train and evaluate the ML model.
    The output of this function will be minimized by optimization algorithm.

    Parameters
    ----------
    prefix : str
        prefix to save the results. This argument will only be used after
        the optimization is complete
    return_model : bool, optional (default=False)
        if True, then objective function will return the built model. This
        argument will only be used after the optimization is complete
    epochs : int, optional
        the number of epochs for which to train the model
    verbosity : int, optional (default=1)
        determines the amount of information to be printed
    predict : bool, optional (default=False)
        whether to make predictions on training and validation data or not.
    suggestions : dict
        a dictionary with values of hyperparameters at the iteration when
        this objective function is called. The objective function will be
        called as many times as the number of iterations in optimization
        algorithm.

    Returns
    -------
    float or Model
    """
    suggestions = jsonize(suggestions)
    global ITER

    # build model
    _model = Model(
        model=LSTM(units=suggestions['units'],
                   activation=suggestions['activation'],
                   dropout=0.2),
        batch_size=suggestions["batch_size"],
        lr=suggestions["lr"],
        prefix=prefix or PREFIX,
        train_fraction=1.0,
        split_random=True,
        epochs=epochs,
        ts_args={"lookback": 14},
        input_features=data.columns.tolist()[0:-1],
        output_features=data.columns.tolist()[-1:],
        x_transformation="zscore",
        y_transformation={"method": "log", "replace_zeros": True, "treat_negatives": True},
        verbosity=verbosity)

    # ai4water's Model class does not fix numpy seed
    # below we fix all the seeds including numpy but this seed it itself randomly generated
    if seed is None:
        seed = SEEDS[ITER]
        SEEDS_USED.append(seed)

    _model.seed_everything(seed)
    SUGGESTIONS[ITER] = suggestions

    # train model
    _model.fit(data=data)

    # evaluate model
    t, p = _model.predict_on_validation_data(data=data, return_true=True)
    metrics = RegressionMetrics(t, p)
    val_score = metrics.rmse()

    for metric in MONITOR.keys():
        val = getattr(metrics, metric)()
        MONITOR[metric].append(val)

    # here we are evaluating model with respect to mse, therefore
    # we don't need to subtract it from 1.0
    if not math.isfinite(val_score):
        val_score = 9999

    print(f"{ITER} {val_score} {seed}")

    ITER += 1

    if predict:
        _model.predict_on_training_data(data=data)
        _model.predict_on_validation_data(data=data)
        _model.predict_on_all_data(data=data)

    if return_model:
        return _model

    return val_score

###########################################
# 2) define parameter space
#-------------------------------
# parameter space
param_space = [
    Integer(10, 20, name="units"),
    Categorical(["relu", "elu", "tanh"], name="activation"),
    Real(0.00001, 0.01, name="lr"),
    Categorical([4, 8, 12, 16, 24], name="batch_size")
]

###########################################
# 3) initial state
#-------------------------------
# initial values
x0 = [16, "relu", 0.001, 8]

#############################################
# 4) run optimization algorithm
#---------------------------------------------
# initialize the HyperOpt class and call fit method on it
optimizer = HyperOpt(
    algorithm="bayes",
    objective_fn=objective_fn,
    param_space=param_space,
    x0=x0,
    num_iterations=num_iterations,
    opt_path=f"results{SEP}{PREFIX}"
)

results = optimizer.fit()

###########################################

best_iteration = optimizer.best_iter()

seed_on_best_iter = SEEDS_USED[int(best_iteration)]

print(f"optimized parameters are \n{optimizer.best_paras()} at {best_iteration} seed {seed_on_best_iter}")

##################################################
# we are interested in the minimum value of following metrics

for key in ['mse', 'nrmse', 'pbias']:
    print(key, np.nanmin(MONITOR[key]), np.nanargmin(MONITOR[key]))

#%%
# we are interested in the maximum value of following metrics

for key in ['r2', 'nse']:
    print(key, np.nanmax(MONITOR[key]), np.nanargmax(MONITOR[key]))

#%%

# we can now again call the objective function with best/optimium parameters

###########################################
# train with best hyperparameters
#---------------------------------------------

model = objective_fn(prefix=f"{PREFIX}{SEP}best",
                     seed=seed_on_best_iter,
                     return_model=True,
                     epochs=200,
                     verbosity=0,
                     predict=True,
                     **optimizer.best_paras())
