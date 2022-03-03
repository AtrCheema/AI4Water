"""
================
cross validation
================
"""

# This example shows how to app different cross
# validation techniques using ai4water

from ai4water import Model
from ai4water.preprocessing import DataSet
from ai4water.datasets import busan_beach


data = busan_beach()

#########################################
# TimeSeriesSplit
# ------------------

ds = DataSet(data=data)

splits = ds.TimeSeriesSplit_splits(n_splits=5)

for fold, ((train_x, train_y), (test_x, test_y)) in enumerate(splits):
    print(fold, train_x.shape, train_y.shape, test_x.shape, test_y.shape)

#####################################################

ds.plot_TimeSeriesSplit_splits(n_splits=5)

#########################################

model = Model(
    model={'RandomForestRegressor': {}},
    cross_validator={'TimeSeriesSplit': {'n_splits': 5}},
    val_metric="r2",
    verbosity=0,
)

tssplit_score = model.cross_val_score(data=data)

#####################################################
# KFold
# ------------------

ds = DataSet(data=data)

splits = ds.KFold_splits(n_splits=5)

for fold, ((train_x, train_y), (test_x, test_y)) in enumerate(splits):
    print(fold, train_x.shape, train_y.shape, test_x.shape, test_y.shape)

########################################################

ds.plot_KFold_splits(n_splits=5)

########################################################

model = Model(
    model={'RandomForestRegressor': {}},
    cross_validator={'KFold': {'n_splits': 5}},
    val_metric="r2",
    verbosity=0,
)

kfold_score = model.cross_val_score(data=data)


#####################################################
# LeaveOneOut
# ------------------

ds = DataSet(data=data.iloc[0:600, :])  # not using all data because it takes more time

splits = ds.LeaveOneOut_splits()

for fold, ((train_x, train_y), (test_x, test_y)) in enumerate(splits):
    print(fold, train_x.shape, train_y.shape, test_x.shape, test_y.shape)

########################################################

ds.plot_LeaveOneOut_splits()

########################################################

model = Model(
    model={'RandomForestRegressor': {}},
    cross_validator={'LeaveOneOut': {}},
    val_metric="mse",
    verbosity=0,
)

loo_score = model.cross_val_score(data=data.iloc[0:600, :])
