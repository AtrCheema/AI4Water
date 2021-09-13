

from ai4water import Model
from ai4water.datasets import arg_beach

model = Model(
    model = {'randomforestregressor': {}},
    data = arg_beach(),
    cross_validator = {'TimeSeriesSplit': {'n_splits': 5}},
    val_metric="r2"
)

tssplit_score = model.cross_val_score()

model = Model(
    model = {'randomforestregressor': {}},
    data = arg_beach(),
    cross_validator = {'KFold': {'n_splits': 5}},
    val_metric="r2"
)

kfold_score = model.cross_val_score()

model = Model(
    model = {'randomforestregressor': {}},
    data = arg_beach(),
    cross_validator = {'LeaveOneOut': {}},
    val_metric="mse"
)

loo_score = model.cross_val_score()

