

from AI4Water import Model
from AI4Water.utils.datasets import arg_beach

model = Model(
    model = {'randomforestregressor': {}},
    data = arg_beach(),
    cross_validator = {'kfold': {'n_splits': 5}}
)

val_score = model.cross_val_score()

model = Model(
    model = {'randomforestregressor': {}},
    data = arg_beach(),
    cross_validator = {'LeaveOneOut': {}}
)

loo_score = model.cross_val_score()

