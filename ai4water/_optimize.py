import importlib
import os.path
from typing import Union

from .utils.utils import dateandtime_now, jsonize, MATRIC_TYPES
from .postprocessing.SeqMetrics import RegressionMetrics


PREFIX = f"trans_hpo_{dateandtime_now()}"


def optimize_transformations(
        model,
        num_iterations=12,
        algorithm="bayes",
        include=None,
        exclude=None,
        append=None
):

    hpo = importlib.import_module("ai4water.hyperopt")

    data = model.data
    val_metric = model.config['val_metric']
    metric_type = MATRIC_TYPES.get(val_metric, 'min')
    cross_validator = model.config['cross_validator']
    config = jsonize(model.config)

    space = make_space(data.columns.to_list(), include=include, exclude=exclude, append=append)

    def objective_fn(
            seed=None,
            **suggestions):

        transformations = []

        for feature, method in suggestions.items():

            if method == "none":
                pass
            else:

                t = {"method": method, "features": [feature]}

                if method.startswith("log"):
                    t["replace_nans"] = True
                    t["replace_zeros"] = True
                transformations.append(t)

        # following parameters must be overwritten even if they were provided by the user.
        config['verbosity'] = 0
        config['prefix'] = PREFIX
        config['transformation'] = transformations
        config['seed'] = seed

        _model = model.from_config(
            config.copy(),
            data=data,
            make_new_path=True,
        )

        _model.fit()

        if cross_validator is None:

            t, p = _model.predict(return_true=True)
            metrics = RegressionMetrics(t, p)
            val_score = getattr(metrics, val_metric)()
        else:
            val_score = model.cross_val_score()

        if metric_type != "min":
            val_score = 1.0 - val_score

        print("val_score", val_score)

        return val_score

    optimizer = hpo.HyperOpt(
        algorithm,
        objective_fn=objective_fn,
        param_space=space,
        num_iterations=num_iterations,
        opt_path=f"results\\{PREFIX}"
    )

    optimizer.fit()

    return optimizer


def make_space(
        features:list,
        include:Union[str, list, dict]=None,
        exclude:Union[str, list]=None,
        append:dict=None
)->list:
    """
    Arguments:
        features :
        include: the names of features to include
        exclude: the name/names of features to exclude
        append: the features with custom candidate transformations
    """
    hpo = importlib.import_module("ai4water.hyperopt")
    Categorical = hpo.Categorical

    categories = ["minmax", "zscore", "log", "robust", "quantile", "log2", "log10", "none"]

    # although we need space as list at the end, it is better to create a dictionary of it
    # because manipulating dictionary is easier
    space = {
        name: Categorical(categories, name=name) for name in features
    }

    if include is not None:
        if isinstance(include, str):
            include = {include: categories}
        elif isinstance(include, list):
            _include = {}
            for feat in include:
                if isinstance(feat, str):
                    _include[feat] = categories
                else:
                    assert isinstance(feat, dict) and len(feat) == 1
                    _include[list(feat.keys())[0]] = list(feat.values())[0]
            include = _include
        else:
            assert isinstance(include, dict) and len(include) == 1

        # since include is given, we will ignore default case when all features are considered
        space = {}
        for k,v in include.items():
            if not isinstance(v, Categorical):
                assert isinstance(v, list), f"space for {k} must be list but it is {v.__class__.__name__}"
                v = Categorical(v, name=k)
            space[k] = v

    if exclude is not None:
        if isinstance(exclude, str):
            exclude = [exclude]
        assert isinstance(exclude, list)
        for feat in exclude:
            space.pop(feat)

    if append is not None:
        assert isinstance(append, dict)
        for k,v in append.items():
            if not isinstance(v, Categorical):
                assert isinstance(v, list), f"space for {k} must be list but it is {v.__class__.__name__}"
                v = Categorical(v, name=k)
            space[k] = v

    return list(space.values())