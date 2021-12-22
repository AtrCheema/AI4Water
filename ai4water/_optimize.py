import importlib
from typing import Union

from .postprocessing.SeqMetrics import RegressionMetrics
from .utils.utils import dateandtime_now, jsonize, MATRIC_TYPES, update_model_config, TrainTestSplit


class ModelOptimizerMixIn(object):

    def __init__(
            self,
            model,
            algorithm,
            num_iterations,
            process_results,
            prefix="hpo",
            data=None
    ):
        self.model = model
        self.algorithm = algorithm
        self.num_iterations = num_iterations
        self.process_results = process_results
        self.prefix = prefix
        if data is not None:
            if not isinstance(data, tuple):
                assert isinstance(data, list) and len(data) == 2
        self.data = data

    def fit(self):

        PREFIX = f"{self.prefix}_{dateandtime_now()}"

        hpo = importlib.import_module("ai4water.hyperopt")

        data = self.model.data
        val_metric = self.model.config['val_metric']
        metric_type = MATRIC_TYPES.get(val_metric, 'min')
        cross_validator = self.model.config['cross_validator']
        config = jsonize(self.model.config)

        def objective_fn(
                seed=None,
                **suggestions,
        ):
            config['seed'] = seed
            config['verbosity'] = -1
            config['prefix'] = PREFIX

            getattr(self, f'update')(config, suggestions)

            _model = self.model.from_config(
                config.copy(),
                data=data,
                make_new_path=True,
            )

            if self.data is not None:  # todo, it is better to split data outside objective_fn
                splitter = TrainTestSplit(*self.data, test_fraction=config['test_fraction'])
                train_x, test_x, train_y, test_y = splitter.split_by_slicing()
                _model.fit(x=train_x, y=train_y)
            else:
                _model.fit()

            if cross_validator is None:

                if self.data is not None:

                    p = _model.predict(test_x)
                else:
                    test_y, p = _model.predict(return_true=True, process_results=False)
                metrics = RegressionMetrics(test_y, p)
                val_score = getattr(metrics, val_metric)()
            else:
                val_score = self.model.cross_val_score(data=self.data)

            if metric_type != "min":
                val_score = 1.0 - val_score

            print("val_score", val_score)

            return val_score

        optimizer = hpo.HyperOpt(
            self.algorithm,
            objective_fn=objective_fn,
            param_space=self.space,
            num_iterations=self.num_iterations,
            process_results=self.process_results,
            opt_path=f"results\\{PREFIX}"
        )

        optimizer.fit()

        return optimizer


class OptimizeHyperparameters(ModelOptimizerMixIn):

    def __init__(
            self,
            model,
            space,
            algorithm,
            num_iterations,
            process_results=False,
            data=None,
            **kwargs
    ):
        super().__init__(
            model=model,
            algorithm=algorithm,
            num_iterations=num_iterations,
            process_results=process_results,
            data=data,
            prefix="hpo"
        )

        self.space = space
        config = jsonize(model.config)
        model_config = config['model']
        # algo type is name of algorithm, e.g. xgboost, randomforest or layers
        self.algo_type = list(model_config.keys())[0]
        self.original_model = model._original_model_config

    def update(self, config, suggestions):
        # first update the model config parameters
        new_model_config = update_model_config(self.original_model['model'].copy(), suggestions)
        config['model'] = {self.algo_type: new_model_config}

        # now update hyperparameters which are not part of model config
        new_other_config = update_model_config(self.original_model['other'].copy(), suggestions)
        config.update(jsonize(new_other_config))

        return config


class OptimizeTransformations(ModelOptimizerMixIn):

    def __init__(
            self,
            model,
            categories,
            num_iterations=12,
            algorithm="bayes",
            include=None,
            exclude=None,
            append=None,
            process_results=False,
            data=None,
    ):
        super().__init__(
            model=model,
            num_iterations=num_iterations,
            algorithm=algorithm,
            process_results=process_results,
            prefix="trans_hpo",
            data=data
        )

        self.space = make_space(self.model.data.columns.to_list(),
                                include=include, exclude=exclude, append=append,
                                categories=categories)

    def update(self, config, suggestions):

        transformations = []

        for feature, method in suggestions.items():

            if method == "none":
                pass
            else:

                t = {"method": method, "features": [feature]}

                if method.startswith("log"):
                    t["treat_negatives"] = True
                elif method == "box-cox":
                    t["treat_negatives"] = True
                    t["replace_zeros"] = True
                elif method == "sqrt":
                    t['treat_negatives'] = True

                transformations.append(t)

        # following parameters must be overwritten even if they were provided by the user.
        config['transformation'] = transformations
        return


def make_space(
        features: list,
        categories: list,
        include: Union[str, list, dict] = None,
        exclude: Union[str, list] = None,
        append: dict = None,
) -> list:
    """
    Arguments:
        features :
        categories :
        include: the names of features to include
        exclude: the name/names of features to exclude
        append: the features with custom candidate transformations
    """
    hpo = importlib.import_module("ai4water.hyperopt")
    Categorical = hpo.Categorical

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
        for k, v in include.items():
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
        for k, v in append.items():
            if not isinstance(v, Categorical):
                assert isinstance(v, list), f"space for {k} must be list but it is {v.__class__.__name__}"
                v = Categorical(v, name=k)
            space[k] = v

    return list(space.values())
