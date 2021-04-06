from collections import OrderedDict
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy

# import optuna
# from optuna.distributions import BaseDistribution
# #from optuna.importance._base import _get_distributions
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
#from optuna.importance._base import BaseImportanceEvaluator
from optuna.logging import get_logger
from optuna._transform import _SearchSpaceTransform
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.importance._fanova import FanovaImportanceEvaluator
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args
from optuna.visualization._plotly_imports import go
from optuna.importance import get_param_importances

import plotly



logger = get_logger(__name__)

Blues = plotly.colors.sequential.Blues

_distribution_colors = {
    UniformDistribution: Blues[-1],
    LogUniformDistribution: Blues[-1],
    DiscreteUniformDistribution: Blues[-1],
    IntUniformDistribution: Blues[-2],
    IntLogUniformDistribution: Blues[-2],
    CategoricalDistribution: Blues[-4],
}

def _get_distributions(study, params):
    assert params is None
    trial = study.trials[0]
    return OrderedDict(trial.distributions)

class ImportanceEvaluator(FanovaImportanceEvaluator):

    def evaluate(
        self,
        study: Study,
        params: Optional[List[str]] = None,
        *,
        target: Optional[Callable[[FrozenTrial], float]] = None,
    ) -> Dict[str, float]:

        if target is None and study._is_multi_objective():
            raise ValueError(
                "If the `study` is being used for multi-objective optimization, "
                "please specify the `target`."
            )

        distributions = _get_distributions(study, params)
        if len(distributions) == 0:
            return OrderedDict()

        trials = []
        for trial in study.trials:
            if trial.state != TrialState.COMPLETE:
                continue
            if any(name not in trial.params for name in distributions.keys()):
                continue
            trials.append(trial)

        trans = _SearchSpaceTransform(distributions, transform_log=False, transform_step=False)

        n_trials = len(trials)
        trans_params = numpy.empty((n_trials, trans.bounds.shape[0]), dtype=numpy.float64)
        trans_values = numpy.empty(n_trials, dtype=numpy.float64)

        for trial_idx, trial in enumerate(trials):
            trans_params[trial_idx] = trans.transform(trial.params)
            trans_values[trial_idx] = trial.value if target is None else target(trial)

        trans_bounds = trans.bounds
        column_to_encoded_columns = trans.column_to_encoded_columns

        if trans_params.size == 0:  # `params` were given but as an empty list.
            return OrderedDict()

        # Many (deep) copies of the search spaces are required during the tree traversal and using
        # Optuna distributions will create a bottleneck.
        # Therefore, search spaces (parameter distributions) are represented by a single
        # `numpy.ndarray`, coupled with a list of flags that indicate whether they are categorical
        # or not.

        evaluator = self._evaluator
        evaluator.fit(
            X=trans_params,
            y=trans_values,
            search_spaces=trans_bounds,
            column_to_encoded_columns=column_to_encoded_columns,
        )

        importances = {}
        for i, name in enumerate(distributions.keys()):
            importance, _ = evaluator.get_importance((i,))
            importances[name] = importance

        total_importance = sum(importances.values())
        for name in importances:
            importances[name] /= total_importance

        sorted_importances = OrderedDict(
            reversed(
                sorted(importances.items(), key=lambda name_and_importance: name_and_importance[1])
            )
        )

        return sorted_importances



# def get_param_importances(
#     study: Study,
#     *,
#     evaluator  = None,
#     params: Optional[List[str]] = None,
#     target: Optional[Callable[[FrozenTrial], float]] = None,
# ) -> Dict[str, float]:
#
#     if evaluator is None:
#         evaluator = ImportanceEvaluator()
#
#     return evaluator.evaluate(study, params=params, target=target)

def plot_param_importances(
    study: Study,
    evaluator = None,
    params: Optional[List[str]] = None,
    *,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
):

    _imports.check()
    _check_plot_args(study, target, target_name)

    layout = go.Layout(
        title="Hyperparameter Importances",
        xaxis={"title": f"Importance for {target_name}"},
        yaxis={"title": "Hyperparameter"},
        showlegend=False,
    )

    # Importances cannot be evaluated without completed trials.
    # Return an empty figure for consistency with other visualization functions.
    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if len(trials) == 0:
        logger.warning("Study instance does not contain completed trials.")
        return go.Figure(data=[], layout=layout)

    if evaluator is None:
        evaluator = ImportanceEvaluator()
    importances = get_param_importances(
        study, evaluator=evaluator, params=params, target=target
    )

    importances = OrderedDict(reversed(list(importances.items())))
    importance_values = list(importances.values())
    param_names = list(importances.keys())

    fig = go.Figure(
        data=[
            go.Bar(
                x=importance_values,
                y=param_names,
                text=importance_values,
                texttemplate="%{text:.2f}",
                textposition="outside",
                cliponaxis=False,  # Ensure text is not clipped.
                hovertemplate=[
                    _make_hovertext(param_name, importance, study)
                    for param_name, importance in importances.items()
                ],
                marker_color=[_get_color(param_name, study) for param_name in param_names],
                orientation="h",
            )
        ],
        layout=layout,
    )

    return importances, fig


def _get_distribution(param_name: str, study: Study):
    for trial in study.trials:
        if param_name in trial.distributions:
            return trial.distributions[param_name]
    assert False


def _get_color(param_name: str, study: Study) -> str:
    return _distribution_colors[type(_get_distribution(param_name, study))]


def _make_hovertext(param_name: str, importance: float, study: Study) -> str:
    return "{} ({}): {}<extra></extra>".format(
        param_name, _get_distribution(param_name, study).__class__.__name__, importance
    )


