# This file contains code mostly or 90% from optuna library. The reason to rewrite this code is because
# I was unable to create an Optuna Study instance without `Storage` attribute. But it appears that
# we can calculate parameter importance without Storage attribute from Study instance. Thus
# the importance is calculated from Study instance without it having Storage attribute.
# The optuna library comes with following MIT licence.
"""
MIT License

Copyright (c) 2018 Preferred Networks, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from collections import OrderedDict
from typing import Callable
from typing import List
from typing import Optional

import numpy

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


from easy_mpl import bar_chart


logger = get_logger(__name__)



def _get_distributions(study, params):
    # based on supposition that get_distributions only returns an ordered dictionary and requiring `storage` attribute
    # of study is redundant
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
    ):

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
        variance = {}
        for i, name in enumerate(distributions.keys()):
            _mean, _std = evaluator.get_importance((i,))
            importances[name] = _mean
            variance[name] = {'mean': _mean, 'std': _std}

        total_importance = sum(importances.values())
        for name in importances:
            importances[name] /= total_importance

        sorted_importances = OrderedDict(
            reversed(
                sorted(importances.items(), key=lambda name_and_importance: name_and_importance[1])
            )
        )

        return sorted_importances, variance


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
    try:
        importances, importance_paras = get_param_importances(
            study, evaluator=evaluator, params=params, target=target
        )
    except RuntimeError:  # sometimes it is returning error e.g. when number of trials are < 4
        return None, None, None

    importances = OrderedDict(reversed(list(importances.items())))
    importance_values = list(importances.values())
    param_names = list(importances.keys())

    ax = bar_chart(importance_values, param_names, orient='h', show=False)

    return importances, importance_paras, ax


def _get_distribution(param_name: str, study: Study):
    for trial in study.trials:
        if param_name in trial.distributions:
            return trial.distributions[param_name]
    assert False


