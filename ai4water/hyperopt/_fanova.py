"""
This implementation is carried out by following fanova package from automl and
the implementation of optuna
"""

__all__ = ["fANOVA"]

import itertools
import warnings
from typing import Tuple, List, Set, Union

from sklearn.preprocessing import OneHotEncoder
from sklearn.tree._tree import Tree
from sklearn.ensemble import RandomForestRegressor

from ai4water.backend import np, pd


class fANOVA(object):
    """
    Calculation of parameter importance using FANOVA (Hutter et al., 2014).

    Parameters
    ----------
    X :
        input data of shape (n_iterations, n_parameters). For hyperparameter optimization,
        iterations represent number of optimization iterations and parameter represent
        number of hyperparameters
    Y :
        objective value corresponding to X. Its length should be same as that of ``X``
    dtypes : list
        list of strings determining the type of hyperparameter. Allowed values are only
        ``categorical`` and ``numerical``.
    bounds : list
        list of tuples, where each tuple defines the upper and lower limit of corresponding
        parameter
    parameter_names : list
        names of features/parameters/hyperparameters
    cutoffs : tuple
    n_estimators : int
        number of trees
    max_depth : int (default=64)
        maximum depth of trees
    **rf_kws :
        keyword arguments to sklearn.ensemble.RandomForestRegressor

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from ai4water.hyperopt import fANOVA
    >>> x = np.arange(20).reshape(10, 2).astype(float)
    >>> y = np.linspace(1, 30, 10).astype(float)
    ... # X are hyperparameters and Y are objective function values at corresponding iterations
    >>> f = fANOVA(X=x, Y=y,
    ...            bounds=[(-2, 20), (-5, 50)],
    ...            dtypes=["numerical", "numerical"],
    ...            random_state=313, max_depth=3)
    ... # calculate importance
    >>> imp = f.feature_importance()

    # for categorical parameters

    >>> x = pd.DataFrame(['2', '2', '3', '1', '1', '2', '2', '1', '3', '3', '3'], columns=['a'])
    >>> x['b'] = ['3', '3', '1', '3', '1', '2', '4', '4', '3', '3', '4']
    >>> y = np.linspace(-1., 1.0, len(x))
    >>> f = fANOVA(X=x, Y=y, bounds=[None, None], dtypes=['categorical', 'categorical'],
    ...            random_state=313, max_depth=3, n_estimators=1)
    ... # calculate importance
    >>> imp = f.feature_importance()

    # for mix types

    >>> x = pd.DataFrame(['2', '2', '3', '1', '1', '2', '2', '1', '3', '3', '3'], columns=['a'])
    >>> x['b'] = np.arange(100, 100+len(x))
    >>> y = np.linspace(-1., 2.0, len(x))
    >>> f = fANOVA(X=x, Y=y, bounds=[None, (10, 150)], dtypes=['categorical', 'numerical'],
    ...           random_state=313, max_depth=5, n_estimators=5)
    ... # calculate importance
    >>> imp = f.feature_importance()

    """
    def __init__(
            self,
            X:Union[np.ndarray, pd.DataFrame],
            Y:np.ndarray,
            dtypes:List[str],
            bounds:List[Union[tuple, None]],
            parameter_names=None,
            cutoffs=(-np.inf, np.inf),
            n_estimators=64,
            max_depth=64,
            random_state=313,
            **rf_kws
    ):

        X_ = []
        encoders = {}
        cols = {}
        _bounds = []

        if isinstance(X, pd.DataFrame):
            if parameter_names is None:
                parameter_names = X.columns.tolist()
            X = X.values

        if parameter_names is None:
            parameter_names = [f"F{i}" for i in range(X.shape[1])]

        assert len(parameter_names) == len(bounds) == len(dtypes) == X.shape[1]

        assert len(X) == len(Y), f"X and Y should have same length"

        self.para_names = parameter_names

        if np.isnan(Y).sum()>0:
            warnings.warn("Removing nans encountered in Y.")
            df = pd.DataFrame(np.column_stack((X, Y))).dropna()
            X, Y = df.values[:, 0:-1], df.values[:, -1]

        for idx, dtype in enumerate(dtypes):
            if dtype.lower() == "categorical":
                x = X[:, idx]
                ohe = OneHotEncoder(sparse=False)
                x_ = ohe.fit_transform(x.reshape(-1,1))
                X_.append(x_)
                encoders[idx] = ohe
                cols[idx] = x_.shape[1]
                __bounds = [(0., 1.) for _ in range(x_.shape[1])]
                assert bounds[idx] is None, f"Cannot set bounds for categorical column"
                _bounds += __bounds
            else:
                X_.append(X[:, idx])
                cols[idx] = 1
                assert isinstance(bounds[idx], tuple), f"for non categorical parameters bounds must be given as tuple as (min,max)"
                assert len(bounds[idx]), 2
                assert bounds[idx][0] < bounds[idx][1]
                _bounds.append(bounds[idx])

        self.encoded_columns = encoded_columns(cols)
        X_ = np.column_stack(X_)
        self._n_features = X_.shape[1]
        self._N_features = X.shape[1]

        self.bounds = _bounds

        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **rf_kws
        )

        self.rf.fit(X_, Y)

        # initialize a dictionary with parameter dims
        self.variance_dict = dict()

        # all midpoints and interval sizes treewise for the whole forest
        self.all_midpoints = []
        self.all_sizes = []

        # compute midpoints and interval sizes for variables in each tree
        for dt in self.rf.estimators_:
            sizes = []
            midpoints = []

            tree_split_values = self._tree_split_values(dt.tree_)

            for i, split_vals in enumerate(tree_split_values):
                #if np.isnan(bounds[i][1]):  # categorical parameter
                #    pass
                #else:
                # add bounds to split values
                sv = np.array([_bounds[i][0]] + list(split_vals) + [_bounds[i][1]])
                # compute midpoints and sizes
                midpoints.append((1 / 2) * (sv[1:] + sv[:-1]))
                sizes.append(sv[1:] - sv[:-1])

            self.all_midpoints.append(midpoints)
            self.all_sizes.append(sizes)

        # capital V in the paper
        self.trees_total_variance = []
        # dict of lists where the keys are tuples of the dimensions
        # and the value list contains \hat{f}_U for the individual trees
        # reset all the variance fractions computed
        self.trees_variance_fractions = {}
        self.V_U_total = {}
        self.V_U_individual = {}

        self.cutoffs = cutoffs
        self.set_cutoffs(cutoffs)

    def _tree_split_values(self, tree:Tree):
        """calculates split values for a decision tree"""
        split_values = [set() for _ in range(self._n_features)]

        for node_index in range(tree.node_count):

            feature = tree.feature[node_index]
            if feature >= 0:  # Not leaf.
                threshold = tree.threshold[node_index]
                split_values[feature].add(threshold)

        sorted_split_values = []

        for split_val in split_values:
            split_values_array = np.array(list(split_val), dtype=np.float64)
            split_values_array.sort()
            sorted_split_values.append(split_values_array)

        return sorted_split_values

    def set_cutoffs(self, cutoffs=(-np.inf, np.inf), quantile=None):
        """
        Setting the cutoffs to constrain the input space

        To properly do things like 'improvement over default' the
        fANOVA now supports cutoffs on the y values. These will exclude
        parts of the parameters space where the prediction is not within
        the provided cutoffs. This is is specialization of
        "Generalized Functional ANOVA Diagnostics for High Dimensional
        Functions of Dependent Variables" by Hooker.
        """

        # reset all the variance fractions computed
        self.trees_variance_fractions = {}
        self.V_U_total = {}
        self.V_U_individual = {}

        # recompute the trees' total variance
        self.trees_total_variance = self.get_trees_total_variances()
        return

    def get_trees_total_variances(self)->tuple:
        """get variance of all trees"""
        variance = []
        for dt in self.rf.estimators_:
            variance.append(self._tree_variance(dt.tree_))
        return tuple(variance)

    def _tree_variance(self, tree:Tree):
        leaf_node_indices = np.argwhere(tree.feature<0).reshape(-1,)

        statistics = self._tree_statistics(tree)

        values, weights = [], []
        for node_index in leaf_node_indices:
            val, weight = statistics[node_index]
            values.append(val)
            weights.append(weight)

        avg_values = np.average(values, weights=weights)

        variance = np.average((np.array(values) - avg_values) ** 2, weights=weights)

        return variance

    def _tree_statistics(self, tree:Tree) -> np.ndarray:
        n_nodes = tree.node_count

        # Holds for each node, its weighted average value and the sum of weights.
        statistics = np.empty((n_nodes, 2), dtype=np.float64)

        subspaces = [None for _ in range(n_nodes)]
        subspaces[0] = np.array(self.bounds)

        # Compute marginals for leaf nodes.
        for node_index in range(n_nodes):
            subspace = subspaces[node_index]

            if tree.feature[node_index] < 0:
                value = tree.value[node_index]
                weight = _get_cardinality(subspace)
                statistics[node_index] = [value, weight]
            else:
                for child_node_index, child_subspace in zip(
                    _get_node_children(node_index, tree),
                    _get_node_children_subspaces(node_index, subspace, tree),
                ):
                    assert subspaces[child_node_index] is None
                    subspaces[child_node_index] = child_subspace

        # Compute marginals for internal nodes.
        for node_index in reversed(range(n_nodes)):
            if not tree.feature[node_index] < 0:  # if not node leaf
                child_values = []
                child_weights = []
                for child_node_index in _get_node_children(node_index, tree):
                    child_values.append(statistics[child_node_index, 0])
                    child_weights.append(statistics[child_node_index, 1])
                value = np.average(child_values, weights=child_weights)
                weight = np.sum(child_weights)
                statistics[node_index] = [value, weight]

        return statistics

    def _compute_marginals(self, dimensions):
        """
        Returns the marginal of selected parameters

        Parameters
        ----------
        dimensions: tuple
            Contains the indices of ConfigSpace for the selected parameters (starts with 0)
        """
        dimensions = tuple(dimensions)

        # check if values has been previously computed
        if dimensions in self.V_U_individual:
            return

        # otherwise make sure all lower order marginals have been
        # computed, if not compute them
        for k in range(1, len(dimensions)):
            for sub_dims in itertools.combinations(dimensions, k):
                if sub_dims not in self.V_U_total:
                    self._compute_marginals(sub_dims)

        assert len(dimensions)==1
        raw_dimensions = self.encoded_columns[dimensions[0]]

        # now all lower order terms have been computed
        self.V_U_individual[dimensions] = []
        self.V_U_total[dimensions] = []

        for tree_idx in range(len(self.all_midpoints)):

            # collect all the midpoints and corresponding sizes for that tree
            midpoints = [self.all_midpoints[tree_idx][dim] for dim in raw_dimensions]
            sizes = [self.all_sizes[tree_idx][dim] for dim in raw_dimensions]

            prod_midpoints = itertools.product(*midpoints)
            prod_sizes = itertools.product(*sizes)

            sample = np.full(self._n_features, np.nan, dtype=np.float)

            values: Union[List[float], np.ndarray] = []
            weights: Union[List[float], np.ndarray] = []

            # make prediction for all midpoints and weigh them by the corresponding size
            for i, (m, s) in enumerate(zip(prod_midpoints, prod_sizes)):
                sample[list(raw_dimensions)] = list(m)

                value, weight = self._tree_marginalized_statistics(sample, self.rf.estimators_[tree_idx].tree_)

                weight *= np.prod(s)

                values.append(value)
                weights.append(weight)

            weights = np.array(weights)
            values = np.array(values)
            average_values = np.average(values, weights=weights)
            variance = np.average((values - average_values) ** 2, weights=weights)

            assert variance >= 0.0, f"Non convergence of variance {variance}"

            # line 10 in algorithm 2
            # note that V_U^2 can be computed by var(\hat a)^2 - \sum_{subU} var(f_subU)^2
            # which is why, \hat{f} is never computed in the code, but
            # appears in the pseudocode
            V_U_total = np.nan
            V_U_individual = np.nan

            if weights.sum()>0:

                V_U_total = variance
                V_U_individual = variance

                for k in range(1, len(dimensions)):
                    for sub_dims in itertools.combinations(dimensions, k):
                        V_U_individual -= self.V_U_individual[sub_dims][tree_idx]

                V_U_individual = np.clip(V_U_individual, 0, np.inf)

            self.V_U_individual[dimensions].append(V_U_individual)
            self.V_U_total[dimensions].append(V_U_total)

        return

    def _tree_marginalized_statistics(self,
                                     feature_vector: np.ndarray,
                                     tree:Tree
                                     ) -> Tuple[float, float]:

        assert feature_vector.size == self._n_features

        statistics = self._tree_statistics(tree)

        marginalized_features = np.isnan(feature_vector)
        active_features = ~marginalized_features

        # Reduce search space cardinalities to 1 for non-active features.
        search_spaces = np.array(self.bounds.copy())
        search_spaces[marginalized_features] = [0.0, 1.0]

        # Start from the root and traverse towards the leafs.
        active_nodes = [0]
        active_search_spaces = [search_spaces]

        node_indices = []
        active_features_cardinalities = []

        tree_active_features = self._tree_active_features(tree)

        while len(active_nodes) > 0:
            node_index = active_nodes.pop()
            search_spaces = active_search_spaces.pop()

            feature = tree.feature[node_index]
            if feature >= 0:  # Not leaf. Avoid unnecessary call to `_is_node_leaf`.
                # If node splits on an active feature, push the child node that we end up in.
                response = feature_vector[feature]
                if not np.isnan(response):
                    if response <= tree.threshold[node_index]:
                        next_node_index = tree.children_left[node_index]
                        next_subspace = _get_node_left_child_subspaces(
                            node_index, search_spaces, tree
                        )
                    else:
                        next_node_index = tree.children_right[node_index]
                        next_subspace = _get_node_right_child_subspaces(
                            node_index, search_spaces, tree
                        )

                    active_nodes.append(next_node_index)
                    active_search_spaces.append(next_subspace)
                    continue

                # If subtree starting from node splits on an active feature, push both child nodes.
                if (active_features & tree_active_features[node_index]).any():
                    for child_node_index in _get_node_children(node_index, tree):
                        active_nodes.append(child_node_index)
                        active_search_spaces.append(search_spaces)
                    continue

            # If node is a leaf or the subtree does not split on any of the active features.
            node_indices.append(node_index)
            active_features_cardinalities.append(_get_cardinality(search_spaces))

        node_indices = np.array(node_indices, dtype=np.int32)
        active_features_cardinalities = np.array(active_features_cardinalities)

        statistics = statistics[node_indices]
        values = statistics[:, 0]
        weights = statistics[:, 1]
        weights = weights / active_features_cardinalities

        value = np.average(values, weights=weights)
        weight = weights.sum()

        return value, weight

    def _tree_active_features(self, tree:Tree) -> List[Set[int]]:

        subtree_active_features = np.full((tree.node_count, self._n_features), fill_value=False)

        for node_index in reversed(range(tree.node_count)):
            feature = tree.feature[node_index]
            if feature >= 0:  # Not leaf. Avoid unnecessary call to `_is_node_leaf`.
                subtree_active_features[node_index, feature] = True
                for child_node_index in _get_node_children(node_index, tree):
                    subtree_active_features[node_index] |= subtree_active_features[
                        child_node_index
                    ]

        return subtree_active_features

    def quantify_importance(self, dims):
        if type(dims[0]) == str:
            idx = []
            for i, param in enumerate(dims):
                idx.append(self.get_idx_by_hyperparameter_name(param))
            dimensions = tuple(idx)
        # make sure that all the V_U values are computed for each tree
        else:
            dimensions = dims

        self._compute_marginals(dimensions)

        importance_dict = {}

        for k in range(1, len(dimensions) + 1):
            for sub_dims in itertools.combinations(dimensions, k):
                if type(dims[0]) == str:
                    dim_names = []
                    for j, dim in enumerate(sub_dims):
                        dim_names.append(self.get_hyperparameter_by_idx(dim))
                    dim_names = tuple(dim_names)
                    importance_dict[dim_names] = {}
                else:
                    importance_dict[sub_dims] = {}
                # clean here to catch zero variance in a trees
                non_zero_idx = np.nonzero([self.trees_total_variance[t] for t in range(self.rf.n_estimators)])
                if len(non_zero_idx[0]) == 0:
                    raise RuntimeError('Encountered zero total variance in all trees.')

                fractions_total = np.array([self.V_U_total[sub_dims][t] / self.trees_total_variance[t]
                                            for t in non_zero_idx[0]])
                fractions_individual = np.array([self.V_U_individual[sub_dims][t] / self.trees_total_variance[t]
                                                 for t in non_zero_idx[0]])

                if type(dims[0]) == str:
                    importance_dict[dim_names]['individual importance'] = np.mean(fractions_individual)
                    importance_dict[dim_names]['total importance'] = np.mean(fractions_total)
                    importance_dict[dim_names]['individual std'] = np.std(fractions_individual)
                    importance_dict[dim_names]['total std'] = np.std(fractions_total)
                else:
                    importance_dict[sub_dims]['individual importance'] = np.mean(fractions_individual)
                    importance_dict[sub_dims]['total importance'] = np.mean(fractions_total)
                    importance_dict[sub_dims]['individual std'] = np.std(fractions_individual)
                    importance_dict[sub_dims]['total std'] = np.std(fractions_total)

        return importance_dict

    def feature_importance(self, dimensions=None, interaction_level=1, return_raw:bool=False):

        if dimensions is None:
            dimensions = self.para_names

        if isinstance(dimensions, (int, str)):
            dimensions = (dimensions,)

        importances_mean = {}
        importances_std = {}
        for dim in dimensions:
            imp = self.quantify_importance((dim, ))
            importances_mean[dim] = imp[(dim,)]['individual importance']
            importances_std[dim] = imp[(dim,)]['individual std']

        if len(dimensions) == self._N_features:
            importances_sum = sum(importances_mean.values())

            for name in importances_mean:
                importances_mean[name] /= importances_sum

        importances = {k: v for k, v in reversed(sorted(importances_mean.items(), key=lambda item: item[1]))}

        if return_raw:
            return importances_mean, importances_std

        return importances

    def get_hyperparameter_by_idx(self, idx:int)->str:
        return self.para_names[idx]

    def get_idx_by_hyperparameter_name(self, name:str)->int:
        return self.para_names.index(name)


def _get_node_right_child_subspaces(
        node_index: int,
        search_spaces: np.ndarray,
        tree:Tree
) -> np.ndarray:
    return _get_subspaces(
        search_spaces,
        search_spaces_column=0,
        feature=tree.feature[node_index],
        threshold=tree.threshold[node_index],
    )

def _get_node_children(node_index: int, tree:Tree) -> Tuple[int, int]:
    return tree.children_left[node_index], tree.children_right[node_index]


def _get_cardinality(search_spaces: np.ndarray) -> float:
    return np.prod(search_spaces[:, 1] - search_spaces[:, 0]).item()


def _get_subspaces(
        search_spaces: np.ndarray, *,
        search_spaces_column: int, feature: int, threshold: float
) -> np.ndarray:
    search_spaces_subspace = np.copy(search_spaces)
    search_spaces_subspace[feature, search_spaces_column] = threshold
    return search_spaces_subspace


def _get_node_children_subspaces(
    node_index: int, search_spaces: np.ndarray, tree
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        _get_node_left_child_subspaces(node_index, search_spaces, tree),
        _get_node_right_child_subspaces(node_index, search_spaces, tree),
    )


def _get_node_left_child_subspaces(
    node_index: int, search_spaces: np.ndarray, tree
) -> np.ndarray:
    return _get_subspaces(
        search_spaces,
        search_spaces_column=1,
        feature=tree.feature[node_index],
        threshold=tree.threshold[node_index],
    )


def encoded_columns(cols:dict):
    # c = {'a': 2, 'b': 1, 'c': 3}
    # -> {'a': [0,1], 'b': [2], 'c': [3,4,5]}
    en = 0
    st = 0
    columns = {}
    for k, v in cols.items():
        en += v
        columns[k] = [i for i in range(st, en)]
        st += v
    return columns

if __name__ == "__main__":

    pass