import os
from typing import Union, Callable, List

import numpy as np
from easy_mpl import imshow
import scipy.stats as stats
import matplotlib.pyplot as plt

from ._explain import ExplainerMixin
from ai4water.utils.utils import reset_seed, ERROR_LABELS


class PermutationImportance(ExplainerMixin):
    """
    permutation importance answers the question, how much the model's prediction
    performance is influenced by a feature? It defines the feature importance as
    the decrease in model performance when one feature is corrupted
    ([Molnar et al., 2021](https://christophm.github.io/interpretable-ml-book/feature-importance.html#feature-importance))

    Attributes:
        importances

    """
    def __init__(
            self,
            model: Callable,
            inputs: Union[np.ndarray, List[np.ndarray]],
            target: np.ndarray,
            scoring: Union[str, Callable] = "r2",
            n_repeats: int = 14,
            noise: Union[str, np.ndarray] = None,
            use_noise_only: bool = False,
            feature_names: list = None,
            path: str = None,
            seed: int = None,
            weights=None,
            **kwargs
    ):
        """
        initiates a the class and calculates the importances

        Arguments:
            model:
                the trained model object which is callable e.g. if you have Keras
                or sklearn model then you should pass `model.predict` instead
                of `model`.
            inputs:
                arrays or list of arrays which will be given as input to `model`
            target:
                the true outputs or labels for corresponding `inputs`
                It must be a numpy array
            scoring:
                the peformance metric to use. It can be any metric from
                [RegressionMetrics][ai4water.postprocessing.SeqMetrics.RegressionMetrics],
                [ClassificationMetrics][ai4water.postprocessing.SeqMetrics.ClassificationMetrics]
                or a callable. If callable, then this must take true and predicted
                as input and sprout a float as output
            n_repeats:
                number of times the permutation for each feature is performed. Number
                of calls to the `model` will be `num_features * n_repeats`
            noise:
                The noise to add in the feature. It should be either an array of noise
                or a string of [scipy distribution name](https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions)
                defining noise.
            use_noise_only:
                If True, the original feature will be replaced by the noise.
            weights:
            feature_names:
                names of features
            seed:
                random seed for reproducibility. Permutation importance is
                strongly affected by random seed. Therfore, if you want to
                reproduce your results, set this value to some integer value.
            path:
                path to save the plots
            kwargs:
                any additional keyword arguments for `model`

        """
        assert callable(model), f"model must be callable"
        self.model = model

        if inputs.__class__.__name__ in ["Series", "DataFrame"]:
            inputs = inputs.values

        self.x = inputs
        self.y = target
        self.scoring = scoring
        self.noise = noise

        if use_noise_only:
            if noise is None:
                raise ValueError("you must define the noise in order to replace it with feature")

        self.use_noise_only = use_noise_only
        self.n_repeats = n_repeats
        self.weights = weights
        self.kwargs = kwargs

        self.importances = None

        super().__init__(features=feature_names, data=inputs, path=path or os.getcwd())

        self.seed = seed

        self._calculate(**kwargs)

    @property
    def noise(self):
        return self._noise

    @noise.setter
    def noise(self, x):
        if x is not None:
            if isinstance(x, str):
                x = getattr(stats, x)().rvs(len(self.y))
            else:
                assert isinstance(x, np.ndarray) and len(x) == len(self.y)

        self._noise = x

    def _base_score(self) -> float:
        """calculates the base score"""
        return self._score(self.model(self.x, **self.kwargs))

    def _score(self, pred) -> float:
        """given the prediction, it calculates the score"""
        from ..SeqMetrics import RegressionMetrics, ClassificationMetrics

        if callable(self.scoring):
            return self.scoring(self.y, pred)
        else:
            if hasattr(RegressionMetrics, self.scoring):
                errors = RegressionMetrics(self.y, pred)
            else:
                errors = ClassificationMetrics(self.y, pred)

            return getattr(errors, self.scoring)()

    def _calculate(
            self,
            **kwargs
    ):
        """Calculates permutation importance using self.x"""

        if self.single_source:

            if self.x.ndim == 2:  # 2d input
                results = self._permute_importance_2d(self.x, **kwargs)

            else:
                results = {}
                for lb in range(self.x.shape[1]):
                    results[lb] = self._permute_importance_2d(self.x,
                                                              time_step=lb,
                                                              **kwargs)

        else:
            results = {}

            for idx in range(len(self.x)):

                if self.x[idx].ndim == 2:  # current input is 2d

                    results[idx] = self._permute_importance_2d(
                        self.x,
                        idx,
                        **kwargs
                    )

                elif self.x[idx].ndim == 3:  # current input is 3d

                    _results = {}
                    for lb in range(self.x[idx].shape[1]):
                        _results[lb] = self._permute_importance_2d(self.x,
                                                                   inp_idx=idx,
                                                                   time_step=lb,
                                                                   **kwargs)

                    results[idx] = _results

                else:
                    raise NotImplementedError

        setattr(self, 'importances', results)

        return results

    def plot_as_heatmap(
            self,
            annotate=True,
            show: bool = True,
            **kwargs
    ):
        """plots the permutation importance as heatmap.

        The input data must be 3d.

        Arguments:
            annotate:
                whether to annotate the heat map with
            show:
                whether to show the plot or not
            kwargs:
                any keyword arguments for [imshow][ai4water.utils.easy_mpl.imshow] function.
        """
        assert self.data_is_3d, f"data must be 3d but it is has {self.x.shape}"

        imp = np.stack([np.mean(v, axis=1) for v in self.importances.values()])

        fig, axis = plt.subplots()

        lookback = imp.shape[0]
        ytick_labels = [f"t-{int(i)}" for i in np.linspace(lookback - 1, 0, lookback)]
        axis, im = imshow(
            imp,
            ax=axis,
            yticklabels=ytick_labels,
            xticklabels=self.features if len(self.features) <= 10 else None,
            ylabel="Looback steps",
            xlabel="Input Features",
            annotate=annotate,
            title=f"Base Score {round(self._base_score(), 3)} with {ERROR_LABELS[self.scoring]}",
            show=False,
            **kwargs
        )

        fig.colorbar(im, orientation='vertical')

        if show:
            plt.show(

            )
        return axis

    def plot_as_boxplot(
            self,
            show: bool = True,
            save: bool = False
    ) -> plt.Axes:
        """Plots the permutation importance as box plots

        Arguments:
            show:
                whether to show the plot or now
            save:
                whether to save the plot or not

        Returns:
            matplotlib AxesSubplot
        """

        if isinstance(self.importances, np.ndarray):
            fig, ax = plt.subplots()
            self._plot_importance(self.importances, self.features, ax,
                                  show=show, save=save)
        else:
            for idx,  importance in enumerate(self.importances.values()):
                if self.data_is_3d:
                    features = self.features
                else:
                    features = self.features[idx]
                ax = self._plot_importance(importance, features, show=show,
                                           save=save, name=idx)

        return ax

    def _permute_importance_2d(
            self,
            inputs,
            inp_idx=None,
            time_step=None,
            **kwargs
    ):
        """
        calculates permutation importance by permuting columns in inputs
        which is supposed to be 2d array. args are optional inputs to model
        """
        original_inp_idx = inp_idx
        if inp_idx is None:
            inputs = [inputs]
            inp_idx = 0

        permuted_x = inputs[inp_idx].copy()
        # reset seed for reproducibility
        reset_seed(self.seed, np=np)

        feat_dim = 1  # feature dimension (0, 1, 2)
        if time_step is not None:
            feat_dim = 2

        # empty container to keep results
        results = np.full((permuted_x.shape[feat_dim], self.n_repeats), np.nan)  # (num_features, n_repeats)

        # todo, instead of having two for loops, we can perturb the
        #  inputs at once and concatenate
        # them as one input and thus call the `model` only once

        for col_index in range(permuted_x.shape[feat_dim]):

            scores = np.full(self.n_repeats, np.nan)

            for n_round in range(self.n_repeats):

                # sklearn sets the random state before permuting each feature
                # also sklearn sets the RandomState insite a function therefore
                # the results from this function will not be reproducible with
                # sklearn and vice versa
                if time_step is None:
                    perturbed_feature = np.random.permutation(
                        permuted_x[:, col_index])
                else:
                    perturbed_feature = np.random.permutation(
                        permuted_x[:, time_step, col_index]
                    )

                if self.noise is not None:
                    if self.use_noise_only:
                        perturbed_feature = self.noise
                    else:
                        perturbed_feature += self.noise

                if time_step is None:
                    permuted_x[:, col_index] = perturbed_feature
                else:
                    permuted_x[:, time_step, col_index] = perturbed_feature

                # put the permuted input back in the list
                inputs[inp_idx] = permuted_x

                if original_inp_idx is None:  # inputs were not list so unpack the list
                    prediction = self.model(*inputs, **kwargs)
                else:
                    prediction = self.model(inputs, **kwargs)

                scores[n_round] = self._score(prediction)

            results[col_index] = scores

        # permutation importance is how much performance decreases by permutation
        results = self._base_score() - results

        return results

    def _permute_importance_2d1(
            self,
            inputs
    ):
        """
        todo inorder to reproduce sklearn's results, use this function
        """

        def _func(_inputs, col_idx):

            permuted_x = _inputs.copy()

            scores = np.full(self.n_repeats, np.nan)
            random_state = np.random.RandomState(self.seed)

            for n_round in range(self.n_repeats):

                perturbed_feature = permuted_x[:, col_idx]

                random_state.shuffle(perturbed_feature)

                if self.noise is not None:
                    if self.use_noise_only:
                        perturbed_feature = self.noise
                    else:
                        perturbed_feature += self.noise
                permuted_x[:, col_idx] = perturbed_feature

                prediction = self.model(permuted_x)

                scores[n_round] = self._score(prediction)

            return scores

        # empty container to keep results
        results = np.full((inputs.shape[1], self.n_repeats), np.nan)

        for col_index in range(inputs.shape[1]):

            results[col_index, :] = _func(inputs, col_index)

        # permutation importance is how much performance decreases by permutation
        results = self._base_score() - results

        return results

    def _plot_importance(self, imp,
                         features, axes=None, show=False,
                         save=False,
                         name=None):
        if axes is None:
            axes = plt.gca()

        importances_mean = np.mean(imp, axis=1)
        perm_sorted_idx = importances_mean.argsort()
        axes.boxplot(
            imp[perm_sorted_idx].T,  # (num_features, n_repeats) -> (n_repeats, num_features)
            vert=False,
            labels=np.array(features)[perm_sorted_idx],
        )

        axes.set_xlabel(ERROR_LABELS.get(self.scoring, self.scoring))

        axes.set_title(f"Base Score {round(self._base_score(), 3)}")

        if save:
            name = name or ''
            fname = os.path.join(self.path, f"box_plot_{name}_{self.n_repeats}_{self.scoring}")
            plt.savefig(fname, bbox_inches="tight")

        if show:
            plt.show()

        return axes
