import os
from typing import Union, List, Callable

import numpy as np
import matplotlib.pyplot as plt

from ._explain import ExplainerMixin


# todo, optionally show predicted value as dots on plots

def compute_bounds(xmin, xmax, xv):
    """
    from shap
    Handles any setting of xmax and xmin.

    Note that we handle None, float, or "percentile(float)" formats.
    """

    if xmin is not None or xmax is not None:
        if type(xmin) == str and xmin.startswith("percentile"):
            xmin = np.nanpercentile(xv, float(xmin[11:-1]))
        if type(xmax) == str and xmax.startswith("percentile"):
            xmax = np.nanpercentile(xv, float(xmax[11:-1]))

        if xmin is None or xmin == np.nanmin(xv):
            xmin = np.nanmin(xv) - (xmax - np.nanmin(xv))/20
        if xmax is None or xmax == np.nanmax(xv):
            xmax = np.nanmax(xv) + (np.nanmax(xv) - xmin)/20

    return xmin, xmax


class PartialDependencePlot(ExplainerMixin):
    """
    Partial dependence plots as introduced by Friedman_ et al., 2001

    .. _Friedman:
        https://doi.org/10.1214/aos/1013203451
    """
    def __init__(
            self,
            model: Callable,
            data,
            feature_names=None,
            num_points: int = 100,
            path=None,
            **kwargs
    ):
        """Initiates the class

        Parameters
        ----------
            model :
                the trained/calibrated model which must be callable. It must take the
                `data` as input and sprout an array of predicted values. For example
                if you are using Keras/sklearn model, then you must pass model.predict
            data :
                The inputs to the `model`.
            feature_names :
                Names of features. Used for labeling.
            num_points :
                determines the grid for evaluation of `model`
            path:
                path to save the plots. By default the results are saved in current directory
            kwargs:
                any additional keyword arguments for `model`
        """

        self.model = model
        self.num_points = num_points
        self.xmin = "percentile(0)"
        self.xmax = "percentile(100)"
        self.kwargs = kwargs

        super().__init__(data=data, features=feature_names, path=path or os.getcwd())

    def nd_interactions(
            self,
            height: int = 2,
            ice: bool = False,
            show_dist: bool = False,
            show_minima: bool = False,
            show: bool = True,
            save: bool = False
    ) -> plt.Figure:
        """Plots 2d interaction plots of all features as done in skopt

        Arguments:
            height:
                height of each subplot in inches
            ice:
                whether to show the ice lines or not
            show_dist:
                whether to show the distribution of data as histogram or not
            show_minima:
                whether to show the function minima or not
            show:
                whether to show the plot or not
            save:
                whether to save the plot or not
        Returns:
            matplotlib Figure
        """

        n_dims = len(self.features)

        fig, ax = plt.subplots(n_dims, n_dims, figsize=(height * n_dims, height * n_dims))

        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                            hspace=0.1, wspace=0.1)

        for i in range(n_dims):
            for j in range(n_dims):
                # diagonal
                if i == j:

                    if n_dims > 1:
                        ax_ = ax[i, i]
                    else:
                        ax_ = ax

                    self._plot_pdp_1dim(*self._pdp_for_2d(self.data, self.features[i]),
                                        self.data, self.features[i],
                                        show_dist=show_dist, show_minima=show_minima,
                                        ice=ice, show=False, ax=ax_)
                    # resetting the label
                    # ax_.set_xlabel(self.features[i])
                    ax_.set_ylabel(self.features[i])
                    process_axis(ax_, xlabel=self.features[i], top_spine=True, right_spine=True)

                # lower triangle
                elif i > j:
                    self.plot_interaction(
                        features=[self.features[j], self.features[i]],
                        show=False, ax=ax[i, j], colorbar=False
                    )

                elif j > i:
                    if not ax[i, j].lines:  # empty axes
                        ax[i, j].axis("off")

                if j > 0:  # not the left most column
                    ax[i, j].yaxis.set_ticks([])
                    ax[i, j].yaxis.set_visible(False)
                    ax[i, j].yaxis.label.set_visible(False)

                if i < n_dims-1:  # not the bottom most row
                    ax[i, j].xaxis.set_ticks([])
                    ax[i, j].xaxis.set_visible(False)
                    ax[i, j].xaxis.label.set_visible(False)

        if save:
            fname = os.path.join(self.path, f"pdp_interact_nd")
            plt.savefig(fname, bbox_inches="tight", dpi=100*n_dims)

        if show:
            plt.show()

        return fig

    def plot_interaction(
            self,
            features: list,
            lookback: int = None,
            ax: plt.Axes = None,
            plot_type: str = "2d",
            cmap=None,
            colorbar: bool = True,
            show: bool = True,
            save: bool = False,
            **kwargs
    ) -> plt.Axes:
        """Shows interaction between two features

        Arguments:
            features:
                a list or tuple of two feature names to use
            lookback:

            ax:
                matplotlib axes on which to draw. If not given, current axes will
                be used.
            plot_type:
                either "2d" or "surface"
            cmap:
                color map to use
            colorbar:
                whether to show the colorbar or not
            show:
                whether to show the plot or not
            save:
                whether to save the plot or not
            kwargs:
                any keyword argument for axes.plot_surface or axes.contourf

        Returns:
              matplotlib Axes
        """

        if not self.data_is_2d:
            raise NotImplementedError

        assert isinstance(features, list) and len(features) == 2

        x0, x1, pd_vals = self._interaction(features, self.data, lookback)

        kwds = {}
        if plot_type == "surface":
            kwds['projection'] = '3d'

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, **kwds)

        if plot_type == "surface":
            _add_surface(ax, x0, x1, pd_vals, cmap, features[0], features[1], **kwargs)
        else:
            self._plot_interaction(ax, x0, x1, pd_vals,
                                   cmap=cmap,
                                   features=features,
                                   lookback=lookback,
                                   colorbar=colorbar, **kwargs)

        if save:
            fname = os.path.join(self.path, f"pdp_interact{features[0]}_{features[1]}")
            plt.savefig(fname, bbox_inches="tight", dpi=300)

        if show:
            plt.show()

        return ax

    def _plot_interaction(self, ax, x0, x1, pd_vals, cmap, features, lookback,
                          colorbar=True,
                          **kwargs):
        """adds a 2d interaction plot"""
        cntr = ax.contourf(x0, x1, pd_vals, cmap=cmap, **kwargs)

        xv0 = self.xv(self.data, features[0], lookback)
        xv1 = self.xv(self.data, features[1], lookback)

        ax.scatter(xv0, xv1, c='k', s=10, lw=0.)
        process_axis(ax, xlabel=features[0], ylabel=features[1])

        if colorbar:
            cbar = plt.colorbar(cntr, ax=ax)
            cbar.set_label(f"E[f(x) | {features[0]} {features[1]} ]", rotation=90)

        return

    def _interaction(self, features, data, lookback):

        ind0 = self._feature_to_ind(features[0])
        ind1 = self._feature_to_ind(features[1])

        xs0 = self.grid(self.data, features[0], lookback)
        xs1 = self.grid(self.data, features[1], lookback)

        features_tmp = data.copy()
        x0 = np.zeros((self.num_points, self.num_points))
        x1 = np.zeros((self.num_points, self.num_points))
        pd_vals = np.zeros((self.num_points, self.num_points))

        for i in range(self.num_points):
            for j in range(self.num_points):
                features_tmp[:, ind0] = xs0[i]
                features_tmp[:, ind1] = xs1[j]
                x0[i, j] = xs0[i]
                x1[i, j] = xs1[j]
                pd_vals[i, j] = self.model(features_tmp).mean()

        return x0, x1, pd_vals

    def plot_1d(
            self,
            feature,
            show_dist: bool = True,
            show_dist_as: str = "hist",
            ice: bool = True,
            feature_expected_value: bool = False,
            model_expected_value: bool = False,
            show_ci: bool = False,
            show_minima: bool = False,
            show: bool = True,
            save: bool = False
    ):
        """partial dependence plot in one dimension

        Arguments:
            feature:
                the feature name for which to plot the partial dependence
            show_dist:
                whether to show actual distribution of data or not
            show_dist_as:
                one of "hist" or "grid"
            ice:
                whether to show individual component elements on plot or not
            feature_expected_value:
                whether to show the average value of feature on the plot or not
            model_expected_value:
                whether to show average prediction on plot or not
            show_ci:
                whether to show confidence interval of pdp or not
            show_minima:
                whether to indicate the minima or not
            show:
                whether to show the plot or not
            save:
                whether to save the plot or not
        """
        if isinstance(feature, list) or isinstance(feature, tuple):
            raise NotImplementedError
        else:
            if self.single_source:
                if self.data_is_2d:
                    self._plot_pdp_1dim(*self._pdp_for_2d(self.data, feature), self.data, feature,
                                        show_dist=show_dist, show_dist_as=show_dist_as,
                                        ice=ice, feature_expected_value=feature_expected_value,
                                        show_ci=show_ci, show_minima=show_minima,
                                        model_expected_value=model_expected_value, show=show, save=save)
                elif self.data_is_3d:
                    for lb in range(self.data.shape[1]):
                        self._plot_pdp_1dim(*self._pdp_for_2d(self.data, feature, lb),
                                            data=self.data, feature=feature, lookback=lb,
                                            show_ci=show_ci, show_minima=show_minima,
                                            show_dist=show_dist, show_dist_as=show_dist_as,
                                            ice=ice, feature_expected_value=feature_expected_value,
                                            model_expected_value=model_expected_value, show=show, save=save)
            else:
                for data in self.data:
                    if self.data_is_2d:
                        self._pdp_for_2d(data, feature)
                    else:
                        for lb in []:
                            self._pdp_for_2d(data, feature, lb)
        return

    def xv(self, data, feature, lookback):

        ind = self._feature_to_ind(feature)
        if data.ndim == 3:
            xv = data[:, lookback, ind]
        else:
            xv = data[:, ind]

        return xv

    def grid(self, data, feature, lookback=None):
        """generates the grid for evaluation of model"""
        xmin, xmax = compute_bounds(self.xmin, self.xmax, self.xv(data, feature, lookback))
        return np.linspace(xmin, xmax, self.num_points)

    def _pdp_for_2d(self, data, feature, lookback=None):
        ind = self._feature_to_ind(feature)

        xs = self.grid(data, feature, lookback)

        data_temp = data.copy()
        pd_vals = np.full(self.num_points, np.nan)
        ice_vals = np.full((self.num_points, data.shape[0]), np.nan)

        for i in range(self.num_points):

            if data.ndim == 3:
                data_temp[:, lookback, ind] = xs[i]
            else:
                data_temp[:, ind] = xs[i]

            preds = self.model(data_temp, **self.kwargs)
            pd_vals[i] = preds.mean()
            ice_vals[i, :] = preds.reshape(-1,)

        return pd_vals, ice_vals

    def _feature_to_ind(self, feature) -> int:
        ind = feature
        if isinstance(feature, str):
            if self.single_source:
                ind = self.features.index(feature)
            else:
                raise NotImplementedError
        elif not isinstance(feature, int):
            raise ValueError

        return ind

    def _plot_pdp_1dim(self,
                       pd_vals, ice_vals, data, feature, lookback=None,
                       show_dist=True, show_dist_as="hist", ice=True, show_ci=False,
                       show_minima=False,
                       feature_expected_value=False, model_expected_value=False,
                       show=True, save=False, ax=None):

        xmin, xmax = compute_bounds(self.xmin, self.xmax, self.xv(data, feature, lookback))

        if ax is None:
            fig = plt.figure()
            ax = fig.add_axes((0.1, 0.3, 0.8, 0.6))

        xs = self.grid(data, feature, lookback)

        ylabel = "E[f(x) | " + feature + "]"
        if ice:
            ice_linewidth = min(1, 50 / ice_vals.shape[1])  # pylint: disable=unsubscriptable-object
            ax.plot(xs, ice_vals, color='lightblue', linewidth=ice_linewidth, alpha=1)
            ylabel = "f(x) | " + feature

        if show_ci:
            std = np.std(ice_vals, axis=1)
            upper = pd_vals + std
            lower = pd_vals - std
            ax.fill_between(xs, upper, lower, alpha=0.14, color='#66C2D7')

        # the line plot
        ax.plot(xs, pd_vals, color='blue', linewidth=2, alpha=1)

        title = None
        if lookback is not None:
            title = f"lookback: {lookback}"
        process_axis(ax, ylabel=ylabel, ylabel_kws=dict(fontsize=20), right_spine=False, top_spine=False,
                     tick_params=dict(labelsize=11), xlabel=feature, xlabel_kws=dict(fontsize=20), title=title)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax2 = ax.twinx()

        if show_dist:
            xv = self.xv(data, feature, lookback)
            if show_dist_as == "hist":

                ax2.hist(xv, 50, density=False, facecolor='black', alpha=0.1, range=(xmin, xmax))
            else:
                _add_dist_as_grid(fig, xv, other_axes=ax, xlabel=feature, xlabel_kws=dict(fontsize=20))

        process_axis(ax2, right_spine=False, top_spine=False, left_spine=False, bottom_spine=False,
                     ylim=(0, data.shape[0]))
        ax2.xaxis.set_ticks_position('bottom')
        ax2.yaxis.set_ticks_position('left')
        ax2.yaxis.set_ticks([])

        if feature_expected_value:
            self._add_feature_exp_val(ax2, ax, xmin, xmax, data, feature, lookback)

        if model_expected_value:
            self._add_model_exp_val(ax2, ax, data)

        if show_minima:
            minina = self.model(data, **self.kwargs).min()
            ax.axvline(minina, linestyle="--", color="r", lw=1)

        if save:
            lookback = lookback or ''
            fname = os.path.join(self.path, f"pdp_{feature}_{lookback}")
            plt.savefig(fname, bbox_inches="tight", dpi=400)

        if show:
            plt.show()

        return ax

    def _add_model_exp_val(self, ax, original_axis, data):
        """adds model expected value on a duplicate axis of ax"""
        model_expected_val = self.model(data, **self.kwargs).mean()

        ax2 = ax.twinx()

        ymin, ymax = original_axis.get_ylim()
        process_axis(ax2, ylim=(ymin, ymax),
                     yticks=[model_expected_val],
                     yticklabels=["E[f(x)]"],
                     right_spine=False,
                     top_spine=False,
                     tick_params=dict(length=0, labelsize=11)
                     )

        original_axis.axhline(model_expected_val, color="#999999", zorder=-1, linestyle="--", linewidth=1)
        return

    def _add_feature_exp_val(self, ax, original_axis, xmin, xmax, data, feature, lookback=None):

        xv = self.xv(data=data, feature=feature, lookback=lookback)
        mval = xv.mean()

        ax3 = ax.twiny()

        process_axis(ax3,
                     xlim=(xmin, xmax),
                     xticks=[mval], xticklabels=["E[" + feature + "]"],
                     tick_params={'length': 0, 'labelsize': 11}, top_spine=False, right_spine=False)
        original_axis.axvline(mval, color="#999999", zorder=-1, linestyle="--", linewidth=1)
        return


def process_axis(
        ax: plt.Axes,
        title=None, title_kws=None,
        xlabel=None, xlabel_kws=None,
        ylabel=None, ylabel_kws=None,
        xticks=None, xticklabels=None,
        yticks=None, yticklabels=None,
        tick_params=None,
        top_spine=None, right_spine=None, bottom_spine=None, left_spine=None,
        xlim=None, ylim=None
):
    """processes a matplotlib axes"""
    if title:
        title_kws = title_kws or {}
        ax.set_title(title, **title_kws)

    if ylabel:
        ylabel_kws = ylabel_kws or {}
        ax.set_ylabel(ylabel, **ylabel_kws)

    if xlabel:
        xlabel_kws = xlabel_kws or {}
        ax.set_xlabel(xlabel, **xlabel_kws)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if xticks:
        ax.set_xticks(xticks)

    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)

    if yticks:
        ax.set_yticks(yticks)

    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    if tick_params:
        ax.tick_params(**tick_params)

    if top_spine is False:
        ax.spines['top'].set_visible(False)
    elif top_spine is True:
        ax.spines['top'].set_visible(True)

    if right_spine is False:
        ax.spines['right'].set_visible(False)
    elif right_spine is True:
        ax.spines['right'].set_visible(True)

    if bottom_spine is False:
        ax.spines['bottom'].set_visible(False)

    if left_spine is False:
        ax.spines['left'].set_visible(False)
    return


def _add_dist_as_grid(fig: plt.Figure, hist_data, other_axes: plt.Axes,
                      xlabel=None, xlabel_kws=None,  **plot_params):
    """Data point distribution plot for numeric feature"""

    ax = fig.add_axes((0.1, 0.1, 0.8, 0.14), sharex=other_axes)
    process_axis(ax, top_spine=False, xlabel=xlabel, xlabel_kws=xlabel_kws, bottom_spine=False,
                 right_spine=False, left_spine=False)
    ax.yaxis.set_visible(False)  # hide the yaxis
    ax.xaxis.set_visible(False)  # hide the x-axis

    color = plot_params.get('pdp_color', '#1A4E5D')
    ax.plot(hist_data, [1] * len(hist_data), '|', color=color, markersize=20)
    return


def _add_surface(ax, x0, x1, pd_vals, cmap, feature0, feature2, **kwargs):

    ax.plot_surface(x0, x1, pd_vals, cmap=cmap, **kwargs)

    ax.set_xlabel(feature0, fontsize=11)
    ax.set_ylabel(feature2, fontsize=11)
    ax.set_zlabel(f"E[f(x) | {feature0} {feature2} ]", fontsize=11)

    return
