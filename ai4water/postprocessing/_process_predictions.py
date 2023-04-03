

from typing import Union, List, Tuple

from SeqMetrics import RegressionMetrics, ClassificationMetrics
from SeqMetrics.utils import plot_metrics

from ai4water.backend import easy_mpl as ep
from ai4water.backend import np, pd, mpl, plt, os, wandb, sklearn

from ai4water.utils.utils import AttribtueSetter
from ai4water.utils.utils import get_values
from ai4water.utils.utils import dateandtime_now, ts_features, dict_to_file

from ai4water.utils.visualizations import Plot, init_subplots
from ai4water.utils.visualizations import murphy_diagram, fdc_plot, edf_plot


# competitive skill score plot/ bootstrap skill score plot as in MLAir
# rank histogram and reliability diagram for probabilitic forecasting model.
# show availability plot of data
# classification report as in yellow brick
# class prediction error as in yellow brick
# discremination threshold as in yellow brick
# Friedman's H statistic https://blog.macuyiko.com/post/2019/discovering-interaction-effects-in-ensemble-models.html
# silhouette analysis
#  KS Statistic plot from labels and scores/probabilities
# reliability curves
# cumulative gain
# lift curve
#

mdates = mpl.dates


# in order to unify the use of metrics
Metrics = {
    'regression': lambda t, p, multiclass=False, **kwargs: RegressionMetrics(t, p, **kwargs),
    'classification': lambda t, p, multiclass=False, **kwargs: ClassificationMetrics(t, p,
                                                                                     multiclass=multiclass, **kwargs)
}


class ProcessPredictions(Plot):
    """post processing of results after training."""

    available_plots = [
        'regression', 'prediction', 'residual',
        'murphy', 'fdc', 'errors', "edf"
    ]

    def __init__(
            self,
            mode: str,
            forecast_len: int = None,
            output_features: Union[list, str] = None,
            wandb_config: dict = None,
            path: str = None,
            dpi: int = 300,
            show=1,
            save: bool = True,
            plots: Union[list, str] = None,
            quantiles:int=None,
    ):
        """
        Parameters
        ----------
            mode : str
                either "regression" or "classification"
            forecast_len : int, optional (default=None)
                forecast length, only valid when mode is regression
            output_features : str, optional
                names of output features
            plots : int/list, optional (default=None)
                the names of plots to draw. Following plots are avialble.

                    ``residual``
                    ``regression``
                    ``prediction``
                    ``errors``
                    ``fdc``
                    ``murphy``
                    ``edf``

            path : str
                folder in which to save the results/plots
            show : bool
                whether to show the plots or not
            save : bool
                whether to save the plots or not
            wandb_config :
                weights and bias configuration dictionary
            dpi : int
                determines resolution of saved figure

        Examples
        --------
        >>> import numpy as np
        >>> from ai4water.postprocessing import ProcessPredictions
        >>> true = np.random.random(100)
        >>> predicted = np.random.random(100)
        >>> processor = ProcessPredictions("regression", forecast_len=1,
        ...    plots=['prediction', 'regression', 'residual'])
        >>> processor(true, predicted)

        # for postprocessing of classification, we need to set the mode

        >>> true = np.random.randint(0, 2, (100, 1))
        >>> predicted = np.random.randint(0, 2, (100, 1))
        >>> processor = ProcessPredictions("classification")
        >>> processor(true, predicted)
        """
        self.mode = mode
        self.forecast_len = forecast_len
        self.output_features = output_features
        self.wandb_config = wandb_config
        self.quantiles = quantiles
        self.show = show
        self.save = save
        self.dpi = dpi

        if plots is None:
            if mode == "regression":
                plots = ['regression', 'prediction', "residual", "errors", "edf"]
            else:
                plots = []
        elif not isinstance(plots, list):
            plots = [plots]

        assert all([plot in self.available_plots for plot in plots]), f"""
        {plots} not allowed."""
        self.plots = plots

        super().__init__(path, save=save)

    @property
    def quantiles(self):
        return self._quantiles

    @quantiles.setter
    def quantiles(self, x):
        self._quantiles = x

    def _classes(self, array):
        if self.mode == "classification":
            return np.unique(array)
        return []

    def n_classes(self, array):
        if self.mode == "classification":
            return len(self._classes(array))
        return None

    def save_or_show(self, show=None, **kwargs):
        if show is None:
            show = self.show
        return super().save_or_show(save=self.save, show=show, **kwargs)

    def __call__(
            self,
            true_outputs,
            predicted,
            metrics="minimal",
            prefix="test",
            index=None,
            inputs=None,
            model=None,
    ):

        if self.quantiles:
            return self.process_quantiles(true_outputs, predicted)

        # it true_outputs and predicted are dictionary of len(1) then just get the values
        true_outputs = get_values(true_outputs)
        predicted = get_values(predicted)

        true_outputs = np.array(true_outputs)
        predicted = np.array(predicted)

        AttribtueSetter(self, true_outputs)

        key = {"regression": "rgr", "classification": "cls"}

        getattr(self, f"process_{key[self.mode]}_results")(
            true_outputs,
            predicted,
            inputs=inputs,
            metrics=metrics,
            prefix=prefix,
            index=index,
        )

        return

    def process_quantiles(self, true, predicted):
        #assert self.num_outs == 1

        if true.ndim == 2:  # todo, this should be avoided
            true = np.expand_dims(true, axis=-1)

        self.quantiles = self.quantiles

        self.plot_quantiles1(true, predicted)
        self.plot_quantiles2(true, predicted)
        self.plot_all_qs(true, predicted)

        return

    def horizon_plots(self, errors: dict, fname=''):
        plt.close('')
        _, axis = plt.subplots(len(errors), sharex='all')

        legends = {'r2': "$R^2$", 'rmse': "RMSE", 'nse': "NSE"}
        idx = 0
        for metric_name, val in errors.items():
            ax = axis[idx]
            ax.plot(val, '--o', label=legends.get(metric_name, metric_name))
            ax.legend(fontsize=14)
            if idx >= len(errors) - 1:
                ax.set_xlabel("Horizons", fontsize=14)
            ax.set_ylabel(legends.get(metric_name, metric_name), fontsize=14)
            idx += 1
        self.save_or_show(fname=fname)
        return

    def plot_results(
            self,
            true,
            predicted: pd.DataFrame,
            prefix,
            where,
            inputs=None,
    ):
        """
        # kwargs can be any/all of followings
            # fillstyle:
            # marker:
            # linestyle:
            # markersize:
            # color:
        """

        for plot in self.plots:
            if plot == "murphy":
                self.murphy_plot(true, predicted, prefix=prefix, where=where, inputs=inputs)
            else:
                getattr(self, f"{plot}_plot")(true, predicted, prefix=prefix, where=where)
        return

    def average_target_across_feature(self, true, predicted, feature):
        raise NotImplementedError

    def prediction_distribution_across_feature(self, true, predicted, feature):
        raise NotImplementedError

    def edf_plot(
            self,
            true, predicted,
            for_prediction:bool = False,
            prefix='', where='',
            **kwargs):
        """cumulative distribution function of absolute error between true and
        predicted.

        Parameters
        -----------
        true :
            array like
        predicted :
            array like
        for_prediction : bool
            whether to plot edf of prediction as well or not
        prefix :
        where :
        """

        true = _to_1darray(true)
        predicted = _to_1darray(predicted)

        error = np.abs(true - predicted)

        _plot_kws = dict(
            xlabel="Absolute Error",
            color="#005066", show=False
        )

        if for_prediction:
            _plot_kws['label'] = "Error"

        ax = edf_plot(
            error, **_plot_kws
        )

        if for_prediction:
            edf_plot(error, xlabel="Absolute Error", show=False,
                     color = "#B3331D", label="Prediction", ax=ax)

        return self.save_or_show(fname=f"{prefix}_error_dist", where=where)

    def murphy_plot(
            self, true,
            predicted,
            inputs,
            prefix='',
            where='',
            **kwargs):

        murphy_diagram(true,
                       predicted,
                       reference_model="LinearRegression",
                       plot_type="diff",
                       inputs=inputs,
                       show=False,
                       **kwargs)

        return self.save_or_show(fname=f"{prefix}_murphy", where=where)

    def fdc_plot(self, true, predicted, prefix='', where='', **kwargs):

        fdc_plot(predicted, true, show=False, **kwargs)

        return self.save_or_show(fname=f"{prefix}_fdc",
                                 where=where)

    def residual_plot(
            self,
            true,
            predicted,
            axes:Union[List[plt.Axes], Tuple[plt.Axes]] = None,
            figsize=None,
            hist_kws:dict = None,
            plot_kws:dict = None,
            prefix='',
            where='',
            xlabel=None,
            **kwargs
    ):
        """
        Makes residual plot

        Parameters
        ----------
        true :
            array like
        predicted :
            array like
        axes : tuple
        figsize : tuple
        plot_kws : dict
        prefix :
        where :
        xlabel :
        hist_kws :

        """
        true = _to_1darray(true)
        predicted = _to_1darray(predicted)

        y = true - predicted

        if axes is None:
            fig, axes = plt.subplots(1, 2,
                                     figsize=figsize,
                                     sharey="all",
                                     gridspec_kw={'width_ratios': [3, 1]})
        else:
            assert len(axes)==2, axes

        _hist_kws = dict(bins=20,
                         linewidth=0.5,
                         edgecolor="k",
                         grid=False,
                         color="#009E73",
                         orientation='horizontal')

        if hist_kws:
            _hist_kws.update(hist_kws)

        ep.hist(y, show=False, ax=axes[1],
             **_hist_kws)

        _plot_kws = dict(
            color="#009E73",
            markerfacecolor="#009E73",
            markeredgecolor="black",
            markeredgewidth=0.5,
            alpha=0.7,
            ax_kws=dict(
                xlabel= xlabel or f"Predicted {prefix}",
                ylabel="Residual"),
        )
        if plot_kws:
            _plot_kws.update(plot_kws)

        ep.plot(predicted, y, 'o', show=False,
             ax=axes[0],
             **_plot_kws
             )

        axes[0].axhline(0.0, color="black")
        return self.save_or_show(fname=f"{prefix}_residual",
                                 where=where)

    def errors_plot(
            self, true,
            predicted,
            prefix='',
            where='', **kwargs):

        errors = Metrics[self.mode](true, predicted, multiclass=self.is_multiclass_)

        return plot_metrics(
            errors.calculate_all(),
            show=self.show,
            save_path=os.path.join(self.path, where),
            save=self.save,
            text_kws = {"fontsize": 16},
            max_metrics_per_fig=20,

        )

    def regression_plot(
            self,
            true,
            predicted,
            prefix = '',
            where='',
            annotate_with="r2",
            **kwargs,
    ):
        annotation_val = getattr(RegressionMetrics(true, predicted), annotate_with)()

        metric_names = {'r2': "$R^2$"}

        annotation_key = metric_names.get(annotate_with, annotate_with)

        RIDGE_LINE_KWS = {'color': 'firebrick', 'lw': 1.0}

        if isinstance(predicted, (pd.DataFrame, pd.Series)):
            predicted = predicted.values

        marginals = True
        if np.isnan(np.array(true)).any() or np.isnan(predicted).any():
            marginals = False

        # if all the values in predicted are same, calculation of kde gives error
        if (predicted == predicted[0]).all():
            marginals = False

        try:
            axes = ep.regplot(true,
                       predicted,
                       marker_color='crimson',
                       line_color='k',
                       scatter_kws={'marker': "o", 'edgecolors': 'black', 'linewidth':0.5},
                       show=False,
                       marginals=marginals,
                       marginal_ax_pad=0.25,
                       marginal_ax_size=0.7,
                       ridge_line_kws=RIDGE_LINE_KWS,
                       hist=False,
                       )
        except np.linalg.LinAlgError:
            axes = ep.regplot(true,
                       predicted,
                       marker_color='crimson',
                       line_color='k',
                       scatter_kws={'marker': "o", 'edgecolors': 'black', 'linewidth': 0.5},
                       show=False,
                       marginals=False
                       )

        axes.annotate(f'{annotation_key}: {round(annotation_val, 3)}',
                     xy=(0.3, 0.95),
                     xycoords='axes fraction',
                     horizontalalignment='right', verticalalignment='top',
                     fontsize=16)

        return self.save_or_show(fname=f"{prefix}_regression",
                                 where=where)

    def prediction_plot(
            self,
            true,
            predicted,
            prefix='',
            where=''
    ):

        _, axis = init_subplots(width=12, height=8)

        # it is quite possible that when data is datetime indexed, then it is not
        # equalidistant and large amount of graph
        # will have not data in that case lines plot will create a lot of useless
        # interpolating lines where no data is present.
        datetime_axis = False
        if isinstance(true.index, pd.DatetimeIndex) and pd.infer_freq(true.index) is not None:
            style = '.'
            true = true
            predicted = predicted
            datetime_axis = True
        else:
            if np.isnan(true.values).sum() > 0:
                # For Nan values we should be using this style otherwise nothing is plotted.
                style = '.'
            else:
                style = '-'
                true = true.values
                predicted = predicted.values

        ms = 4 if style == '.' else 2

        # because the data is very large, so better to use small marker size
        if len(true) > 1000:
            ms = 2

        axis.plot(predicted, style, color='r', label='Prediction')

        axis.plot(true, style, color='b', marker='o', fillstyle='none',
                  markersize=ms, label='True')

        axis.legend(loc="best", fontsize=22, markerscale=4)

        if datetime_axis:
            loc = mdates.AutoDateLocator(minticks=4, maxticks=6)
            axis.xaxis.set_major_locator(loc)
            fmt = mdates.AutoDateFormatter(loc)
            axis.xaxis.set_major_formatter(fmt)

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel("Time", fontsize=18)

        return self.save_or_show(fname=f"{prefix}_prediction", where=where)

    def plot_all_qs(self, true_outputs, predicted, save=False):
        plt.close('all')
        plt.style.use('ggplot')

        st, en = 0, true_outputs.shape[0]

        plt.plot(np.arange(st, en), true_outputs[st:en, 0], label="True", color='navy')

        for idx, q in enumerate(self.quantiles):
            q_name = "{:.1f}".format(q * 100)
            plt.plot(np.arange(st, en), predicted[st:en, idx], label="q {} %".format(q_name))

        plt.legend(loc="best")
        self.save_or_show(save, fname="all_quantiles", where='results')

        return

    def plot_quantiles1(self, true_outputs, predicted, st=0, en=None, save=True):
        plt.close('all')
        plt.style.use('ggplot')
        assert true_outputs.shape[-2:] == (1, 1)
        if en is None:
            en = true_outputs.shape[0]
        for q in range(len(self.quantiles) - 1):
            st_q = "{:.1f}".format(self.quantiles[q] * 100)
            en_q = "{:.1f}".format(self.quantiles[-q] * 100)

            plt.plot(np.arange(st, en), true_outputs[st:en, 0], label="True",
                     color='navy')
            plt.fill_between(np.arange(st, en), predicted[st:en, q].reshape(-1, ),
                             predicted[st:en, -q].reshape(-1, ), alpha=0.2,
                             color='g', edgecolor=None, label=st_q + '_' + en_q)
            plt.legend(loc="best")
            self.save_or_show(save, fname='q' + st_q + '_' + en_q, where='results')
        return

    def plot_quantiles2(
            self, true_outputs,
            predicted,
            st=0,
            en=None,
            save=True
    ):
        plt.close('all')
        plt.style.use('ggplot')

        if en is None:
            en = true_outputs.shape[0]
        for q in range(len(self.quantiles) - 1):
            st_q = "{:.1f}".format(self.quantiles[q] * 100)
            en_q = "{:.1f}".format(self.quantiles[q + 1] * 100)

            plt.plot(np.arange(st, en), true_outputs[st:en, 0], label="True",
                     color='navy')
            plt.fill_between(np.arange(st, en),
                             predicted[st:en, q].reshape(-1, ),
                             predicted[st:en, q + 1].reshape(-1, ),
                             alpha=0.2,
                             color='g', edgecolor=None, label=st_q + '_' + en_q)
            plt.legend(loc="best")
            self.save_or_show(save, fname='q' + st_q + '_' + en_q + ".png",
                              where='results')
        return

    def plot_quantile(self, true_outputs, predicted, min_q: int, max_q, st=0,
                      en=None, save=False):
        plt.close('all')
        plt.style.use('ggplot')

        if en is None:
            en = true_outputs.shape[0]
        q_name = "{:.1f}_{:.1f}_{}_{}".format(self.quantiles[min_q] * 100,
                                              self.quantiles[max_q] * 100, str(st),
                                              str(en))

        plt.plot(np.arange(st, en), true_outputs[st:en, 0], label="True", color='navy')
        plt.fill_between(np.arange(st, en),
                         predicted[st:en, min_q].reshape(-1, ),
                         predicted[st:en, max_q].reshape(-1, ),
                         alpha=0.2,
                         color='g', edgecolor=None, label=q_name + ' %')
        plt.legend(loc="best")
        self.save_or_show(save, fname="q_" + q_name + ".png", where='results')
        return

    def roc_curve(self, estimator, x, y, prefix=None):

        if hasattr(estimator, '_model'):
            if estimator._model.__class__.__name__ in ["XGBClassifier", "XGBRFClassifier"] and isinstance(x,
                                                                                                          np.ndarray):
                x = pd.DataFrame(x, columns=estimator.input_features)

        plot_roc_curve(estimator, x, y.reshape(-1, ))
        self.save_or_show(fname=f"{prefix}_roc")
        return

    def confusion_matrix(self, true, predicted, prefix=None, cmap="Blues", **kwargs):
        """plots confusion matrix.

        cmap :
        **kwargs :
            any keyword arguments for imshow
        """
        cm = ClassificationMetrics(
            true,
            predicted,
            multiclass=self.is_multiclass_).confusion_matrix()

        kws = {
            'annotate': True,
            'colorbar': True,
            'cmap': cmap,
            'xticklabels': self.classes_,
            'yticklabels': self.classes_,
            'ax_kws': {'xlabel': "Predicted Label",
            'ylabel': "True Label"},
            'show': False,
            'annotate_kws': {'fontsize': 14, "fmt": '%.f', 'ha':"left"}
        }

        kws.update(kwargs)

        ep.imshow(cm, **kws)

        self.save_or_show(fname=f"{prefix}_confusion_matrix", where=prefix)
        return

    def precision_recall_curve(self, estimator, x, y, prefix=None):

        if hasattr(estimator, '_model'):
            if estimator._model.__class__.__name__ in ["XGBClassifier", "XGBRFClassifier"] and isinstance(x,
                                                                                                          np.ndarray):
                x = pd.DataFrame(x, columns=estimator.input_features)
        plot_precision_recall_curve(estimator, x, y.reshape(-1, ))
        self.save_or_show(fname=f"{prefix}_plot_precision_recall_curve")
        return

    def process_rgr_results(
            self,
            true: np.ndarray,
            predicted: np.ndarray,
            metrics="minimal",
            prefix=None,
            index=None,
            remove_nans=True,
            inputs=None,
    ):
        """
        predicted, true are arrays of shape (examples, outs, forecast_len).
        """
        # if user_defined_data:
        if self.output_features is None:
            # when data is user_defined, we don't know what out_cols, and forecast_len are
            if predicted.size == len(predicted):
                out_cols = ['output']
                forecast_len = 1
            else:
                out_cols = [f'output_{i}' for i in range(predicted.shape[-1])]

            forecast_len = 1
            true, predicted = self.maybe_not_3d_data(true, predicted)
        else:
            # for cases if they are 2D/1D, add the third dimension.
            true, predicted = self.maybe_not_3d_data(true, predicted)

            forecast_len = self.forecast_len
            if isinstance(forecast_len, dict):
                forecast_len = np.unique(list(forecast_len.values())).item()

            out_cols = self.output_features
            if isinstance(out_cols, dict):
                _out_cols = []
                for cols in out_cols.values():
                    _out_cols = _out_cols + cols
                out_cols = _out_cols
                if len(out_cols) > 1 and not isinstance(predicted, np.ndarray):
                    raise NotImplementedError("""
                    can not process results with more than 1 output arrays""")

        for idx, out in enumerate(out_cols):

            horizon_errors = {metric_name: [] for metric_name in ['nse', 'rmse']}
            for h in range(forecast_len):

                errs = dict()

                fpath = os.path.join(self.path, out)
                if not os.path.exists(fpath):
                    os.makedirs(fpath)

                t = pd.DataFrame(true[:, idx, h], index=index, columns=['true_' + out])
                p = pd.DataFrame(predicted[:, idx, h], index=index, columns=['pred_' + out])

                if wandb is not None and self.wandb_config is not None:
                    _wandb_scatter(t.values, p.values, out)

                df = pd.concat([t, p], axis=1)
                df = df.sort_index()
                fname = f"{prefix}_{out}_{h}"
                df.to_csv(os.path.join(fpath, fname + ".csv"), index_label='index')

                self.plot_results(t, p, prefix=fname, where=out, inputs=inputs)

                if remove_nans:
                    nan_idx = np.isnan(t)
                    t = t.values[~nan_idx]
                    p = p.values[~nan_idx]

                errors = RegressionMetrics(t, p)
                errs[out + '_errors_' + str(h)] = getattr(errors, f'calculate_{metrics}')()
                errs[out + 'true_stats_' + str(h)] = ts_features(t)
                errs[out + 'predicted_stats_' + str(h)] = ts_features(p)

                dict_to_file(fpath, errors=errs, name=prefix)

                for p in horizon_errors.keys():
                    horizon_errors[p].append(getattr(errors, p)())

            if forecast_len > 1:
                self.horizon_plots(horizon_errors, f'{prefix}_{out}_horizons.png')
        return

    def process_cls_results(
            self,
            true: np.ndarray,
            predicted: np.ndarray,
            metrics="minimal",
            prefix=None,
            index=None,
            inputs=None,
            model=None,
    ):
        """post-processes classification results."""

        plt.close('all')
        if self.is_multilabel_:
            return self.process_multilabel(true, predicted, metrics, prefix, index)

        if self.is_multiclass_:
            return self.process_multiclass(true, predicted, metrics, prefix, index)
        else:
            return self.process_binary(true, predicted, metrics, prefix, index, model=None)

    def process_multilabel(self, true, predicted, metrics, prefix, index):

        for label in range(true.shape[1]):
            if self.n_classes(true[:, label]) == 2:
                self.process_binary(true[:, label], predicted[:, label], metrics, f"{prefix}_{label}", index)
            else:
                self.process_multiclass(true[:, label], predicted[:, label], metrics, f"{prefix}_{label}", index)
        return

    def process_multiclass(self, true, predicted, metrics, prefix, index):

        if len(predicted) == predicted.size:
            predicted = predicted.reshape(-1, 1)
        else:
            predicted = np.argmax(predicted, axis=1).reshape(-1, 1)
        if len(true) == true.size:
            true = true.reshape(-1, 1)
        else:
            true = np.argmax(true, axis=1).reshape(-1, 1)

        if self.output_features is None:
            self.output_features = [f'feature_{i}' for i in range(self.n_classes(true))]

        fpath = os.path.join(self.path, prefix)
        if not os.path.exists(fpath):
            os.makedirs(fpath)

        self.confusion_matrix(true, predicted, prefix=prefix)

        fname = os.path.join(fpath, f"{prefix}_prediction.csv")
        pd.DataFrame(np.concatenate([true, predicted], axis=1),
                     columns=['true', 'predicted'], index=index).to_csv(fname)
        class_metrics = ClassificationMetrics(true, predicted, multiclass=True)

        dict_to_file(fpath,
                     errors=class_metrics.calculate_all(),
                     name=f"{prefix}_{dateandtime_now()}.json")
        return

    def process_binary(self, true, predicted, metrics, prefix, index, model=None):

        assert self.n_classes(true) == 2

        if model is not None:
            try:  # todo, also plot for DL
                self.precision_recall_curve(model, x=true, y=predicted, prefix=prefix)
                self.roc_curve(model, x=true, y=predicted, prefix=prefix)
            except NotImplementedError:
                pass

        if predicted.ndim == 1:
            predicted = predicted.reshape(-1, 1)
        elif predicted.size != len(predicted):
            predicted = np.argmax(predicted, axis=1).reshape(-1, 1)

        if true.ndim == 1:
            true = true.reshape(-1, 1)
        elif true.size != len(true):
            true = np.argmax(true, axis=1).reshape(-1, 1)

        self.confusion_matrix(true, predicted, prefix=prefix)

        fpath = os.path.join(self.path, prefix)
        if not os.path.exists(fpath):
            os.makedirs(fpath)

        metrics_instance = ClassificationMetrics(true, predicted, multiclass=False)
        metrics = getattr(metrics_instance, f"calculate_{metrics}")()
        dict_to_file(fpath,
                     errors=metrics,
                     name=f"{prefix}.json"
                     )

        fname = os.path.join(fpath, f"{prefix}_.csv")
        array = np.concatenate([true.reshape(-1, 1), predicted.reshape(-1, 1)], axis=1)
        pd.DataFrame(array, columns=['true', 'predicted'], index=index).to_csv(fname)

        return

    def maybe_not_3d_data(self, true, predicted):

        forecast_len = self.forecast_len

        if true.ndim < 3:
            if isinstance(forecast_len, dict):
                forecast_len = set(list(forecast_len.values()))
                assert len(forecast_len) == 1
                forecast_len = forecast_len.pop()

            assert forecast_len == 1, f'{forecast_len}'
            axis = 2 if true.ndim == 2 else (1, 2)
            true = np.expand_dims(true, axis=axis)

        if predicted.ndim < 3:
            assert forecast_len == 1
            axis = 2 if predicted.ndim == 2 else (1, 2)
            predicted = np.expand_dims(predicted, axis=axis)

        return true, predicted


def plot_roc_curve(*args, **kwargs):
    try:
        func = sklearn.metrics.RocCurveDisplay.from_estimator
    except AttributeError:
        func = sklearn.metrics.plot_roc_curve

    return func(*args, **kwargs)


def plot_precision_recall_curve(*args, **kwargs):
    try:
        func = sklearn.metrics.PrecisionRecallDisplay.from_estimator
    except AttributeError:
        func = sklearn.metrics.plot_precision_recall_curve
    return func(*args, **kwargs)


def _wandb_scatter(true: np.ndarray, predicted: np.ndarray, name: str) -> None:
    """Adds a scatter plot on wandb."""
    data = [[x, y] for (x, y) in zip(true.reshape(-1, ), predicted.reshape(-1, ))]
    table = wandb.Table(data=data, columns=["true", "predicted"])
    wandb.log({
        "scatter_plot": wandb.plot.scatter(table, "true", "predicted",
                                           title=name)
    })
    return

def _to_1darray(array):
    if isinstance(array, (pd.DataFrame, pd.Series)):
        array = array.values

    assert len(array) == array.size
    return array.reshape(-1,)