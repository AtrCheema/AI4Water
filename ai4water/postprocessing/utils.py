
from typing import Union

from SeqMetrics import RegressionMetrics, ClassificationMetrics
from SeqMetrics.utils import plot_metrics

from ai4water.backend import easy_mpl as ep
from ai4water.backend import np, pd, mpl, plt, os, sklearn
from ai4water.utils.visualizations import Plot, init_subplots
from ai4water.utils.utils import dateandtime_now, ts_features, dict_to_file
from ai4water.utils.visualizations import murphy_diagram, fdc_plot, plot_edf

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

mdates = mpl.dates
plot_roc_curve = sklearn.metrics.plot_roc_curve
plot_precision_recall_curve = sklearn.metrics.plot_precision_recall_curve

# in order to unify the use of metrics
Metrics = {
    'regression': lambda t, p, multiclass=False, **kwargs: RegressionMetrics(t, p, **kwargs),
    'classification': lambda t, p, multiclass=False, **kwargs: ClassificationMetrics(t, p,
        multiclass=multiclass, **kwargs)
}

# TODO add Murphy's plot as shown in MLAir
# prediction_distribution aka actual_plot of PDPbox
# average_target_value aka target_plot of PDPbox
# competitive skill score plot/ bootstrap skill score plot as in MLAir
# rank histogram and reliability diagram for probabilitic forecasting model.
# show availability plot of data
# residual plot as in yellow brick
# classification report as in yellow brick
# class prediction error as in yellow brick
# discremination threshold as in yellow brick


class ProcessPredictions(Plot):
    """post processing of results after training"""

    available_plots = [
        'regression', 'prediction', 'residual',
        'murphy', 'fdc', 'errors', "edf"
    ]

    def __init__(
            self,
            mode: str,
            forecast_len: int = None,
            output_features: Union[list, str]=None,
            is_multiclass:bool = None,
            is_binary:bool = None,
            is_multilabel:bool = None,
            wandb_config: dict = None,
            path:str = None,
            dpi: int = 300,
            show = 1,
            save: bool = True,
            plots: Union[list, str] = None,
    ):
        """
        Parameters
        ----------
            mode : str
                either "regression" or "classification"
            forecast_len : int
                forecast length, only valid when mode is regression
            output_features : str, optional
                names of output features
            is_binary : bool, optional (default=None)
                whether the results correspond to binary classification problem.
            is_multiclass : bool
                whether the results correspond to multiclass classification problem.
                Only valid if mode is classification
            is_multilabel : bool, optional (default=None)
                whether the results correspond to multilabel classification problem.
                Only valid if mode is classification
            plots : int, list
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
        >>> true = np.random.random(100)
        >>> predicted = np.random.random(100)
        >>> processor = ProcessPredictions("regression", plots=['prediction', 'regression', 'residual', 'murphy'])
        >>> processor(true, predicted)

        # for classification

        >>> true = np.random.randint(0, 2, (100, 1))
        >>> predicted = np.random.randint(0, 2, (100, 1))
        >>> processor = ProcessPredictions("classification", is_binary=True)
        >>> processor(true, predicted)
        """
        self.mode = mode
        self.forecast_len = forecast_len
        self.output_features = output_features
        self.is_multiclass = is_multiclass
        self.is_binary = is_binary
        self.is_multilabel = is_multilabel
        self.wandb_config = wandb_config
        self.quantiles = None
        self.show = show
        self.save = save
        self.dpi = dpi

        if plots is None:
            plots = ['regression', 'prediction']
        elif not isinstance(plots, list):
            plots = [plots]

        assert all([plot in self.available_plots for plot in plots])
        self.plots = plots

        super().__init__(path, save=save)

    @property
    def quantiles(self):
        return self._quantiles

    @quantiles.setter
    def quantiles(self, x):
        self._quantiles = x

    def classes(self, array):
        if self.mode == "classification":
            return np.unique(array)
        return []

    def n_classes(self, array):
        if self.mode == "classification":
            return len(self.classes(array))
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
            inputs=None
    ):

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

    def horizon_plots(self, errors: dict, fname='', save=True):
        plt.close('')
        _, axis = plt.subplots(len(errors), sharex='all')

        legends = {'r2': "$R^2$", 'rmse': "RMSE", 'nse': "NSE"}
        idx = 0
        for metric_name, val in errors.items():
            ax = axis[idx]
            ax.plot(val, '--o', label=legends.get(metric_name, metric_name))
            ax.legend(fontsize=14)
            if idx >= len(errors)-1:
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
            if plot=="murphy":
                self.murphy_plot(true, predicted, prefix, where, inputs)
            else:
                getattr(self, f"{plot}_plot")(true, predicted, prefix, where)
        return

    def average_target_across_feature(self, true, predicted, feature):
        raise NotImplementedError

    def prediction_distribution_across_feature(self, true, predicted, feature):
        raise NotImplementedError

    def edf_plot(self, true, predicted, prefix, where, **kwargs):
        """cummulative distribution function of absolute error between true and predicted."""

        if isinstance(true, (pd.DataFrame, pd.Series)):
            true = true.values
        if isinstance(predicted, (pd.DataFrame, pd.Series)):
            predicted = predicted.values

        error = np.abs(true - predicted)
        
        plot_edf(error, xlabel="Absolute Error")
        return self.save_or_show(fname=f"{prefix}_error_dist", where=where)

    def murphy_plot(self, true, predicted, prefix, where, inputs, **kwargs):

        murphy_diagram(true,
                       predicted,
                       reference_model="LinearRegression",
                       plot_type="diff",
                       inputs=inputs,
                       show = False,
                       **kwargs)

        return self.save_or_show(fname=f"{prefix}_murphy", where=where)

    def fdc_plot(self, true, predicted, prefix, where, **kwargs):

        fdc_plot(predicted, true, show=False, **kwargs)

        return self.save_or_show(fname=f"{prefix}_fdc",
                          where=where)

    def residual_plot(self, true, predicted, prefix, where, **kwargs):

        fig, axis = plt.subplots(2)

        x = predicted.values
        y = true.values-predicted.values

        ep.hist(y, show=False, ax=axis[0])

        ep.plot(x, y, 'o', show=False, ax=axis[1],
                color="darksalmon", xlabel="Predicted", ylabel="Residual")
        # draw horizontal line on y=0
        axis[1].axhline(0.0)
        plt.suptitle("Residual")

        return self.save_or_show(fname=f"{prefix}_residual",
                          where=where)

    def errors_plot(self, true, predicted, prefix, where, **kwargs):

        errors = Metrics[self.mode](true, predicted, multiclass=self.is_multiclass)

        return plot_metrics(
            errors.calculate_all(),
            show=False,
            save_path=os.path.join(self.path, where),
            save=self.save,
            statistics=False)

    def regression_plot(
            self,
            true,
            predicted,
            target_name,
            where,
            annotate_with="r2"
    ):
        annotation_val = getattr(RegressionMetrics(true, predicted), annotate_with)()

        metric_names = {'r2': "$R^2$"}

        annotation_key = metric_names.get(annotate_with, annotate_with)

        ep.regplot(true,
                predicted,
                title="Regression Plot",
                annotation_key=annotation_key,
                annotation_val=annotation_val,
                show=False
                )

        return self.save_or_show(fname=f"{target_name}_regression",
                          where=where)

    def prediction_plot(self, true, predicted, prefix, where):
        mpl.rcParams.update(mpl.rcParamsDefault)

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
                style = '.'  # For Nan values we should be using this style otherwise nothing is plotted.
            else:
                style = '-'
                true = true.values
                predicted = predicted.values

        ms = 4 if style == '.' else 2

        if len(true) > 1000:  # because the data is very large, so better to use small marker size
            ms = 2

        axis.plot(predicted, style, color='r',  label='Prediction')

        axis.plot(true, style, color='b', marker='o', fillstyle='none',  markersize=ms, label='True')

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

    def plot_loss(self, history: dict, name="loss_curve"):
        """Considering history is a dictionary of different arrays, possible
        training and validation loss arrays, this method plots those arrays."""

        plt.clf()
        plt.close('all')
        fig = plt.figure()
        plt.style.use('ggplot')
        i = 1

        legends = {
            'mean_absolute_error': 'Mean Absolute Error',
            'mape': 'Mean Absolute Percentage Error',
            'mean_squared_logarithmic_error': 'Mean Squared Logrithmic Error',
            'pbias': "Percent Bias",
            "nse": "Nash-Sutcliff Efficiency",
            "kge": "Kling-Gupta Efficiency",
            "tf_r2": "$R^{2}$",
            "r2": "$R^{2}$"
        }

        sub_plots = {1: {'axis': (1, 1), 'width': 9, 'height': 6},
                     2: {'axis': (1, 2), 'width': 9, 'height': 6},
                     3: {'axis': (1, 3), 'width': 9, 'height': 6},
                     4: {'axis': (2, 2), 'width': 9, 'height': 6},
                     5: {'axis': (5, 1), 'width': 8, 'height': 12},
                     6: {'axis': (3, 2), 'width': 8, 'height': 12},
                     7: {'axis': (3, 2), 'width': 20, 'height': 20},
                     8: {'axis': (4, 2), 'width': 20, 'height': 20},
                     9: {'axis': (5, 2), 'width': 20, 'height': 20},
                     10: {'axis': (5, 2), 'width': 20, 'height': 20},
                     12: {'axis': (4, 3), 'width': 20, 'height': 20},
                     }

        epochs = range(1, len(history['loss']) + 1)
        axis_cache = {}

        for key, val in history.items():

            m_name = key.split('_')[1:] if 'val' in key and '_' in key else key

            if isinstance(m_name, list):
                m_name = '_'.join(m_name)
            if m_name in list(axis_cache.keys()):
                axis = axis_cache[m_name]
                axis.plot(epochs, val, color=[0.96707953, 0.46268314, 0.45772886],
                          label='Validation ')
                axis.legend()
            else:
                axis = fig.add_subplot(*sub_plots[len(history)]['axis'], i)
                axis.plot(epochs, val, color=[0.13778617, 0.06228198, 0.33547859],
                          label='Training ')
                axis.legend()
                axis.set_xlabel("Epochs")
                axis.set_ylabel(legends.get(key, key))
                axis_cache[key] = axis
                i += 1
            axis.set(frame_on=True)

        fig.set_figheight(sub_plots[len(history)]['height'])
        fig.set_figwidth(sub_plots[len(history)]['width'])
        self.save_or_show(fname=name, show=self.show)
        mpl.rcParams.update(mpl.rcParamsDefault)
        return

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
            plt.fill_between(np.arange(st, en), predicted[st:en, q].reshape(-1,),
                             predicted[st:en, -q].reshape(-1,), alpha=0.2,
                             color='g', edgecolor=None, label=st_q + '_' + en_q)
            plt.legend(loc="best")
            self.save_or_show(save, fname='q' + st_q + '_' + en_q, where='results')
        return

    def plot_quantiles2(self, true_outputs, predicted, st=0, en=None,
                        save=True):
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
                             predicted[st:en, q].reshape(-1,),
                             predicted[st:en, q + 1].reshape(-1,),
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
                         predicted[st:en, min_q].reshape(-1,),
                         predicted[st:en, max_q].reshape(-1,),
                         alpha=0.2,
                         color='g', edgecolor=None, label=q_name + ' %')
        plt.legend(loc="best")
        self.save_or_show(save, fname="q_" + q_name + ".png", where='results')
        return

    def roc_curve(self, estimator, x, y):

        if hasattr(estimator, '_model'):
            if estimator._model.__class__.__name__ in ["XGBClassifier", "XGBRFClassifier"] and isinstance(x, np.ndarray):
                x = pd.DataFrame(x, columns=estimator.input_features)

        plot_roc_curve(estimator, x, y.reshape(-1, ))
        self.save_or_show(fname="roc")
        return

    def confusion_matrx(self, true, predicted, **kwargs):

        cm = ClassificationMetrics(true, predicted, multiclass=self.is_multiclass).confusion_matrix()

        ep.imshow(cm, annotate=True, colorbar=True, show=False, **kwargs)

        self.save_or_show(fname="confusion_matrix")
        return

    def precision_recall_curve(self, estimator, x, y):

        if hasattr(estimator, '_model'):
            if estimator._model.__class__.__name__ in ["XGBClassifier", "XGBRFClassifier"] and isinstance(x, np.ndarray):
                x = pd.DataFrame(x, columns=estimator.input_features)
        plot_precision_recall_curve(estimator, x, y.reshape(-1, ))
        self.save_or_show(fname="plot_precision_recall_curve")
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
        #if user_defined_data:
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
                if len(out_cols)>1 and not isinstance(predicted, np.ndarray):
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
                df.to_csv(os.path.join(fpath, fname+".csv"), index_label='index')

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
    ):
        """post-processes classification results."""

        if self.is_multiclass is None and self.is_binary is None and self.is_multilabel is None:
            if self.n_classes(true)==2:
                self.is_binary = True
            else:
                self.is_multiclass = True

        if self.is_multilabel:
            return self.process_multilabel(true, predicted, metrics, prefix, index)

        if self.is_multiclass:
            return self.process_multiclass(true, predicted, metrics, prefix, index)
        else:
            return self.process_binary(true, predicted, metrics, prefix, index)

    def process_multilabel(self, true, predicted, metrics, prefix, index):

        for label in range(true.shape[1]):
            if self.n_classes(true[:, label])==2:
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

        self.confusion_matrx(true, predicted)

        fname = os.path.join(self.path, f"{prefix}_prediction.csv")
        pd.DataFrame(np.concatenate([true, predicted], axis=1),
                     columns=['true', 'predicted'], index=index).to_csv(fname)
        class_metrics = ClassificationMetrics(true, predicted, multiclass=True)

        dict_to_file(self.path,
                     errors=class_metrics.calculate_all(),
                     name=f"{prefix}_{dateandtime_now()}.json")
        return

    def process_binary(self, true, predicted, metrics, prefix, index):

        assert self.n_classes(true) == 2

        if predicted.ndim == 1:
            predicted = predicted.reshape(-1, 1)
        elif predicted.size != len(predicted):
            predicted = np.argmax(predicted, axis=1).reshape(-1,1)

        if true.ndim == 1:
            true = true.reshape(-1, 1)
        elif true.size != len(true):
            true = np.argmax(true, axis=1).reshape(-1,1)

        self.confusion_matrx(true, predicted)

        fpath = os.path.join(self.path, prefix)
        if not os.path.exists(fpath):
            os.makedirs(fpath)

        metrics_instance = ClassificationMetrics(true, predicted, multiclass=False)
        metrics = getattr(metrics_instance, f"calculate_{metrics}")()
        dict_to_file(fpath,
                     errors=metrics,
                     name=f"{prefix}_{dateandtime_now()}.json"
                     )

        fname = os.path.join(fpath, f"{prefix}_.csv")
        array = np.concatenate([true.reshape(-1, 1), predicted.reshape(-1, 1)], axis=1)
        pd.DataFrame(array, columns=['true', 'predicted'],  index=index).to_csv(fname)

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


def choose_examples(x, examples_to_use, y=None):
    """Chooses exampels from x and y"""
    if isinstance(examples_to_use, int):
        if len(x) == examples_to_use:
            # since using all the examples, don't randomize them
            x, index = x, np.arange(examples_to_use)
        else:
            # randomly chose x values from test_x
            x, index = choose_n_imp_exs(x, examples_to_use, y)
    elif isinstance(examples_to_use, float):
        assert examples_to_use < 1.0
        # randomly choose x fraction from test_x
        x, index = choose_n_imp_exs(x, int(examples_to_use * len(x)), y)

    elif hasattr(examples_to_use, '__len__'):
        index = np.array(examples_to_use)
        x = x[index]
    else:
        raise ValueError(f"unrecognized value of examples_to_use: {examples_to_use}")

    return x, index


def choose_n_imp_exs(x: np.ndarray, n: int, y=None):
    """Chooses the n important examples from x and y"""

    n = min(len(x), n)

    st = n // 2
    en = n - st

    if y is None:
        idx = np.random.randint(0, len(x), n)
    else:
        st = np.argsort(y, axis=0)[0:st].reshape(-1,)
        en = np.argsort(y, axis=0)[-en:].reshape(-1,)
        idx = np.hstack([st, en])

    x = x[idx]

    return x, idx


def _wandb_scatter(true: np.ndarray, predicted: np.ndarray, name: str) -> None:
    """Adds a scatter plot on wandb."""
    data = [[x, y] for (x, y) in zip(true.reshape(-1,), predicted.reshape(-1,))]
    table = wandb.Table(data=data, columns=["true", "predicted"])
    wandb.log({
        "scatter_plot": wandb.plot.scatter(table, "true", "predicted",
                                           title=name)
               })
    return
