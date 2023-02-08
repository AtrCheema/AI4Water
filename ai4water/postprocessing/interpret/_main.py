

import warnings

from ai4water.backend import xgboost, tf, np, pd, mpl, plt, os
from ai4water.backend import easy_mpl as ep
from ai4water.utils.visualizations import Plot
from ai4water.utils.utils import plot_activations_along_inputs


class Interpret(Plot):
    """Interprets the ai4water Model."""

    def __init__(self, model, save:bool = False, show:bool = True):
        """
        Arguments
        ---------
            model :
                an instance of ai4water's Model
            save : bool
            show : bool
        """
        self.model = model
        self.save = save
        self.show = show

        super().__init__(model.path)

        if self.model.category.upper() == "DL":

            if hasattr(model, 'interpret') and not model.__class__.__name__ == "Model":
                model.interpret()

        elif self.model.category == 'ML':
            use_xgb = False
            if self.model._model.__class__.__name__ == "XGBRegressor":
                use_xgb = True
            self.plot_feature_importance(use_xgb = use_xgb)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, x):
        self._model = x

    def feature_importance(self):
        if self.model.category.upper() == "ML":

            estimator = self.model._model

            if not is_fitted(estimator):
                print(f"the model {estimator} is not fitted yet so not feature importance")
                return

            model_name = list(self.model.config['model'].keys())[0]
            if model_name.upper() in ["SVC", "SVR"]:
                if estimator.kernel == "linear":
                    # https://stackoverflow.com/questions/41592661/determining-the-most-contributing-features-for-svm-classifier-in-sklearn
                    return estimator.coef_
            elif hasattr(estimator, "feature_importances_"):
                return estimator.feature_importances_

    def f_importances_svm(self, coef, names):

        plt.close('all')
        mpl.rcParams.update(mpl.rcParamsDefault)
        classes = coef.shape[0]
        features = coef.shape[1]
        _, axis = plt.subplots(classes, sharex='all')
        axis = axis if hasattr(axis, "__len__") else [axis]

        for idx, ax in enumerate(axis):
            # colors = ['red' if c < 0 else 'blue' for c in self._model.coef_[idx]]
            ax.bar(range(features), self._model.coef_[idx], 0.4)

        plt.xticks(ticks=range(features), labels=self.model.input_features, rotation=90, fontsize=12)
        self.save_or_show(save=self.save, fname=f"{list(self.model.config['model'].keys())[0]}_feature_importance")
        return

    def plot_feature_importance(
            self,
            importance=None,
            use_xgb=False,
            max_num_features=20,
            figsize=None,
            **kwargs):
        """
        plots feature importance when the model is tree based.
        """
        figsize = figsize or (8, 8)

        if importance is None:
            importance = self.feature_importance()

        if self.model.category == "ML":
            model_name = list(self.model.config['model'].keys())[0]
            if model_name.upper() in ["SVC", "SVR"]:
                if self.model._model.kernel == "linear":
                    return self.f_importances_svm(importance, self.model.input_features)
                else:
                    warnings.warn(f"for {self.model._model.kernel} kernels of {model_name}, feature "
                                  f"importance can not be plotted.")
                return

        if isinstance(importance, np.ndarray):
            assert importance.ndim <= 2

        if importance is None:
            return

        all_cols = self.model.input_features + self.model.output_features
        if self.model.teacher_forcing:
            all_cols = self.model.input_features

        imp_sort = np.sort(importance)[::-1]
        all_cols = np.array(all_cols)
        all_cols = all_cols[np.argsort(importance)[::-1]]

        # save the whole importance before truncating it
        fname = os.path.join(self.model.path, 'feature_importance.csv')
        pd.DataFrame(imp_sort, index=all_cols,
                     columns=['importance_sorted']).to_csv(fname)

        imp = np.concatenate([imp_sort[0:max_num_features], [imp_sort[max_num_features:].sum()]])
        all_cols = list(all_cols[0:max_num_features]) + [f'rest_{len(all_cols) - max_num_features}']

        if use_xgb:
            self._feature_importance_xgb(max_num_features=max_num_features)
        else:
            plt.close('all')
            _, axis = plt.subplots(figsize=figsize)
            ep.bar_chart(labels=all_cols,
                      values=imp,
                      ax=axis,
                      ax_kws={'title':"Feature importance",
                              'xlabel_kws': {'fontsize': 12}},
                      show=False)
            self.save_or_show(save=self.save, show=self.show,
                              fname="feature_importance.png")
        return

    def _feature_importance_xgb(self, max_num_features=None, **kwargs):
        # https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.plot_importance
        if xgboost is None:
            warnings.warn("install xgboost to plot plot_importance using xgboost", UserWarning)
        else:
            booster = self.model._model.get_booster()
            booster.feature_names = self.model.input_features
            plt.close('all')
            # global feature importance with xgboost comes with different types
            xgboost.plot_importance(booster, max_num_features=max_num_features)
            self.save_or_show(save=self.save, show=self.show,
                              fname="feature_importance_weight.png")
            plt.close('all')
            xgboost.plot_importance(booster, importance_type="cover",
                                    max_num_features=max_num_features, **kwargs)
            self.save_or_show(save=self.save, show=self.show,
                              fname="feature_importance_type_cover.png")
            plt.close('all')
            xgboost.plot_importance(booster, importance_type="gain",
                                    max_num_features=max_num_features, **kwargs)
            self.save_or_show(save=self.save, show=self.show,
                              fname="feature_importance_type_gain.png")

        return

    def compare_xgb_f_imp(
            self,
            calculation_method="all",
            rescale=True,
            figsize:tuple=None,
            backend:str = 'matplotlib',
            **kwargs
    ):
        """compare various feature importance calculations methods that are built
        in in XGBoost"""

        inp_features = self.model.input_features
        assert isinstance(inp_features, list)

        booster = self.model._model.get_booster()
        booster.feature_names = self.model.input_features

        _importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
        importance_types = _importance_types.copy()

        if calculation_method != "all":
            if isinstance(calculation_method, str):
                calculation_method = [calculation_method]
            assert isinstance(calculation_method, list)
            # remove those which are not desired
            for imp in _importance_types:
                if imp not in calculation_method:
                    importance_types.remove(imp)

        # container to hold importances with each method
        importance = []

        for idx, imp_type in enumerate(importance_types):
            score = pd.Series(booster.get_score(importance_type=imp_type))
            score = pd.DataFrame(score, columns=[imp_type])

            if rescale:
                # so that the sum of all feature importance is 1.0 and the
                # scale is relative
                score = score / score.sum()

            importance.append(score)

        importance = pd.concat(importance, axis=1)

        if backend=="plotly":
            width = figsize[0] if figsize else 1200
            height = figsize[1] if figsize else 1200
            return xgb_fimp_with_plotly(importance,
                                        importance_types,
                                        fig_width=width,
                                        fig_height=height,
                                        path=self.model.path)

        plt.close('all')

        fig, axis = plt.subplots(importance.shape[1],
                                 sharex="all",
                                 figsize=figsize)

        for ax, imp in zip(axis.flat, importance.columns):

            ax = ep.bar_chart(
                importance[imp],
                labels=importance.index,
                orient="vertical",
                show=False,
                rotation=90,
                label=imp,
                ax=ax,
                **kwargs)
            ax.legend()

        fname = os.path.join(self.model.path, "xgb_f_imp_comp")

        if self.save:
            plt.savefig(fname, bbox_inches="tight")

        if self.show:
            plt.show()

        return fig

    def tft_attention_components(
            self,
            x = None,
            data=None,
            data_type:str = "test",
    ):
        """
        Gets attention components of tft layer from ai4water's Model.

        Parameters
        ----------
            x :
                the input data to the model
            data :
                raw data from which ``x``/inputs are extracted.
            data_type :
                the data to use to calculate attention components

        Returns
        -------
        dict
            dictionary containing attention components of tft as numpy arrays.
            Following four attention components are present in the dictionary
                - decoder_self_attn: (attention_heads, ?, total_time_steps, 22)
                - static_variable_selection_weights:
                - encoder_variable_selection_weights: (?, encoder_steps, input_features)
                - decoder_variable_selection_weights: (?, decoder_steps, input_features)
        str
            a string indicating which data was used
        """

        maybe_create_path(self.model.path)

        if x is None:
            x, _, = getattr(self.model, f'{data_type}_data')(data=data)

        if len(x) == 0 and data_type == "test":
            warnings.warn("No test data found. using validation data instead",
                          UserWarning)
            x, _, = getattr(self.model, 'validation_data')(data=data)
            assert len(x) >0

        attentions = self.model.TemporalFusionTransformer_attentions
        if self.model.api == 'subclassing':
            inputs = self.model.inputs
        else:
            inputs = self.model._model.inputs

        attention_components = {}

        for k, v in attentions.items():
            if v is not None:
                temp_model = tf.keras.Model(inputs=inputs,
                                            outputs=v)
                attention_components[k] = temp_model.predict(x=x, verbose=0, steps=1)

        return attention_components, data

    def get_enc_var_selection_weights(self, data, data_type:str='test'):
        """Returns encoder variable selection weights of TFT model"""

        ac, _ = self.tft_attention_components(data=data, data_type=data_type)
        return ac['encoder_variable_selection_weights']

    def interpret_example_tft(
            self,
            example_index:int,
            x = None,
            data=None,
            data_type='test'
    ):
        """interprets a single example using TFT model.

        Parameters
        ---------
            example_index : int
                index of example to be explained
            x :
                input data, if not given, ``data`` must be given
            data :
                the data whose example to interpret.
            data_type : str
                either ``training``, ``test``, ``validation`` or ``all``.
                It is only useful when ``data`` argument is used.
        """

        assert data_type in ("training", "test", "validation", "all")

        if x is None:
            data_name = data_type
        else:
            data_name = "data"

        enc_var_selection_weights = self.get_enc_var_selection_weights(
            data=data, data_type=data_type)

        plt.close('all')

        im = ep.imshow(
            enc_var_selection_weights[example_index],
            aspect="auto",
            ax_kws=dict(title=example_index,
                        ylabel="lookback steps"),
            show=False
        )

        plt.xticks(np.arange(self.model.num_ins), self.model.input_features,
                   rotation=90)
        plt.colorbar(im, orientation='vertical', pad=0.05)
        fname = os.path.join(maybe_create_path(self.model.path),
                             f'{data_name}_enc_var_selec_{example_index}.png')
        if self.save:
            plt.savefig(fname, bbox_inches='tight', dpi=300)
        if self.show:
            plt.show()
        return

    def interpret_tft(
            self,
            x=None,
            y=None,
            data=None,
            data_type="test"
    ):
        """global interpretation of TFT model.

        Arguments:
            x :
                input data. If not given, ``data`` argument must be given.
            y :
                labels/target/true data corresponding to ``x``. It is only
                used for plotting.
            data :
                the data to use to interpret model. It is only required
                when ``x`` is not given.
            data_type :
                either ``training``, ``test``, ``validation`` or ``all``.
                It is only useful when ``data`` argument is used.
        """

        if x is None:
            predictions = getattr(self.model, f"predict_on_{data_type}_data")(
                data=data,
                process_results=False, verbose=0)
            x, y, = getattr(self.model, f'{data_type}_data')(data=data)
        else:
            predictions = self.model.predict(x=x, verbose=0)

        ac, data = self.tft_attention_components(data=data)

        encoder_variable_selection_weights = ac['encoder_variable_selection_weights']

        plot_activations_along_inputs(
            activations=encoder_variable_selection_weights,
            data=x[:, -1],
            observations=y,
            predictions=predictions,
            in_cols=self.model.input_features,
            out_cols=self.model.output_features,
            lookback=self.model.lookback,
            name=f'tft_encoder_weights_{data}',
            path=maybe_create_path(self.model.path)
        )
        return

    def interpret_attention_lstm(
            self,
            x=None,
            data = None,
            data_type:str = "test"
    ):
        """
        Arguments:
            x :
                input data. If not given, ``data`` argument must be given.
            data :
                the data to use to interpret model. It is only required
                when ``x`` is not given.
            data_type :
                either ``training``, ``test``, ``validation`` or ``all``.
                It is only useful when ``data`` argument is used.
        """
        raise NotImplementedError

    def interpret_tab_transformer(
            self,
            x=None,
            data = None,
            data_type:str = "test"
    ):
        """
        Arguments:
            x :
                input data. If not given, ``data`` argument must be given.
            data :
                the data to use to interpret model. It is only required
                when ``x`` is not given.
            data_type :
                either ``training``, ``test``, ``validation`` or ``all``.
                It is only useful when ``data`` argument is used.
        """
        raise NotImplementedError

    def interpret_ft_transformer(
            self,
            x=None,
            data = None,
            data_type:str = "test"
    ):
        """
        Arguments:
            x :
                input data. If not given, ``data`` argument must be given.
            data :
                the data to use to interpret model. It is only required
                when ``x`` is not given.
            data_type :
                either ``training``, ``test``, ``validation`` or ``all``.
                It is only useful when ``data`` argument is used.
        """
        raise NotImplementedError

def xgb_fimp_with_plotly(
        importance:pd.DataFrame,
        importance_types,
        fig_width,
        fig_height,
        path,
):

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # initiate figure with subplots
    fig = make_subplots(
        rows=len(importance_types) + 1, cols=1,
        vertical_spacing=0.02
        # shared_xaxes=True
    )

    for idx, col in enumerate(importance.columns):
        fig.add_trace(go.Bar(
            x=importance.index.tolist(),
            y=importance[col],
            name=col
        ), row=idx + 1, col=1)

    fig.update_xaxes(showticklabels=False)  # hide all the xticks
    fig.update_xaxes(showticklabels=True,
                     row=len(importance_types),
                     col=1,
                     tickangle=-45,
                     title="Input Features"
                     )

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(
        height=fig_height,
        width=fig_width,
        legend_title="Calculation Method",
        title_text="XGBoost Feature Importance",
        title_x=0.42,
        font=dict(
            family="Times New Roman",
            size=26,
        )
    )
    fname = os.path.join(path, "xgb_f_imp_comp.html")
    fig.write_html(fname)
    return fig


def maybe_create_path(path):
    path = os.path.join(path, "interpret")
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def is_fitted(estimator):

    if hasattr(estimator, 'is_fitted'):  # for CATBoost
        return estimator.is_fitted

    attrs = [v for v in vars(estimator)
             if v.endswith("_") and not v.startswith("__")]

    if not attrs:
        return False

    return True
