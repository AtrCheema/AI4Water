import os
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import lime
except ModuleNotFoundError:
    lime = None

from ._explain import ExplainerMixin


class LimeExplainer(ExplainerMixin):

    """
    Wrapper around LIME module.

    Example:
        >>>from ai4water import Model
        >>>from ai4water.datasets import arg_beach
        >>>model = Model(model="GradientBoostingRegressor", data=arg_beach())
        >>>model.fit()
        >>>lime_exp = LimeExplainer(model=model._model,
        ...                       train_data=model.training_data()[0],
        ...                       test_data=model.test_data()[0],
        ...                       mode="regression")
        >>>lime_exp()

    Attributes:
        explaination_objects : location explaination objects for each individual example/instance
    """
    def __init__(
            self,
            model,
            data,
            train_data,
            mode: str,
            explainer=None,
            path=None,
            features: list = None,
            verbosity: Union[int, bool] = True
    ):
        """
        Arguments:
            model : the model to explain. The model must have `predict` method.
            data : the data to explain.
            train_data : the data on which the model was trained.
            mode : either of `regression` or `classification`
            explainer : The explainer to use. By default, LimeTabularExplainer is used.
            path : path where to save all the plots. By default, plots will be saved in
                current working directory.
            features : name/names of features.
            verbosity : whether to print information or not.
        """
        self.model = model
        self.train_data = to_np(train_data)

        super(LimeExplainer, self).__init__(path=path or os.getcwd(), data=to_np(data), features=features)

        self.mode = mode
        self.verbosity = verbosity
        self.explainer = self._get_explainer(explainer)

        self.explaination_objects = {}

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, x):
        if x is not None:
            assert x in ["regression", "classification"], f"mode must be either regression or classification not {x}"
        self._mode = x

    def _get_explainer(self, proposed_explainer=None):

        import lime.lime_tabular

        if proposed_explainer is None and self.data.ndim <= 2:
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(self.train_data,
                                                                    feature_names=self.features,
                                                                    # class_names=['price'],
                                                                    # categorical_features=categorical_features,
                                                                    verbose=self.verbosity,
                                                                    mode=self.mode
                                                                    )
        elif proposed_explainer in lime.lime_tabular.__dict__.keys():
            lime_explainer = getattr(lime.lime_tabular, proposed_explainer)(self.train_data,
                                                                            feature_names=self.features,
                                                                            mode=self.mode,
                                                                            verbose=self.verbosity)
        elif self.data.ndim == 3:
            lime_explainer = lime.lime_tabular.RecurrentTabularExplainer(self.train_data,
                                                                         mode=self.mode,
                                                                         feature_names=self.features,
                                                                         verbose=self.verbosity)
        elif proposed_explainer is not None:
            lime_explainer = getattr(lime, proposed_explainer)(self.train_data,
                                                               features=self.features, mode=self.mode)
        else:
            raise ValueError(f"Can not infer explainer. Please specify explainer to use.")
        return lime_explainer

    def __call__(self, *args, **kwargs):

        self.explain_all_examples(*args, **kwargs)

        return

    def explain_all_examples(self,
                             plot_type="pyplot",
                             name="lime_explaination",
                             num_features=None,
                             **kwargs
                             ):
        """
        Draws and saves plot for all examples of test_data.

        Arguments:
            plot_type :
            name :
            num_features :
            kwargs : any keyword argument for `explain_instance`

        An example here means an instance/sample/data point.
        """
        for i in range(len(self.data)):
            self.explain_example(i, plot_type=plot_type, name=f"{name}_{i}",
                                 num_features=num_features, **kwargs)
        return

    def explain_example(
            self,
            index: int,
            plot_type: str = "pyplot",
            name: str = "lime_explaination",
            num_features: int = None,
            colors=None,
            annotate=False,
            show=False,
            save=True,
            **kwargs
    ):
        """
        Draws and saves plot for a single example of test_data.

        Arguments:
            index : index of test_data
            plot_type : either pyplot or html
            name : name with which to save the file
            num_features :
            colors :
            annotate : whether to annotate figure or not
            show : whether to show figure or not
            save : wheter to save figure or not
            kwargs : any keyword argument for `explain_instance`

        Returns:
            matplotlib figure if plot_type="pyplot" and show is False.
        """
        assert plot_type in ("pyplot", "html")

        exp = self.explainer.explain_instance(self.data[index],
                                              self.model.predict,
                                              num_features=num_features or len(self.features),
                                              **kwargs
                                              )

        self.explaination_objects[index] = exp

        fig = None
        if plot_type == "pyplot":
            plt.close()
            fig = as_pyplot_figure(exp, colors=colors, example_index=index, annotate=annotate)
            if save:
                plt.savefig(os.path.join(self.path, f"{name}_{index}"), bbox_inches="tight")
            if show:
                plt.show()
        else:
            exp.save_to_file(os.path.join(self.path, f"{name}_{index}"))

        return fig


def to_np(x) -> np.ndarray:

    if isinstance(x, pd.DataFrame):
        x = x.values
    else:
        assert isinstance(x, np.ndarray)

    return x


def as_pyplot_figure(
        inst_explainer,
        label=1,
        example_index=None,
        colors: [str, tuple, list] = None,
        annotate=False,
        **kwargs):
    """Returns the explanation as a pyplot figure.

    Will throw an error if you don't have matplotlib installed
    Args:
        inst_explainer : instance explainer
        label: desired label. If you ask for a label for which an
               explanation wasn't computed, will throw an exception.
               Will be ignored for regression explanations.
        colors : if tuple it must be names of two colors for +ve and -ve
        example_index :
        annotate : whether to annotate the figure or not?
        kwargs: keyword arguments, passed to domain_mapper

    Returns:
        pyplot figure (barchart).
    """
    textstr = f"""Prediction: {round(inst_explainer.predicted_value, 2)}
Local prediction: {round(inst_explainer.local_pred.item(), 2)}"""

    if colors is None:
        colors = ([0.9375, 0.01171875, 0.33203125], [0.23828125, 0.53515625, 0.92578125])
    elif isinstance(colors, str):
        colors = (colors, colors)

    exp = inst_explainer.as_list(label=label, **kwargs)
    fig = plt.figure()
    vals = [x[1] for x in exp]
    names = [x[0] for x in exp]
    vals.reverse()
    names.reverse()

    if isinstance(colors, tuple):
        colors = [colors[0] if x > 0 else colors[1] for x in vals]

    pos = np.arange(len(exp)) + .5
    h = plt.barh(pos, vals, align='center', color=colors)
    plt.yticks(pos, names)
    if inst_explainer.mode == "classification":
        title = 'Local explanation for class %s' % inst_explainer.class_names[label]
    else:
        title = f'Local explanation for example {example_index}'
    plt.title(title)
    plt.grid(linestyle='--', alpha=0.5)

    if annotate:
        # https://stackoverflow.com/a/59109053/5982232
        plt.legend(h, [textstr], loc="best",
                   fancybox=True, framealpha=0.7,
                   handlelength=0, handletextpad=0)
    return fig
