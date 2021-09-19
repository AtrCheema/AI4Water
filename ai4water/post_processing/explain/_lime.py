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


class LimeMLExplainer(ExplainerMixin):

    """
    Wrapper around LIME module.

    Example
    -------
    >>>from ai4water import Model
    >>>from ai4water.datasets import arg_beach
    >>>model = Model(model="GradientBoostingRegressor", data=arg_beach())
    >>>model.fit()
    >>>lime_exp = LimeMLExplainer(model=model._model,
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
            test_data,
            train_data,
            mode: str,
            explainer=None,
            path=os.getcwd(),
            features: list = None,
            verbosity: Union[int, bool] = True
    ):
        """
        Arguments:
            model : the model to explain
            test_data : the data to explain.
            train_data : training data
            mode : either of regression or classification
            explainer : The explainer to use. By default, LimeTabularExplainer is used.
            path : path where to save all the plots
            features : name/names of features.
            verbosity : whether to print information or not.
        """
        self.model = model
        self.train_data = to_np(train_data)

        super(LimeMLExplainer, self).__init__(path=path, data=to_np(test_data), features=features)

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

        if proposed_explainer is None:
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(self.train_data,
                                                                    feature_names=self.features,
                                                                    # class_names=['price'],
                                                                    # categorical_features=categorical_features,
                                                                    verbose=self.verbosity,
                                                                    mode=self.mode
                                                                    )
        else:
            lime_explainer = getattr(lime, proposed_explainer)(self.train_data, self.features, mode=self.mode)
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
            self.explain_example(i, plot_type=plot_type, name=name, num_features=num_features, **kwargs)
        return

    def explain_example(self,
                        index: int,
                        plot_type: str = "pyplot",
                        name: str = "lime_explaination",
                        num_features: int = None,
                        **kwargs
                        ):
        """
        Draws and saves plot for a single example of test_data.

        Arguments:
            index : index of test_data
            plot_type : either pyplot or html
            name : name with which to save the file
            num_features :
            kwargs : any keyword argument for `explain_instance`
        """
        exp = self.explainer.explain_instance(self.data[index],
                                              self.model.predict,
                                              num_features=num_features or len(self.features),
                                              **kwargs
                                              )

        self.explaination_objects[index] = exp

        if plot_type == "pyplot":
            plt.close()
            exp.as_pyplot_figure()
            plt.savefig(os.path.join(self.path, f"{name}_{index}"), bbox_inches="tight")
        else:
            exp.save_to_file(os.path.join(self.path, f"{name}_{index}"))

        return


def to_np(x) -> np.ndarray:

    if isinstance(x, pd.DataFrame):
        x = x.values
    else:
        assert isinstance(x, np.ndarray)

    return x
