import unittest

import pandas as pd

from ai4water import Model
from ai4water.datasets import arg_beach
from ai4water.datasets import MtropicsLaos
from ai4water.post_processing.explain import LimeMLExplainer

laos = MtropicsLaos()

data = laos.make_regression()


def get_lime(to_dataframe=False, examples_to_explain=5):

    model = Model(
        model="GradientBoostingRegressor",
        data=data,
        cross_validator={'TimeSeriesSplit': {'n_splits': 10}},
        transformation={'method': 'quantile', 'features': ['Ecoli_mpn100'],
                        },
        val_metric='r2',
        verbosity=0,
    )

    model.fit()

    train_x, train_y = model.training_data()
    test_x, test_y = model.test_data()

    if to_dataframe:
        train_x = pd.DataFrame(train_x, columns=model.in_cols)
        test_x = pd.DataFrame(test_x[0:examples_to_explain], columns=model.in_cols)
    else:
        test_x = test_x[0:examples_to_explain]

    lime_exp = LimeMLExplainer(model=model._model,
                               train_data=train_x,
                               test_data=test_x,
                               mode="regression",
                               features=list(model.in_cols),
                               path=model.path,
                               verbosity=False,
                               )
    return lime_exp


class TestLimeExplainer(unittest.TestCase):

    def test_all_examples(self):
        lime_exp = get_lime(examples_to_explain=5)
        lime_exp.explain_all_examples()
        assert len(lime_exp.explaination_objects) == 5

        return

    def test_single_example(self):

        lime_exp = get_lime()
        lime_exp.explain_example(0)
        assert len(lime_exp.explaination_objects) == 1

        return

    def test_wth_pandas(self):
        lime_exp = get_lime(True)
        lime_exp.explain_example(0)

        assert len(lime_exp.explaination_objects) == 1

        return

    def test_docstring_example(self):

        model = Model(model="GradientBoostingRegressor", data=arg_beach(), verbosity=0)
        model.fit()
        lime_exp = LimeMLExplainer(model=model._model,
                                   train_data=model.training_data()[0],
                                   test_data=model.test_data()[0][0:3],
                                   mode="regression",
                                   verbosity=False
                                   )
        lime_exp()

        assert len(lime_exp.explaination_objects) == 3

        return


if __name__ == "__main__":

    unittest.main()
