
import unittest
import site
import os
site.addsitedir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from SeqMetrics import ClassificationMetrics

from ai4water.experiments import MLClassificationExperiments

from ai4water.datasets import MtropicsLaos


data = MtropicsLaos(
    path=r'D:\data\MtropicsLaos'
).make_classification(
    input_features=['air_temp', 'rel_hum'],
    lookback_steps=1)

inputs = data.columns.tolist()[0:-1]
outputs = data.columns.tolist()[-1:]


def f1_score_(t,p)->float:
    return ClassificationMetrics(t, p).f1_score(average="macro")


def make_multiclass_classification(
    n_samples,
    n_features,
    n_classes
):
    input_features = [f'input_{n}' for n in range(n_features)]

    x = np.random.random((n_samples, n_features))
    y = np.random.randint(0, n_classes, size=(n_samples, 1))

    df = pd.DataFrame(
        np.concatenate([x, y], axis=1),
        columns=input_features + ['target']
    )

    return df, input_features


data_multiclass, input_features_cls = make_multiclass_classification(100, 10, 5)


class TestCls(unittest.TestCase):

    def test_basic(self): # todo
        exp = MLClassificationExperiments(
            input_features=inputs,
            output_features=outputs,
            monitor = [f1_score_, "accuracy", "f1_score", "precision"],
            show=False, save=False
        )

        exp.fit(data=data, exclude=[
            # giving nan predictions
            'LabelPropagation', 'LabelSpreading', 'QuadraticDiscriminantAnalysis',
            'LinearDiscriminantAnalysis',
        ],
                include=['model_GradientBoostingClassifier',
                         'model_HistGradientBoostingClassifier']
                )
        exp.compare_errors('accuracy', data=data)
        exp.compare_errors('f1_score', data=data)
        exp.compare_precision_recall_curves(data[inputs].values, data[outputs].values)
        exp.compare_roc_curves(data[inputs].values, data[outputs].values)

        return

    def test_binary_cv(self):
        """run MLClassificationExperiments for binary classification with optimization"""
        exp = MLClassificationExperiments(
            input_features=inputs,
            output_features=outputs,
            train_fraction=1.0,
            val_fraction=0.3,
            cross_validator = {"KFold": {"n_splits": 3}},
            show=False, save=False
        )

        exp.fitcv(data=data,
                exclude=[
                    'LabelPropagation', 'LabelSpreading',
                    'QuadraticDiscriminantAnalysis',
                    'LinearDiscriminantAnalysis'
                ],
                include=['model_GradientBoostingClassifier', 'RandomForestClassifier'],
                )

        exp.plot_cv_scores()
        return

    def test_binary_optimize(self): # todo
        """run MLClassificationExperiments for binary classification with optimization"""
        exp = MLClassificationExperiments(
            input_features=inputs,
            output_features=outputs,
            train_fraction=1.0,
            val_fraction=0.3,
            show=False, save=False
        )

        exp.fit(data=data,
                include=[
                    # "model_AdaBoostClassifier", TODO
                    "model_HistGradientBoostingClassifier",
                    "RandomForestClassifier"
                ],
                run_type="optimize",
                )
        exp.compare_precision_recall_curves(data[inputs].values, data[outputs].values)
        return

    def test_multiclass(self):
        """multiclass classification"""

        exp = MLClassificationExperiments(
            input_features=input_features_cls,
            output_features=['target'],
            show=False, save=False
        )

        exp.fit(data=data_multiclass,
                            include=[
                                "model_HistGradientBoostingClassifier",
                                "RandomForestClassifier"
                            ],
                )
        return

    def test_multiclass_cv(self):
        """multiclass classification with cross validation"""

        exp = MLClassificationExperiments(
            input_features=input_features_cls,
            cross_validator={"KFold": {"n_splits": 3}},
            output_features=['target'],
            show=False, save=False
        )

        exp.fit(data=data_multiclass,
                include=[
                    "model_HistGradientBoostingClassifier",
                    "RandomForestClassifier"
                ],
                cross_validate=True
                )

        exp.plot_cv_scores()

        return

    def test_optimize_wtih_cv(self):

        exp = MLClassificationExperiments(
            input_features=inputs,
            output_features=outputs,
            cross_validator={'KFold': {'n_splits': 3}},
            show=False, save=False
        )

        exp.fit(data=data,
                run_type="optimize",
                opt_method="random",
                num_iterations=5,
                cross_validate=True,
                post_optimize="train_best",
                include=[
                    'model_LinearSVC',
                ]
                )
        exp.plot_cv_scores()
        return


if __name__ == "__main__":
    unittest.main()
