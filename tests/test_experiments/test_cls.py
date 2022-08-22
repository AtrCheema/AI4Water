
import unittest
import site
import os
site.addsitedir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from ai4water.experiments import MLClassificationExperiments

from ai4water.datasets import MtropicsLaos


data = MtropicsLaos().make_classification(
    input_features=['air_temp', 'rel_hum'],
    lookback_steps=1)

inputs = data.columns.tolist()[0:-1]
outputs = data.columns.tolist()[-1:]


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

    def test_basic(self):
        exp = MLClassificationExperiments(input_features=inputs,
                                          output_features=outputs)

        exp.fit(data=data, exclude=[
            # giving nan predictions
            'LabelPropagation', 'LabelSpreading', 'QuadraticDiscriminantAnalysis',
            'LinearDiscriminantAnalysis',
        ]
                )
        exp.compare_errors('accuracy', show=False)
        return

    def test_binary_cv(self):
        """run MLClassificationExperiments for binary classification with optimization"""
        exp = MLClassificationExperiments(
            input_features=inputs,
            output_features=outputs,
            train_fraction=1.0,
            val_fraction=0.3,
            cross_validator = {"KFold": {"n_splits": 3}}
        )

        exp.fit(data=data,
                exclude=[
                    'LabelPropagation', 'LabelSpreading',
                    'QuadraticDiscriminantAnalysis',
                    'LinearDiscriminantAnalysis'
                ],
                cross_validate=True,
                )

        exp.plot_cv_scores(show=False)
        return

    def test_binary_optimize(self):
        """run MLClassificationExperiments for binary classification with optimization"""
        exp = MLClassificationExperiments(
            input_features=inputs,
            output_features=outputs,
            train_fraction=1.0,
            val_fraction=0.3,
        )

        exp.fit(data=data,
                include=[
                    # "model_AdaBoostClassifier", TODO
                    "model_CatBoostClassifier",
                    "model_LGBMClassifier",
                    "model_XGBClassifier",
                    "RandomForestClassifier"
                ],
                run_type="optimize",
                )
        return

    def test_multiclass(self):
        """multiclass classification"""

        exp = MLClassificationExperiments(
            input_features=input_features_cls,
            output_features=['target']
        )

        exp.fit(data=data_multiclass,
                exclude=[
            # giving nan predictions
            'LabelPropagation', 'LabelSpreading', 'QuadraticDiscriminantAnalysis',
                    'NuSVC',
        ]
                )
        return

    def test_multiclass_cv(self):
        """multiclass classification with cross validation"""

        exp = MLClassificationExperiments(
            input_features=input_features_cls,
            cross_validator={"KFold": {"n_splits": 3}},
            output_features=['target']
        )

        exp.fit(data=data_multiclass, exclude=[
            # giving nan predictions
            'LabelPropagation', 'LabelSpreading', 'QuadraticDiscriminantAnalysis',
            'NuSVC',  # ValueError: specified nu is infeasible
        ],
                cross_validate=True
                )

        exp.plot_cv_scores(show=False)

        return

if __name__ == "__main__":
    unittest.main()