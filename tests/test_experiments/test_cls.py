
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
    x = np.random.random((n_samples, n_features))
    y = np.random.randint(0, n_classes, size=(n_samples, 1))
    return x, y


class TestCls(unittest.TestCase):

    def test_basic(self):
        exp = MLClassificationExperiments(input_features=inputs,
                                          output_features=outputs)

        exp.fit(data=data, exclude=[
            # giving nan predictions
            'LabelPropagation', 'LabelSpreading', 'QuadraticDiscriminantAnalysis'
        ]
                )
        exp.compare_errors('accuracy', show=False)
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

        n_classes = 5
        input_features = [f'input_{n}' for n in range(10)]
        #outputs = [f'target_{n}' for n in range(n_classes)]
        x, y = make_multiclass_classification(n_samples=100,
                                              n_features=len(input_features),
                                              n_classes=n_classes)
        df = pd.DataFrame(
            np.concatenate([x, y], axis=1),
            columns=input_features + ['target']
        )

        exp = MLClassificationExperiments(input_features=inputs,
                                    output_features=outputs)

        exp.fit(data=data, exclude=[
            # giving nan predictions
            'LabelPropagation', 'LabelSpreading', 'QuadraticDiscriminantAnalysis'
        ]
                )

if __name__ == "__main__":
    unittest.main()