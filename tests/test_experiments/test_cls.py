
import unittest
import site
import os
site.addsitedir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from ai4water.experiments import MLClassificationExperiments

from ai4water.datasets import MtropicsLaos


data = MtropicsLaos().make_classification(lookback_steps=2)

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
            'LabelPropagation', 'LabelSpreading', 'QuadraticDiscriminantAnalysis'  # giving nan predictions
        ]
                )
        exp.compare_errors('accuracy', show=False)
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
            'LabelPropagation', 'LabelSpreading', 'QuadraticDiscriminantAnalysis'  # giving nan predictions
        ]
                )

if __name__ == "__main__":
    unittest.main()