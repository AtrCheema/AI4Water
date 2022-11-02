import unittest
import site
import os

cwd = os.path.dirname(os.path.abspath(__file__))
site.addsitedir(os.path.dirname(os.path.dirname(cwd)))

from ai4water.experiments import DLRegressionExperiments, DLClassificationExperiments
from ai4water.datasets import busan_beach
from ai4water.hyperopt import Categorical
from ai4water.datasets import MtropicsLaos


data = busan_beach()

cls_data = MtropicsLaos().make_classification(
    input_features=['air_temp', 'rel_hum'],
    lookback_steps=5)

inputs = data.columns.tolist()[0:-1]
outputs = data.columns.tolist()[-1:]

class TestClassification(unittest.TestCase):

    def test_basic(self):
        exp = DLClassificationExperiments(
            input_features=cls_data.columns.tolist()[0:-1],
            output_features=cls_data.columns.tolist()[-1:],
            epochs=5,
            ts_args={"lookback": 5}
        )

        exp.fit(data=cls_data, include=["TFT",
                                    "MLP"])
        return


class TestDLExeriments(unittest.TestCase):

    def test_rgr_dry_run(self):

        exp = DLRegressionExperiments(
            input_features = data.columns.tolist()[0:-1],
            output_features = data.columns.tolist()[-1:],
            epochs=5,
            ts_args={"lookback": 9}
        )

        exp.fit(data=data, include=["TFT",
                                    "TCN",
                                    "CNNLSTM",
                                    "LSTMAutoEncoder"])

        exp.loss_comparison(save=False, show=False)
        exp.compare_errors('r2', data=data, save=False, show=False)

        return

    def test_rgr_optimize(self):

        exp = DLRegressionExperiments(
            input_features = data.columns.tolist()[0:-1],
            output_features = data.columns.tolist()[-1:],
            epochs=5,
            ts_args={"lookback": 5}
        )

        exp.batch_size_space = Categorical(categories=[4, 8, 12, 16, 32],
                                           name="batch_size")

        exp.fit(data=data,
                include=["MLP", "CNN"],
                run_type="optimize",
                num_iterations=12)
        return


if __name__ == "__main__":
    unittest.main()