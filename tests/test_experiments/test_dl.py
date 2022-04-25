import unittest
import site
import os

cwd = os.path.dirname(os.path.abspath(__file__))
site.addsitedir(os.path.dirname(os.path.dirname(cwd)))

from ai4water.experiments import DLRegressionExperiments
from ai4water.datasets import busan_beach
from ai4water.hyperopt import Categorical

data = busan_beach()


class TestDLExeriments(unittest.TestCase):


    # def test_rgr_dry_run(self):
    #
    #     exp = DLRegressionExperiments(
    #         input_features = data.columns.tolist()[0:-1],
    #         output_features = data.columns.tolist()[-1:],
    #         epochs=50,
    #         ts_args={"lookback": 9}
    #     )
    #
    #     exp.fit(data=data, include=["TFT",
    #                                 "TCN",
    #                                 "CNNLSTM",
    #                                 "LSTMAutoEncoder"])
    #
    #     exp.loss_comparison(save=False, show=False)
    #     exp.compare_errors('r2', save=False, show=False)
    #
    #     return

    def test_rgr_optimize(self):

        exp = DLRegressionExperiments(
            input_features = data.columns.tolist()[0:-1],
            output_features = data.columns.tolist()[-1:],
            epochs=50,
            ts_args={"lookback": 5}
        )

        exp.batch_size_space = Categorical(categories=[4, 8, 12, 16, 32],
                                           name="batch_size")

        exp.fit(data=data,
                include=["MLP", "CNN"],
                run_type="optimize",
                num_iterations=12)

        exp.loss_comparison(save=False, show=False)
        exp.compare_errors('r2', save=False, show=False)
        exp.compare_convergence(save=False, show=False)

        return

if __name__ == "__main__":
    unittest.main()