import os
import sys
import site
import unittest
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

from ai4water.eda import EDA
from ai4water.datasets import MtropicsLaos, arg_beach

laos = MtropicsLaos()

pcp = laos.fetch_pcp(
    st="20110101", en="20110104"
)


class TestEDA(unittest.TestCase):

    def test_series(self):
        eda = EDA(data=pcp, save=True)
        eda()
        return

    def test_dataframe(self):
        eda = EDA(data=arg_beach(), dpi=50, save=True)
        eda()
        return

    def test_with_input_features(self):
        eda = EDA(data=arg_beach(), in_cols=arg_beach().columns.to_list()[0:-1],
              dpi=50,
              save=True)
        eda()
        return

    def test_autocorr(self):

        eda = EDA(data=arg_beach())
        eda.autocorrelation(nlags=10, show=False)
        return

    def test_partial_autocorr(self):
        eda = EDA(data=arg_beach())
        eda.partial_autocorrelation(nlags=10, show=False)
        return


if __name__ == "__main__":

    unittest.main()