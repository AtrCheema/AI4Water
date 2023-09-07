
import unittest

import numpy as np
import matplotlib.pyplot as plt

from ai4water.utils import edf_plot
from ai4water.utils.visualizations import murphy_diagram, fdc_plot


class TestMurphyDiagrams(unittest.TestCase):

    y = np.random.randint(1, 1000, 100)
    f1 = np.random.randint(1, 1000, 100)
    f2 = np.random.randint(1, 1000, 100)

    def test_basic(self):

        ax = murphy_diagram(self.y, self.f1, self.f2, show=False)
        assert isinstance(ax, plt.Axes)
        return

    def test_basic_diff(self):
        murphy_diagram(self.y, self.f1, self.f2, plot_type="diff", show=False)
        return

    def test_raise_error(self):

        self.assertRaises(AssertionError, murphy_diagram,
                          observed=self.y,
                          predicted=self.f1,
                          reference_model="LinearRegression",
                          plot_type="diff")
        return

    def test_with_reference_model(self):
        inputs = np.random.random((100, 2))
        ax = murphy_diagram(self.y,
                            self.f1,
                            reference_model="LinearRegression",
                            inputs=inputs, plot_type="diff",
                       show=False)
        assert isinstance(ax, plt.Axes)
        return


class TestFDC(unittest.TestCase):

    def test_basic(self):
        simulated = np.random.random(100)
        observed = np.random.random(100)
        ax = fdc_plot(simulated, observed,
                       show=False)
        assert isinstance(ax, plt.Axes)
        return


class TestEDFPlot(unittest.TestCase):

    def test_nparray(self):
        ax = edf_plot(np.random.random(100), show=False)
        assert isinstance(ax, plt.Axes)
        plt.close()
        return

    def test_ax(self):
        _, ax = plt.subplots()
        ax = edf_plot(np.random.random(100), show=False, ax=ax)
        assert isinstance(ax, plt.Axes)
        plt.close()
        return

    def test_kwargs(self):
        ax = edf_plot(np.random.random(100), show=False, color='r')
        assert isinstance(ax, plt.Axes)
        plt.close()
        return


if __name__ == "__main__":
    unittest.main()