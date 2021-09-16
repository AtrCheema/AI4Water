import unittest

import numpy as np

from ai4water.utils.visualizations import PlotResults


def get_history(keys, add_val=False):
    history = {}
    for k in keys:
        history[k] = np.random.random(10)
        if add_val:
            history[f"val_{k}"] = np.random.random(10)
    return history


class TestLossCurve(unittest.TestCase):

    def test_plot_loss_1(self):
        visualizer = PlotResults()
        visualizer.plot_loss(get_history(['loss']))
        visualizer.plot_loss(get_history(['loss'], True))
        return

    def test_plot_loss_2(self):
        visualizer = PlotResults()
        visualizer.plot_loss(get_history(['loss', 'nse']))
        visualizer.plot_loss(get_history(['loss', 'nse'], True))
        return

    def test_plot_loss_3(self):
        visualizer = PlotResults()
        visualizer.plot_loss(get_history(['loss', 'nse', 'r2']))
        visualizer.plot_loss(get_history(['loss', 'nse', 'r2'], True))
        return

    def test_plot_loss_4(self):
        visualizer = PlotResults()
        visualizer.plot_loss(get_history(['loss', 'nse', 'r2', 'kge']))
        visualizer.plot_loss(get_history(['loss', 'nse', 'r2', 'kge'], True))
        return

    def test_plot_loss_5(self):
        visualizer = PlotResults()
        visualizer.plot_loss(get_history(['loss', 'nse', 'r2', 'kge', 'pbias']))
        visualizer.plot_loss(get_history(['loss', 'nse', 'r2', 'kge', 'pbias'], True))
        return

    def test_plot_loss_6(self):
        visualizer = PlotResults()
        visualizer.plot_loss(get_history(['loss', 'nse', 'r2', 'kge', 'pbias', 'bias']))
        visualizer.plot_loss(get_history(['loss', 'nse', 'r2', 'kge', 'pbias', 'bias'], True))
        return

if __name__ == "__main__":
    unittest.main()