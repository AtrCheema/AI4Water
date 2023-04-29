import unittest
import os
import sys
import site
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

import numpy as np
import matplotlib.pyplot as plt

from ai4water.utils import LossCurve


def get_history(keys, add_val=False):
    history = {}
    for k in keys:
        history[k] = np.random.random(10)
        if add_val:
            history[f"val_{k}"] = np.random.random(10)
    return history


class TestLossCurve(unittest.TestCase):
    show = False

    def test_plot_loss_1(self):
        visualizer = LossCurve(show=self.show, save=False)
        visualizer.plot_loss(get_history(['loss']))
        ax = visualizer.plot_loss(get_history(['loss'], True))
        assert isinstance(ax, plt.Axes)
        return
    
    def test_plot_1(self):
        visualizer = LossCurve(show=self.show, save=False)
        visualizer.plot(get_history(['loss']))
        ax = visualizer.plot(get_history(['loss'], True))
        assert isinstance(ax, plt.Axes)
        return

    def test_figsize(self):
        visualizer = LossCurve(show=self.show, save=False)
        visualizer.plot(get_history(['loss']), figsize=(10, 10))
        ax = visualizer.plot(get_history(['loss'], True))
        assert isinstance(ax, plt.Axes)
        return

    def test_plot_loss_2(self):
        visualizer = LossCurve(show=self.show, save=False)
        visualizer.plot(get_history(['loss', 'nse']))
        ax = visualizer.plot(get_history(['loss', 'nse'], True))
        assert isinstance(ax, plt.Axes)
        return

    def test_plot_loss_3(self):
        visualizer = LossCurve(show=self.show, save=False)
        visualizer.plot(get_history(['loss', 'nse', 'r2']))
        ax = visualizer.plot(get_history(['loss', 'nse', 'r2'], True))
        assert isinstance(ax, plt.Axes)
        return

    def test_plot_loss_4(self):
        visualizer = LossCurve(show=self.show, save=False)
        visualizer.plot(get_history(['loss', 'nse', 'r2', 'kge']))
        ax = visualizer.plot(get_history(['loss', 'nse', 'r2', 'kge'], True))
        assert isinstance(ax, plt.Axes)
        return

    def test_plot_loss_5(self):
        visualizer = LossCurve(show=self.show, save=False)
        visualizer.plot(get_history(['loss', 'nse', 'r2', 'kge', 'pbias']))
        ax = visualizer.plot(get_history(['loss', 'nse', 'r2', 'kge', 'pbias'], True))
        assert isinstance(ax, plt.Axes)
        return

    def test_plot_loss_6(self):
        visualizer = LossCurve(show=self.show, save=False)
        visualizer.plot(get_history(['loss', 'nse', 'r2', 'kge', 'pbias', 'bias']))
        ax = visualizer.plot(get_history(['loss', 'nse', 'r2', 'kge', 'pbias', 'bias'], True))
        assert isinstance(ax, plt.Axes)
        return

    def test_plot_loss_6_smoothing(self):
        visualizer = LossCurve(show=self.show, save=False)
        visualizer.plot(get_history(['loss', 'nse', 'r2', 'kge', 'pbias', 'bias']),
                             smoothing="ewma")
        ax = visualizer.plot(get_history(['loss', 'nse', 'r2', 'kge', 'pbias', 'bias'], True),
                             smoothing="ewma")
        assert isinstance(ax, plt.Axes)
        return


if __name__ == "__main__":
    unittest.main()