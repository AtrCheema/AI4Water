import os
import sys
import site
import random
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

import unittest

import numpy as np
import matplotlib.pyplot as plt

from ai4water.datasets import arg_beach
from ai4water.utils.visualizations import PlotResults, regplot
from ai4water.utils.plotting_tools import bar_chart, BAR_CMAPS, cmap


data = arg_beach()

def get_history(keys, add_val=False):
    history = {}
    for k in keys:
        history[k] = np.random.random(10)
        if add_val:
            history[f"val_{k}"] = np.random.random(10)
    return history


def get_chart_data(n):
    d = np.random.randint(2, 50, n)
    return d, [f'feature_{i}' for i in d]


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


class TestRegplot(unittest.TestCase):

    def test_reg_plot_with_line(self):
        regplot(data['pcp3_mm'], data['pcp6_mm'], ci=None)
        return

    def test_regplot_with_line_and_ci(self):
        regplot(data['pcp3_mm'], data['pcp6_mm'])
        return
    def test_regplot_with_line_ci_and_annotation(self):
        regplot(data['pcp3_mm'], data['pcp6_mm'], annotation_key="MSE", annotation_val=0.2)

    def test_with_list_as_inputs(self):
        regplot(data['pcp3_mm'].values.tolist(), data['pcp6_mm'].values.tolist())
        return


class TestBarChart(unittest.TestCase):

    def test_bar_h(self):
        d, names = get_chart_data(5)
        cm = cmap(random.choice(BAR_CMAPS), len(d), 0.2)

        plt.close('all')
        _, axis = plt.subplots()
        bar_chart(values=d, labels=names, axis=axis, color=cm)


    def test_bar_v_without_axis(self):
        d, names = get_chart_data(5)
        cm = cmap(random.choice(BAR_CMAPS), len(d), 0.2)

        bar_chart(values=d, labels=names, color=cm, sort=True)

    def test_h_sorted(self):
        d, names = get_chart_data(5)
        cm = cmap(random.choice(BAR_CMAPS), len(d), 0.2)

        bar_chart(values=d, labels=names, color=cm, orient='v')


    def test_vertical_without_axis(self):
        d, names = get_chart_data(5)
        cm = cmap(random.choice(BAR_CMAPS), len(d), 0.2)
        bar_chart(values=d, labels=names, color=cm, sort=True, orient='v')
        return


if __name__ == "__main__":
    unittest.main()