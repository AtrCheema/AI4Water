
import random
import unittest

import numpy as np
import matplotlib.pyplot as plt

from ai4water.datasets import busan_beach
from ai4water.utils.easy_mpl import bar_chart, BAR_CMAPS, get_cmap, imshow, hist
from ai4water.utils.visualizations import regplot


data = busan_beach()

def get_chart_data(n):
    d = np.random.randint(2, 50, n)
    return d, [f'feature_{i}' for i in d]


class TestBarChart(unittest.TestCase):

    def test_bar_h(self):
        d, names = get_chart_data(5)
        cm = get_cmap(random.choice(BAR_CMAPS), len(d), 0.2)

        plt.close('all')
        _, axis = plt.subplots()
        bar_chart(values=d, labels=names, axis=axis, color=cm, show=False)
        return

    def test_bar_v_without_axis(self):
        d, names = get_chart_data(5)
        cm = get_cmap(random.choice(BAR_CMAPS), len(d), 0.2)

        bar_chart(values=d, labels=names, color=cm, sort=True, show=False)

    def test_h_sorted(self):
        d, names = get_chart_data(5)
        cm = get_cmap(random.choice(BAR_CMAPS), len(d), 0.2)

        bar_chart(values=d, labels=names, color=cm, orient='v', show=False)
        return

    def test_vertical_without_axis(self):
        d, names = get_chart_data(5)
        cm = get_cmap(random.choice(BAR_CMAPS), len(d), 0.2)
        bar_chart(values=d, labels=names, color=cm, sort=True, orient='v', show=False)
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


class TestImshow(unittest.TestCase):

    def test_imshow(self):
        imshow(np.random.random((10, 10)), colorbar=True, show=False)
        return

class Testhist(unittest.TestCase):

    def test_hist(self):
        hist(np.random.random((10, 1)), show=False)
        return

    def test_hist_with_axes(self):
        _, ax = plt.subplots()
        hist(np.random.random((10, 1)), ax=ax, show=False)
        return

if __name__ == "__main__":
    unittest.main()