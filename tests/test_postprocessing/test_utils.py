import unittest
import os
import sys
import site
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

import numpy as np

from ai4water.postprocessing import ProcessPredictions

t = np.random.random(100)
p = np.random.random(100)
t_cls = np.random.randint(0, 2, (100, 1))
p_cls = np.random.randint(0, 2, (100, 1))
t_m_cls = np.random.randint(0, 4, (100, 1))
p_m_cls = np.random.randint(0, 4, (100, 1))
x = np.random.random((100, 4))
x_2d = np.random.random((100, 4, 2))
t2 = np.random.random((100, 2))
p2 = np.random.random((100, 2))


def get_history(keys, add_val=False):
    history = {}
    for k in keys:
        history[k] = np.random.random(10)
        if add_val:
            history[f"val_{k}"] = np.random.random(10)
    return history


class TestLossCurve(unittest.TestCase):

    def test_plot_loss_1(self):
        visualizer = ProcessPredictions(mode="regression", show=False)
        visualizer.plot_loss(get_history(['loss']))
        visualizer.plot_loss(get_history(['loss'], True))
        return

    def test_plot_loss_2(self):
        visualizer = ProcessPredictions(mode="regression", show=False)
        visualizer.plot_loss(get_history(['loss', 'nse']))
        visualizer.plot_loss(get_history(['loss', 'nse'], True))
        return

    def test_plot_loss_3(self):
        visualizer = ProcessPredictions(mode="regression", show=False)
        visualizer.plot_loss(get_history(['loss', 'nse', 'r2']))
        visualizer.plot_loss(get_history(['loss', 'nse', 'r2'], True))
        return

    def test_plot_loss_4(self):
        visualizer = ProcessPredictions(mode="regression", show=False)
        visualizer.plot_loss(get_history(['loss', 'nse', 'r2', 'kge']))
        visualizer.plot_loss(get_history(['loss', 'nse', 'r2', 'kge'], True))
        return

    def test_plot_loss_5(self):
        visualizer = ProcessPredictions(mode="regression", show=False)
        visualizer.plot_loss(get_history(['loss', 'nse', 'r2', 'kge', 'pbias']))
        visualizer.plot_loss(get_history(['loss', 'nse', 'r2', 'kge', 'pbias'], True))
        return

    def test_plot_loss_6(self):
        visualizer = ProcessPredictions(mode="regression", show=False)
        visualizer.plot_loss(get_history(['loss', 'nse', 'r2', 'kge', 'pbias', 'bias']))
        visualizer.plot_loss(get_history(['loss', 'nse', 'r2', 'kge', 'pbias', 'bias'], True))
        return


class TestProcessPrediction(unittest.TestCase):

    show = False
    def test_rgr_1_output(self):
        pp = ProcessPredictions(mode="regression",
                                forecast_len=1,
                                output_features=['a'],
                                is_multiclass=False,
                                plots=["residual", "murphy"],
                                show=self.show,
                                save=False,
                                )

        pp(t, p,  inputs=x)
        return

    def test_rgr_1_output_2dinp(self):
        pp = ProcessPredictions(mode="regression",
                                forecast_len=1,
                                output_features=['a'],
                                is_multiclass=False,
                                plots=["residual", "murphy"],
                                show=self.show,
                                save=False,
                                )

        self.assertRaises(ValueError, pp, t, p,  inputs=x_2d)
        return

    def test_rgr_2_output(self):
        pp = ProcessPredictions(mode="regression",
                                forecast_len=1,
                                output_features=['a', 'b'],
                                is_multiclass=False,
                                plots=ProcessPredictions.available_plots,
                                show=self.show,
                                save=False,
                                )

        pp(t2, p2, inputs=x)
        return

    def test_binary(self):
        pp = ProcessPredictions(mode="classification",
                                forecast_len=1,
                                output_features=['a'],
                                is_multiclass=False,
                                plots=ProcessPredictions.available_plots,
                                show=self.show,
                                save=False,
                                )

        pp(t_cls, p_cls,  inputs=x)
        return

    def test_binary_1d(self):
        pp = ProcessPredictions(mode="classification",
                                forecast_len=1,
                                output_features=['a'],
                                is_multiclass=False,
                                plots=ProcessPredictions.available_plots,
                                show=self.show,
                                save=False,
                                )

        pp(t_cls.reshape(-1,), p_cls.reshape(-1,),  inputs=x)
        return

    def test_binary_(self):
        """don't tell explicitly that it is multiclass"""
        pp = ProcessPredictions(mode="classification",
                                forecast_len=1,
                                output_features=['a'],
                                plots=ProcessPredictions.available_plots,
                                show=self.show,
                                save=False,
                                )

        pp(t_cls.reshape(-1,), p_cls.reshape(-1,),  inputs=x)
        return

    def test_multiclass(self):
        """don't tell explicitly that it is multiclass"""
        pp = ProcessPredictions(mode="classification",
                                forecast_len=1,
                                output_features=['a', 'b'],
                                plots=ProcessPredictions.available_plots,
                                show=self.show,
                                save=False,
                                )

        pp(t_m_cls.reshape(-1,), p_m_cls.reshape(-1,),  inputs=x)
        return


if __name__ == "__main__":
    unittest.main()