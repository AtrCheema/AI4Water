import unittest
import os
import sys
import site

import matplotlib
import matplotlib.pyplot as plt

ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

import numpy as np

from ai4water.postprocessing import ProcessPredictions


t = np.random.random(100)
p = np.random.random(100)
t1 = np.random.random(100)
p1 = np.random.random(100)
t_nan = t
t_nan[20] = np.nan
t_cls = np.random.randint(0, 2, (100, 1))
p_cls = np.random.randint(0, 2, (100, 1))
t_m_cls = np.random.randint(0, 4, (100, 1))
p_m_cls = np.random.randint(0, 4, (100, 1))
x = np.random.random((100, 4))
x_2d = np.random.random((100, 4, 2))
t2 = np.random.random((100, 2))
p2 = np.random.random((100, 2))


class TestProcessPrediction(unittest.TestCase):

    show = False
    save = True

    def test_rgr_1_output(self):
        pp = ProcessPredictions(mode="regression",
                                forecast_len=1,
                                output_features=['a'],
                                plots=["residual", "murphy"],
                                show=self.show,
                                save=self.save,
                                )

        pp(t, p,  inputs=x)
        return

    def test_rgr_plot_output(self):
        pp = ProcessPredictions(mode="regression",
                                forecast_len=1,
                                show=self.show,
                                save=self.save,
                                )

        ax = pp.regression_plot(t, p)
        assert isinstance(ax, plt.Axes)
        return

    def test_edf_plot_output(self):
        pp = ProcessPredictions(mode="regression",
                                forecast_len=1,
                                show=self.show,
                                save=self.save,
                                )

        ax = pp.edf_plot(t, p, for_prediction=False)
        assert isinstance(ax, plt.Axes)
        return

    def test_edf_plot_output_with_prediction(self):
        pp = ProcessPredictions(mode="regression",
                                forecast_len=1,
                                show=self.show,
                                save=self.save,
                                )

        output = pp.edf_plot(t, p)
        assert isinstance(output, list)
        assert isinstance(output[0], plt.Axes)
        assert isinstance(output[1], plt.Axes)
        return

    def test_reusability_of_edf_output(self):
        pp = ProcessPredictions(mode="regression",
                                forecast_len=1,
                                show=self.show,
                                save=self.save,
                                )

        output = pp.edf_plot(t, p)
        output[0].get_legend().remove()
        output[1].get_legend().remove()
        output = pp.edf_plot(t1, p1, marker='*', ax=output[0], pred_axes=output[1],
                             label=("Absolute Error 2", "Prediction 2"))
        assert isinstance(output, list)
        assert isinstance(output[0], plt.Axes)
        assert isinstance(output[1], plt.Axes)
        return

    def test_rgr_1_output_nan(self):
        pp = ProcessPredictions(mode="regression",
                                forecast_len=1,
                                output_features=['a'],
                                plots=["regression", "residual", "murphy"],
                                show=self.show,
                                save=self.save,
                                )

        pp(t, p,  inputs=x)
        return

    def test_rgr_1_output_2dinp(self):
        pp = ProcessPredictions(mode="regression",
                                forecast_len=1,
                                output_features=['a'],
                                plots=["residual", "murphy"],
                                show=self.show,
                                save=self.save,
                                )

        self.assertRaises(ValueError, pp, t, p,  inputs=x_2d)
        return

    def test_rgr_2_output(self):
        pp = ProcessPredictions(mode="regression",
                                forecast_len=1,
                                output_features=['a', 'b'],
                                plots=ProcessPredictions.available_plots,
                                show=self.show,
                                save=self.save,
                                )

        pp(t2, p2, inputs=x)
        return

    def test_binary(self):
        pp = ProcessPredictions(mode="classification",
                                output_features=['a'],
                                plots=ProcessPredictions.available_plots,
                                show=self.show,
                                save=self.save,
                                )

        pp(t_cls, p_cls,  inputs=x)
        return

    def test_binary_1d(self):
        pp = ProcessPredictions(mode="classification",
                                forecast_len=1,
                                output_features=['a'],
                                plots=ProcessPredictions.available_plots,
                                show=self.show,
                                save=self.save,
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
                                save=self.save,
                                )

        pp(t_cls.reshape(-1,), p_cls.reshape(-1,),  inputs=x)
        return

    def test_multiclass(self):
        """don't tell explicitly that it is multiclass"""
        pp = ProcessPredictions(mode="classification",
                                #output_features=['a', 'b'],
                                plots=ProcessPredictions.available_plots,
                                show=self.show,
                                save=self.save,
                                )

        pp(t_m_cls.reshape(-1,), p_m_cls.reshape(-1,),  inputs=x)
        return

    def test_multiclass1(self):
        """don't tell explicitly that it is multiclass"""
        pp = ProcessPredictions(mode="classification",
                                plots=ProcessPredictions.available_plots,
                                show=self.show,
                                save=self.save,
                                )

        pp(t_m_cls.reshape(-1,), p_m_cls.reshape(-1,),  inputs=x)
        return

    def test_binary_confusion(self):
        true = np.random.randint(0, 2, 100)
        pred = np.random.randint(0, 2, 100)
        proc = ProcessPredictions('classification', save=False,
                           show=False)
        img = proc.confusion_matrix(true, pred)
        assert isinstance(img, matplotlib.image.AxesImage)
        plt.close('all')
        return


if __name__ == "__main__":
    unittest.main()
