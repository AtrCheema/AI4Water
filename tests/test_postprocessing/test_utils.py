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
t2 = np.random.random((100, 2))
p2 = np.random.random((100, 2))


class TestProcessPrediction(unittest.TestCase):

    def test_rgr_1_output(self):
        pp = ProcessPredictions(mode="regression",
                                forecast_len=1,
                                output_features=['a'],
                                is_multiclass=False,
                                plots=ProcessPredictions.available_plots,
                                show=False,
                                save=False,
                                )

        pp(t,p,  inputs=x)
        return

    def test_rgr_2_output(self):
        pp = ProcessPredictions(mode="regression",
                                forecast_len=1,
                                output_features=['a', 'b'],
                                is_multiclass=False,
                                plots=ProcessPredictions.available_plots,
                                show=False,
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
                                show=False,
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
                                show=False,
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
                                show=False,
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
                                show=False,
                                save=False,
                                )

        pp(t_m_cls.reshape(-1,), p_m_cls.reshape(-1,),  inputs=x)
        return


if __name__ == "__main__":
    unittest.main()