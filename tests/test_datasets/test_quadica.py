
import unittest

from ai4water.datasets import Quadica


class TestQuadica(unittest.TestCase):

    dataset = Quadica()

    def test_avg_temp(self):
        assert self.dataset.avg_temp().shape == (828, 1386)
        return

    def test_pet(self):
        assert self.dataset.pet().shape == (828, 1386)
        return

    def test_precipitation(self):
        assert self.dataset.precipitation().shape == (828, 1386)
        return

    def test_monthly_medians(self):

        assert self.dataset.monthly_medians().shape == (16629, 18)
        return

    def test_wrtds_monthly(self):
        assert self.dataset.wrtds_monthly().shape == (50186, 47)
        return

    def test_catchment_attrs(self):
        assert self.dataset.catchment_attributes().shape == (1386, 113)
        assert self.dataset.catchment_attributes(stations=[1,2,3]).shape == (3, 113)
        return

    def test_fetch_monthly(self):
        dyn, cat = self.dataset.fetch_monthly(max_nan_tol=None)
        assert dyn.shape == (29484, 33)
        assert cat.shape == (29484, 113)
        mon_dyn_tn, mon_cat_tn = self.dataset.fetch_monthly(features="TN", max_nan_tol=0)
        assert mon_dyn_tn.shape == (6300, 9)
        assert mon_cat_tn.shape == (6300, 113)
        mon_dyn_tp, mon_cat_tp = self.dataset.fetch_monthly(features="TP", max_nan_tol=0)
        assert mon_dyn_tp.shape == (21420, 9)
        assert mon_cat_tp.shape == (21420, 113)
        mon_dyn_toc, mon_cat_toc = self.dataset.fetch_monthly(features="TOC", max_nan_tol=0)
        assert mon_dyn_toc.shape == (5796, 9)
        assert mon_cat_toc.shape == (5796, 113)
        mon_dyn_doc, mon_cat_doc = self.dataset.fetch_monthly(features="DOC", max_nan_tol=0)
        assert mon_dyn_doc.shape == (6804, 9)
        assert mon_cat_doc.shape == (6804, 113)

        return


if __name__=="__main__":
    unittest.main()