
import unittest
from typing import Union

import pandas as pd

from ai4water.datasets import WQJordan, WQJordan2, YamaguchiClimateJp, FlowBenin, HydrometricParana
from ai4water.datasets import Weisssee, RiverTempSpain, WQCantareira, RiverIsotope, EtpPcpSamoylov
from ai4water.datasets import FlowSamoylov, FlowSedDenmark, StreamTempSpain, RiverTempEroo
from ai4water.datasets import HoloceneTemp, FlowTetRiver, SedimentAmersee, HydrocarbonsGabes
from ai4water.datasets import WaterChemEcuador, WaterChemVictoriaLakes, HydroChemJava, PrecipBerlin
from ai4water.datasets import GeoChemMatane, WeatherJena, SWECanada
from ai4water.datasets import GRQA
from ai4water.datasets import Quadica


def check_data(dataset, num_datasets=1, min_len_data=1, index_col: Union[None, str] = 'index'):
    data = dataset.fetch(index_col=index_col)
    assert len(data) == num_datasets, f'data is of length {len(data)}'

    for k, v in data.items():
        assert len(v) >= min_len_data, f'{v} if length {len(v)}'
        if index_col is not None:
            assert isinstance(v.index, pd.DatetimeIndex), f'for {k} index is of type {type(v.index)}'
    return


def test_jena_weather():
    wj = WeatherJena()
    df = wj.fetch()

    assert df.shape[0] >= 919551
    assert df.shape[1] >= 21

    #wj = WeatherJena(obs_loc='soil')
    #df = wj.fetch()
    #assert isinstance(df, pd.DataFrame)
    return


def test_swe_canada():
    swe = SWECanada()

    stns = swe.stations()

    df1 = swe.fetch(1)
    assert len(df1) == 1

    df10 = swe.fetch(10, st='20110101')
    assert len(df10) == 10

    df2 = swe.fetch(0.001, st='20110101')
    assert len(df2) == 2

    df3 = swe.fetch('ALE-05AE810', st='20110101')
    assert df3['ALE-05AE810'].shape == (3500, 3)

    df4 = swe.fetch(stns[0:10], st='20110101')
    assert len(df4) == 10

    return

class TestPangaea(unittest.TestCase):

    def test_Weisssee(self):
        dataset = Weisssee()
        check_data(dataset, 21, 29)

    def test_jordanwq(self):
        dataset = WQJordan()
        check_data(dataset, 1, 428)

    def test_jordanwq2(self):
        dataset = WQJordan2()
        check_data(dataset, 1, 189)

    def test_YamaguchiClimateJp(self):
        dataset = YamaguchiClimateJp()
        check_data(dataset, 1, 877)

    def test_FlowBenin(self):
        dataset = FlowBenin()
        check_data(dataset, 4, 600)

    def test_HydrometricParana(self):
        dataset = HydrometricParana()
        check_data(dataset, 2, 1700)

    def test_RiverTempSpain(self):
        dataset = RiverTempSpain()
        check_data(dataset, 21, 400)

    def test_WQCantareira(self):
        dataset = WQCantareira()
        check_data(dataset, 1, 67)

    def test_RiverIsotope(self):
        dataset = RiverIsotope()
        check_data(dataset, 1, 398, index_col=None)

    def test_EtpPcpSamoylov(self):
        dataset = EtpPcpSamoylov()
        check_data(dataset, 1, 4214)

    def test_FlowSamoylov(self):
        dataset = FlowSamoylov()
        check_data(dataset, 1, 3292)

    def test_FlowSedDenmark(self):
        dataset = FlowSedDenmark()
        check_data(dataset, 1, 29663)

    def test_StreamTempSpain(self):
        dataset = StreamTempSpain()
        check_data(dataset, 1, 1)

    def test_RiverTempEroo(self):
        dataset = RiverTempEroo()
        check_data(dataset, 1, 138442)

    def test_HoloceneTemp(self):
        dataset = HoloceneTemp()
        check_data(dataset, 1, 1030, index_col=None)

    def test_FlowTetRiver(self):
        dataset = FlowTetRiver()
        check_data(dataset, 1, 7649)

    def test_SedimentAmersee(self):
        dataset = SedimentAmersee()
        check_data(dataset, 1, 455, index_col=None)

    def test_HydrocarbonsGabes(self):
        dataset = HydrocarbonsGabes()
        check_data(dataset, 1, 14, index_col=None)

    def test_WaterChemEcuador(self):
        dataset = WaterChemEcuador()
        check_data(dataset, 1, 10, index_col=None)

    def test_WaterChemVictoriaLakes(self):
        dataset = WaterChemVictoriaLakes()
        check_data(dataset, 1, 4, index_col=None)

    def test_HydroChemJava(self):
        dataset = HydroChemJava()
        check_data(dataset, 1, 40)

    def test_PrecipBerlin(self):
        dataset = PrecipBerlin()
        check_data(dataset, 1, 113952)

    def test_GeoChemMatane(self):
        dataset = GeoChemMatane()
        check_data(dataset, 1, 166)

    def test_swe_canada(self):
        test_swe_canada()
        return

    def test_jena_weather(self):
        test_jena_weather()
        return

    def test_gqra(self):
        ds = GRQA()
        return

class TestQuadica(unittest.TestCase):
    dataset = Quadica()

    def test_avg_temp(self):
        assert self.dataset.avg_temp().shape == (828, 1386)

    def test_pet(self):
        assert self.dataset.pet().shape == (828, 1386)

    def test_precipitation(self):
        assert self.dataset.precipitation().shape == (828, 1386)

    def test_monthly_medians(self):

        assert self.dataset.monthly_medians().shape == (16629, 18)

    def test_wrtds_monthly(self):
        assert self.dataset.wrtds_monthly().shape == (50186, 47)

    def test_catchment_attrs(self):
        assert self.dataset.catchment_attributes().shape == (1386, 113)
        assert self.dataset.catchment_attributes(stations=[1,2,3]).shape == (3, 113)

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
