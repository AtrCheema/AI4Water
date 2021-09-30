import os
import sys
import unittest
import site   # so that AI4Water directory is in path
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)

import pandas as pd

from ai4water.datasets import WeatherJena, SWECanada, MtropicsLaos


laos = MtropicsLaos()


def test_jena_weather():
    wj = WeatherJena()
    df = wj.fetch()

    assert df.shape[0] >= 919551
    assert df.shape[1] >= 21

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


class TestMtropicsLaos(unittest.TestCase):

    def test_pcp(self):
        pcp = laos.fetch_pcp()
        assert isinstance(pcp.index, pd.DatetimeIndex)
        assert pcp.shape == (1665361, 1)
        assert pcp.index.freqstr == '6T'

    def test_weather_station_data(self):
        w = laos.fetch_weather_station_data()
        assert isinstance(w.index, pd.DatetimeIndex)
        assert w.index.freq == 'H'
        assert w.shape == (166536, 4)
        assert int(w.isna().sum().sum()) == 82114

    def test_fetch_hydro(self):
        wl, spm = laos.fetch_hydro()
        assert wl.shape == (454699, 1)
        assert isinstance(wl.index, pd.DatetimeIndex)
        assert spm.shape == (6428, 1)
        assert isinstance(spm.index, pd.DatetimeIndex)

    def test_fetch_ecoli(self):
        ecoli = laos.fetch_ecoli()
        assert ecoli.shape == (409, 1)
        assert int(ecoli.isna().sum()) == 42
        assert isinstance(ecoli.index, pd.DatetimeIndex)
        ecoli_all = laos.fetch_ecoli(features='all')
        assert ecoli_all.shape == (409, 3)
        assert int(ecoli_all.isna().sum().sum()) == 162
        assert isinstance(ecoli_all.index, pd.DatetimeIndex)

    def test_fetch_physio_chem(self):
        phy_chem = laos.fetch_physiochem('T_deg')
        assert phy_chem.shape == (411, 1)
        assert int(phy_chem.isna().sum()) == 63
        assert isinstance(phy_chem.index, pd.DatetimeIndex)
        phy_chem_all = laos.fetch_physiochem(features='all')
        assert phy_chem_all.shape == (411, 8)
        assert int(phy_chem_all.isna().sum().sum()) == 640
        assert isinstance(phy_chem_all.index, pd.DatetimeIndex)

    def test_rain_gauges(self):
        rg = laos.fetch_rain_gauges()
        assert isinstance(rg.index, pd.DatetimeIndex)
        assert rg.index.freq == 'D'
        assert rg.shape == (6939, 7)
        assert int(rg.isna().sum().sum()) == 34510

    def test_regression(self):

        df = laos.make_regression()
        assert isinstance(df.index, pd.DatetimeIndex)
        assert int(df.isna().sum().sum()) == 650483
        self.assertEqual(df.shape[-1], 8)

        return

    def test_classification(self):

        df = laos.make_classification()
        assert isinstance(df.index, pd.DatetimeIndex)
        assert int(df.isna().sum().sum()) == 650483
        s = df['Ecoli_mpn100']
        assert (s == 0).sum() == 102
        self.assertEqual((s == 1).sum(), 156)

        return

    def test_swe_canada(self):
        # test_swe_canada()
        return

    def test_jena_weather(self):
        # test_jena_weather()
        return


if __name__ == "__main__":
    unittest.main()
