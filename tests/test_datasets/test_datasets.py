
import unittest
from typing import Union

import pandas as pd

from ai4water.datasets import busan_beach
from ai4water.datasets import WQJordan, WQJordan2, YamaguchiClimateJp, FlowBenin
from ai4water.datasets import HydrometricParana, EtpPcpSamoylov, HydrocarbonsGabes
from ai4water.datasets import Weisssee, RiverTempSpain, WQCantareira, RiverIsotope
from ai4water.datasets import FlowSamoylov, FlowSedDenmark, StreamTempSpain
from ai4water.datasets import HoloceneTemp, FlowTetRiver, SedimentAmersee
from ai4water.datasets import PrecipBerlin, RiverTempEroo
from ai4water.datasets import WaterChemEcuador, WaterChemVictoriaLakes, HydroChemJava
from ai4water.datasets import GeoChemMatane, WeatherJena, SWECanada, gw_punjab
from ai4water.datasets import ec_removal_biochar, mg_photodegradation


def check_data(dataset, num_datasets=1,
               min_len_data=1, index_col: Union[None, str] = 'index'):
    data = dataset.fetch(index_col=index_col)
    assert len(data) == num_datasets, f'data is of length {len(data)}'

    for k, v in data.items():
        assert len(v) >= min_len_data, f'{v} if length {len(v)}'
        if index_col is not None:
            assert isinstance(v.index, pd.DatetimeIndex), f"""
            for {k} index is of type {type(v.index)}"""
    return


def test_jena_weather():
    wj = WeatherJena(path=r'F:\data\WeatherJena')
    df = wj.fetch()

    assert df.shape[0] >= 919551
    assert df.shape[1] >= 21

    #wj = WeatherJena(obs_loc='soil')
    #df = wj.fetch()
    #assert isinstance(df, pd.DataFrame)
    return


def test_swe_canada():
    swe = SWECanada(path=r'F:\data\SWECanada')

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
        dataset = Weisssee(path=r'F:\data\Weisssee')
        check_data(dataset, 21, 29)

    def test_jordanwq(self):
        dataset = WQJordan(path=r'F:\data\WQJordan')
        check_data(dataset, 1, 428)

    def test_jordanwq2(self):
        dataset = WQJordan2(path=r'F:\data\WQJordan2')
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

    def test_gw_punjab(self):
        df = gw_punjab()
        assert df.shape == (68782, 5)
        assert isinstance(df.index, pd.DatetimeIndex)
        df_lts = gw_punjab("LTS")
        assert df_lts.shape == (7546, 4), df_lts.shape
        assert isinstance(df_lts.index, pd.DatetimeIndex)

        df = gw_punjab(country="IND")
        assert df.shape == (29172, 5)
        df = gw_punjab(country="PAK")
        assert df.shape == (39610, 5)
        return

    def test_qe_biochar(self):
        data, _ = ec_removal_biochar()
        assert data.shape == (3757, 27)
        data, encoders = ec_removal_biochar(encoding="le")
        assert data.shape == (3757, 27)
        assert data.sum().sum() >= 10346311.47
        adsorbents = encoders['adsorbent'].inverse_transform(data.iloc[:, 22])
        assert len(set(adsorbents)) == 15
        pollutants = encoders['pollutant'].inverse_transform(data.iloc[:, 23])
        assert len(set(pollutants)) == 14
        ww_types = encoders['ww_type'].inverse_transform(data.iloc[:, 24])
        assert len(set(ww_types)) == 4
        adsorption_types = encoders['adsorption_type'].inverse_transform(data.iloc[:, 25])
        assert len(set(adsorption_types)) == 2
        data, encoders = ec_removal_biochar(encoding="ohe")
        assert data.shape == (3757, 58)
        adsorbents = encoders['adsorbent'].inverse_transform(data.iloc[:, 22:37].values)
        assert len(set(adsorbents)) == 15
        pollutants =  encoders['pollutant'].inverse_transform(data.iloc[:, 37:51].values)
        assert len(set(pollutants)) == 14
        ww_types = encoders['ww_type'].inverse_transform(data.iloc[:, 51:55].values)
        assert len(set(ww_types)) == 4
        adsorption_types = encoders['adsorption_type'].inverse_transform(data.iloc[:, -3:-1].values)
        assert len(set(adsorption_types)) == 2
        return

    def test_mg_photodegradation(self):
        data, *_ = mg_photodegradation()
        assert data.shape == (1200, 12)
        data, cat_enc, an_enc = mg_photodegradation(encoding="le")
        assert data.shape == (1200, 12)
        assert data.sum().sum() >= 406354.95
        cat_enc.inverse_transform(data.iloc[:, 9].values.astype(int))
        an_enc.inverse_transform(data.iloc[:, 10].values.astype(int))
        data, cat_enc, an_enc = mg_photodegradation(encoding="ohe")
        assert data.shape == (1200, 31)
        cat_enc.inverse_transform(data.iloc[:, 9:24].values)
        an_enc.inverse_transform(data.iloc[:, 24:30].values)

        return

    def test_busan(self):
        data = busan_beach()
        assert data.shape == (1446, 14)
        return


if __name__=="__main__":
    unittest.main()
