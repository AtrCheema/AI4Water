import unittest
import os
import site   # so that AI4Water directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )
from typing import Union

import pandas as pd

from AI4Water.utils.datasets import CAMELS_GB, CAMELS_BR, CAMELS_AUS, CAMELS_CL, CAMELS_US, LamaH, HYSETS, HYPE
from AI4Water.utils.datasets import WQJordan, WQJordan2, YamaguchiClimateJp, FlowBenin, HydrometricParana
from AI4Water.utils.datasets import Weisssee, RiverTempSpain, WQCantareira, RiverIsotope, EtpPcpSamoylov
from AI4Water.utils.datasets import FlowSamoylov, FlowSedDenmark, StreamTempSpain, RiverTempEroo
from AI4Water.utils.datasets import HoloceneTemp, FlowTetRiver, SedimentAmersee, HydrocarbonsGabes
from AI4Water.utils.datasets import WaterChemEcuador, WaterChemVictoriaLakes, HydroChemJava, PrecipBerlin
from AI4Water.utils.datasets import GeoChemMatane



def check_data(dataset, num_datasets=1, min_len_data=1, index_col: Union[None, str] = 'index'):
    data = dataset.fetch(index_col=index_col)
    assert len(data) == num_datasets, f'data is of length {len(data)}'

    for k,v in data.items():
        assert len(v) >= min_len_data, f'{v} if length {len(v)}'
        if index_col is not None:
            assert isinstance(v.index, pd.DatetimeIndex), f'for {k} index is of type {type(v.index)}'
    return


def test_dynamic_data(dataset, stations, target, stn_data_len):
    print(f"test_dynamic_data for {dataset.name}")
    df = dataset.fetch(stations=stations, static_attributes=None)

    assert len(df) == target, f'dataset lenth is {len(df)} while target is {target}'
    for k,v in df.items():
        assert isinstance(v, pd.DataFrame)
        # data for each station must minimum be of this length
        assert len(v) >= stn_data_len, f'{k} for {dataset.name} is of length {len(v)} and not {stn_data_len}'
    return


def test_static_data(dataset, stations, target, stn_data_len):
    print(f"test_static_data for {dataset.name}")
    df = dataset.fetch(stations=stations, dynamic_attributes=None, static_attributes='all')

    assert len(df) == target
    for k,v in df.items():
        assert any(isinstance(v, _type) for _type in [pd.Series, pd.DataFrame]), f'for {k}, v is of type {type(v)}'
        # static data for each station must be of this shape
        assert v.shape[1] == stn_data_len, f'for {k}, v is of shape {v.shape} and not of {stn_data_len}'
    return


def test_attributes(dataset, static_attr_len, dyn_attr_len, stations):
    print(f"test_attributes for {dataset.name}")
    static_attributes = dataset.static_attributes
    assert len(static_attributes) == static_attr_len
    assert isinstance(static_attributes, list)
    assert all([isinstance(i, str) for i in static_attributes])

    assert os.path.exists(dataset.ds_dir)

    dynamic_attributes = dataset.dynamic_attributes
    assert len(dynamic_attributes) == dyn_attr_len, f'length of dynamic attributes: {len(dynamic_attributes)}'
    assert isinstance(dynamic_attributes, list)
    assert all([isinstance(i, str) for i in dynamic_attributes])

    test_stations(dataset, stations)

    return


def test_stations(dataset, stations_len):
    print(f"test_stations for {dataset.name}")
    stations = dataset.stations()
    assert len(stations) == stations_len, f'number of stations for {dataset.name} are {len(stations)}'

    assert all([isinstance(i, str) for i in stations])
    return


def test_fetch_dynamic_attributes(dataset, stn_id, num_dyn_attrs):
    print(f"test_fetch_dynamic_attributes for {dataset.name}")
    df = dataset.fetch_dynamic_attributes(stn_id)
    assert df.shape[1] == num_dyn_attrs, f'for {dataset.name}, num_dyn_attributes are {df.shape[1]}'


def test_fetch_dynamic_multiple_stations(dataset, n_stns, target):
    print(f"test_fetch_dynamic_multiple_stations for {dataset.name}")
    stations = dataset.stations()
    d = dataset.fetch(stations[0:n_stns])
    for k, v in d.items():
        assert v.shape[1] == target, f'for {dataset.name} shape is {v.shape}'
    return


def test_fetch_static_attribue(dataset, stn_id, target):
    print(f"test_fetch_static_attribue for {dataset.name}")
    df = dataset.fetch(stn_id, dynamic_attributes=None, static_attributes='all')
    assert df[stn_id].shape[1] == target, f'shape is: {df[stn_id].shape}'
    return

def test_st_en_with_static_and_dynamic(dataset, station, len_dyn_attrs, len_static_attrs):
    data = dataset.fetch([station], as_ts=True, static_attributes='all', st='19880101', en='19881231')
    for k, v in data.items():
        assert isinstance(v.index, pd.DatetimeIndex)
        assert v.shape == (366, len_dyn_attrs + len_static_attrs), f'{v.shape}'

    data = dataset.fetch([station], static_attributes='all', st='19880101', en='19881231')
    for k, v in data.items():
        assert isinstance(v, dict)
        assert isinstance(v['dynamic'], pd.DataFrame)
        assert isinstance(v['static'], pd.DataFrame)
        assert len(v['dynamic']) == 366
        assert v['static'].shape[1] == len_static_attrs

    data = dataset.fetch_dynamic_attributes(station, st='19880101', en='19881231')
    assert len(data) == 366

    data = dataset.fetch([station], st='19880101', en='19881231')
    for k,v in data.items():
        assert len(v) == 366
    return

ds_br = CAMELS_BR()
ds_aus = CAMELS_AUS()
ds_cl = CAMELS_CL()
ds_us = CAMELS_US()
hy = HYSETS(path=r'D:\mytools\AI4Water\AI4Water\utils\datasets\data\HYSETS', source='ERA5')
s = hy.stations()
ds_hype = HYPE()
st = hy.fetch_static_attributes(station=s[0])
#
class TestLamaH(unittest.TestCase):

    def test_all(self):
        stations = [859, 859, 454]
        static = [61, 62, 61]
        for idx, dt in enumerate(LamaH._data_types):
            ds_eu = LamaH(time_step='daily', data_type=dt)
            test_dynamic_data(ds_eu, None, stations[idx], 14244)
            test_dynamic_data(ds_eu, 0.1, int(stations[idx]*0.1), 14244)
            test_static_data(ds_eu, None, stations[idx], static[idx])
            test_static_data(ds_eu, 0.1,  int(stations[idx]*0.1), static[idx])
            test_attributes(ds_eu, static[idx], 22, stations[idx])
            test_fetch_dynamic_attributes(ds_eu, '2', 22)
            test_fetch_dynamic_multiple_stations(ds_eu, 3, 22)
            test_fetch_static_attribue(ds_eu, '2', static[idx])
            data = ds_eu.fetch_dynamic_attributes('2', st='19880101', en='19881231')
            self.assertEqual(len(data), 366)
            data = ds_eu.fetch(['2'], static_attributes=None, st='19880101', en='19881231')
            for k,v in data.items():
                self.assertEqual(len(v), 366)


ds_gb = CAMELS_GB(path=r"D:\mytools\AI4Water\AI4Water\utils\datasets\data\CAMELS\CAMELS-GB")
class TestCamelsGB(unittest.TestCase):

    def test_all_dynamic_data(self):
        test_dynamic_data(ds_gb, None, 671, 16436)  # check that dynamic attribues from all data can be retrieved.

    def test_random_dynamic_data(self):
        test_dynamic_data(ds_gb, 0.1, 67, 16436)    # check that dynamic data of 10% of stations can be retrieved

    def test_all_static_data(self):
        test_static_data(ds_gb, None, 671, 145)  # check that static data of all stations can be retrieved

    def test_random_static_data(self):
         test_static_data(ds_gb, 0.1, 67, 145)    # check that static data of 10% of stations can be retrieved

    def test_attributes(self):
        test_attributes(ds_gb, 145, 10, 671)  # check length of static attribute categories
        return

    def test_single_station_dynamic(self):
        test_fetch_dynamic_attributes(ds_gb, '97002', 10)  # make sure dynamic data from one station have 10 attributes

    def test_multiple_station_dynamic(self):
        test_fetch_dynamic_multiple_stations(ds_gb, 3, 10)  # make sure that dynamic data from 3 stations each have 10 attributes

    def test_single_station_static(self):
        test_fetch_static_attribue(ds_gb, '97002', 145)  # make sure that static data from one station can be retrieved

    def test_st_en(self):
        test_st_en_with_static_and_dynamic(ds_gb, '97002', 10, 145)
        return


class TestHYPE(unittest.TestCase):

    def test_all_dynamic_data(self):
        test_dynamic_data(ds_hype, None, 564, 12783)  # check that dynamic attribues from all data can be retrieved.

    def test_random_dynamic_data(self):
        test_dynamic_data(ds_hype, 0.1, 56, 12783)    # check that dynamic data of 10% of stations can be retrieved

    # def test_all_static_data(self):
    #     test_static_data(ds_hype, None, 564, 68)  # check that static data of all stations can be retrieved
    #
    # def test_random_static_data(self):
    #     test_static_data(ds_br, 0.1, 56, 68)    # check that static data of 10% of stations can be retrieved

    # def test_static_attributes(self):
    #     test_static_attributes(ds_br, 9)  # check length of static attribute categories

    def test_stations(self):
        test_stations(ds_hype, 564)  # check length of stations

    def test_fetch_dynamic_attributes(self):
        test_fetch_dynamic_attributes(ds_hype, '5', 9)  # make sure dynamic data from one station have 17 attributes

    def test_fetch_dynamic_multiple_stations(self):
        test_fetch_dynamic_multiple_stations(ds_hype, 3, 9)  # make sure that dynamic data from 3 stations each have 17 attributes

    # def test_fetch_static_attribue(self):
    #     test_fetch_static_attribue(ds_br, '64620000', 68)  # make sure that static data from one station can be retrieved

    def test_st_en(self):
        data = ds_hype.fetch_dynamic_attributes('12', st='19880101', en='19881231')
        self.assertEqual(len(data), 366)
        data = ds_hype.fetch(['12'], categories=None, st='19880101', en='19881231')
        for k,v in data.items():
            self.assertEqual(len(v), 366)

    def test_st_en_with_static_and_dynamic(self):
        #data = ds_hype.fetch(['64620000'], as_ts=True, st='19880101', en='19881231')
        #for k,v in data.items():
        #    assert isinstance(v.index, pd.DatetimeIndex)
        #    assert v.shape == (366, 85)
        return


class TestCamelsBR(unittest.TestCase):

    def test_all_dynamic_data(self):
        test_dynamic_data(ds_br, None, 593, 14245)  # check that dynamic attribues from all data can be retrieved.

    def test_random_dynamic_data(self):
        test_dynamic_data(ds_br, 0.1, 59, 14245)    # check that dynamic data of 10% of stations can be retrieved

    def test_all_static_data(self):
        test_static_data(ds_br, None, 593, 67)  # check that static data of all stations can be retrieved

    def test_random_static_data(self):
        test_static_data(ds_br, 0.1, 59, 67)    # check that static data of 10% of stations can be retrieved

    def test_attributes(self):
        test_attributes(ds_br, 67, 12, 593)  # check length of static attribute categories
        return

    def test_fetch_dynamic_attributes(self):
        test_fetch_dynamic_attributes(ds_br, '64620000', 12)  # make sure dynamic data from one station have 17 attributes

    def test_fetch_dynamic_multiple_stations(self):
        test_fetch_dynamic_multiple_stations(ds_br, 3, 12)  # make sure that dynamic data from 3 stations each have 17 attributes

    def test_fetch_static_attribue(self):
        test_fetch_static_attribue(ds_br, '64620000', 67)  # make sure that static data from one station can be retrieved

    def test_st_en_with_static_and_dynamic(self):
        test_st_en_with_static_and_dynamic(ds_br, '64620000', 12, 67)
        return


class TestCamelsAus(unittest.TestCase):

    def test_all_dynamic_data(self):
        test_dynamic_data(ds_aus, None,  222,  21184)  # check that dynamic attribues from all data can be retrieved.

    def test_random_dynamic_data(self):
        test_dynamic_data(ds_aus, 0.1,  22,  21184)    # check that dynamic data of 10% of stations can be retrieved

    def test_all_static_data(self):
        test_static_data(ds_aus, None,  222,  110)  # check that static data of all stations can be retrieved

    def test_random_static_data(self):
        test_static_data(ds_aus, 0.1,  22,  110)    # check that static data of 10% of stations can be retrieved

    def test_attributes(self):
        test_attributes(ds_aus, 110, 26, 222)  # check length of static attribute categories
        return

    def test_fetch_dynamic_attributes(self):
        test_fetch_dynamic_attributes(ds_aus, '912101A', 26)  # make sure dynamic data from one station have 26 attributes

    def test_fetch_dynamic_multiple_stations(self):
        test_fetch_dynamic_multiple_stations(ds_aus, 3, 26)  # make sure that dynamic data from 3 stations each have 26 attributes

    def test_fetch_static_attribue(self):
        test_fetch_static_attribue(ds_aus, '912101A', 110)  # make sure that static data from one station can be retrieved

    def test_st_en_with_static_and_dynamic(self):  # todo to slow even for one station
        test_st_en_with_static_and_dynamic(ds_aus, '912101A', 26, 110)
        return


class TestCamelsCL(unittest.TestCase):

    def test_all_dynamic_data(self):
        test_dynamic_data(ds_cl, None,  516,  38374)  # check that dynamic attribues from all data can be retrieved.

    def test_random_dynamic_data(self):
        test_dynamic_data(ds_cl, 0.1,  51,  38374)    # check that dynamic data of 10% of stations can be retrieved

    def test_all_static_data(self):
        test_static_data(ds_cl, None,  516,  104)  # check that static data of all stations can be retrieved

    def test_static_data(self):
        test_static_data(ds_cl, 0.1,  51,  104)    # check that static data of 10% of stations can be retrieved

    def test_attributes(self):
        test_attributes(ds_cl, 104, 12, 516)
        return

    def test_fetch_dynamic_attributes(self):
        test_fetch_dynamic_attributes(ds_cl, '6003001', 12)

    def test_fetch_dynamic_multiple_stations(self):
        test_fetch_dynamic_multiple_stations(ds_cl, 5, 12)

    def test_fetch_static_attribue(self):
        test_fetch_static_attribue(ds_cl, '6003001', 104)

    def test_st_en_with_static_and_dynamic(self):
        test_st_en_with_static_and_dynamic(ds_cl, '6003001', 12, 104)
        return


class TestCamelsUS(unittest.TestCase):

    def test_all_dynamic_data(self):
        test_dynamic_data(ds_us, None,  671,  12784)  # check that dynamic attribues from all data can be retrieved.

    def test_random_dynamic_data(self):
        test_dynamic_data(ds_us, 0.1,  67,  12784)    # check that dynamic data of 10% of stations can be retrieved

    def test_all_static_data(self):
        test_static_data(ds_us, None,  671,  59)  # check that static data of all stations can be retrieved

    def test_random_static_data(self):
        test_static_data(ds_us, 0.1,  67,  59)    # check that static data of 10% of stations can be retrieved

    def test_attributes(self):
        test_attributes(ds_us, 59, 8, 671)  # check length of static attribute categories
        return

    def test_fetch_dynamic_attributes(self):
        test_fetch_dynamic_attributes(ds_us, '11473900', 7)

    def test_fetch_dynamic_multiple_stations(self):
        test_fetch_dynamic_multiple_stations(ds_us, 5, 7)
#
    def test_fetch_static_attribute(self):
        test_fetch_static_attribue(ds_us, '11473900', 59)

    def test_st_en_with_static_and_dynamic(self):
        test_st_en_with_static_and_dynamic(ds_us, '11473900', 8, 59)
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



if __name__=="__main__":
    unittest.main()
    #import time
    #st = time.time()
    #df = ds_aus.fetch(categories=None)
    #print(time.time() - st)






