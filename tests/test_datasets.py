import unittest
import os
import site   # so that dl4seq directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )
from typing import Union

import pandas as pd

from dl4seq.utils.datasets import CAMELS_GB, CAMELS_BR, CAMELS_AUS, CAMELS_CL, CAMELS_US
from dl4seq.utils.datasets import WQJordan, WQJordan2, YamaguchiClimateJp, FlowBenin, HydrometricParana
from dl4seq.utils.datasets import Weisssee, RiverTempSpain, WQCantareira, RiverIsotope, EtpPcpSamoylov
from dl4seq.utils.datasets import FlowSamoylov, FlowSedDenmark, StreamTempSpain, RiverTempEroo
from dl4seq.utils.datasets import HoloceneTemp, FlowTetRiver, SedimentAmersee, HydrocarbonsGabes
from dl4seq.utils.datasets import WaterChemEcuador, WaterChemVictoriaLakes, HydroChemJava, PrecipBerlin
from dl4seq.utils.datasets import GeoChemMatane



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
    df = dataset.fetch(stations=stations, categories=None)

    assert len(df) == target
    for k,v in df.items():
        assert isinstance(v, pd.DataFrame)
        # data for each station must minimum be of this length
        assert len(v) >= stn_data_len, f'{k} for {dataset.name} is of length {len(v)} and not {stn_data_len}'
    return


def test_static_data(dataset, stations, target, stn_data_len):
    print(f"test_static_data for {dataset.name}")
    df = dataset.fetch(stations=stations, dynamic_attributes=None)

    assert len(df) == target
    for k,v in df.items():
        assert any(isinstance(v, _type) for _type in [pd.Series, pd.DataFrame]), f'for {k}, v is of type {type(v)}'
        # static data for each station must be of this shape
        assert v.shape[1] == stn_data_len, f'for {k}, v is of shape {v.shape} and not of {stn_data_len}'
    return


def test_static_attributes(dataset, static_attr_len):
    print(f"test_static_attributes for {dataset.name}")
    static_attributes = dataset.static_attribute_categories
    assert len(static_attributes) == static_attr_len
    return


def test_stations(dataset, stations_len):
    print(f"test_stations for {dataset.name}")
    stations = dataset.stations()
    assert len(stations) == stations_len, f'number of stations for {dataset.name} are {len(stations)}'
    return


def test_fetch_dynamic_attributes(dataset, stn_id, num_dyn_attrs):
    print(f"test_fetch_dynamic_attributes for {dataset.name}")
    df = dataset.fetch_dynamic_attributes(stn_id)
    assert df.shape[1] == num_dyn_attrs, f'for {dataset.name}, num_dyn_attributes are {df.shape[1]}'


def test_fetch_dynamic_multiple_stations(dataset, n_stns, target):
    print(f"test_fetch_dynamic_multiple_stations for {dataset.name}")
    stations = dataset.stations()
    d = dataset.fetch(stations[0:n_stns], categories=None)
    for k, v in d.items():
        assert v.shape[1] == target, f'for {dataset.name} shape is {v.shape}'
    return


def test_fetch_static_attribue(dataset, stn_id, target):
    print(f"test_fetch_static_attribue for {dataset.name}")
    df = dataset.fetch(stn_id, dynamic_attributes=None, categories='all')
    assert df[stn_id].shape[1] == target, f'shape is: {df[stn_id].shape}'
    return


ds_br = CAMELS_BR()
ds_aus = CAMELS_AUS()
ds_cl = CAMELS_CL()
ds_us = CAMELS_US()


# ds_gb = CAMELS_GB(path=r"D:\mytools\dl4seq\dl4seq\utils\datasets\CAMELS\CAMELS-GB")
# class TestCamelsGB(unittest.TestCase):
#
#     def test_all_dynamic_data(self):
#         test_dynamic_data(ds_gb, None, 671, 16436)  # check that dynamic attribues from all data can be retrieved.
#
#     def test_random_dynamic_data(self):
#         test_dynamic_data(ds_gb, 0.1, 67, 16436)    # check that dynamic data of 10% of stations can be retrieved
#
#     def test_all_static_data(self):
#         test_static_data(ds_gb, None, 671, 153)  # check that static data of all stations can be retrieved
#
#     def test_random_static_data(self):
#         test_static_data(ds_gb, 0.1, 67, 153)    # check that static data of 10% of stations can be retrieved
#
#     def test_static_attributes(self):
#         test_static_attributes(ds_gb, 8)  # check length of static attribute categories
#
#     def test_stations(self):
#         test_stations(ds_gb, 671)  # check length of stations
#
#     def test_single_station_dynamic(self):
#         test_fetch_dynamic_attributes(ds_gb, '97002', 10)  # make sure dynamic data from one station have 10 attributes
#
#     def test_multiple_station_dynamic(self):
#         test_fetch_dynamic_multiple_stations(ds_gb, 3, 10)  # make sure that dynamic data from 3 stations each have 10 attributes
#
#     def test_single_station_static(self):
#         test_fetch_static_attribue(ds_gb, '97002', 153)  # make sure that static data from one station can be retrieved
#
#     def test_st_en(self):
#         data = ds_gb.fetch_dynamic_attributes('97002', st='19880101', en='19881231')
#         self.assertEqual(len(data), 366)
#         data = ds_gb.fetch(['97002'], categories=None, st='19880101', en='19881231')
#         for k,v in data.items():
#             self.assertEqual(len(v), 366)


class TestCamelsBR(unittest.TestCase):

    def test_all_dynamic_data(self):
        test_dynamic_data(ds_br, None, 593, 14245)  # check that dynamic attribues from all data can be retrieved.

    def test_random_dynamic_data(self):
        test_dynamic_data(ds_br, 0.1, 59, 14245)    # check that dynamic data of 10% of stations can be retrieved

    def test_all_static_data(self):
        test_static_data(ds_br, None, 593, 68)  # check that static data of all stations can be retrieved

    def test_random_static_data(self):
        test_static_data(ds_br, 0.1, 59, 68)    # check that static data of 10% of stations can be retrieved

    def test_static_attributes(self):
        test_static_attributes(ds_br, 9)  # check length of static attribute categories

    def test_stations(self):
        test_stations(ds_br, 593)  # check length of stations

    def test_fetch_dynamic_attributes(self):
        test_fetch_dynamic_attributes(ds_br, '64620000', 17)  # make sure dynamic data from one station have 10 attributes

    def test_fetch_dynamic_multiple_stations(self):
        test_fetch_dynamic_multiple_stations(ds_br, 3, 17)  # make sure that dynamic data from 3 stations each have 10 attributes

    def test_fetch_static_attribue(self):
        test_fetch_static_attribue(ds_br, '64620000', 68)  # make sure that static data from one station can be retrieved

    def test_st_en(self):
        data = ds_br.fetch_dynamic_attributes('64620000', st='19880101', en='19881231')
        self.assertEqual(len(data), 366)
        data = ds_br.fetch(['64620000'], categories=None, st='19880101', en='19881231')
        for k,v in data.items():
            self.assertEqual(len(v), 366)

    def test_st_en_with_static_and_dynamic(self):
        data = ds_br.fetch(['64620000'], as_ts=True, st='19880101', en='19881231')
        for k,v in data.items():
            assert isinstance(v.index, pd.DatetimeIndex)
            assert v.shape == (366, 85)
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

    def test_static_attributes(self):
        test_static_attributes(ds_aus, 5)  # check length of static attribute categories

    def test_stations(self):
        test_stations(ds_aus, 222)  # check length of stations

    def test_fetch_dynamic_attributes(self):
        test_fetch_dynamic_attributes(ds_aus, '912101A', 23)  # make sure dynamic data from one station have 23 attributes

    def test_fetch_dynamic_multiple_stations(self):
        test_fetch_dynamic_multiple_stations(ds_aus, 3, 23)  # make sure that dynamic data from 3 stations each have 23 attributes

    def test_fetch_static_attribue(self):
        test_fetch_static_attribue(ds_aus, '912101A', 110)  # make sure that static data from one station can be retrieved

    def test_st_en(self):
        data = ds_aus.fetch_dynamic_attributes('912101A', st='19880101', en='19881231')
        self.assertEqual(len(data), 366)
        data = ds_aus.fetch(['912101A'], st='19880101', en='19881231')
        for k,v in data.items():
            self.assertEqual(len(v), 366)

    def test_st_en_with_static_and_dynamic(self):
        data = ds_aus.fetch(['912101A'], as_ts=True, st='19880101', en='19881231')
        for k,v in data.items():
            assert isinstance(v.index, pd.DatetimeIndex)
            assert v.shape == (366, 133)
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

    def test_stations(self):
        test_stations(ds_cl, 516)  # check length of stations

    def test_static_attributes(self):
        test_static_attributes(ds_cl, 104)

    def test_fetch_dynamic_attributes(self):
        test_fetch_dynamic_attributes(ds_cl, '6003001', 12)

    def test_fetch_dynamic_multiple_stations(self):
        test_fetch_dynamic_multiple_stations(ds_cl, 5, 12)

    def test_fetch_static_attribue(self):
        test_fetch_static_attribue(ds_cl, '6003001', 104)

    def test_st_en(self):
        data = ds_cl.fetch_dynamic_attributes('6003001', st='19880101', en='19881231')
        self.assertEqual(len(data), 366)
        data = ds_cl.fetch(['6003001'], st='19880101', en='19881231')
        for k,v in data.items():
            self.assertEqual(len(v), 366)


class TestCamelsUS(unittest.TestCase):

    def test_all_dynamic_data(self):
        test_dynamic_data(ds_us, None,  674,  12784)  # check that dynamic attribues from all data can be retrieved.

    def test_random_dynamic_data(self):
        test_dynamic_data(ds_us, 0.1,  67,  12784)    # check that dynamic data of 10% of stations can be retrieved

    def test_stations(self):
        test_stations(ds_us, 674)  # check length of stations

    def test_fetch_dynamic_attributes(self):
        test_fetch_dynamic_attributes(ds_us, '11473900', 7)

    def test_fetch_dynamic_multiple_stations(self):
        test_fetch_dynamic_multiple_stations(ds_us, 5, 7)

    def test_st_en(self):
        data = ds_us.fetch_dynamic_attributes('11473900', st='19880101', en='19881231')
        self.assertEqual(len(data), 366)
        data = ds_us.fetch(['11473900'], st='19880101', en='19881231')
        for k,v in data.items():
            self.assertEqual(len(v), 366)

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






