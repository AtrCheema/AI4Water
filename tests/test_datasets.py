import unittest
import os
import random
import site   # so that AI4Water directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )
from typing import Union

import pandas as pd
import xarray as xr

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


def test_dynamic_data(dataset, stations, num_stations, stn_data_len, as_dataframe=False):
    print(f"test_dynamic_data for {dataset.name}")
    df = dataset.fetch(stations=stations, static_attributes=None, as_dataframe=as_dataframe)

    if as_dataframe:
        check_dataframe(dataset, df, num_stations, stn_data_len)
    else:
        check_dataset(dataset, df, num_stations, stn_data_len)

    return

def test_all_data(dataset, stations, stn_data_len, as_dataframe=False):

    df = dataset.fetch(stations, static_attributes='all', as_ts=False, as_dataframe=as_dataframe)

    if as_dataframe:
        check_dataframe(dataset, df['dynamic'], stations, stn_data_len)
    else:
        check_dataset(dataset, df['dynamic'], stations, stn_data_len)

    assert df['static'].shape == (stations, len(dataset.static_attributes)), f"shape is {df['static'].shape}"

    return

def check_dataframe(dataset, df, num_stations, data_len):
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == num_stations, f'dataset lenth is {df.shape[1]} while target is {num_stations}'
    for col in df.columns:
        #     for dyn_attr in dataset.dynamic_attributes:
        #         stn_data = df[col]  # (stn_data_len*dynamic_features, )
        #         _stn_data_len = len(stn_data.iloc[stn_data.index.get_level_values('dynamic_features') == dyn_attr])
        #         assert _stn_data_len>=stn_data_len, f"{col} for {dataset.name} is not of length {stn_data_len}"
        stn_data = df[col].unstack()
        # data for each station must minimum be of this shape
        assert stn_data.shape == (data_len, len(dataset.dynamic_attributes)), f"""
            for {col} station of {dataset.name} the shape is {stn_data.shape}"""
    return


def check_dataset(dataset, xds, num_stations, data_len):
    assert isinstance(xds, xr.Dataset)
    assert len(xds.data_vars) == num_stations, f'for {dataset.name}, {len(xds.data_vars)} data_vars are present'
    for var in xds.data_vars:
        assert xds[var].data.shape == (data_len, len(dataset.dynamic_attributes)), f"""shape of data is 
        {xds[var].data.shape} and not {data_len, len(dataset.dynamic_attributes)}"""

    for dyn_attr in xds.coords['dynamic_features'].data:
        assert dyn_attr in dataset.dynamic_attributes
    return


def test_static_data(dataset, stations, target):
    print(f"test_static_data for {dataset.name}")
    df = dataset.fetch(stations=stations, dynamic_attributes=None, static_attributes='all')

    assert len(df) == target, f'length of data is {len(df)} and not {target}'
    assert df.shape == (target, len(dataset.static_attributes)), f'for {dataset.name}, v is of shape {df.shape} and not of {len(dataset.static_attributes)}'

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


def test_fetch_dynamic_attributes(dataset, stn_id, as_dataframe=False):
    print(f"test_fetch_dynamic_attributes for {dataset.name}")
    df = dataset.fetch_dynamic_attributes(stn_id, as_dataframe=as_dataframe)
    if as_dataframe:
        assert df.unstack().shape[1] == len(dataset.dynamic_attributes), f'for {dataset.name}, num_dyn_attributes are {df.shape[1]}'
    else:
        assert isinstance(df, xr.Dataset)
        assert len(df.data_vars) == 1


def test_fetch_dynamic_multiple_stations(dataset, n_stns, stn_data_len, as_dataframe=False):
    print(f"test_fetch_dynamic_multiple_stations for {dataset.name}")
    stations = dataset.stations()
    data = dataset.fetch(stations[0:n_stns], as_dataframe=as_dataframe)

    if as_dataframe:
        check_dataframe(dataset, data, n_stns, stn_data_len)
    else:
        check_dataset(dataset, data, n_stns, stn_data_len)

    return


def test_fetch_static_attribue(dataset, stn_id):
    print(f"test_fetch_static_attribue for {dataset.name}")
    df = dataset.fetch(stn_id, dynamic_attributes=None, static_attributes='all')
    assert len(df.loc[stn_id, :]) == len(dataset.static_attributes), f'shape is: {df[stn_id].shape}'
    return


def test_st_en_with_static_and_dynamic(dataset, station, as_dataframe=False, yearly_steps=366):
    data = dataset.fetch([station], static_attributes='all', st='19880101', en='19881231', as_dataframe=as_dataframe)
    if as_dataframe:
        check_dataframe(dataset, data['dynamic'], 1, yearly_steps)
    else:
        check_dataset(dataset, data['dynamic'], 1, yearly_steps)

    assert data['static'].shape == (1, len(dataset.static_attributes))

    data = dataset.fetch_dynamic_attributes(station, st='19880101', en='19881231', as_dataframe=as_dataframe)
    if as_dataframe:
        check_dataframe(dataset, data, 1, yearly_steps)
    else:
        check_dataset(dataset, data, 1, yearly_steps)
    return

#
# ds_cl = CAMELS_CL()
# data = ds_cl.fetch(as_dataframe=True)
hy = HYSETS(path=r'D:\mytools\AI4Water\AI4Water\utils\datasets\data\HYSETS')

# def test_hysets():
#     s = hy.stations()
#     xds1 = hy.fetch(100, static_attributes=None, st="2000")
#     #xds2 = hy.fetch(20, static_attributes='all', st="2000")
#     #xds3 = hy.fetch(s)
#     #xds4 = hy.fetch_dynamic_attributes(2, st="1980")
#     #xds5 = hy.fetch_static_attributes(2, st="1980")

# st = hy.fetch_static_attributes(station=s[0])



def test_dataset(dataset, num_stations, dyn_data_len, num_static_attrs, num_dyn_attrs,
                 test_df=True, yearly_steps=366):

    # # check that dynamic attribues from all data can be retrieved.
    test_dynamic_data(dataset, None, num_stations, dyn_data_len)
    if test_df:
        test_dynamic_data(dataset, None, num_stations, dyn_data_len, True)

    # check that dynamic data of 10% of stations can be retrieved
    test_dynamic_data(dataset, 0.1, int(num_stations*0.1), dyn_data_len)
    test_dynamic_data(dataset, 0.1, int(num_stations*0.1), dyn_data_len, True)

    test_static_data(dataset, None, num_stations)  # check that static data of all stations can be retrieved

    test_static_data(dataset, 0.1, int(num_stations*0.1))  # check that static data of 10% of stations can be retrieved

    test_all_data(dataset, 3, dyn_data_len)
    test_all_data(dataset, 3, dyn_data_len, True)

    # check length of static attribute categories
    test_attributes(dataset, num_static_attrs, num_dyn_attrs, num_stations)

    # make sure dynamic data from one station have 10 attributes
    test_fetch_dynamic_attributes(dataset, random.choice(dataset.stations()))
    test_fetch_dynamic_attributes(dataset, random.choice(dataset.stations()), True)

    # make sure that dynamic data from 3 stations each have 10 attributes
    test_fetch_dynamic_multiple_stations(dataset, 3,  dyn_data_len)
    test_fetch_dynamic_multiple_stations(dataset, 3, dyn_data_len, True)

    # make sure that static data from one station can be retrieved
    #test_fetch_static_attribue(dataset, random.choice(dataset.stations()))

    test_st_en_with_static_and_dynamic(dataset, random.choice(dataset.stations()), yearly_steps=yearly_steps)
    test_st_en_with_static_and_dynamic(dataset, random.choice(dataset.stations()), True, yearly_steps=yearly_steps)
    return

class TestCamels(unittest.TestCase):

    def test_gb(self):
        ds_gb = CAMELS_GB(path=r"D:\mytools\AI4Water\AI4Water\utils\datasets\data\CAMELS\CAMELS-GB")
        test_dataset(ds_gb, 671, 16436, 145, 10)
        return

    def test_aus(self):
        ds_aus = CAMELS_AUS()
        test_dataset(ds_aus, 222, 21184, 110, 26)
        return

    def test_hype(self):
        ds_hype = HYPE()
        test_dataset(ds_hype, 564, 12783, 0, 9)
        return

    def test_cl(self):
        ds_cl = CAMELS_CL()
        #test_dataset(ds_cl, 516, 38374, 104, 12)
        return

    def test_lamah(self):
        stations = {'daily': [859, 859, 454], 'hourly': [859, 859, 454]}
        static = {'daily': [61, 62, 61], 'hourly': [61, 62, 61]}
        num_dyn_attrs = {'daily': 22, 'hourly': 17}
        len_dyn_data = {'daily': 14244, 'hourly': 341856}
        test_df = True
        yearly_steps = {'daily': 366, 'hourly': 8784}

        for idx, dt in enumerate(LamaH._data_types):

            for ts in LamaH.time_steps:

                if ts in ['daily']:

                    print(f'checking for {dt} at {ts} time step')

                    ds_eu = LamaH(time_step=ts, data_type=dt)

                    if ts =='hourly':
                        test_df=False

                    # test_dataset(ds_eu, stations[ts][idx], len_dyn_data[ts], static[ts][idx], num_dyn_attrs[ts],
                    #              test_df, yearly_steps=yearly_steps[ts])

        return

    def test_br(self):
        ds_br = CAMELS_BR()
        test_dataset(ds_br, 593, 14245, 67, 12)
        return

    def test_us(self):
        ds_us = CAMELS_US()
        test_dataset(ds_us, 671, 12784, 59, 8)
        return


# class TestCamelsCL(unittest.TestCase):
#
#     def test_all_dynamic_data(self):
#         test_dynamic_data(ds_cl, None,  516,  38374)  # check that dynamic attribues from all data can be retrieved.
#
#     def test_random_dynamic_data(self):
#         test_dynamic_data(ds_cl, 0.1,  51,  38374)    # check that dynamic data of 10% of stations can be retrieved
#
#     def test_all_static_data(self):
#         test_static_data(ds_cl, None,  516,  104)  # check that static data of all stations can be retrieved
#
#     def test_static_data(self):
#         test_static_data(ds_cl, 0.1,  51,  104)    # check that static data of 10% of stations can be retrieved
#
#     def test_attributes(self):
#         test_attributes(ds_cl, 104, 12, 516)
#         return
#
#     def test_fetch_dynamic_attributes(self):
#         test_fetch_dynamic_attributes(ds_cl, '6003001', 12)
#
#     def test_fetch_dynamic_multiple_stations(self):
#         test_fetch_dynamic_multiple_stations(ds_cl, 5, 12)
#
#     def test_fetch_static_attribue(self):
#         test_fetch_static_attribue(ds_cl, '6003001', 104)
#
#     def test_st_en_with_static_and_dynamic(self):
#         test_st_en_with_static_and_dynamic(ds_cl, '6003001', 12, 104)
#         return


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
