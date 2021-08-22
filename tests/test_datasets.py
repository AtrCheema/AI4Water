import unittest
import os
import random
import site   # so that AI4Water directory is in path
site.addsitedir(os.path.dirname(os.path.dirname(__file__)) )
from typing import Union

import pandas as pd
import xarray as xr

from ai4water.utils.datasets import CAMELS_GB, CAMELS_BR, CAMELS_AUS, CAMELS_CL, CAMELS_US, LamaH, HYSETS, HYPE
from ai4water.utils.datasets import WQJordan, WQJordan2, YamaguchiClimateJp, FlowBenin, HydrometricParana
from ai4water.utils.datasets import Weisssee, RiverTempSpain, WQCantareira, RiverIsotope, EtpPcpSamoylov
from ai4water.utils.datasets import FlowSamoylov, FlowSedDenmark, StreamTempSpain, RiverTempEroo
from ai4water.utils.datasets import HoloceneTemp, FlowTetRiver, SedimentAmersee, HydrocarbonsGabes
from ai4water.utils.datasets import WaterChemEcuador, WaterChemVictoriaLakes, HydroChemJava, PrecipBerlin
from ai4water.utils.datasets import GeoChemMatane



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
    df = dataset.fetch(stations=stations, static_features=None, as_dataframe=as_dataframe)

    if as_dataframe:
        check_dataframe(dataset, df, num_stations, stn_data_len)
    else:
        check_dataset(dataset, df, num_stations, stn_data_len)

    return

def test_all_data(dataset, stations, stn_data_len, as_dataframe=False):

    df = dataset.fetch(stations, static_features='all', as_ts=False, as_dataframe=as_dataframe)

    if as_dataframe:
        check_dataframe(dataset, df['dynamic'], stations, stn_data_len)
    else:
        check_dataset(dataset, df['dynamic'], stations, stn_data_len)

    assert df['static'].shape == (stations, len(dataset.static_features)), f"shape is {df['static'].shape}"

    return

def check_dataframe(dataset, df, num_stations, data_len):
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == num_stations, f'dataset lenth is {df.shape[1]} while target is {num_stations}'
    for col in df.columns:
        #     for dyn_attr in dataset.dynamic_features:
        #         stn_data = df[col]  # (stn_data_len*dynamic_features, )
        #         _stn_data_len = len(stn_data.iloc[stn_data.index.get_level_values('dynamic_features') == dyn_attr])
        #         assert _stn_data_len>=stn_data_len, f"{col} for {dataset.name} is not of length {stn_data_len}"
        stn_data = df[col].unstack()
        # data for each station must minimum be of this shape
        assert stn_data.shape == (data_len, len(dataset.dynamic_features)), f"""
            for {col} station of {dataset.name} the shape is {stn_data.shape}"""
    return


def check_dataset(dataset, xds, num_stations, data_len):
    assert isinstance(xds, xr.Dataset), f'xds is of type {xds.__class__.__name__}'
    assert len(xds.data_vars) == num_stations, f'for {dataset.name}, {len(xds.data_vars)} data_vars are present'
    for var in xds.data_vars:
        assert xds[var].data.shape == (data_len, len(dataset.dynamic_features)), f"""shape of data is 
        {xds[var].data.shape} and not {data_len, len(dataset.dynamic_features)}"""

    for dyn_attr in xds.coords['dynamic_features'].data:
        assert dyn_attr in dataset.dynamic_features
    return


def test_static_data(dataset, stations, target):
    print(f"test_static_data for {dataset.name}")
    if len(dataset.static_features)>0:
        df = dataset.fetch(stations=stations, dynamic_features=None, static_features='all')
        assert isinstance(df, pd.DataFrame)
        assert len(df) == target, f'length of data is {len(df)} and not {target}'
        assert df.shape == (target, len(dataset.static_features)), f'for {dataset.name}, v is of shape {df.shape} and not of {len(dataset.static_features)}'

    return


def test_attributes(dataset, static_attr_len, dyn_attr_len, stations):
    print(f"test_attributes for {dataset.name}")
    static_features = dataset.static_features
    assert len(static_features) == static_attr_len, f'for {dataset.name} static_features are {len(static_features)} and not {static_attr_len}'
    assert isinstance(static_features, list)
    assert all([isinstance(i, str) for i in static_features])

    assert os.path.exists(dataset.ds_dir)

    dynamic_features = dataset.dynamic_features
    assert len(dynamic_features) == dyn_attr_len, f'length of dynamic attributes: {len(dynamic_features)}'
    assert isinstance(dynamic_features, list)
    assert all([isinstance(i, str) for i in dynamic_features])

    test_stations(dataset, stations)

    return


def test_stations(dataset, stations_len):
    print(f"test_stations for {dataset.name}")
    stations = dataset.stations()
    assert len(stations) == stations_len, f'number of stations for {dataset.name} are {len(stations)}'

    for stn in stations:
        assert isinstance(stn, str)

    assert all([isinstance(i, str) for i in stations])
    return


def test_fetch_dynamic_features(dataset, stn_id, as_dataframe=False):
    print(f"test_fetch_dynamic_features for {dataset.name}")
    df = dataset.fetch_dynamic_features(stn_id, as_dataframe=as_dataframe)
    if as_dataframe:
        assert df.unstack().shape[1] == len(dataset.dynamic_features), f'for {dataset.name}, num_dyn_attributes are {df.shape[1]}'
    else:
        assert isinstance(df, xr.Dataset), f'data is of type {df.__class__.__name__}'
        assert len(df.data_vars) == 1, f'{len(df.data_vars)}'


def test_fetch_dynamic_multiple_stations(dataset, n_stns, stn_data_len, as_dataframe=False):
    print(f"test_fetch_dynamic_multiple_stations for {dataset.name}")
    stations = dataset.stations()
    data = dataset.fetch(stations[0:n_stns], as_dataframe=as_dataframe)

    if as_dataframe:
        check_dataframe(dataset, data, n_stns, stn_data_len)
    else:
        check_dataset(dataset, data, n_stns, stn_data_len)

    return


def test_fetch_static_feature(dataset, stn_id):
    print(f"test_fetch_static_attribue for {dataset.name}")
    if len(dataset.static_features)>0:
        df = dataset.fetch(stn_id, dynamic_features=None, static_features='all')
        assert isinstance(df, pd.DataFrame)
        assert len(df.loc[stn_id, :]) == len(dataset.static_features), f'shape is: {df[stn_id].shape}'

        df = dataset.fetch_static_features(stn_id, features='all')

        assert isinstance(df, pd.DataFrame), f'fetch_static_features for {dataset.name} returned of type {df.__class__.__name__}'
        assert len(df.loc[stn_id, :]) == len(dataset.static_features), f'shape is: {df[stn_id].shape}'

    return


def test_st_en_with_static_and_dynamic(dataset, station, as_dataframe=False, yearly_steps=366):
    if len(dataset.static_features)>0:
        data = dataset.fetch([station], static_features='all', st='19880101', en='19881231', as_dataframe=as_dataframe)
        if as_dataframe:
            check_dataframe(dataset, data['dynamic'], 1, yearly_steps)
        else:
            check_dataset(dataset, data['dynamic'], 1, yearly_steps)

        assert data['static'].shape == (1, len(dataset.static_features))

        data = dataset.fetch_dynamic_features(station, st='19880101', en='19881231', as_dataframe=as_dataframe)
        if as_dataframe:
            check_dataframe(dataset, data, 1, yearly_steps)
        else:
            check_dataset(dataset, data, 1, yearly_steps)
    return


def test_selected_dynamic_features(dataset):

    features = dataset.dynamic_features[0:2]
    data = dataset.fetch(dataset.stations()[0], dynamic_features=features, as_dataframe=True)
    data = data.unstack()
    assert data.shape[1] == 2
    return


def test_hysets():
    hy = HYSETS(path=r'D:\mytools\AI4Water\AI4Water\utils\datasets\data\HYSETS')
    test_static_data(hy, None, 14425)
    test_static_data(hy, 0.1, int(14425*0.1))
    test_attributes(hy, 28, 5, 14425)
    test_fetch_dynamic_features(hy, random.choice(hy.stations()))
    test_fetch_dynamic_multiple_stations(hy, 3,  25202)
    test_fetch_static_feature(hy, random.choice(hy.stations()))
    test_selected_dynamic_features(hy)
    xds1 = hy.fetch(100, static_features=None, st="2000")
    #xds2 = hy.fetch(20, static_features='all', st="2000")
    #xds3 = hy.fetch(s)
    #xds4 = hy.fetch_dynamic_features(2, st="1980")


def test_dataset(dataset, num_stations, dyn_data_len, num_static_attrs, num_dyn_attrs,
                 test_df=True, yearly_steps=366):

    # # check that dynamic attribues from all data can be retrieved.
    test_dynamic_data(dataset, None, num_stations, dyn_data_len)
    if test_df:
        test_dynamic_data(dataset, None, num_stations, dyn_data_len, as_dataframe=True)

    # check that dynamic data of 10% of stations can be retrieved
    test_dynamic_data(dataset, 0.1, int(num_stations*0.1), dyn_data_len)
    test_dynamic_data(dataset, 0.1, int(num_stations*0.1), dyn_data_len, True)

    test_static_data(dataset, None, num_stations)  # check that static data of all stations can be retrieved

    test_static_data(dataset, 0.1, int(num_stations*0.1))  # check that static data of 10% of stations can be retrieved

    test_all_data(dataset, 3, dyn_data_len)
    test_all_data(dataset, 3, dyn_data_len, True)

    # check length of static attribute categories
    test_attributes(dataset, num_static_attrs, num_dyn_attrs, num_stations)

    # make sure dynamic data from one station have num_dyn_attrs attributes
    test_fetch_dynamic_features(dataset, random.choice(dataset.stations()))
    test_fetch_dynamic_features(dataset, random.choice(dataset.stations()), True)

    # make sure that dynamic data from 3 stations each have 10 attributes
    test_fetch_dynamic_multiple_stations(dataset, 3,  dyn_data_len)
    test_fetch_dynamic_multiple_stations(dataset, 3, dyn_data_len, True)

    # make sure that static data from one station can be retrieved
    test_fetch_static_feature(dataset, random.choice(dataset.stations()))

    test_st_en_with_static_and_dynamic(dataset, random.choice(dataset.stations()), yearly_steps=yearly_steps)
    test_st_en_with_static_and_dynamic(dataset, random.choice(dataset.stations()), True, yearly_steps=yearly_steps)

    # test that selected dynamic features can be retrieved successfully
    test_selected_dynamic_features(dataset)
    return

class TestCamels(unittest.TestCase):

    def test_gb(self):
        ds_gb = CAMELS_GB(path=r"D:\mytools\AI4Water\AI4Water\utils\datasets\data\CAMELS\CAMELS-GB")
        test_dataset(ds_gb, 671, 16436, 290, 10)
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
        test_dataset(ds_cl, num_stations=516, dyn_data_len=38374, num_static_attrs=104, num_dyn_attrs=12)
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

                    test_dataset(ds_eu, stations[ts][idx],
                                 len_dyn_data[ts], static[ts][idx], num_dyn_attrs[ts],
                                 test_df, yearly_steps=yearly_steps[ts])
        return

    def test_br(self):
        ds_br = CAMELS_BR()
        test_dataset(ds_br, 593, 14245, 67, 12)
        return

    def test_us(self):
        ds_us = CAMELS_US()
        test_dataset(ds_us, 671, 12784, 59, 8)
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
