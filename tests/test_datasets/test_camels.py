
import os
import sys
import site   # so that AI4Water directory is in path
import random
import unittest
ai4_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
site.addsitedir(ai4_dir)


import pandas as pd
import xarray as xr

from ai4water.datasets import CAMELS_DK
from ai4water.datasets import CAMELS_GB, CAMELS_BR, CAMELS_AUS
from ai4water.datasets import CAMELS_CL, CAMELS_US, LamaH, HYSETS, HYPE
from ai4water.datasets import WaterBenchIowa


def test_dynamic_data(dataset, stations, num_stations, stn_data_len, as_dataframe=False):
    print(f"test_dynamic_data for {dataset.name}")
    df = dataset.fetch(stations=stations, static_features=None, as_dataframe=as_dataframe)

    if as_dataframe:
        check_dataframe(dataset, df, num_stations, stn_data_len)
    else:
        check_dataset(dataset, df, num_stations, stn_data_len)

    return


def test_all_data(dataset, stations, stn_data_len, as_dataframe=False):

    if len(dataset.static_features) > 0:
        df = dataset.fetch(stations, static_features='all', as_ts=False, as_dataframe=as_dataframe)
        assert df['static'].shape == (stations, len(dataset.static_features)), f"shape is {df['static'].shape}"
    else:
        df = dataset.fetch(stations, static_features=None, as_ts=False, as_dataframe=as_dataframe)
        df = {'dynamic': df}

    if as_dataframe:
        check_dataframe(dataset, df['dynamic'], stations, stn_data_len)
    else:
        check_dataset(dataset, df['dynamic'], stations, stn_data_len)

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
    print(f"testing fetch_dynamic_multiple_stations for {dataset.name}")
    stations = dataset.stations()
    data = dataset.fetch(stations[0:n_stns], as_dataframe=as_dataframe)

    if as_dataframe:
        check_dataframe(dataset, data, n_stns, stn_data_len)
    else:
        check_dataset(dataset, data, n_stns, stn_data_len)

    return


def test_fetch_static_feature(dataset, stn_id):
    print(f"testing `fetch_static_features` method for {dataset.name}")
    if len(dataset.static_features)>0:
        df = dataset.fetch(stn_id, dynamic_features=None, static_features='all')
        assert isinstance(df, pd.DataFrame)
        assert len(df.loc[stn_id, :]) == len(dataset.static_features), f'shape is: {df[stn_id].shape}'

        df = dataset.fetch_static_features(stn_id, features='all')

        assert isinstance(df, pd.DataFrame), f'fetch_static_features for {dataset.name} returned of type {df.__class__.__name__}'
        assert len(df.loc[stn_id, :]) == len(dataset.static_features), f'shape is: {df[stn_id].shape}'

    return


def test_st_en_with_static_and_dynamic(dataset, station,
                                       as_dataframe=False,
                                       yearly_steps=366):
    print(f"testing {dataset.name} with st and en with both static and dynamic")

    if len(dataset.static_features)>0:
        data = dataset.fetch([station], static_features='all',
                             st='19880101',
                             en='19881231', as_dataframe=as_dataframe)
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
    hy = HYSETS(path=r'F:\data\HYSETS')

    # because it takes very long time, we dont test with all the data
    test_dynamic_data(hy, 0.003, int(14425 * 0.003), 25202)

    test_static_data(hy, None, 14425)
    test_static_data(hy, 0.1, int(14425*0.1))

    test_all_data(hy, 3, 25202)
    test_all_data(hy, 3, 25202, True)

    test_attributes(hy, 28, 5, 14425)

    test_fetch_dynamic_features(hy, random.choice(hy.stations()))
    test_fetch_dynamic_features(hy, random.choice(hy.stations()), True)

    test_fetch_dynamic_multiple_stations(hy, 3,  25202)
    test_fetch_dynamic_multiple_stations(hy, 3, 25202, True)

    test_fetch_static_feature(hy, random.choice(hy.stations()))

    test_st_en_with_static_and_dynamic(hy, random.choice(hy.stations()), yearly_steps=366)
    test_st_en_with_static_and_dynamic(hy, random.choice(hy.stations()), True, yearly_steps=366)

    test_selected_dynamic_features(hy)

    return



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
        path = r"F:\data\CAMELS\CAMELS-GB"
        if os.path.exists(path):
            ds_gb = CAMELS_GB(path=path)
            test_dataset(ds_gb, 671, 16436, 290, 10)
        return

    def test_aus(self):
        ds_aus = CAMELS_AUS(path=r'F:\data\CAMELS\CAMELS_AUS')
        test_dataset(ds_aus, 222, 21184, 161, 26)
        return

    def test_hype(self):
        ds_hype = HYPE(path=r'F:\data\CAMELS\HYPE')
        test_dataset(ds_hype, 564, 12783, 0, 9)
        return

    def test_cl(self):
        ds_cl = CAMELS_CL(r'F:\data\CAMELS\CAMELS_CL')
        test_dataset(ds_cl, num_stations=516, dyn_data_len=38374,
                     num_static_attrs=104, num_dyn_attrs=12)
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

                    ds_eu = LamaH(time_step=ts, data_type=dt, path=r'F:\data\CAMELS\LamaH')

                    if ts =='hourly':
                        test_df=False

                    test_dataset(ds_eu, stations[ts][idx],
                                 len_dyn_data[ts], static[ts][idx], num_dyn_attrs[ts],
                                 test_df, yearly_steps=yearly_steps[ts])
        return

    def test_br(self):
        ds_br = CAMELS_BR(path=r'F:\data\CAMELS\CAMELS-BR')
        test_dataset(ds_br, 593, 14245, 67, 12)
        return

    def test_us(self):
        ds_us = CAMELS_US(path=r'F:\data\CAMELS\CAMELS_US')
        test_dataset(ds_us, 671, 12784, 59, 8)
        return

    def test_dk(self):
        ds_us = CAMELS_DK(path="F:\\data\\Denmark")
        test_dataset(ds_us, 308, 14609, 211, 39)
        return

    def test_hysets(self):
        test_hysets()
        return

    def test_waterbenchiowa(self):

        dataset = WaterBenchIowa(path=r'F:\data\CAMELS\WaterBenchIowa')

        data = dataset.fetch(static_features=None)
        assert len(data) == 125
        for k, v in data.items():
            assert v.shape == (61344, 3)

        data = dataset.fetch(5, as_dataframe=True)
        assert data.shape == (184032, 5)

        data = dataset.fetch(5, static_features="all", as_dataframe=True)
        assert data['static'].shape == (5, 7)
        data = dataset.fetch_dynamic_features('644', as_dataframe=True)
        assert data.unstack().shape == (61344, 3)

        stns = dataset.stations()
        assert len(stns) == 125

        static_data = dataset.fetch_static_features(stns)
        assert static_data.shape == (125, 7)

        static_data = dataset.fetch_static_features('592')
        assert static_data.shape == (1, 7)

        static_data = dataset.fetch_static_features(stns, ['slope', 'area'])
        assert static_data.shape == (125, 2)

        data = dataset.fetch_static_features('592', features=['slope', 'area'])
        assert data.shape == (1,2)
        return

    def test_camels_dk_docs(self):

        dataset = CAMELS_DK()

        assert len(dataset.stations()) == 308
        assert dataset.fetch_static_features(dataset.stations()).shape == (308, 211)
        assert dataset.fetch_static_features('80001').shape == (1, 211)
        assert dataset.fetch_static_features(features=['gauge_lat', 'area']).shape == (308, 2)
        assert dataset.fetch_static_features('80001', features=['gauge_lat', 'area']).shape == (1, 2)

        df = dataset.fetch(stations=0.1, as_dataframe=True)
        assert df.index.names == ['time', 'dynamic_features']
        df = dataset.fetch(stations=1, as_dataframe=True)
        assert df.unstack().shape == (14609, 39)
        assert dataset.fetch(stations='80001', as_dataframe=True).unstack().shape == (14609, 39)

        df = dataset.fetch(1, as_dataframe=True,
                           dynamic_features=['snow_depth_water_equivalent_mean', 'temperature_2m_mean',
                                             'potential_evaporation_sum', 'total_precipitation_sum',
                                             'streamflow']).unstack()
        assert df.shape == (14609, 5)
        df = dataset.fetch(10, as_dataframe=True)
        assert df.shape == (569751, 10)

        data = dataset.fetch(stations='80001', static_features="all", as_dataframe=True)
        assert data['static'].shape == (1, 211)
        assert data['dynamic'].shape == (569751, 1)
        return


if __name__=="__main__":
    unittest.main()
