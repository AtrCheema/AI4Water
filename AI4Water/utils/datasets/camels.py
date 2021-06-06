import os
import glob
import random
from typing import Union

try:
    import netCDF4
except ModuleNotFoundError:
    netCDF4 = None

import numpy as np
import pandas as pd

from .datasets import Datasets
from .utils import check_attributes, download, sanity_check

try:  # shapely may not be installed, as it may be difficult to isntall and is only needed for plotting data.
    from AI4Water.utils.spatial_utils import plot_shapefile
except ModuleNotFoundError:
    plot_shapefile = None


SEP = os.sep


def gb_message():
    link = "https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9"
    raise ValueError(f"Dwonlaoad the data from {link} and provide the directory path as dataset=Camels(data=data)")


class Camels(Datasets):

    """
    Get CAMELS dataset.
    This class first downloads the CAMELS dataset if it is not already downloaded. Then the selected attribute
    for a selected id are fetched and provided to the user using the method `fetch`.

    Attributes:
        ds_dir str/path: diretory of the dataset
        dynamic_attributes list: tells which dynamic attributes are available in this dataset
        static_attributes list: a list of static attributes.
        static_attribute_categories list: tells which kinds of static attributes
            are present in this category.

    Methods:
        stations : returns the stations for which the data (dynamic attributes)
            exists as list of strings.

        fetch : fetches all attributes (both static and dynamic type) of all
            station/gauge_ids or a speficified station. It can also be used to
            fetch all attributes of a number of stations ids either by providing
            their guage_id or  by just saying that we need data of 20 stations
            which will then be chosen randomly.

        fetch_dynamic_attributes :
            fetches speficied dynamic attributes of one specified station. If the
            dynamic attribute is not specified, all dynamic attributes will be
            fetched for the specified station. If station is not specified, the
            specified dynamic attributes will be fetched for all stations.

        fetch_static_attributes :
            works same as `fetch_dynamic_attributes` but for `static` attributes.
            Here if the `category` is not specified then static attributes of
            the specified station for all categories are returned.
        stations : returns list of stations
    """

    DATASETS = {
        'CAMELS-BR': {'url': "https://zenodo.org/record/3964745#.YA6rUxZS-Uk",
                      },
        'CAMELS-GB': {'url': gb_message},
    }

    def stations(self):
        raise NotImplementedError

    def fetch_dynamic_attributes(self, station, dynamic_attributes, **kwargs):
        raise NotImplementedError

    def fetch_static_attributes(self, station, static_attributes, st=None, en=None, as_ts=False):
        raise NotImplementedError

    @property
    def start(self):  # start of data
        raise NotImplementedError

    @property
    def end(self):  # end of data
        raise NotImplementedError

    def _check_length(self, st, en):
        if st is None:
            st = self.start
        if en is None:
            en = self.end
        return st, en

    def to_ts(self, static, st, en, as_ts=False, freq='D'):

        st, en = self._check_length(st, en)

        if as_ts:
            idx = pd.date_range(st, en, freq=freq)
            static = pd.DataFrame(np.repeat(static.values, len(idx), axis=0), index=idx, columns=static.columns)
            return static
        else:
            return static

    @property
    def camels_dir(self):
        """Directory where all camels datasets will be saved. This will under datasets directory"""
        return os.path.join(self.base_ds_dir, "CAMELS")

    @property
    def ds_dir(self):
        """Directory where a particular dataset will be saved. """
        return self._ds_dir

    @ds_dir.setter
    def ds_dir(self, x):
        if x is None:
            x = os.path.join(self.camels_dir, self.__class__.__name__)

        if not os.path.exists(x):
            os.makedirs(x)
        # sanity_check(self.name, x)
        self._ds_dir = x

    def fetch(self,
              stations: Union[str, list, int, float, None] = None,
              dynamic_attributes: Union[list, str, None] = 'all',
              static_attributes: Union[str, list, None] = None,
              st: Union[None, str] = None,
              en: Union[None, str] = None,
              **kwargs
              ) -> dict:
        """
        Fetches the attributes of one or more stations.

        Arguments:
            stations : if string, it is supposed to be a station name/gauge_id.
                If list, it will be a list of station/gauge_ids. If int, it will
                be supposed that the user want data for this number of
                stations/gauge_ids. If None (default), then attributes of all
                available stations. If float, it will be supposed that the user
                wants data of this fraction of stations.
            dynamic_attributes : If not None, then it is the attributes to be
                fetched. If None, then all available attributes are fetched
            static_attributes : list of static attributes to be fetches. None
                means no static attribute will be fetched.
            st : starting date of data to be returned. If None, the data will be
                returned from where it is available.
            en : end date of data to be returned. If None, then the data will be
                returned till the date data is available.
            kwargs : keyword arguments to read the files

        returns:
            dictionary whose keys are station/gauge_ids and values are the attributes and dataframes.

        """
        if isinstance(stations, int):
            # the user has asked to randomly provide data for some specified number of stations
            stations = random.sample(self.stations(), stations)
        elif isinstance(stations, list):
            pass
        elif isinstance(stations, str):
            stations = [stations]
        elif isinstance(stations, float):
            num_stations = int(len(self.stations()) * stations)
            stations = random.sample(self.stations(), num_stations)
        elif stations is None:
            # fetch for all stations
            stations = self.stations()
        else:
            raise TypeError(f"Unknown value provided for stations {stations}")

        return self.fetch_stations_attributes(stations, dynamic_attributes, static_attributes,
                                              st=st, en=en,
                                              **kwargs)

    def fetch_stations_attributes(self,
                                  stations: list,
                                  dynamic_attributes: Union[str, list, None] = 'all',
                                  static_attributes: Union[str, list, None] = None,
                                  **kwargs) -> dict:
        """fetches attributes of multiple stations.
        Arguments:
            stations : list of stations for which data is to be fetched.
            dynamic_attributes : list of dynamic attributes to be fetched.
                if 'all', then all dynamic attributes will be fetched.
            static_attributes : list of static attributes to be fetched.
                If `all`, then all static attributes will be fetched. If None,
                then no static attribute will be fetched.
            kwargs dict: additional keyword arguments
        """
        assert isinstance(stations, list)

        stations_attributes = {}
        for station in stations:
            stations_attributes[station] = self.fetch_station_attributes(
                station, dynamic_attributes, static_attributes, **kwargs)

        return stations_attributes

    def fetch_station_attributes(self,
                                 station: str,
                                 dynamic_attributes: Union[str, list, None] = 'all',
                                 static_attributes: Union[str, list, None] = None,
                                 as_ts: bool = False,
                                 st: Union[str, None] = None,
                                 en: Union[str, None] = None,
                                 **kwargs) -> pd.DataFrame:
        """
        Fetches attributes for one station.
        Arguments:
            station : station id/gauge id for which the data is to be fetched.
            dynamic_attributes
            static_attributes
            as_ts : whether static attributes are to be converted into a time
                series or not. If yes then the returned time series will be of
                same length as that of dynamic attribtues.
            st : starting point from which the data to be fetched. By default
                the data will be fetched from where it is available.
            en : end point of data to be fetched. By default the dat will be fetched
        Return:
            dataframe if as_ts is True else it returns a dictionary of static and
                dynamic attributes for a station/gauge_id
            """
        st, en = self._check_length(st, en)

        station_df = pd.DataFrame()
        if dynamic_attributes:
            dynamic = self.fetch_dynamic_attributes(station, dynamic_attributes, st=st, en=en, **kwargs)
            station_df = pd.concat([station_df, dynamic])

            if static_attributes is not None:
                static = self.fetch_static_attributes(station, static_attributes, st=st, en=en, as_ts=as_ts)

                if as_ts:
                    station_df = pd.concat([station_df, static], axis=1)
                else:
                    station_df ={'dynamic': station_df, 'static': static}

        elif static_attributes is not None:
            station_df = self.fetch_static_attributes(station, static_attributes, st=st, en=en, as_ts=as_ts)

        return station_df


class LamaH(Camels):
    """
    Large-Sample Data for Hydrology and Environmental Sciences for Central Europe
    from     url = "https://zenodo.org/record/4609826#.YFNp59zt02w"
    paper: https://essd.copernicus.org/preprints/essd-2021-72/
    """
    url = "https://zenodo.org/record/4609826#.YFNp59zt02w"
    _data_types = ['total_upstrm', 'diff_upstrm_all', 'diff_upstrm_lowimp']

    static_attribute_categories = ['']

    def __init__(self, *, time_step: str, data_type: str, **kwargs):
        assert time_step in ['daily', 'hourly'], f"invalid time_step {time_step} given"
        assert data_type in self._data_types
        self.time_step = time_step
        self.data_type = data_type
        super().__init__(**kwargs)

        self._download()

    """
    Arguments:
        time_step str:
        data_type str:
    """

    @property
    def dynamic_attributes(self):
        station = self.stations()[0]
        df = self.read_ts_of_station(station)
        return df.columns.to_list()

    @property
    def static_attributes(self) -> list:
        fname = os.path.join(self.data_type_dir, f'1_attributes{SEP}Catchment_attributes.csv')
        df = pd.read_csv(fname, sep=';', index_col='ID')
        return df.columns.to_list()

    @property
    def ds_dir(self):
        """Directory where a particular dataset will be saved. """
        return os.path.join(self.camels_dir, self.name)

    @property
    def data_type_dir(self):
        # self.ds_dir/CAMELS_AT/data_type_dir
        f = [f for f in os.listdir(os.path.join(self.ds_dir, 'CAMELS_AT')) if self.data_type in f][0]
        return os.path.join(self.ds_dir, f'CAMELS_AT{SEP}{f}')

    def stations(self)->list:
        # assuming file_names of the format ID_{stn_id}.csv
        _dirs = os.listdir(os.path.join(self.data_type_dir, f'2_timeseries{SEP}{self.time_step}'))
        s = [f.split('_')[1].split('.csv')[0] for f in _dirs]
        return s

    def fetch_station_attributes(self,
                                 station,
                                 dynamic_attributes='all',
                                 static_attributes='all',
                                 as_ts=False,
                                 st=None,
                                 en=None,
                                 **kwargs) ->pd.DataFrame:
        """Reads attributes of one station"""

        station_df = pd.DataFrame()

        if dynamic_attributes is not None:
            dynamic_df = self.fetch_dynamic_attributes(station, dynamic_attributes, st, en, **kwargs)
            station_df = pd.concat([station_df, dynamic_df])

        if static_attributes is not None:
            static = self.fetch_static_attributes(station, static_attributes, st=st, en=en, as_ts=as_ts)
            if as_ts:
                station_df = pd.concat([station_df, static], axis=1)

            else:
                station_df = pd.concat([station_df, static])

        return station_df

    def fetch_static_attributes(self,
                                station,
                                static_attributes=None,
                                st=None,
                                en=None,
                                as_ts=False):

        fname = os.path.join(self.data_type_dir, f'1_attributes{SEP}Catchment_attributes.csv')

        df = pd.read_csv(fname, sep=';', index_col='ID')

        if static_attributes is not None:
            static_attributes = check_attributes(static_attributes, self.static_attributes)

        df = df[static_attributes]

        return self.to_ts(df, st, en, as_ts)

    def fetch_dynamic_attributes(self,
                                 station,
                                 attributes='all',
                                 st=None,
                                 en=None,
                                 **kwargs):
        st, en = self._check_length(st, en)

        dynamic_attributes = check_attributes(attributes, self.dynamic_attributes)

        df = self.read_ts_of_station(station)

        return df[dynamic_attributes][st:en]

    def read_ts_of_station(self, station) -> pd.DataFrame:
        # read a file containing timeseries data for one station
        fname = os.path.join(self.data_type_dir, f'2_timeseries{SEP}{self.time_step}{SEP}ID_{station}.csv')
        df = pd.read_csv(fname, sep=';')
        periods = pd.PeriodIndex(year=df["YYYY"], month=df["MM"], day=df["DD"], freq="D")
        df.index = periods.to_timestamp()
        [df.pop(item) for item in ['YYYY', 'MM', 'DD']]
        return df

    @property
    def start(self):
        return "19810101"

    @property
    def end(self):
        return "20191231"


class HYSETS(Camels):
    """
    database for hydrometeorological modeling of 14,425 North American watersheds
    from 1950-2018.
    Following data_source are available.
        SNODAS_SWE:
        SCDNA:
        nonQC_stations:
        Livneh:
        ERA5:
    SWE data_types do not have tasmin and tasmax otherwise all datatypes have following dynamic_attributes
    with following shapes
        time                           (25202,)
        watershedID                    (14425,)
        drainage_area                  (14425,)
        drainage_area_GSIM             (14425,)
        flag_GSIM_boundaries           (14425,)
        flag_artificial_boundaries     (14425,)
        centroid_lat                   (14425,)
        centroid_lon                   (14425,)
        elevation                      (14425,)
        slope                          (14425,)
        discharge                      (14425, 25202)
        pr                             (14425, 25202)
        tasmax                         (14425, 25202)
        tasmin                         (14425, 25202)
    """
    doi = "https://doi.org/10.1038/s41597-020-00583-2"
    url = "https://osf.io/rpc3w/"
    DATA_SOURCES = ['ERA5', 'ERA5Land', 'ERA5Land_SWE', 'Livneh', 'nonQC_stations', 'SCDNA', 'SNODAS_SWE']
    dynamic_attributes = ['discharge', 'swe', 'tasmin', 'tasmax', 'pr']

    def __init__(self, path, source, **kwargs):

        assert source in self.DATA_SOURCES, f'source must be one of {self.DATA_SOURCES}'
        self.source = source

        super().__init__(**kwargs)

        self.ds_dir = path
        """
        Arguments:
            path str: path where all the data files are saved.
            source str: source of data
        """

    @property
    def ds_dir(self):
        return self._ds_dir

    @ds_dir.setter
    def ds_dir(self, x):
        sanity_check('HYSETS', x)
        self._ds_dir = x

    @property
    def static_attributes(self):
        df = self.read_static_data()
        return df.columns.to_list()

    def stations(self) -> list:
        return self.read_static_data().index.to_list()

    @property
    def start(self):
        return "19500101"

    @property
    def end(self):
        return "20181231"

    def fetch_dynamic_attributes(self,
                                 station,
                                 dynamic_attributes='all',
                                 st=None,
                                 en=None,
                                 **kwargs):

        st, en = self._check_length(st, en)
        attrs = check_attributes(dynamic_attributes, self.dynamic_attributes)

        nc = netCDF4.Dataset(os.path.join(self.ds_dir, f'HYSETS_2020_{self.source}.nc'))

        stn_df = pd.DataFrame(columns=attrs)

        for var in nc.variables:
            if var in attrs:
                ma = np.array(nc[var][:])
                ma[ma == nc[var]._FillValue] = np.nan
                ta = ma[station, :]  # target array of on station
                s = pd.Series(ta, index=pd.date_range(self.start, self.end, freq='D'), name=var)
                stn_df[var] = s[st:en]
        nc.close()

        return

    def fetch_static_attributes(self,
                                station,
                                static_attributes='all',
                                st=None,
                                en=None,
                                as_ts=False):

        df = self.read_static_data()

        static_attributes = check_attributes(static_attributes, self.static_attributes)

        return self.to_ts(df.loc[station][static_attributes], st=st, en=en, as_ts=as_ts)

    def read_static_data(self):
        fname = os.path.join(self.ds_dir, 'HYSETS_watershed_properties.txt')
        return pd.read_csv(fname, index_col='Watershed_ID', sep=';')


class CAMELS_US(Camels):
    """
    Downloads and processes CAMELS dataset of 671 catchments named as CAMELS
    from https://ral.ucar.edu/solutions/products/camels
    https://doi.org/10.5194/hess-19-209-2015
    """
    DATASETS = ['CAMELS_US']
    url = "https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/basin_timeseries_v1p2_metForcing_obsFlow.zip"
    catchment_attr_url = "https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/camels_attributes_v2.0.zip"

    folders = {'basin_mean_daymet': f'basin_mean_forcing{SEP}daymet',
               'basin_mean_maurer': f'basin_mean_forcing{SEP}maurer',
               'basin_mean_nldas': f'basin_mean_forcing{SEP}nldas',
               'basin_mean_v1p15_daymet': f'basin_mean_forcing{SEP}v1p15{SEP}daymet',
               'basin_mean_v1p15_nldas': f'basin_mean_forcing{SEP}v1p15{SEP}nldas',
               'elev_bands': f'elev{SEP}daymet',
               'hru': f'hru_forcing{SEP}daymet'}

    dynamic_attributes = ['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)',
                          'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)', 'Flow']

    def __init__(self, data_source='basin_mean_daymet'):

        assert data_source in self.folders, f'allwed data sources are {self.folders.keys()}'
        self.data_source = data_source

        super().__init__("CAMELS_US")

        if os.path.exists(self.ds_dir):
            print(f"dataset is already downloaded at {self.ds_dir}")
        else:
            download(self.url, os.path.join(self.camels_dir, f'CAMELS_US{SEP}CAMELS_US.zip'))
            download(self.catchment_attr_url, os.path.join(self.camels_dir, f"CAMELS_US{SEP}catchment_attrs.zip"))
            self._unzip()

        self.attr_dir = os.path.join(self.ds_dir, f'catchment_attrs{SEP}camels_attributes_v2.0')
        self.dataset_dir = os.path.join(self.ds_dir, f'CAMELS_US{SEP}basin_dataset_public_v1p2')

    @property
    def ds_dir(self):
        """Directory where a particular dataset will be saved. """
        return os.path.join(self.camels_dir, self.name)

    @property
    def start(self):
        return "19800101"

    @property
    def end(self):
        return "20141231"

    @property
    def static_attributes(self):
        static_fpath = os.path.join(self.ds_dir, 'static_attributes.csv')
        if not os.path.exists(static_fpath):
            files = glob.glob(f"{os.path.join(self.ds_dir, 'catchment_attrs', 'camels_attributes_v2.0')}/*.txt")
            cols = []
            for f in files:
                _df = pd.read_csv(f, sep=';', index_col='gauge_id', nrows=1)
                cols += list(_df.columns)
        else:
            df = pd.read_csv(static_fpath, index_col='gauge_id', nrows=1)
            cols = list(df.columns)

        return cols

    def stations(self) -> list:
        stns = []
        for _dir in os.listdir(os.path.join(self.dataset_dir, 'usgs_streamflow')):
            cat = os.path.join(self.dataset_dir, f'usgs_streamflow{SEP}{_dir}')
            stns += [fname.split('_')[0] for fname in os.listdir(cat)]

        # remove stations for which static values are not available
        for stn in ['06775500', '06846500', '09535100']:
            stns.remove(stn)

        return stns

    def fetch_dynamic_attributes(self,
                                 station,
                                 dynamic_attributes='all',
                                 st=None,
                                 en=None,
                                 **kwargs):

        st, en = self._check_length(st, en)
        attributes = check_attributes(dynamic_attributes, self.dynamic_attributes)

        assert isinstance(station, str)
        df = None
        df1 = None
        dir_name = self.folders[self.data_source]
        for cat in os.listdir(os.path.join(self.dataset_dir, dir_name)):
            cat_dirs = os.listdir(os.path.join(self.dataset_dir, f'{dir_name}{SEP}{cat}'))
            stn_file = f'{station}_lump_cida_forcing_leap.txt'
            if stn_file in cat_dirs:
                df = pd.read_csv(os.path.join(self.dataset_dir, f'{dir_name}{SEP}{cat}{SEP}{stn_file}'),
                                 sep="\s+|;|:",
                                 skiprows=4,
                                 engine='python',
                                 names=['Year', 'Mnth', 'Day', 'Hr', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m2)',
                                        'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)'],
                                 )
                df.index = pd.to_datetime(df['Year'].map(str) + '-' + df['Mnth'].map(str) + '-' + df['Day'].map(str))

        flow_dir = os.path.join(self.dataset_dir, 'usgs_streamflow')
        for cat in os.listdir(flow_dir):
            cat_dirs = os.listdir(os.path.join(flow_dir, cat))
            stn_file = f'{station}_streamflow_qc.txt'
            if stn_file in cat_dirs:
                fpath = os.path.join(flow_dir, f'{cat}{SEP}{stn_file}')
                df1 = pd.read_csv(fpath,  sep="\s+|;|:'", names=['station', 'Year', 'Month', 'Day', 'Flow', 'Flag'],
                                  engine='python')
                df1.index = pd.to_datetime(
                    df1['Year'].map(str) + '-' + df1['Month'].map(str) + '-' + df1['Day'].map(str))

        out_df = pd.concat([df[['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']],
                            df1['Flow']],
                           axis=1)
        return out_df[attributes][st:en]

    def fetch_static_attributes(self,
                                station,
                                static_attributes,
                                st=None,
                                en=None,
                                as_ts=False
                                ):
        attributes = check_attributes(static_attributes, self.static_attributes)
        st, en = self._check_length(st, en)

        static_fpath = os.path.join(self.ds_dir, 'static_attributes.csv')
        if not os.path.exists(static_fpath):
            files = glob.glob(f"{os.path.join(self.ds_dir, 'catchment_attrs', 'camels_attributes_v2.0')}/*.txt")
            static_df = pd.DataFrame()
            for f in files:
                idx = pd.read_csv(f, sep=';', usecols=['gauge_id'], dtype=str)  # index should be read as string
                _df = pd.read_csv(f, sep=';', index_col='gauge_id')
                _df.index = idx['gauge_id']
                static_df = pd.concat([static_df, _df], axis=1)
            static_df.to_csv(static_fpath, index_label='gauge_id')
        else:   # index should be read as string bcs it has 0s at the start
            idx = pd.read_csv(static_fpath, usecols=['gauge_id'], dtype=str)
            static_df = pd.read_csv(static_fpath, index_col='gauge_id')
            static_df.index = idx['gauge_id']

        return self.to_ts(pd.DataFrame(static_df.loc[station][attributes]).transpose(), st, en, as_ts)


class CAMELS_BR(Camels):
    """
    Downloads and processes CAMELS dataset of Brazil
    """
    url = "https://zenodo.org/record/3964745#.YA6rUxZS-Uk"

    folders = {'streamflow_m3s': '02_CAMELS_BR_streamflow_m3s',
               'streamflow_mm': '03_CAMELS_BR_streamflow_mm_selected_catchments',
               'simulated_streamflow_m3s': '04_CAMELS_BR_streamflow_simulated',
               'precipitation_cpc': '07_CAMELS_BR_precipitation_cpc',
               'precipitation_mswep': '06_CAMELS_BR_precipitation_mswep',
               'precipitation_chirps': '05_CAMELS_BR_precipitation_chirps',
               'evapotransp_gleam': '08_CAMELS_BR_evapotransp_gleam',
               'evapotransp_mgb': '09_CAMELS_BR_evapotransp_mgb',
               'potential_evapotransp_gleam': '10_CAMELS_BR_potential_evapotransp_gleam',
               'temperature_min': '11_CAMELS_BR_temperature_min_cpc',
               'temperature_mean': '12_CAMELS_BR_temperature_mean_cpc',
               'temperature_max': '13_CAMELS_BR_temperature_max_cpc'
               }

    def __init__(self):

        super().__init__("CAMELS-BR")

        self._download()

    @property
    def ds_dir(self):
        """Directory where a particular dataset will be saved. """
        return os.path.join(self.camels_dir, self.name)

    @property
    def _all_dirs(self):
        """All the folders in the dataset_directory"""
        return [f for f in os.listdir(self.ds_dir) if os.path.isdir(os.path.join(self.ds_dir, f))]

    @property
    def static_dir(self):
        path = None
        for _dir in self._all_dirs:
            if "attributes" in _dir:  # supposing that 'attributes' axist in only one file/folder in self.ds_dir
                path = os.path.join(self.ds_dir, f'{_dir}{SEP}{_dir}')
        return path

    @property
    def static_files(self):
        all_files = None
        if self.static_dir is not None:
            all_files = glob.glob(f"{self.static_dir}/*.txt")
        return all_files

    @property
    def dynamic_attributes(self) -> list:
        return list(CAMELS_BR.folders.keys())

    @property
    def static_attribute_categories(self):
        static_attrs = []
        for f in self.static_files:
            ff = str(os.path.basename(f).split('.txt')[0])
            static_attrs.append('_'.join(ff.split('_')[2:]))
        return static_attrs

    @property
    def static_attributes(self):
        static_fpath = os.path.join(self.ds_dir, 'static_attributes.csv')
        if not os.path.exists(static_fpath):
            files = glob.glob(f"{os.path.join(self.ds_dir, '01_CAMELS_BR_attributes','01_CAMELS_BR_attributes')}/*.txt")
            cols = []
            for f in files:
                _df = pd.read_csv(f, sep=' ', index_col='gauge_id', nrows=1)
                cols += list(_df.columns)
        else:
            df = pd.read_csv(static_fpath, index_col='gauge_id', nrows=1)
            cols = list(df.columns)

        return cols

    @property
    def start(self):
        return "19800101"

    @property
    def end(self):
        return "20181231"

    def all_stations(self, attribute) -> list:
        """Tells all station ids for which a data of a specific attribute is available."""
        all_files = []
        for _attr, _dir in self.folders.items():
            if attribute in _attr:
                all_files = os.listdir(os.path.join(self.ds_dir, f'{_dir}{SEP}{_dir}'))

        stations = []
        for f in all_files:
            stations.append(f.split('_')[0])

        return stations

    def stations(self, to_exclude=None):
        """Returns a list of station ids which are common among all dynamic attributes.
        >>>dataset = CAMELS_BR()
        >>>stations = dataset.stations()
        """
        if to_exclude is not None:
            if not isinstance(to_exclude, list):
                assert isinstance(to_exclude, str)
                to_exclude = [to_exclude]
        else:
            to_exclude = []

        stations = {}
        for dyn_attr in self.dynamic_attributes:
            if dyn_attr not in to_exclude:
                stations[dyn_attr] = self.all_stations(dyn_attr)

        return list(set.intersection(*map(set, list(stations.values()))))

    def fetch_dynamic_attributes(self,
                                 stn_id,
                                 attributes='all',
                                 st=None,
                                 en=None,
                                 **kwargs) -> pd.DataFrame:
        """
        returns the dynamic/time series attribute/attributes for one station id.
        ```python
        >>>dataset = CAMELS_BR()
        >>>pcp = dataset.fetch_dynamic_attributes('10500000', 'precipitation_cpc')
        ...# fetch all time series data associated with a station.
        >>>x = dataset.fetch_dynamic_attributes('51560000', dataset.dynamic_attributes)
        ```
        """

        st, en = self._check_length(st, en)

        if not kwargs:
            kwargs = {'sep': ' '}

        attributes = check_attributes(attributes, self.dynamic_attributes)

        data = pd.DataFrame()
        for attr, _dir in self.folders.items():

            if attr in attributes:
                path = os.path.join(self.ds_dir, f'{_dir}{SEP}{_dir}')
                # supposing that the filename starts with stn_id and has .txt extension.
                fname = [f for f in os.listdir(path) if f.startswith(str(stn_id)) and f.endswith('.txt')]
                fname = fname[0]
                if os.path.exists(os.path.join(path, fname)):
                    df = pd.read_csv(os.path.join(path, fname), **kwargs)
                    df.index = pd.to_datetime(df[['year', 'month', 'day']])
                    df.index.freq = pd.infer_freq(df.index)
                    df = df[st:en]
                    [df.pop(item) for item in ['year', 'month', 'day']]
                    data = pd.concat([data, df], axis=1)
                else:
                    raise FileNotFoundError(f"file {fname} not found at {path}")

        return data[attributes]

    def fetch_static_attributes(self,
                                stn_id: int,
                                attributes=None,
                                st=None,
                                en=None,
                                as_ts=False
                                ) -> pd.DataFrame:
        """
        Arguments:
        stn_id int:
            station id whose attribute to fetch
        attributes str/list:
            name of attribute to fetch. Default is None, which will return all the
            attributes for a particular station of the specified category.
        index_col_name str:
            name of column containing station names

        as_ts bool:

        Example:
        -------
        ```python
        >>>dataset = Camels('CAMELS-BR')
        >>>df = dataset.fetch_static_attributes(11500000, 'climate')
        ```
        """

        attributes = check_attributes(attributes, self.static_attributes)

        static_fpath = os.path.join(self.ds_dir, 'static_attributes.csv')
        if not os.path.exists(static_fpath):
            files = glob.glob(f"{os.path.join(self.ds_dir, '01_CAMELS_BR_attributes','01_CAMELS_BR_attributes')}/*.txt")
            static_df = pd.DataFrame()
            for f in files:
                _df = pd.read_csv(f, sep=' ', index_col='gauge_id')
                static_df = pd.concat([static_df, _df], axis=1)
            static_df.to_csv(static_fpath, index_label='gauge_id')
        else:
            static_df = pd.read_csv(static_fpath, index_col='gauge_id')

        return self.to_ts(pd.DataFrame(static_df.loc[int(stn_id)][attributes]).transpose(), st, en, as_ts)


class CAMELS_GB(Camels):
    """
    This dataset must be manually downloaded by the user.
    The path of the downloaded folder must be provided while initiating this class.
    """
    dynamic_attributes = ["precipitation", "pet", "temperature", "discharge_spec", "discharge_vol", "peti",
                          "humidity", "shortwave_rad", "longwave_rad", "windspeed"]

    def __init__(self, path=None):
        super().__init__(name="CAMELS-GB")
        self.ds_dir = path

    @property
    def ds_dir(self):
        """Directory where a particular dataset will be saved. """
        return self._ds_dir

    @ds_dir.setter
    def ds_dir(self, x):
        sanity_check('CAMELS-GB', x)
        self._ds_dir = x

    @property
    def static_attribute_categories(self) -> list:
        attributes = []
        path = os.path.join(self.ds_dir, 'data')
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)) and f.endswith('csv'):
                attributes.append(f.split('_')[2])

        return attributes

    @property
    def start(self):
        return "19701001"

    @property
    def end(self):
        return "20150930"

    @property
    def static_attributes(self):
        files = glob.glob(f"{os.path.join(self.ds_dir, 'data')}/*.csv")
        cols = []
        for f in files:
            if 'static_attributes.csv' not in f:
                df = pd.read_csv(f, nrows=1, index_col='gauge_id')
                cols += (list(df.columns))
        return cols

    def stations(self, to_exclude=None):
        # CAMELS_GB_hydromet_timeseries_StationID_number
        path = os.path.join(self.ds_dir, f'data{SEP}timeseries')
        gauge_ids = []
        for f in os.listdir(path):
            gauge_ids.append(f.split('_')[4])

        return gauge_ids

    def fetch_dynamic_attributes(self,
                                 stn_id,
                                 attributes='all',
                                 st=None,
                                 en=None,
                                 **kwargs) -> pd.DataFrame:
        """Fetches dynamic attribute/attributes of one station."""
        st, en = self._check_length(st, en)

        path = os.path.join(self.ds_dir, f"data{SEP}timeseries")
        fname = None
        for f in os.listdir(path):
            if stn_id in f:
                fname = f
                break

        if not kwargs:
            kwargs = {'index_col': 'date'}

        df = pd.read_csv(os.path.join(path, fname), **kwargs)
        df.index = pd.to_datetime(df.index)
        df.index.freq = pd.infer_freq(df.index)
        df = df[st:en]
        if attributes != 'all':
            return df[attributes]
        else:
            return df

    def fetch_static_attributes(self,
                                stn_id,
                                attributes='all',
                                st=None,
                                en=None,
                                as_ts=False,
                                **kwargs) -> pd.DataFrame:
        """Fetches static attributes of one station for one or more category as dataframe."""

        attributes = check_attributes(attributes, self.static_attributes)
        static_fname = 'static_attributes.csv'
        static_fpath = os.path.join(self.ds_dir, 'data', static_fname)
        if os.path.exists(static_fpath):
            static_df = pd.read_csv(static_fpath, index_col='gauge_id')
        else:
            files = glob.glob(f"{os.path.join(self.ds_dir, 'data')}/*.csv")
            static_df = pd.DataFrame()
            for f in files:
                _df = pd.read_csv(f, index_col='gauge_id')
                static_df = pd.concat([static_df, _df], axis=1)
            static_df.to_csv(static_fpath)

        return self.to_ts(pd.DataFrame(static_df.loc[int(stn_id)][attributes]).transpose(), st, en, as_ts)


class CAMELS_AUS(Camels):
    """
    Inherits from Camels class. Fetches CAMELS-AUS dataset.
    """
    url = 'https://doi.pangaea.de/10.1594/PANGAEA.921850'
    urls = {
        "01_id_name_metadata.zip": "https://download.pangaea.de/dataset/921850/files/",
        "02_location_boundary_area.zip": "https://download.pangaea.de/dataset/921850/files/",
        "03_streamflow.zip": "https://download.pangaea.de/dataset/921850/files/",
        "04_attributes.zip": "https://download.pangaea.de/dataset/921850/files/",
        "05_hydrometeorology.zip": "https://download.pangaea.de/dataset/921850/files/",
        "CAMELS_AUS_Attributes-Indices_MasterTable.csv": "https://download.pangaea.de/dataset/921850/files/",
        "Units_01_TimeseriesData.pdf": "https://download.pangaea.de/dataset/921850/files/",
        "Units_02_AttributeMasterTable.pdf": "https://download.pangaea.de/dataset/921850/files/",
    }

    folders = {
        'streamflow_MLd': f'03_streamflow{SEP}03_streamflow{SEP}streamflow_MLd',
        'streamflow_MLd_inclInfilled': f'03_streamflow{SEP}03_streamflow{SEP}streamflow_MLd_inclInfilled',
        'streamflow_mmd': f'03_streamflow{SEP}03_streamflow{SEP}streamflow_mmd',

        'et_morton_actual_SILO':  f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries{SEP}et_morton_actual_SILO',
        'et_morton_point_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries{SEP}et_morton_point_SILO',
        'et_morton_wet_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries{SEP}et_morton_wet_SILO',
        'et_short_crop_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries{SEP}et_short_crop_SILO',
        'et_tall_crop_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries{SEP}et_tall_crop_SILO',
        'evap_morton_lake_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries{SEP}evap_morton_lake_SILO',
        'evap_pan_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries{SEP}evap_pan_SILO',
        'evap_syn_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries{SEP}evap_syn_SILO',

        'precipitation_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries{SEP}precipitation_AWAP',
        'precipitation_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries{SEP}precipitation_SILO',
        'precipitation_var_SWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries{SEP}precipitation_var_AWAP',

        'solarrad_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AWAP{SEP}solarrad_AWAP',
        'tmax_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AWAP{SEP}tmax_AWAP',
        'tmin_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AWAP{SEP}tmin_AWAP',
        'vprp_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AWAP{SEP}vprp_AWAP',

        'mslp_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO{SEP}mslp_SILO',
        'radiation_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO{SEP}radiation_SILO',
        'rh_tmax_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO{SEP}rh_tmax_SILO',
        'rh_tmin_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO{SEP}rh_tmin_SILO',
        'tmax_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO{SEP}tmax_SILO',
        'tmin_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO{SEP}tmin_SILO',
        'vp_deficit_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO{SEP}vp_deficit_SILO',
        'vp_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO{SEP}vp_SILO',
    }

    def __init__(self, path=None):
        """
        Arguments:
            path: path where the CAMELS-AUS dataset has been downloaded. This path
                must contain five zip files and one xlsx file. If None, then the
                data will downloaded.
        """
        if path is not None:
            if not os.path.exists(path) or len(os.listdir(path)) < 2:
                raise FileNotFoundError(f"The path {path} does not exist")
        self.ds_dir = path

        super().__init__()
        if not os.path.exists(self.ds_dir):
            os.makedirs(self.ds_dir)

        for _file, url in self.urls.items():
            fpath = os.path.join(self.ds_dir, _file)
            if not os.path.exists(fpath):
                download(url + _file, fpath)

        self._unzip()

    @property
    def start(self):
        return "19570101"

    @property
    def end(self):
        return "20181231"

    @property
    def location(self):
        return "Australia"

    def stations(self, as_list=True):
        fname = os.path.join(self.ds_dir, f"01_id_name_metadata{SEP}01_id_name_metadata{SEP}id_name_metadata.csv")
        df = pd.read_csv(fname)
        if as_list:
            return df['station_id'].to_list()
        else:
            return df

    @property
    def static_attribute_categories(self):
        attributes = []
        path = os.path.join(self.ds_dir, f'04_attributes{SEP}04_attributes')
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)) and f.endswith('csv'):
                f = str(f.split('.csv')[0])
                attributes.append(''.join(f.split('_')[2:]))
        return attributes

    @property
    def static_attributes(self) -> list:
        static_fpath = os.path.join(self.ds_dir, 'static_attributes.csv')
        if not os.path.exists(static_fpath):
            files = glob.glob(f"{os.path.join(self.ds_dir, '04_attributes', '04_attributes')}/*.csv")
            cols = []
            for f in files:
                _df = pd.read_csv(f, index_col='station_id', nrows=1)
                cols += list(_df.columns)
        else:
            df = pd.read_csv(static_fpath, index_col='station_id', nrows=1)
            cols = list(df.columns)

        return cols

    @property
    def dynamic_attributes(self) -> list:
        return list(self.folders.keys())

    def fetch_stations_attributes(self,
                                  stations: list,
                                  dynamic_attributes='all',
                                  static_attributes=None,
                                  st=None,
                                  en=None,
                                  as_ts=False,
                                  **kwargs) -> dict:
        stns = {}
        dyn_attrs = {}

        st, en = self._check_length(st, en)

        if dynamic_attributes is not None:
            dynamic_attributes = check_attributes(dynamic_attributes, self.dynamic_attributes)

            for _attr in dynamic_attributes:
                df = self._read_df(_attr, **kwargs)
                dyn_attrs[_attr] = df

            # making one separate dataframe for one station
            for stn in stations:
                stn_df = pd.DataFrame()
                for attr, attr_df in dyn_attrs.items():
                    if attr in dynamic_attributes:
                        stn_df[attr] = attr_df[stn][st:en]
                stns[stn] = stn_df

            if static_attributes is not None:
                static = self._read_static(stations, static_attributes, st, en, as_ts=as_ts)
                for k, v in stns.items():
                    if as_ts:
                        stns[k] = pd.concat([stns[k], static], axis=1)
                    else:
                        stns[k] = {'dynamic': stns[k], 'static': static}

        elif static_attributes is not None:
            _stns = self._read_static(stations, static_attributes, st, en, as_ts=as_ts)
            for k in stations:
                stns[k] = pd.DataFrame(_stns.loc[k]).transpose()

        return stns

    def _read_static(self, stations, attributes,
                     st=None, en=None, as_ts=False):

        attributes = check_attributes(attributes, self.static_attributes)
        static_fname = 'static_attributes.csv'
        static_fpath = os.path.join(self.ds_dir, static_fname)
        if os.path.exists(static_fpath):
            static_df = pd.read_csv(static_fpath, index_col='station_id')
        else:
            files = glob.glob(f"{os.path.join(self.ds_dir, '04_attributes', '04_attributes')}/*.csv")
            static_df = pd.DataFrame()
            for f in files:
                _df = pd.read_csv(f, index_col='station_id')
                static_df = pd.concat([static_df, _df], axis=1)
            static_df.to_csv(static_fpath)

        return self.to_ts(pd.DataFrame(static_df.loc[stations][attributes]), st, en, as_ts)

    def fetch_dynamic_attributes(self,
                                 stn_id,
                                 attributes='all',
                                 st=None,
                                 en=None,
                                 **kwargs) -> pd.DataFrame:
        """Fetches all or selected dynamic attributes of one station."""

        st, en = self._check_length(st, en)

        attributes = check_attributes(attributes, self.dynamic_attributes)

        assert isinstance(stn_id, str), f"provide only one station_id. You provided{stn_id}"

        df = pd.DataFrame()

        for _attr in attributes:
            _df = self._read_df(_attr, **kwargs)
            df[_attr] = _df[stn_id][st:en]

        return df

    def _read_df(self, _attr, **kwargs):
        _path = os.path.join(self.ds_dir, f'{self.folders[_attr]}.csv')
        _df = pd.read_csv(_path, na_values=['-99.99'], **kwargs)
        _df.index = pd.to_datetime(_df[['year', 'month', 'day']])
        [_df.pop(col) for col in ['year', 'month', 'day']]
        return _df

    def fetch_static_attributes(self,
                                stn_id,
                                attribute='all',
                                **kwargs) -> pd.DataFrame:
        """Fetches static attribuets of one station as dataframe."""

        return self._read_static(stn_id, attribute)

    def plot(self, what, stations=None, **kwargs):
        assert what in ['outlets', 'boundaries']
        f1 = os.path.join(self.ds_dir, f'02_location_boundary_area{SEP}02_location_boundary_area{SEP}shp{SEP}CAMELS_AUS_BasinOutlets_adopted.shp')
        f2 = os.path.join(self.ds_dir, f'02_location_boundary_area{SEP}02_location_boundary_area{SEP}shp{SEP}bonus data{SEP}Australia_boundaries.shp')

        if plot_shapefile is not None:
            return plot_shapefile(f1, bbox_shp=f2, recs=stations, rec_idx=0, **kwargs)
        else:
            raise ModuleNotFoundError("Shapely must be installed in order to plot the datasets.")


class CAMELS_CL(Camels):
    """
    Downloads and processes CAMELS dataset of Chile
    https://doi.org/10.5194/hess-22-5817-2018
    """

    urls = {
        "1_CAMELScl_attributes.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "2_CAMELScl_streamflow_m3s.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "3_CAMELScl_streamflow_mm.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "4_CAMELScl_precip_cr2met.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "5_CAMELScl_precip_chirps.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "6_CAMELScl_precip_mswep.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "7_CAMELScl_precip_tmpa.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "8_CAMELScl_tmin_cr2met.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "9_CAMELScl_tmax_cr2met.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "10_CAMELScl_tmean_cr2met.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "11_CAMELScl_pet_8d_modis.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "12_CAMELScl_pet_hargreaves.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "13_CAMELScl_swe.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "14_CAMELScl_catch_hierarchy.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "CAMELScl_catchment_boundaries.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
    }

    dynamic_attributes = ['streamflow_m3s', 'streamflow_mm',
                          'precip_cr2met', 'precip_chirps', 'precip_mswep', 'precip_tmpa',
                          'tmin_cr2met', 'tmax_cr2met', 'tmean_cr2met',
                          'pet_8d_modis', 'pet_hargreaves',
                          'swe'
                          ]
    """
    Arguments:
        path: path where the CAMELS-AUS dataset has been downloaded. This path must
              contain five zip files and one xlsx file.
    """
    def __init__(self,
                 path: str = None
                 ):

        self.ds_dir = path

        super().__init__()

        if not os.path.exists(self.ds_dir):
            os.makedirs(self.ds_dir)

        for _file, url in self.urls.items():
            fpath = os.path.join(self.ds_dir, _file)
            if not os.path.exists(fpath):
                download(url+_file, fpath)
        self._unzip()

    @property
    def _all_dirs(self):
        """All the folders in the dataset_directory"""
        return [f for f in os.listdir(self.ds_dir) if os.path.isdir(os.path.join(self.ds_dir, f))]

    @property
    def start(self):
        return "19130215"

    @property
    def end(self):
        return "20180309"

    @property
    def static_attributes(self) -> list:
        path = os.path.join(self.ds_dir, f"1_CAMELScl_attributes{SEP}1_CAMELScl_attributes.txt")
        df = pd.read_csv(path, sep='\t', index_col='gauge_id')
        return df.index.to_list()

    def stations(self) -> list:
        """Tells all station ids for which a data of a specific attribute is available."""
        _stations = {}
        for dyn_attr in self.dynamic_attributes:
            for _dir in self._all_dirs:
                if dyn_attr in _dir:
                    fname = os.path.join(self.ds_dir, f"{_dir}{SEP}{_dir}.txt")
                    df = pd.read_csv(fname, sep='\t', nrows=2, index_col='gauge_id')
                    _stations[dyn_attr] = list(df.columns)

        return list(set.intersection(*map(set, list(_stations.values()))))

    def fetch_stations_attributes(self,
                                  stations: list,
                                  dynamic_attributes='all',
                                  static_attributes=None,
                                  st=None,
                                  en=None,
                                  as_ts=False,
                                  **kwargs) -> dict:
        """Overwitten for speed"""
        stns = {}
        st, en = self._check_length(st, en)

        if static_attributes is not None:
            static_attributes = check_attributes(static_attributes, self.static_attributes)

        assert all(stn in self.stations() for stn in stations)

        if dynamic_attributes is not None:
            dynamic_attributes = check_attributes(dynamic_attributes, self.dynamic_attributes)

            # since one file contains data for one dynamic attribute for all stations, thus if we need data
            # for one station only, we will have to read all the files. So we first read all the files

            # reading all dynnamic attributes
            dyn_attrs = {}
            for attr in dynamic_attributes:
                fname = [f for f in self._all_dirs if '_'+attr in f][0]
                fname = os.path.join(self.ds_dir, f'{fname}{SEP}{fname}.txt')
                _df = pd.read_csv(fname, sep='\t', index_col=['gauge_id'])
                _df.index = pd.to_datetime(_df.index)
                dyn_attrs[attr] = _df

            # making one separate dataframe for one station
            for stn in stations:
                stn_df = pd.DataFrame()
                for attr, attr_df in dyn_attrs.items():
                    if attr in dynamic_attributes:
                        stn_df[attr] = attr_df[stn]
                stns[stn] = stn_df[st:en]

            if static_attributes is not None:
                static = self._read_static(stations, static_attributes, st, en, as_ts=as_ts)
                for k, v in stns.items():
                    if as_ts:
                        stns[k] = pd.concat([stns[k], static[k]], axis=1)
                    else:
                        stns[k] = {'dynamic': stns[k], 'static': static[k]}

        elif static_attributes is not None:
            return self._read_static(stations, static_attributes, st, en, as_ts=as_ts)

        return stns

    def _read_static(self, stations, attributes, st, en, as_ts):
        # overwritten for speed
        df = pd.DataFrame()
        path = os.path.join(self.ds_dir, f"1_CAMELScl_attributes{SEP}1_CAMELScl_attributes.txt")
        _df = pd.read_csv(path, sep='\t', index_col='gauge_id')

        stns_df = {}
        for stn in stations:
            if stn in _df:
                df[stn] = _df[stn]
            elif ' ' + stn in _df:
                df[stn] = _df[' ' + stn]

            stns_df[stn] = self.to_ts(df.transpose()[attributes], st, en, as_ts)

        return stns_df

    def fetch_dynamic_attributes(self,
                                 stn_id,
                                 attributes='all',
                                 st=None,
                                 en=None,
                                 **kwargs) -> pd.DataFrame:
        """
        Fetches dynamic attribute/attributes of one station.
        This is provided for homogenity otherwise fetching for single station is very slow this way."""

        st, en = self._check_length(st, en)

        if attributes == 'all':
            attributes = self.dynamic_attributes

        assert isinstance(stn_id, str), f"provide only one station_id. You provided {stn_id}"

        df = pd.DataFrame(columns=attributes)

        fname = None
        for attr in attributes:
            for f in self._all_dirs:
                if attr in f:
                    fname = f
                    break
            fpath = os.path.join(self.ds_dir, f"{fname}{SEP}{fname}.txt")
            _df = pd.read_csv(fpath, sep='\t', index_col='gauge_id', na_values=['" "', '', ' '])
            _df.index = pd.to_datetime(_df.index)
            df[attr] = _df[stn_id][st:en]

        return df

    def fetch_static_attributes(self,
                                station,
                                attributes=None,
                                st=None,
                                en=None,
                                as_ts=False
                                ):

        return self._read_static(station, attributes, st, en, as_ts)


class HYPE(Camels):
    """
    Downloads and preprocesses HYPE dataset from https://zenodo.org/record/4029572.
    This is a rainfall-runoff dataset of 564 stations from 1985 to 2019 at daily
    monthly and yearly time steps.
    paper : https://doi.org/10.2166/nh.2010.007
    """
    url = [
        "https://zenodo.org/record/581435",
        "https://zenodo.org/record/4029572"
    ]
    dynamic_attributes = [
        'AET_mm',
        'Baseflow_mm',
        'Infiltration_mm',
        'SM_mm',
        'Streamflow_mm',
        'Runoff_mm',
        'Qsim_m3-s',
        'Prec_mm',
        'PET_mm'
    ]

    def __init__(self, time_step: str = 'daily', **kwargs):
        assert time_step in ['daily', 'monthly', 'yearly']
        self.time_step = time_step
        self.ds_dir = None
        super().__init__(**kwargs)

        self._download()

    def stations(self):
        return np.arange(1, 565).astype(str).tolist()

    def fetch_dynamic_attributes(self,
                                 station,
                                 attributes='all',
                                 st=None,
                                 en=None,
                                 **kwargs):

        dynamic_attributes = check_attributes(attributes, self.dynamic_attributes)

        _dynamic_attributes = []
        for dyn_attr in dynamic_attributes:
            pref, suff = dyn_attr.split('_')[0], dyn_attr.split('_')[-1]
            _dyn_attr = f"{pref}_{self.time_step}_{suff}"
            _dynamic_attributes.append(_dyn_attr)

        df = pd.DataFrame()
        for dyn_attr in _dynamic_attributes:
            fname = f"{dyn_attr}.csv"
            fpath = os.path.join(self.ds_dir, fname)
            _df = pd.read_csv(fpath, index_col='DATE', usecols=['DATE', str(station)])
            _df.index = pd.to_datetime(_df.index)
            _df.columns = [dyn_attr]
            df = pd.concat([df, _df], axis=1)

        return df[_dynamic_attributes][st:en]

    @property
    def start(self):
        return '19850101'

    @property
    def end(self):
        return '20191231'
