# camels #
# https://confluence.ecmwf.int/display/COPSRV/GloFAS+Climate+Data+Store
# https://www.sciencedirect.com/journal/data-in-brief
# https://data.mendeley.com/datasets/5vzp6svhwh/4
# https://zenodo.org/record/4218413#.YA6w7BZS-Uk
# https://www.sciencedirect.com/search?qs=time%20series&pub=Data%20in%20Brief&cid=311593
# https://doi.pangaea.de/10.1594/PANGAEA.898217
# https://doi.pangaea.de/10.1594/PANGAEA.811992  # protected
# https://doi.pangaea.de/10.1594/PANGAEA.905446
# https://doi.pangaea.de/10.1594/PANGAEA.900958
# https://doi.pangaea.de/10.1594/PANGAEA.831193


# HYSETS https://osf.io/rpc3w/  https://www.nature.com/articles/s41597-020-00583-2


import glob
import zipfile
import random
import urllib.request as ulib
import sys, shutil, os
import urllib.parse as urlparse
import tempfile
from typing import Union

import pandas as pd
import numpy as np

from dl4seq.utils.download_zenodo import download_from_zenodo
from dl4seq.utils.download_pangaea import PanDataSet
from dl4seq.utils.spatial_utils import plot_shapefile

# TODO, add visualization

DATASETS = ['ISWDC', 'SEBAL_ET_CHINA']

# following files must exist withing data folder for CAMELS-GB data
CAMELS_GB_files = [
    'CAMELS_GB_climatic_attributes.csv',
    'CAMELS_GB_humaninfluence_attributes.csv',
    'CAMELS_GB_hydrogeology_attributes.csv',
    'CAMELS_GB_hydrologic_attributes.csv',
    'CAMELS_GB_hydrometry_attributes.csv',
    'CAMELS_GB_landcover_attributes.csv',
    'CAMELS_GB_soil_attributes.csv',
    'CAMELS_GB_topographic_attributes.csv'
]

def gb_message():
    link = "https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9"
    raise ValueError(f"Dwonlaoad the data from {link} and provide the directory path as dataset=Camels(data=data)")


def sanity_check(dataset_name, path):
    if dataset_name == 'CAMELS-GB':
        if not os.path.exists(os.path.join(path, 'data')):
            raise FileNotFoundError(f"No folder named `path` exists inside {path}")
        else:
            data_path = os.path.join(path, 'data')
            for file in CAMELS_GB_files:
                if not os.path.exists(os.path.join(data_path, file)):
                    raise FileNotFoundError(f"File {file} must exist inside {data_path}")
    return


class Datasets(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self.name = name

    @property
    def base_ds_dir(self):
        """Base datasets directory"""
        return os.path.join(os.path.dirname(__file__), 'datasets')

    @property
    def ds_dir(self):
        _dir = os.path.join(self.base_ds_dir, self.__class__.__name__)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        return _dir

    @property
    def DATASETS(self):
        raise NotImplementedError

    @property
    def location(self):  # geographical location of data
        raise NotImplementedError

    @property
    def num_points(self):  # number of points in the data
        raise NotImplementedError

    @property
    def start(self):  # start of data
        raise NotImplementedError

    @property
    def end(self):  # end of data
        raise NotImplementedError

    def plot(self, points):
        # this function should be implemented to either plot data of all points/stations or of selected points/stations.
        raise NotImplementedError(f'The method plot should be written for the class {self.name}')

    def _download(self, overwrite=False):
        """Downloads the dataset. If already downloaded, then"""
        if os.path.exists(self.ds_dir):
            if overwrite:
                print(f"removing previous data directory {self.ds_dir} and downloading new")
                shutil.rmtree(self.ds_dir)
                self._download_and_unzip()
            else:
                print(f"""
Not downloading the data since the directory 
{self.ds_dir} already exists.
Use overwrite=True to remove previously saved files and download again""")
        else:
            self._download_and_unzip()
        return

    def _download_and_unzip(self):
        os.makedirs(self.ds_dir)
        download_from_zenodo(self.ds_dir, self.DATASETS[self.name]['url'])
        self._unzip()
        return

    def _unzip(self, dirname=None):
        """unzip all the zipped files in a directory"""
        if dirname is None:
            dirname = self.ds_dir

        all_files = glob.glob(f"{dirname}/*.zip")
        for f in all_files:
            src = os.path.join(dirname, f)
            trgt = os.path.join(dirname, f.split('.zip')[0])
            if not os.path.exists(trgt):
                print(f"unziping {src} to {trgt}")
                with zipfile.ZipFile(os.path.join(dirname, f), 'r') as zip_ref:
                    try:
                        zip_ref.extractall(os.path.join(dirname, f.split('.zip')[0]))
                    except OSError:
                        filelist = zip_ref.filelist
                        for _file in filelist:
                            if '.txt' in _file.filename or '.csv' in _file.filename or '.xlsx' in _file.filename:
                                zip_ref.extract(_file)
        return

    def download_from_pangaea(self, overwrite=False):

        if os.path.exists(self.ds_dir):
            if overwrite:
                print("removing previously downloaded data and downloading again")
            else:
                print(f"The path {self.ds_dir} already exists.")
                self.data_files = [f for f in os.listdir(self.ds_dir) if f.endswith('.txt')]
                self.metadata_files = [f for f in os.listdir(self.ds_dir) if f.endswith('.json')]
                if len(self.data_files) == 0:
                    print(f"The path {self.ds_dir} is empty so downloading the files again")
                    self._download_from_pangaea()
        else:
            self._download_from_pangaea()
        return

    def _download_from_pangaea(self):
        self.data_files = []
        self.metadata_files = []
        ds = PanDataSet(self.url)
        kids = ds.children()
        if len(kids) > 1:
            for kid in kids:
                kid_ds = PanDataSet(kid)
                fname = kid_ds.download(self.ds_dir)
                self.metadata_files.append(fname + '._metadata.json')
                self.data_files.append(fname + '.txt')
        else:
            fname = ds.download(self.ds_dir)
            self.metadata_files.append(fname + '._metadata.json')
            self.data_files.append(fname + '.txt')
        return


class Camels(Datasets):

    """Get CAMELS dataset.
    This class first downloads the CAMELS dataset if it is not already downloaded. Then the selected attribute
    for a selected id are fetched and provided to the user using the method `fetch`.

    Attributes
        ds_dir: diretory of the dataset
        dynamic_attributes: tells which dynamic attributes are available in this dataset
        static_attribute_categories: tells which kinds of static attributes are present in this category.


    Methods:
        stations: returns the stations for which the data (dynamic attributes) exists as list of strings.

        fetch: fetches all attributes (both static and dynamic type) of all station/gauge_ids or a speficified station.
               It can also be used to fetch all attributes of a number of stations ids either by providing their
               guage_id or by just saying that we need data of 20 stations which will then be chosen randomly.

        fetch_dynamic_attributes: fetches speficied dynamic attributes of one specified station. If the dynamic attribute
                                  is not specified, all dynamic attributes will be fetched for the specified station.
                                  If station is not specified, the specified dynamic attributes will be fetched for all
                                  stations.

        fetch_static_attributes: works same as `fetch_dynamic_attributes` but for `static` attributes. Here if the
                                 `category` is not specified then static attributes of the specified station for all
                                 categories are returned.
    """

    DATASETS = {
        'CAMELS-BR': {'url': "https://zenodo.org/record/3964745#.YA6rUxZS-Uk",
                      },
        'CAMELS-GB': {'url': gb_message},
    }

    def stations(self):
        raise NotImplementedError

    def static_attribute_categories(self):
        raise NotImplementedError

    def fetch_dynamic_attributes(self, station, dynamic_attributes, **kwargs):
        raise NotImplementedError

    def fetch_static_attributes(self, station, categories, static_attributes, as_ts=False):
        raise NotImplementedError

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
        #sanity_check(self.name, x)
        self._ds_dir = x

    @property
    def common_index(self):
        """
        It is possible that multiple data are available during different time-periods.
        This returns the time-pseriod/index which is shared between all the data.
        >>>import random
        >>>dataset = CAMELS_BR
        >>>stations = dataset.stations()
        >>>df = dataset.fetch_stations_attributes(station=random.choice(stations))
        """
        return


    def fetch(self, stations=None, dynamic_attributes='all', categories='all', static_attributes='all',
              st: Union[None, str] = None, en: Union[None, str] = None,
              **kwargs)->dict:
        """
        Fetches the attributes of one or more stations.
        :param stations: str/list/int/float/None, default None. if string, it is supposed to be a station name/gauge_id.
                         If list, it will be a list of station/gauge_ids. If int, it will be supposed that the
                         user want data for this number of stations/gauge_ids. If None (default), then attributes
                         of all available stations. If float, it will be supposed that the user wants data
                         of this fraction of stations.
        :param dynamic_attributes: list default(None), If not None, then it is the attributes to be fetched.
                                   If None, then all available attributes are fetched
        :param categories: list/str, Categories of static attributes to be fetched.If None, then static attributes will
                           not be fetched.
        :param static_attributes: list
        :param st: str, starting date of data to be returned. If None,
                        the data will be returned from where it is available
        :param en: str, end date of data to be returned. If None, then the data will be returned till the
                        date data is available.
        :param kwargs: keyword arguments to read the files

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

        return self.fetch_stations_attributes(stations, dynamic_attributes, categories, static_attributes,
                                              st=st, en=en,
                                              **kwargs)

    def fetch_stations_attributes(self,
                                  stations:list,
                                  dynamic_attributes='all',
                                  categories=None,
                                  static_attributes='all',
                                  **kwargs)->dict:
        """fetches attributes of multiple stations."""
        assert isinstance(stations, list)

        # static attributes should not be None, if we want to fetch some categories.
        if categories:
            assert static_attributes, f"""
static_static attributes can not be {static_attributes}
when categorices is {categories}"""

        stations_attributes = {}
        for station in stations:
            stations_attributes[station] = self.fetch_station_attributes(
                station, dynamic_attributes, categories, static_attributes, **kwargs)

        return stations_attributes

    def fetch_station_attributes(self,
                                 station,
                                 dynamic_attributes='all',
                                 categories='all',
                                 static_attributes='all',
                                 as_ts=False,
                                 st=None,
                                 en=None,
                                 **kwargs)->pd.DataFrame:
        """Fetches attributes for one station.
        Return:
            dataframe if as_ts is True else it returns a dictionary of static and dynamic attributes for
            a station/gauge_id
            """
        if st is None:
            st = self.start
        if en is None:
            en = self.end

        station_df = pd.DataFrame()
        if dynamic_attributes:
            dynamic = self.fetch_dynamic_attributes(station, dynamic_attributes, st=st, en=en, **kwargs)
            station_df = pd.concat([station_df, dynamic])

        if categories:
            if categories == 'all':
                categories = self.static_attribute_categories

            static = self.fetch_static_attributes(station, categories, static_attributes, as_ts=as_ts)
            if as_ts:
                idx = pd.date_range(st, en, freq='D')
                static = pd.DataFrame(np.repeat(static.values, len(idx), axis=0), index=idx)
                station_df = pd.concat([station_df, static], axis=1)

            else:
                station_df = pd.concat([station_df, static])

        return station_df


class CAMELS_US(Camels):
    """Downloads and processes CAMELS dataset of 671 catchments named as CAMELS
    from https://ral.ucar.edu/solutions/products/camels
    """
    DATASETS = ['CAMELS_US']
    url = "https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/basin_timeseries_v1p2_metForcing_obsFlow.zip"
    catchment_attr_url = "https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/camels_attributes_v2.0.zip"

    folders = {'basin_mean_daymet': 'basin_mean_forcing\\daymet',
             'basin_mean_maurer': 'basin_mean_forcing\\maurer',
             'basin_mean_nldas': 'basin_mean_forcing\\nldas',
             'elev_bands': 'elev\\daymet',
             'hru': 'hru_forcing\\daymet'}

    def __init__(self):

        super().__init__("CAMELS_US")

        if os.path.exists(self.ds_dir):
            print(f"dataset is already downloaded at {self.ds_dir}")
        else:
            download(self.url, os.path.join(self.camels_dir, 'CAMELS_US\\CAMELS_US.zip'))
            download(self.catchment_attr_url, os.path.join(self.camels_dir, "CAMELS_US\\catchment_attrs.zip"))
            self._unzip()

        self.attr_dir = os.path.join(self.ds_dir, 'catchment_attrs\\camels_attributes_v2.0')
        self.dataset_dir = os.path.join(self.ds_dir, 'CAMELS_US\\basin_dataset_public_v1p2')

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

    def stations(self)->list:
        stns = []
        for _dir in os.listdir(os.path.join(self.dataset_dir, 'usgs_streamflow')):
            cat = os.path.join(self.dataset_dir, f'usgs_streamflow\\{_dir}')
            stns += [fname.split('_')[0] for fname in os.listdir(cat)]

        return stns

    def fetch_station_attributes(self,
                                 station,
                                 dynamic_attributes='all',
                                 categories='all',
                                 static_attributes='all',
                                 forcing='basin_mean_daymet',
                                 as_ts=False,
                                 **kwargs) -> pd.DataFrame:

        return self.fetch_dynamic_attributes(station,
                                             forcing=forcing,
                                             as_ts=as_ts,
                                             **kwargs)

    def fetch_dynamic_attributes(self,
                                 station,
                                 dynamic_attributes='all',
                                 forcing='basin_mean_daymet',
                                 st=None,
                                 en=None,
                                 **kwargs):
        if st is None:
            st = self.start

        if en is None:
            en = self.end

        assert isinstance(station, str)
        df = None
        df1 = None
        dir_name = self.folders[forcing]
        for cat in os.listdir(os.path.join(self.dataset_dir, dir_name)):
            cat_dirs = os.listdir(os.path.join(self.dataset_dir, f'{dir_name}\\{cat}'))
            stn_file = f'{station}_lump_cida_forcing_leap.txt'
            if stn_file in cat_dirs:
                df = pd.read_csv(os.path.join(self.dataset_dir, f'{dir_name}\\{cat}\\{stn_file}'),
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
                fpath = os.path.join(flow_dir, f'{cat}\\{stn_file}')
                df1 = pd.read_csv(fpath,  sep="\s+|;|:'", names=['station', 'Year', 'Month', 'Day', 'Flow', 'Flag'], engine='python')
                df1.index = pd.to_datetime(
                    df1['Year'].map(str) + '-' + df1['Month'].map(str) + '-' + df1['Day'].map(str))

        out_df = pd.concat([df[['prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']], df1['Flow']],
                           axis=1)
        return  out_df[st:en]


class CAMELS_BR(Camels):
    """Downloads and processes CAMELS dataset of Brazil"""

    dynamic_attributes = ['streamflow_m3s', 'streamflow_mm', 'streamflow_simulated',
                           'precipitation_cpc', 'precipitation_mswep', 'precipitation_chirps',
                           'evapotransp_gleam', 'evapotransp_mgb', 'potential_evapotransp_gleam',
                           'temperature_min_cpc', 'temperature_mean', 'temperature_max']
    def __init__(self):

        super().__init__("CAMELS-BR")

        self._download()

    @property
    def ds_dir(self):
        """Directory where a particular dataset will be saved. """
        return os.path.join(self.camels_dir, self.name)

    @property
    def static_dir(self):
        path = None
        for _dir in self._all_dirs:
            if "attributes" in _dir:  # supposing that 'attributes' axist in only one file/folder in self.ds_dir
                path = os.path.join(self.ds_dir, f'{_dir}\\{_dir}')
        return path

    @property
    def static_files(self):
        all_files = None
        if self.static_dir is not None:
            all_files = glob.glob(f"{self.static_dir}/*.txt")
        return all_files

    @property
    def static_attribute_categories(self):
        static_attrs = []
        for f in self.static_files:
            ff = str(os.path.basename(f).split('.txt')[0])
            static_attrs.append('_'.join(ff.split('_')[2:]))
        return static_attrs

    @property
    def start(self):
        return "19800101"

    @property
    def end(self):
        return "20181231"

    def all_stations(self, attribute)->list:
        """Tells all station ids for which a data of a specific attribute is available."""
        all_files = []
        for _dir in self._all_dirs:
            if attribute in _dir:
                all_files = os.listdir(os.path.join(self.ds_dir, f'{_dir}\\{_dir}'))

        stations = []
        for f in all_files:
            stations.append(f.split('_')[0])

        return stations

    @property
    def _all_dirs(self):
        """All the folders in the dataset_directory"""
        return  [f for f in os.listdir(self.ds_dir) if os.path.isdir(os.path.join(self.ds_dir, f))]

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

        return list(set.intersection(*map(set, list(stations.values()) )))

    def fetch_dynamic_attributes(self,
                                 stn_id,
                                 attributes='all',
                                 st=None,
                                 en=None,
                                 **kwargs)->pd.DataFrame:
        """
        returns the dynamic/time series attribute/attributes for one station id.
        >>>dataset = CAMELS_BR()
        >>>pcp = dataset.fetch_dynamic_attributes('10500000', 'precipitation_cpc')
        ...# fetch all time series data associated with a station.
        >>>x = dataset.fetch_dynamic_attributes('51560000', dataset.dynamic_attributes)
        """

        if st is None:
            st = self.start
        if en is None:
            en = self.end

        if not kwargs:
            kwargs = {'sep': ' '}

        if attributes == 'all':
            attributes = self.dynamic_attributes
        elif not isinstance(attributes, list):
            assert isinstance(attributes, str)
            attributes = [attributes]

        data = pd.DataFrame()
        for attr in attributes:

            for _dir in self._all_dirs:
                if attr in _dir:
                    path = os.path.join(self.ds_dir, f'{_dir}\\{_dir}')
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
                # else:
                #     ValueError(f"{attributes} is not a valid dynamic attribute for {self.name}. Choose any of"
                #            f" { self.DATASETS[self.name]['dynamic_attributes']}")
        return data

    def fetch_static_attributes(self,
                                stn_id:int,
                                categories='all',
                                attributes=None,
                                index_col_name='gauge_id',
                                as_ts=False,
                                **kwargs)->pd.DataFrame:
        """
        :param stn_id: int, station id whose attribute to fetch
        :param categories: str, represents the type of static attribute to fetch e.g. climate, the one of the file
                              name must have this in its name, otherwise this function will return None.
        :param attributes: str, list, name of attribute to fetch. Default is None, which will return all the
                               attributes for a particular station of the specified category.
        :param index_col_name: str, name of column containing station names
        :param kwargs: keyword arguments to be passed to pd.read_csv to read files.
        :param as_ts:
        >>>dataset = Camels('CAMELS-BR')
        >>>df = dataset.fetch_static_attributes(11500000, 'climate')
        """
        if not kwargs:
            kwargs = {'sep': ' '}

        if categories == 'all':
            categories = self.static_attribute_categories
        elif isinstance(categories, str):
            categories = [categories]

        df = pd.DataFrame()

        for category in categories:
            fname = None
            for f in self.static_files:
                if category in f:
                    fname = os.path.join(self.static_dir, f)
                    break
            _df = pd.read_csv(fname, **kwargs)

            if attributes != 'all':
                if isinstance(attributes, str):
                    _attributes = [attributes]
                else:
                    _attributes = attributes
            else:
                _attributes = list(_df.columns)

            _df = _df.loc[_df[index_col_name] == int(stn_id)][_attributes]

            df[_attributes] = _df

        return df


class CAMELS_GB(Camels):

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
    def static_attribute_categories(self)->list:
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

    def stations(self, to_exclude=None):
        # CAMELS_GB_hydromet_timeseries_StationID_number
        path = os.path.join(self.ds_dir, 'data\\timeseries')
        gauge_ids = []
        for f in os.listdir(path):
            gauge_ids.append(f.split('_')[4])

        return gauge_ids

    def fetch_dynamic_attributes(self,
                                 stn_id,
                                 attributes='all',
                                 st=None,
                                 en=None,
                                 **kwargs) ->pd.DataFrame:
        """Fetches dynamic attribute/attributes of one station."""
        if st is None:
            st = self.start
        if en is None:
            en = self.end

        path = os.path.join(self.ds_dir, f"data\\timeseries")
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
                                categories='all',
                                attribute='all',
                                **kwargs)->pd.DataFrame:
        """Fetches static attributes of one station for one or more category as dataframe."""

        if categories == 'all':
            categories = self.static_attribute_categories
        elif isinstance(categories, str):
            categories = [categories]

        df = pd.DataFrame()
        for category in categories:
            cat_df = self.fetch_static_attribute(stn_id, category, **kwargs)
            df = pd.concat([cat_df, df], axis=1)

        return df

    def fetch_static_attribute(self,
                               stn_id,
                               category,
                               attribute='all',
                               index_col_name='gauge_id',
                               as_ts=False,
                               **kwargs)->pd.DataFrame:

        """Fetches static attribute/attributes of one category as dataframe."""
        path = os.path.join(self.ds_dir, "data")
        fname = None
        for f in os.listdir(path):
            if category in f:
                fname = f
                break
        df = pd.read_csv(os.path.join(path, fname), **kwargs)
        if attribute == 'all':
            df = df.loc[df[index_col_name] == int(stn_id)]
        else:
            df = df.loc[df[index_col_name] == int(stn_id)][attribute]
        if as_ts:
            return self._to_ts(df)
        else:
            return df

    def _to_ts(self, df):
        return df


class CAMELS_AUS(Camels):
    """
    Arguments:
        path: path where the CAMELS-AUS dataset has been downloaded. This path must
              contain five zip files and one xlsx file.
    """
    url = 'https://doi.pangaea.de/10.1594/PANGAEA.921850'
    urls = {
        "01_id_name_metadata.zip":"https://download.pangaea.de/dataset/921850/files/",
        "02_location_boundary_area.zip":"https://download.pangaea.de/dataset/921850/files/",
        "03_streamflow.zip":"https://download.pangaea.de/dataset/921850/files/",
        "04_attributes.zip":"https://download.pangaea.de/dataset/921850/files/",
        "05_hydrometeorology.zip":"https://download.pangaea.de/dataset/921850/files/",
        "CAMELS_AUS_Attributes-Indices_MasterTable.csv":"https://download.pangaea.de/dataset/921850/files/",
        "Units_01_TimeseriesData.pdf": "https://download.pangaea.de/dataset/921850/files/",
        "Units_02_AttributeMasterTable.pdf":"https://download.pangaea.de/dataset/921850/files/",
    }

    dynamic_attributes = ['streamflow_MLd', 'streamflow_MLd_inclInfilled', 'streamflow_mmd.csv'
                          'et_morton', 'et_morton_point_SILO',
                          'et_morton_wet_SILO',
                          'et_short_crop_SILO', 'et_tall_crop_SILO', 'evap_morton_lake_SILO',
                          'evap_pan_SILO', 'evap_syn_SILO',
                          'precipitation_AWAP', 'precipitation_SILO', 'precipitation_var_SWAP',
                          'solarrad_AWAP', 'tmax_AWAP', 'tmin_AWAP', 'vprp_AWAP',
                          'mslp_SILO', 'radiation_SILO', 'rh_tmax_SILO', 'rh_tmin_SILO',
                          'tmax_SILO', 'tmin_SILO', 'vp_deficit_SILO', 'vp_SILO'
                           ]

    def __init__(self, path=None):
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
        fname = os.path.join(self.ds_dir, "01_id_name_metadata\\01_id_name_metadata\\id_name_metadata.csv")
        df = pd.read_csv(fname)
        if as_list:
            return df['station_id'].to_list()
        else:
            return df

    @property
    def static_attribute_categories(self):
        attributes = []
        path = os.path.join(self.ds_dir, '04_attributes\\04_attributes')
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)) and f.endswith('csv'):
                f = str(f.split('.csv')[0])
                attributes.append(''.join(f.split('_')[2:]))
        return attributes

    def fetch_stations_attributes(self,
                                  stations:list,
                                  dynamic_attributes='all',
                                  categories=None,
                                  static_attributes='all',
                                  st=None,
                                  en=None,
                                  as_ts=False,
                                  **kwargs) ->dict:
        stns = {}
        dyn_attrs = {}

        if st is None:
            st = self.start

        if en is None:
            en = self.end

        if isinstance(categories, str):
            if categories == 'all':
                categories = self.static_attribute_categories
            else:
                categories = [categories]

        def populate_dynattr(_path, attr):
            fname = os.path.join(_path, attr + ".csv")
            if os.path.exists(fname):
                df = pd.read_csv(fname, na_values=['-99.99'], **kwargs)
                df.index = pd.to_datetime(df[['year', 'month', 'day']])
                [df.pop(col) for col in ['year', 'month', 'day']]
                dyn_attrs[attr] = df

        if dynamic_attributes is not None:
            if dynamic_attributes == 'all':
                dynamic_attributes = self.dynamic_attributes
            elif isinstance(dynamic_attributes, str):
                assert dynamic_attributes in self.dynamic_attributes
                dynamic_attributes = [dynamic_attributes]
            elif isinstance(dynamic_attributes, list):
                assert all(attr in self.dynamic_attributes for attr in dynamic_attributes)
                dynamic_attributes = dynamic_attributes

            for attr in dynamic_attributes:
                if 'SILO' in attr:
                    path = os.path.join(self.ds_dir, "05_hydrometeorology\\05_hydrometeorology\\03_Other\\SILO")
                    populate_dynattr(path, attr)

                if 'AWAP' in attr:
                    path = os.path.join(self.ds_dir, "05_hydrometeorology\\05_hydrometeorology\\03_Other\\AWAP")
                    populate_dynattr(path, attr)

                if 'et_' in attr or 'evap_' in attr:
                    path = os.path.join(self.ds_dir,
                                        "05_hydrometeorology\\05_hydrometeorology\\02_EvaporativeDemand_timeseries")
                    populate_dynattr(path, attr)

                if 'precipitation_' in attr:
                    path = os.path.join(self.ds_dir,
                                        "05_hydrometeorology\\05_hydrometeorology\\01_precipitation_timeseries")
                    populate_dynattr(path, attr)

                if 'streamflow_' in attr:
                    path = os.path.join(self.ds_dir, "03_streamflow\\03_streamflow")
                    populate_dynattr(path, attr)

            # making one separate dataframe for one station
            for stn in stations:
                stn_df = pd.DataFrame()
                for attr, attr_df in dyn_attrs.items():
                    if attr in dynamic_attributes:
                        stn_df[attr] = attr_df[stn][st:en]
                stns[stn] = stn_df

            if categories is not None:
                static = self._read_static(stations, categories, static_attributes, st,en, True)
                for k, v in stns.items():
                    stns[k] = pd.concat([stns[k], static[k]], axis=1)

        elif categories is not None:
            return self._read_static(stations, categories, static_attributes, st, en, as_ts=as_ts)

        return stns

    def _read_static(self, stations, categories, attributes,
                     st=None, en=None, as_ts=False)->dict:

        df_categories = {}
        path = os.path.join(self.ds_dir, "04_attributes\\04_attributes")
        for category in categories:
            for fname in os.listdir(path):
                if category in fname:
                    _fname = os.path.join(path, fname)
                    df_categories[category] = pd.read_csv(_fname, index_col=['station_id'])

        stn_categories = {}
        for stn in stations:
            stn_df = pd.DataFrame()
            for cat, cat_df in df_categories.items():
                stn_df = pd.concat([stn_df, cat_df.loc[cat_df.index==stn]], axis=1)
            if attributes == 'all':
                stn_categories[stn] = self.to_ts(stn_df, st, en, to_ts=as_ts)
            else:
                stn_categories[stn] = self.to_ts(stn_df[attributes], st, en, to_ts=as_ts)

        return stn_categories

    def to_ts(self, df:pd.DataFrame, st, en, to_ts=False):
        if to_ts:
            idx = pd.date_range(st, en, freq='D')
            return pd.DataFrame(np.repeat(df.values, len(idx), axis=0), index=idx)
        else:
            return df

    def fetch_dynamic_attributes(self,
                                 stn_id,
                                 attributes='all',
                                 st=None,
                                 en=None,
                                 **kwargs) ->pd.DataFrame:
        """Fetches all or selected dynamic attributes of one station."""

        if st is None:
            st = self.start

        if en is None:
            en = self.end

        if attributes == 'all':
            attributes = self.dynamic_attributes

        assert isinstance(stn_id, str), f"provide only one station_id. You provided{stn_id}"

        df = pd.DataFrame()

        def populate_df(_path):
            for attr in attributes:
                fname = os.path.join(_path, attr + ".csv")
                if os.path.exists(fname):
                    _df = pd.read_csv(fname, usecols=[stn_id, 'year', 'month', 'day'], na_values=['-99.99'], **kwargs)
                    _df.index = pd.to_datetime(_df[['year', 'month', 'day']].astype(str).apply(' '.join, 1))
                    df[attr] = _df[stn_id][st:en]

        if any(['SILO' in i for i in attributes]):
            path = os.path.join(self.ds_dir, "05_hydrometeorology\\05_hydrometeorology\\03_Other\\SILO")
            populate_df(path)

        if any(['AWAP' in i for i in attributes]):
            path = os.path.join(self.ds_dir, "05_hydrometeorology\\05_hydrometeorology\\03_Other\\AWAP")
            populate_df(path)

        if any(['et_' in i for i in attributes]) or any(['evap_' in i for i in attributes]):
            path = os.path.join(self.ds_dir, "05_hydrometeorology\\05_hydrometeorology\\02_EvaporativeDemand_timeseries")
            populate_df(path)

        if any(['precipitation_' in i for i in attributes]):
            path = os.path.join(self.ds_dir, "05_hydrometeorology\\05_hydrometeorology\\01_precipitation_timeseries")
            populate_df(path)

        if any(['streamflow_' in i for i in attributes]):
            path = os.path.join(self.ds_dir, "03_streamflow\\03_streamflow")
            populate_df(path)

        return df

    def fetch_static_attributes(self,
                                stn_id,
                                categories='all',
                                attribute='all',
                                **kwargs)->pd.DataFrame:
        """Fetches static attribuets of one station for one or more category as dataframe."""

        if categories == 'all':
            categories = self.static_attribute_categories
        elif isinstance(categories, str):
            categories = [categories]

        df = pd.DataFrame()
        for category in categories:
            cat_df = self.fetch_static_attribute(stn_id, category, **kwargs)
            df = pd.concat([cat_df, df], axis=1)

        return df

    def fetch_static_attribute(self,
                               stn_id,
                               category,
                               attribute='all',
                               index_col_name='station_id',
                               as_ts=False,
                               **kwargs)->pd.DataFrame:
        # fetch static attribute of one station
        path = os.path.join(self.ds_dir, "04_attributes\\04_attributes")
        fname = None
        for f in os.listdir(path):
            if category in f:
                fname = f
                break
        df = pd.read_csv(os.path.join(path, fname), **kwargs)
        if attribute == 'all':
            df = df.loc[df[index_col_name] == stn_id]
        else:
            df = df.loc[df[index_col_name] == int(stn_id)][attribute]
        if as_ts:
            return self._to_ts(df)
        else:
            return df

    def plot(self, what, stations=None, **kwargs):
        assert what in ['outlets', 'boundaries']
        f1 = os.path.join(self.ds_dir, f'02_location_boundary_area\\02_location_boundary_area\\shp\\CAMELS_AUS_BasinOutlets_adopted.shp')
        f2 = os.path.join(self.ds_dir, f'02_location_boundary_area\\02_location_boundary_area\\shp\\bonus data\\Australia_boundaries.shp')

        return plot_shapefile(f1, bbox_shp=f2, recs=stations, rec_idx=0, **kwargs)


class CAMELS_CL(Camels):
    """Downloads and processes CAMELS dataset of Chile"""

    urls = {
        "1_CAMELScl_attributes.zip":"https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "2_CAMELScl_streamflow_m3s.zip":"https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "3_CAMELScl_streamflow_mm.zip":"https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "4_CAMELScl_precip_cr2met.zip":"https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "5_CAMELScl_precip_chirps.zip":"https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "6_CAMELScl_precip_mswep.zip":"https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "7_CAMELScl_precip_tmpa.zip":"https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "8_CAMELScl_tmin_cr2met.zip":"https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "9_CAMELScl_tmax_cr2met.zip":"https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "10_CAMELScl_tmean_cr2met.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "11_CAMELScl_pet_8d_modis.zip":"https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "12_CAMELScl_pet_hargreaves.zip":"https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "13_CAMELScl_swe.zip":"https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "14_CAMELScl_catch_hierarchy.zip":"https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "CAMELScl_catchment_boundaries.zip":"https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
    }

    dynamic_attributes = ['streamflow_m3s', 'streamflow_mm',
                           'precip_cr2met', 'precip_chirps', 'precip_mswep', 'precip_tmpa',
                           'tmin_cr2met', 'tmax_cr2met', 'tmean_cr2met',
                           'pet_8d_modis', 'pet_hargreaves',
                           'swe']
    """
    Arguments:
        path: path where the CAMELS-AUS dataset has been downloaded. This path must
              contain five zip files and one xlsx file.
    """
    def __init__(self,
                 path=None):
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
        return  [f for f in os.listdir(self.ds_dir) if os.path.isdir(os.path.join(self.ds_dir, f))]

    @property
    def start(self):
        return "19130215"

    @property
    def end(self):
        return "20180309"

    @property
    def static_attribute_categories(self):
        path = os.path.join(self.ds_dir, "1_CAMELScl_attributes\\1_CAMELScl_attributes.txt")
        df = pd.read_csv(path, sep='\t', index_col='gauge_id')
        return df.index.to_list()

    def stations(self)->list:
        """Tells all station ids for which a data of a specific attribute is available."""
        _stations = {}
        for dyn_attr in self.dynamic_attributes:
            for _dir in self._all_dirs:
                if dyn_attr in _dir:
                    fname = os.path.join(self.ds_dir, f"{_dir}\\{_dir}.txt")
                    df = pd.read_csv(fname, sep='\t', nrows=2, index_col='gauge_id')
                    _stations[dyn_attr] = list(df.columns)

        return list(set.intersection(*map(set, list(_stations.values()) )))

    def fetch_stations_attributes(self,
                                  stations:list,
                                  dynamic_attributes='all',
                                  categories=None,
                                  static_attributes='all',
                                  st=None,
                                  en=None,
                                  **kwargs) ->dict:
        """Overwitten for speed"""
        stns = {}
        if st is None:
            st = self.start
        if en is None:
            en = self.end

        assert all(stn in self.stations() for stn in stations)

        if dynamic_attributes is not None:
            if dynamic_attributes == 'all':
                dynamic_attributes = self.dynamic_attributes
            elif isinstance(dynamic_attributes, str):
                assert dynamic_attributes in self.dynamic_attributes
                dynamic_attributes = [dynamic_attributes]
            elif isinstance(dynamic_attributes, list):
                assert all(attr in self.dynamic_attributes for attr in dynamic_attributes)
                dynamic_attributes = dynamic_attributes

            # since one file contains data for one dynamic attribute for all stations, thus if we need data
            # for one station only, we will have to read all the files. So we first read all the files

            # reading all dynnamic attributes
            dyn_attrs = {}
            for attr in dynamic_attributes:
                fname = [f for f in self._all_dirs if '_'+attr in f][0]
                fname = os.path.join(self.ds_dir, f'{fname}\\{fname}.txt')
                _df = pd.read_csv(fname, sep='\t', index_col=['gauge_id'])
                _df.index=  pd.to_datetime(_df.index)
                dyn_attrs[attr] = _df

            # making one separate dataframe for one statino
            for stn in stations:
                stn_df = pd.DataFrame()
                for attr, attr_df in dyn_attrs.items():
                    if attr in dynamic_attributes:
                        stn_df[attr] = attr_df[stn]
                stns[stn] = stn_df[st:en]

        if categories is not None and dynamic_attributes is None:
            return self._read_static(stations)

        return stns

    def _read_static(self, stations):
        # overwritten for speed
        df = pd.DataFrame()
        path = os.path.join(self.ds_dir, "1_CAMELScl_attributes\\1_CAMELScl_attributes.txt")
        _df = pd.read_csv(path, sep='\t', index_col='gauge_id')

        stns_df = {}
        for stn in stations:
            if stn in _df:
                df[stn] = _df[stn]
            elif ' ' + stn in _df:
                df[stn] = _df[' ' + stn]

            stns_df[stn] = df.transpose()

        return  stns_df # to_dict('series')

    def fetch_dynamic_attributes(self,
                                 stn_id,
                                 attributes='all',
                                 st=None,
                                 en=None,
                                 **kwargs)->pd.DataFrame:
        """Fetches dynamic attribute/attributes of one station.
        This is provided for homogenity otherwise fetching for single station is very slow this way."""

        if st is None:
            st = self.start
        if en is None:
            en = self.end

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
            fpath = os.path.join(self.ds_dir, f"{fname}\\{fname}.txt")
            _df = pd.read_csv(fpath, sep='\t', index_col='gauge_id', na_values=['" "', '', ' '])
            _df.index = pd.to_datetime(_df.index)
            df[attr] = _df[stn_id][st:en]

        return df

    def fetch_static_attributes(self,
                                station,
                                categories=None,
                                static_attributes=None,
                                as_ts=False
                                ):
        if categories == 'all':
            attributes = self.static_attribute_categories
        elif categories:
            attributes = categories

        if static_attributes == 'all':
            attributes = self.static_attribute_categories

        path = os.path.join(self.ds_dir, "1_CAMELScl_attributes\\1_CAMELScl_attributes.txt")
        try:
            df = pd.read_csv(path, sep='\t', index_col='gauge_id', usecols=[' ' + station, 'gauge_id']).transpose()
        except ValueError:
            df = pd.read_csv(path, sep='\t', index_col='gauge_id', usecols=[station, 'gauge_id']).transpose()

        return df[attributes]


class Weisssee(Datasets):

    dynamic_attributes = ['Precipitation_measurements',
                           'long_wave_upward_radiation',
                           'snow_density_at_30cm',
                           'long_wave_downward_radiation'
                           ]

    url = '10.1594/PANGAEA.898217'

    def fetch(self, **kwargs):
        self.download_from_pangaea()
        data = {}
        for f in self.data_files:
            fpath = os.path.join(self.ds_dir, f)
            df = pd.read_csv(fpath, **kwargs)

            if 'index_col' in kwargs:
                df.index = pd.to_datetime(df.index)

            data[f.split('.txt')[0]] = df

        return data


class ETP_CHN_SEBAL(Datasets):

    url = "https://zenodo.org/record/4218413#.YBNhThZS-Ul"


class ISWDC(Datasets):

    url = "https://zenodo.org/record/2616035#.YBNl5hZS-Uk"


class WQJordan(Weisssee):
    """Jordan River water quality data of 9 variables for two variables."""
    url = 'https://doi.pangaea.de/10.1594/PANGAEA.919103'


class WQJordan2(Weisssee):
    """Stage and Turbidity data of Jordan River"""
    url = '10.1594/PANGAEA.919104'


class YamaguchiClimateJp(Weisssee):
    """Daily climate and flow data of Japan from 2006 2018"""
    url = "https://doi.pangaea.de/10.1594/PANGAEA.909880"


class FlowBenin(Weisssee):
     """Flow data"""
     url = "10.1594/PANGAEA.831196"


class HydrometricParana(Weisssee):
    """Daily and monthly water level and flow data of Parana river Argentina
    from 1875 to 2017."""
    url = "https://doi.pangaea.de/10.1594/PANGAEA.882613"


class RiverTempSpain(Weisssee):
    """Daily mean stream temperatures in Central Spain for different periods."""
    url = "https://doi.pangaea.de/10.1594/PANGAEA.879494"


class WQCantareira(Weisssee):
    """Water quality and quantity primary data from field campaigns in the Cantareira Water Supply System,
     period Oct. 2013 - May 2014"""
    url="https://doi.pangaea.de/10.1594/PANGAEA.892384"


class RiverIsotope(Weisssee):
    """399 δ18O and δD values in river surface waters of Indian River"""
    url = "https://doi.pangaea.de/10.1594/PANGAEA.912582"


class EtpPcpSamoylov(Weisssee):
    """Evpotranspiration and Precipitation at station TOWER on Samoylov Island Russia
     from 20110524 to 20110819 with 30 minute frequency"""
    url = "10.1594/PANGAEA.811076"

class FlowSamoylov(Weisssee):
    """Net lateral flow at station INT2 on Samoylov Island Russia
    from 20110612 to 20110819 with 30 minute frequency"""
    url = "10.1594/PANGAEA.811072"


class FlowSedDenmark(Weisssee):
    """Flow and suspended sediment concentration fields over tidal bedforms, ADCP profile"""
    url = "10.1594/PANGAEA.841977"


class StreamTempSpain(Weisssee):
    """Daily Mean Stream Temperature at station Tormes3, Central Spain from 199711 to 199906."""
    url = "https://doi.pangaea.de/10.1594/PANGAEA.879507"


class RiverTempEroo(Weisssee):
    """Water temperature records in the Eroo River and some tributaries (Selenga River basin, Mongolia, 2011-2012)"""
    url = "10.1594/PANGAEA.890070"


class HoloceneTemp(Weisssee):
    """Holocene temperature reconstructions for northeastern North America and the northwestern Atlantic,
     core Big_Round_Lake."""
    url = "10.1594/PANGAEA.905446"


class FlowTetRiver(Weisssee):
    """Daily mean river discharge at meteorological station Perpignan upstream, Têt basin France from 1980
    to 2000."""
    url = "10.1594/PANGAEA.226925"


class SedimentAmersee(Weisssee):
    """Occurence of flood laminae in sediments of Ammersee"""
    url = "10.1594/PANGAEA.746240"


class HydrocarbonsGabes(Weisssee):
    """Concentration and geological parameters of n-alkanes and n-alkenes in surface sediments from the Gulf of Gabes,
     Tunisia"""
    url = "10.1594/PANGAEA.774595"


class WaterChemEcuador(Weisssee):
    """weekly and biweekly Water chemistry of cloud forest streams at baseflow conditions,
     Rio San Francisco, Ecuador"""
    url = "10.1594/PANGAEA.778629"


class WaterChemVictoriaLakes(Weisssee):
    """Surface water chemistry of northern Victoria Land lakes"""
    url = "10.1594/PANGAEA.807883"


class HydroChemJava(Weisssee):
    """Hydrochemical data from subsurface rivers, coastal and submarine springsin a karstic region
     in southern Java."""
    url = "10.1594/PANGAEA.882178"


class PrecipBerlin(Weisssee):
    """Sub-hourly Berlin Dahlem precipitation time-series 2001-2013"""
    url = "10.1594/PANGAEA.883587"


class GeoChemMatane(Weisssee):
    """Geochemical data collected in shallow groundwater and river water in a subpolar environment
     (Matane river, QC, Canada)."""
    url = "10.1594/PANGAEA.908290"



def download(url, out=None):
    """High level function, which downloads URL into tmp file in current
    directory and then renames it to filename autodetected from either URL
    or HTTP headers.

    :param url:
    :param out: output filename or directory
    :return:    filename where URL is downloaded to
    """
    # detect of out is a directory
    if out is not None:
        outdir = os.path.dirname(out)
        out_filename = os.path.basename(out)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    else:
        outdir = os.getcwd()
        out_filename = None

    # get filename for temp file in current directory
    prefix = filename_from_url(url)
    (fd, tmpfile) = tempfile.mkstemp(".tmp", prefix=prefix, dir=".")
    os.close(fd)
    os.unlink(tmpfile)

    # set progress monitoring callback
    def callback_charged(blocks, block_size, total_size):
        # 'closure' to set bar drawing function in callback
        callback_progress(blocks, block_size, total_size, bar_function=bar)

    callback = callback_charged


    # Python 3 can not quote URL as needed
    binurl = list(urlparse.urlsplit(url))
    binurl[2] = urlparse.quote(binurl[2])
    binurl = urlparse.urlunsplit(binurl)

    (tmpfile, headers) = ulib.urlretrieve(binurl, tmpfile, callback)
    filename = filename_from_url(url)

    if out_filename:
        filename = out_filename

    filename = outdir + "/" + filename

    # add numeric ' (x)' suffix if filename already exists
    if os.path.exists(filename):
        filename = filename + '1'
    shutil.move(tmpfile, filename)

    #print headers
    return filename


__current_size = 0

def callback_progress(blocks, block_size, total_size, bar_function):
    """callback function for urlretrieve that is called when connection is
    created and when once for each block

    draws adaptive progress bar in terminal/console

    use sys.stdout.write() instead of "print,", because it allows one more
    symbol at the line end without linefeed on Windows

    :param blocks: number of blocks transferred so far
    :param block_size: in bytes
    :param total_size: in bytes, can be -1 if server doesn't return it
    :param bar_function: another callback function to visualize progress
    """
    global __current_size

    width = 100

    if sys.version_info[:3] == (3, 3, 0):  # regression workaround
        if blocks == 0:  # first call
            __current_size = 0
        else:
            __current_size += block_size
        current_size = __current_size
    else:
        current_size = min(blocks * block_size, total_size)
    progress = bar_function(current_size, total_size, width)
    if progress:
        sys.stdout.write("\r" + progress)


def filename_from_url(url):
    """:return: detected filename as unicode or None"""
    # [ ] test urlparse behavior with unicode url
    fname = os.path.basename(urlparse.urlparse(url).path)
    if len(fname.strip(" \n\t.")) == 0:
        return None
    return fname

def bar(current_size, total_size, width):
    percent = current_size/total_size * 100
    if round(percent % 1, 4) == 0.0:
        print(f"{round(percent)}% of {round(total_size*1e-6, 2)} MB downloaded")
    return