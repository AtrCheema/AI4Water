# camels #
# https://confluence.ecmwf.int/display/COPSRV/GloFAS+Climate+Data+Store
# https://www.sciencedirect.com/journal/data-in-brief
# https://data.mendeley.com/datasets/5vzp6svhwh/4
# https://zenodo.org/record/4218413#.YA6w7BZS-Uk
# https://www.sciencedirect.com/search?qs=time%20series&pub=Data%20in%20Brief&cid=311593

import os
import glob
import zipfile
import shutil

import pandas as pd

from dl4seq.utils.download_zenodo import download_from_zenodo


DATASETS = ['ISWDC', 'SEBAL_ET_CHINA']


class Datasets(object):

    @property
    def base_ds_dir(self):
        """Base datasets directory"""
        return os.path.join(os.path.dirname(__file__), 'datasets')



class Camels(Datasets):

    """Get CAMELS dataset.
    This class first downloads the CAMELS dataset if it is not already downloaded. Then the selected attribute
    for a selected id are fetched and provided to the user using the method `fetch`.
    Arguments:
        name: str, name of the CAMELS dataset. See Camels.DATASETS for available dataset names.

    Attributes
        ds_dir: diretory of the dataset


    Methods:
        fetch: fetches a attribute from a site.

    """

    DATASETS = {
        'CAMELS-BR': {'url': "https://zenodo.org/record/3964745#.YA6rUxZS-Uk",
                      'dynamic_attributes': ['streamflow_m3s', 'streamflow_mm', 'streamflow_simulated',
                                             'precipitation_cpc', 'precipitation_mswep', 'precipitation_chirps',
                                             'evapotransp_gleam', 'evapotransp_mgb', 'potential_evapotransp_gleam',
                                             'temperature_min_cpc', 'temperature_mean', 'temperature_max']}
    }

    def __init__(self, name):
        if name not in self.DATASETS.keys():
            raise ValueError(f"unknown dataset {name}. Available datasets are \n{self.DATASETS.keys()}")
        self.name = name

        self._download()


    @property
    def camels_dir(self):
        """Directory where all camels datasets will be saved. This will under datasets directory"""
        return os.path.join(self.base_ds_dir, "CAMELS")

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
    def dynamic_attributes(self):
        return self.DATASETS[self.name]['dynamic_attributes']

    def station_ids(self, attribute)->list:
        """Tells station ids for which a data of a specific attribute is available."""
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

    @property
    def common_index(self):
        """
        It is possible that multiple data are available during different time-periods.
        This returns the time-pseriod/index which is shared between all the data.
        >>>import random
        >>>dataset = Camels("CAMELS-BR")
        >>>stations = dataset.common_stations()
        >>>df = dataset.all_dynamic_attributes(station=random.choice(stations))
        """
        return

    def fetch(self, identifier)->dict:
        """reads and fetches the attribute for a station_id.
        :param identifier: dict, a dictionary whose keys should be station ids and whose keys should be attributes which
                               are to be extracted from a station.
        Examples:
        >>>camel_br = Camels('CAMELS-BR')
        >>>pcp_101 = camel_br.fetch({'10100000': 'precipitation_cpc'})
        """

        attributes = {}
        for stn_id, attribute in identifier.items():
            attributes[str(stn_id)] = self.fetch_attribute(stn_id, attribute)

        return attributes

    def fetch_attribute(self, stn_id, attribute, **kwargs):
        if not kwargs:
            kwargs = {'sep': ' '}

        if not isinstance(attribute, list):
            assert isinstance(attribute, str)
            attribute = [attribute]

        data = pd.DataFrame()
        for attr in attribute:
            if attr in self.DATASETS[self.name]['dynamic_attributes']:
                _data =  self.fetch_dynamic_attributes(stn_id, attr, **kwargs)
                data = pd.concat([data, _data])
            else:
                _data = self.fetch_static_attributes(stn_id, attr, **kwargs)

        return data

    def fetch_dynamic_attributes(self, stn_id,  attribute, **kwargs)->pd.DataFrame:
        """
        returns the dynamic/time series attribute/attributes for one station id.
        >>>dataset = Camels('CAMELS-BR')
        >>>pcp = dataset.fetch_dynamic_attributes('10500000', 'precipitation_cpc')
        # fetch all time series data associated with a station.
        >>>x =   dataset.fetch_dynamic_attributes('51560000', dataset.dynamic_attributes)
        """

        if not kwargs:
            kwargs = {'sep': ' '}

        if not isinstance(attribute, list):
            assert isinstance(attribute, str)
            attribute = [attribute]

        data = pd.DataFrame()
        for attr in attribute:

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
                        [df.pop(item) for item in ['year', 'month', 'day']]
                        data = pd.concat([data, df])
                    else:
                        raise FileNotFoundError(f"file {fname} not found at {path}")
                else:
                    ValueError(f"{attribute} is not a valid dynamic attribute for {self.name}. Choose any of"
                           f" { self.DATASETS[self.name]['dynamic_attributes']}")
        return data

    def fetch_static_attributes(self, stn_id, category,  attribute=None, index_col_name='gauge_id', **kwargs):
        """
        :param stn_id: int, station id whose attribute to fetch
        :param category: str, represents the type of static attribute to fetch e.g. climate, the one of the file
                              name must have this in its name, otherwise this function will return None.
        :param attribute: str, list, name of attribute to fetch. Default is None, which will return all the
                               attributes for a particular station of the specified category.
        :param index_col_name: str, name of column containing station names
        :param kwargs: keyword arguments to be passed to pd.read_csv to read files.
        >>>dataset = Camels('CAMELS-BR')
        >>>df = dataset.fetch_static_attributes(11500000, 'climate', sep=' ')
        """
        if not kwargs:
            kwargs = {'sep': ' '}

        assert isinstance(category, str)

        for f in self.static_files:
            if category in f:
                df = pd.read_csv(os.path.join(self.static_dir, f), **kwargs)

                if attribute is None:
                    attribute = df.columns
                elif not isinstance(attribute, list):
                    assert isinstance(attribute, str)

                return df.loc[df[index_col_name] == int(stn_id)][attribute]
            else:
                return None

    def common_stations(self, to_exclude=None):
        """Returns a list of station ids which are common among all dynamic attributes.
        >>>dataset = Camels('CAMELS-BR')
        >>>stations = dataset.common_stations()
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
                stations[dyn_attr] = self.station_ids(dyn_attr)

        return list(set.intersection(*map(set, list(stations.values()) )))

    def all_dynamic_attributes(self, station='all'):
        """fetch all attributes of specified stations which are common among all
        >>># featch data of all common statinos
        >>>camel_br = Camels("CAMELS-BR")
        >>>data = camel_br.all_dynamic_attributes()
        """

        if station == 'all':
            stations = self.common_stations()
        elif isinstance(station, list):
            stations = station
        else:
            stations = [int(station)]

        stations = {stn:self.dynamic_attributes for stn in stations}

        data = self.fetch(stations)

        return data

    def _download(self, overwrite=False):
        """Downloads the dataset. If already downloaded, then"""
        if os.path.exists(self.ds_dir):
            if overwrite:
                print(f"removing previous data directory {self.ds_dir} and downloading new")
                shutil.rmtree(self.ds_dir)
                self._download_and_unzip()
            else:
                print(f"""The directory {self.ds_dir} already exists.
                          Use overwrite=True to remove previously saved files and download again""")
        else:
            self._download_and_unzip()
        return

    def _download_and_unzip(self):
        os.makedirs(self.ds_dir)
        download_from_zenodo(self.ds_dir, self.DATASETS[self.name]['url'])
        self._unzip()
        return

    def _unzip(self):
        """unzip all the zipped files in a directory"""
        all_files = glob.glob(f"{self.ds_dir}/*.zip")
        for f in all_files:
            print(f"unziping {os.path.join(self.ds_dir, f)} to {os.path.join(self.ds_dir, f.split('.zip')[0])}")
            with zipfile.ZipFile(os.path.join(self.ds_dir, f), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(self.ds_dir, f.split('.zip')[0]))
        return

