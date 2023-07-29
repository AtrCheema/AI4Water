
import glob
from typing import Union, List

from ai4water.backend import pd, os
from .camels import Camels
from ..utils import check_attributes

# directory separator
SEP = os.sep


class CAMELS_BR(Camels):
    """
    This is a dataset of 593 Brazilian catchments with 67 static features
    and 12 dyanmic features for each catchment. The dyanmic features are
    timeseries from 1980-01-01 to 2018-12-31. This class
    downloads and processes CAMELS dataset of Brazil as provided by
    `VP Changas et al., 2020 <https://doi.org/10.5194/essd-12-2075-2020>`_

    Examples
    --------
    >>> from ai4water.datasets import CAMELS_BR
    >>> dataset = CAMELS_BR()
    >>> df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
    (14245, 12)
    # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    593
    # we can get data of 10% catchments as below
    >>> data = dataset.fetch(0.1, as_dataframe=True)
    >>> data.shape
    (170940, 59)
    # the data is multi-index with ``time`` and ``dynamic_features`` as indices
    >>> data.index.names == ['time', 'dynamic_features']
     True
    # get data by station id
    >>> df = dataset.fetch(stations='46035000', as_dataframe=True).unstack()
    >>> df.shape
    (14245, 12)
    # get names of available dynamic features
    >>> dataset.dynamic_features
    # get only selected dynamic features
    >>> df = dataset.fetch(1, as_dataframe=True,
    ... dynamic_features=['precipitation_cpc', 'evapotransp_mgb', 'temperature_mean', 'streamflow_m3s']).unstack()
    >>> df.shape
    (14245, 4)
    # get names of available static features
    >>> dataset.static_features
    # get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape
    (170940, 10)  # remember this is multi-indexed DataFrame
    # when we get both static and dynamic data, the returned data is a dictionary
    # with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='46035000', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
    ((1, 67), (170940, 1))

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

    def __init__(self, path=None):

        super().__init__(path=path, name="CAMELS_BR")
        self.path = path
        self._download()

        self._maybe_to_netcdf('camels_dyn_br')

    @property
    def _all_dirs(self):
        """All the folders in the dataset_directory"""
        return [f for f in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, f))]

    @property
    def static_dir(self):
        path = None
        for _dir in self._all_dirs:
            if "attributes" in _dir:
                # supposing that 'attributes' axist in only one file/folder in self.path
                path = os.path.join(self.path, f'{_dir}{SEP}{_dir}')
        return path

    @property
    def static_files(self):
        all_files = None
        if self.static_dir is not None:
            all_files = glob.glob(f"{self.static_dir}/*.txt")
        return all_files

    @property
    def dynamic_features(self) -> list:
        return list(CAMELS_BR.folders.keys())

    @property
    def static_attribute_categories(self):
        static_attrs = []
        for f in self.static_files:
            ff = str(os.path.basename(f).split('.txt')[0])
            static_attrs.append('_'.join(ff.split('_')[2:]))
        return static_attrs

    @property
    def static_features(self):
        static_fpath = os.path.join(self.path, 'static_features.csv')
        if not os.path.exists(static_fpath):
            files = glob.glob(
                f"{os.path.join(self.path, '01_CAMELS_BR_attributes', '01_CAMELS_BR_attributes')}/*.txt")
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
                all_files = os.listdir(os.path.join(self.path, f'{_dir}{SEP}{_dir}'))

        stations = []
        for f in all_files:
            stations.append(str(f.split('_')[0]))

        return stations

    def stations(self, to_exclude=None) -> list:
        """Returns a list of station ids which are common among all dynamic
        attributes.

        Example
        -------
        >>> dataset = CAMELS_BR()
        >>> stations = dataset.stations()
        """
        if to_exclude is not None:
            if not isinstance(to_exclude, list):
                assert isinstance(to_exclude, str)
                to_exclude = [to_exclude]
        else:
            to_exclude = []

        stations = {}
        for dyn_attr in self.dynamic_features:
            if dyn_attr not in to_exclude:
                stations[dyn_attr] = self.all_stations(dyn_attr)

        stns = list(set.intersection(*map(set, list(stations.values()))))
        return stns

    def _read_dynamic_from_csv(self,
                               stations,
                               attributes: Union[str, list] = 'all',
                               st=None,
                               en=None,
                               ):
        """
        returns the dynamic/time series attribute/attributes for one station id.

        Example
        -------
        >>> dataset = CAMELS_BR()
        >>> pcp = dataset.fetch_dynamic_features('10500000', 'precipitation_cpc')
        ... # fetch all time series data associated with a station.
        >>> x = dataset.fetch_dynamic_features('51560000', dataset.dynamic_features)

        """

        attributes = check_attributes(attributes, self.dynamic_features)

        dyn = {}
        for stn_id in stations:
            # making one separate dataframe for one station
            data = pd.DataFrame()
            for attr, _dir in self.folders.items():

                if attr in attributes:
                    path = os.path.join(self.path, f'{_dir}{SEP}{_dir}')
                    # supposing that the filename starts with stn_id and has .txt extension.
                    fname = [f for f in os.listdir(path) if f.startswith(str(stn_id)) and f.endswith('.txt')]
                    fname = fname[0]
                    if os.path.exists(os.path.join(path, fname)):
                        df = pd.read_csv(os.path.join(path, fname), sep=' ')
                        df.index = pd.to_datetime(df[['year', 'month', 'day']])
                        df.index.freq = pd.infer_freq(df.index)
                        df = df[st:en]
                        # only read one column which matches the attr
                        # todo, qual_flag maybe important
                        [df.pop(item) for item in df.columns if item != attr]
                        data = pd.concat([data, df], axis=1)
                    else:
                        raise FileNotFoundError(f"file {fname} not found at {path}")

            dyn[stn_id] = data

        return dyn

    def fetch_static_features(
            self,
            stn_id: Union[str, List[str]],
            features:Union[str, List[str]]=None
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
            stn_id : int/list
                station id whose attribute to fetch
            features : str/list
                name of attribute to fetch. Default is None, which will return all the
                attributes for a particular station of the specified category.
        Example
        -------
        >>> dataset = Camels()
        >>> df = dataset.fetch_static_features('11500000', 'climate')
        # read all static features of all stations
        >>> data = dataset.fetch_static_features(dataset.stations(), dataset.static_features)
        >>> data.shape
        (597, 67)

        """
        if isinstance(stn_id, int):
            station = [str(stn_id)]
        elif isinstance(stn_id, list):
            station = [str(stn) for stn in stn_id]
        elif isinstance(stn_id, str):
            station = [stn_id]
        else:
            raise ValueError

        attributes = check_attributes(features, self.static_features)

        static_fpath = os.path.join(self.path, 'static_features.csv')
        if not os.path.exists(static_fpath):
            files = glob.glob(
                f"{os.path.join(self.path, '01_CAMELS_BR_attributes', '01_CAMELS_BR_attributes')}/*.txt")
            static_df = pd.DataFrame()
            for f in files:
                _df = pd.read_csv(f, sep=' ', index_col='gauge_id')
                static_df = pd.concat([static_df, _df], axis=1)
            static_df.to_csv(static_fpath, index_label='gauge_id')
        else:
            static_df = pd.read_csv(static_fpath, index_col='gauge_id')

        static_df.index = static_df.index.astype(str)

        return pd.DataFrame(static_df.loc[station][attributes])
