
import json
import glob
from typing import Union, List

from ai4water.backend import os, pd, np

from .camels import Camels
from ..utils import check_attributes, download, sanity_check, _unzip

try:  # shapely may not be installed, as it may be difficult to isntall and is only needed for plotting data.
    from ai4water.preprocessing.spatial_utils import plot_shapefile
except ModuleNotFoundError:
    plot_shapefile = None

# directory separator
SEP = os.sep


class CAMELS_US(Camels):
    """
    This is a dataset of 671 US catchments with 59 static features
    and 8 dyanmic features for each catchment. The dyanmic features are
    timeseries from 1980-01-01 to 2014-12-31. This class
    downloads and processes CAMELS dataset of 671 catchments named as CAMELS
    from `ucar.edu <https://ral.ucar.edu/solutions/products/camels>`_
    following `Newman et al., 2015 <https://doi.org/10.5194/hess-19-209-2015>`_

    Examples
    --------
    >>> from ai4water.datasets import CAMELS_US
    >>> dataset = CAMELS_US()
    >>> df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
    (12784, 8)
    # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    671
    # we can get data of 10% catchments as below
    >>> data = dataset.fetch(0.1, as_dataframe=True)
    >>> data.shape
    (460488, 51)
    # the data is multi-index with ``time`` and ``dynamic_features`` as indices
    >>> data.index.names == ['time', 'dynamic_features']
     True
    # get data by station id
    >>> df = dataset.fetch(stations='11478500', as_dataframe=True).unstack()
    >>> df.shape
    (12784, 8)
    # get names of available dynamic features
    >>> dataset.dynamic_features
    # get only selected dynamic features
    >>> df = dataset.fetch(1, as_dataframe=True,
    ... dynamic_features=['prcp(mm/day)', 'srad(W/m2)', 'tmax(C)', 'tmin(C)', 'Flow']).unstack()
    >>> df.shape
    (12784, 5)
    # get names of available static features
    >>> dataset.static_features
    # get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape
    (102272, 10)  # remember this is multi-indexed DataFrame
    # when we get both static and dynamic data, the returned data is a dictionary
    # with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='11478500', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
    ((1, 59), (102272, 1))

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

    dynamic_features = ['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)',
                        'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)', 'Flow']

    def __init__(
            self,
            data_source:str='basin_mean_daymet',
            path=None):

        """
        parameters
        ----------
        path : str
            If the data is alredy downloaded then provide the complete
            path to it. If None, then the data will be downloaded.
            The data is downloaded once and therefore susbsequent
            calls to this class will not download the data unless
            ``overwrite`` is set to True.
        data_source : str
            allowed values are
                - basin_mean_daymet
                - basin_mean_maurer
                - basin_mean_nldas
                - basin_mean_v1p15_daymet
                - basin_mean_v1p15_nldas
                - elev_bands
                - hru
        """
        assert data_source in self.folders, f'allwed data sources are {self.folders.keys()}'
        self.data_source = data_source

        super().__init__(path=path, name="CAMELS_US")

        self.path = path

        if os.path.exists(self.path):
            print(f"dataset is already downloaded at {self.path}")
        else:
            download(self.url, os.path.join(self.camels_dir, f'CAMELS_US{SEP}CAMELS_US.zip'))
            download(self.catchment_attr_url, os.path.join(self.camels_dir, f"CAMELS_US{SEP}catchment_attrs.zip"))
            _unzip(self.path)

        self.attr_dir = os.path.join(self.path, f'catchment_attrs{SEP}camels_attributes_v2.0')
        self.dataset_dir = os.path.join(self.path, f'CAMELS_US{SEP}basin_dataset_public_v1p2')

        self._maybe_to_netcdf('camels_us_dyn')

    @property
    def start(self):
        return "19800101"

    @property
    def end(self):
        return "20141231"

    @property
    def static_features(self):
        static_fpath = os.path.join(self.path, 'static_features.csv')
        if not os.path.exists(static_fpath):
            files = glob.glob(f"{os.path.join(self.path, 'catchment_attrs', 'camels_attributes_v2.0')}/*.txt")
            cols = []
            for f in files:
                _df = pd.read_csv(f, sep=';', index_col='gauge_id', nrows=1)
                cols += list(_df.columns)
        else:
            df = pd.read_csv(static_fpath, index_col='gauge_id', nrows=1)
            cols = list(df.columns)

        return cols

    def q_mmd(
            self,
            stations: Union[str, List[str]] = None
    )->pd.DataFrame:
        """
        returns streamflow in the units of milimeter per day. This is obtained
        by diving ``Flow``/area

        parameters
        ----------
        stations : str/list
            name/names of stations. Default is None, which will return
            area of all stations

        Returns
        --------
        pd.DataFrame
            a pandas DataFrame whose indices are time-steps and columns
            are catchment/station ids.

        """
        stations = check_attributes(stations, self.stations())
        q = self.fetch_stations_features(stations,
                                           dynamic_features='Flow',
                                           as_dataframe=True)
        q.index = q.index.get_level_values(0)
        area_m2 = self.area(stations) * 1e6  # area in m2
        q = (q / area_m2) * 86400  # cms to m/day
        return q  * 1e3  # to mm/day

    def area(
            self,
            stations: Union[str, List[str]] = None
    ) ->pd.Series:
        """
        Returns (GAGES II) area (Km2) of all catchments as pandas series

        parameters
        ----------
        stations : str/list
            name/names of stations. Default is None, which will return
            area of all stations

        Returns
        --------
        pd.Series
            a pandas series whose indices are catchment ids and values
            are areas of corresponding catchments.

        Examples
        ---------
        >>> from ai4water.datasets import CAMELS_US
        >>> dataset = CAMELS_US()
        >>> dataset.area()  # returns area of all stations
        >>> dataset.stn_coords('1030500')  # returns area of station whose id is 912101A
        >>> dataset.stn_coords(['1030500', '14400000'])  # returns area of two stations
        """
        stations = check_attributes(stations, self.stations())

        fpath = os.path.join(self.path,
                             'catchment_attrs',
                             'camels_attributes_v2.0',
                             'camels_topo.txt')

        df = pd.read_csv(fpath, sep=";",
                         dtype={
                             'gauge_id': str,
                             'gauge_lat': float,
                             'gauge_lon': float,
                             'elev_mean': float,
                             'slope_mean': float,
                             'area_gages2': float
                         })
        df.index = df['gauge_id']

        s = df.loc[stations, 'area_gages2']
        s.name = 'area'
        return s

    def stn_coords(
            self,
            stations:Union[str, List[str]] = None
    ) ->pd.DataFrame:
        """
        returns coordinates of stations as DataFrame
        with ``long`` and ``lat`` as columns.

        Parameters
        ----------
        stations :
            name/names of stations. If not given, coordinates
            of all stations will be returned.

        Returns
        -------
        coords :
            pandas DataFrame with ``long`` and ``lat`` columns.
            The length of dataframe will be equal to number of stations
            wholse coordinates are to be fetched.

        Examples
        --------
        >>> dataset = CAMELS_US()
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('1030500')  # returns coordinates of station whose id is 912101A
        >>> dataset.stn_coords(['1030500', '14400000'])  # returns coordinates of two stations

        """
        stations = check_attributes(stations, self.stations())
        fpath = os.path.join(self.path,
                             'catchment_attrs',
                             'camels_attributes_v2.0',
                             'camels_topo.txt')

        df = pd.read_csv(fpath, sep=";",
                         dtype={
                             'gauge_id': str,
                             'gauge_lat': float,
                             'gauge_lon': float,
                             'elev_mean': float,
                             'slope_mean': float
                         })
        df.index = df['gauge_id']
        df = df[['gauge_lat', 'gauge_lon']]
        df.columns = ['lat', 'long']
        return df.loc[stations, :]

    def stations(self) -> list:
        stns = []
        for _dir in os.listdir(os.path.join(self.dataset_dir, 'usgs_streamflow')):
            cat = os.path.join(self.dataset_dir, f'usgs_streamflow{SEP}{_dir}')
            stns += [fname.split('_')[0] for fname in os.listdir(cat)]

        # remove stations for which static values are not available
        for stn in ['06775500', '06846500', '09535100']:
            stns.remove(stn)

        return stns

    def _read_dynamic_from_csv(self,
                               stations,
                               dynamic_features: Union[str, list] = 'all',
                               st=None,
                               en=None,
                               ):
        dyn = {}
        for station in stations:

            # attributes = check_attributes(dynamic_features, self.dynamic_features)

            assert isinstance(station, str)
            df = None
            df1 = None
            dir_name = self.folders[self.data_source]
            for cat in os.listdir(os.path.join(self.dataset_dir, dir_name)):
                cat_dirs = os.listdir(os.path.join(self.dataset_dir, f'{dir_name}{SEP}{cat}'))
                stn_file = f'{station}_lump_cida_forcing_leap.txt'
                if stn_file in cat_dirs:
                    df = pd.read_csv(os.path.join(self.dataset_dir,
                                                  f'{dir_name}{SEP}{cat}{SEP}{stn_file}'),
                                     sep="\s+|;|:",
                                     skiprows=4,
                                     engine='python',
                                     names=['Year', 'Mnth', 'Day', 'Hr', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m2)',
                                            'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)'],
                                     )
                    df.index = pd.to_datetime(
                        df['Year'].map(str) + '-' + df['Mnth'].map(str) + '-' + df['Day'].map(str))

            flow_dir = os.path.join(self.dataset_dir, 'usgs_streamflow')
            for cat in os.listdir(flow_dir):
                cat_dirs = os.listdir(os.path.join(flow_dir, cat))
                stn_file = f'{station}_streamflow_qc.txt'
                if stn_file in cat_dirs:
                    fpath = os.path.join(flow_dir, f'{cat}{SEP}{stn_file}')
                    df1 = pd.read_csv(fpath, sep="\s+|;|:'",
                                      names=['station', 'Year', 'Month', 'Day', 'Flow', 'Flag'],
                                      engine='python')
                    df1.index = pd.to_datetime(
                        df1['Year'].map(str) + '-' + df1['Month'].map(str) + '-' + df1['Day'].map(str))

            out_df = pd.concat([df[['dayl(s)',
                                    'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']],
                                df1['Flow']],
                               axis=1)
            dyn[station] = out_df

        return dyn

    def fetch_static_features(
            self,
            stn_id: Union[str, List[str]]="all",
            features:Union[str, List[str]]=None
    ):
        """
        gets one or more static features of one or more stations

        Parameters
        ----------
            stn_id : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Examples
        --------
            >>> from ai4water.datasets import CAMELS_US
            >>> camels = CAMELS_US()
            >>> st_data = camels.fetch_static_features('11532500')
            >>> st_data.shape
               (1, 59)
            get names of available static features
            >>> camels.static_features
            get specific features of one station
            >>> static_data = camels.fetch_static_features('11528700',
            >>> features=['area_gages2', 'geol_porostiy', 'soil_conductivity', 'elev_mean'])
            >>> static_data.shape
               (1, 4)
            get names of allstations
            >>> all_stns = camels.stations()
            >>> len(all_stns)
               671
            >>> all_static_data = camels.fetch_static_features(all_stns)
            >>> all_static_data.shape
               (671, 59)
        """
        features = check_attributes(features, self.static_features)

        static_fpath = os.path.join(self.path, 'static_features.csv')
        if not os.path.exists(static_fpath):
            files = glob.glob(f"{os.path.join(self.path, 'catchment_attrs', 'camels_attributes_v2.0')}/*.txt")
            static_df = pd.DataFrame()
            for f in files:
                # index should be read as string
                idx = pd.read_csv(f, sep=';', usecols=['gauge_id'], dtype=str)
                _df = pd.read_csv(f, sep=';', index_col='gauge_id')
                _df.index = idx['gauge_id']
                static_df = pd.concat([static_df, _df], axis=1)
            static_df.to_csv(static_fpath, index_label='gauge_id')
        else:  # index should be read as string bcs it has 0s at the start
            idx = pd.read_csv(static_fpath, usecols=['gauge_id'], dtype=str)
            static_df = pd.read_csv(static_fpath, index_col='gauge_id')
            static_df.index = idx['gauge_id']

        static_df.index = static_df.index.astype(str)

        if stn_id == "all":
            stn_id = self.stations()

        df = static_df.loc[stn_id][features]
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df).transpose()

        return df


class CAMELS_GB(Camels):
    """
    This is a dataset of 671 catchments with 290 static features
    and 10 dyanmic features for each catchment. The dyanmic features are
    timeseries from 1957-01-01 to 2018-12-31. This dataset must be manually
    downloaded by the user. The path of the downloaded folder must be provided
    while initiating this class.

    >>> from ai4water.datasets import CAMELS_GB
    >>> dataset = CAMELS_GB("path/to/CAMELS_GB")
    >>> data = dataset.fetch(0.1, as_dataframe=True)
    >>> data.shape
     (164360, 67)
    >>> data.index.names == ['time', 'dynamic_features']
    True
    >>> df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
    (16436, 10)
    # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    671
    # get data by station id
    >>> df = dataset.fetch(stations='97002', as_dataframe=True).unstack()
    >>> df.shape
    (16436, 10)
    # get names of available dynamic features
    >>> dataset.dynamic_features
    # get only selected dynamic features
    >>> df = dataset.fetch(1, as_dataframe=True,
    ... dynamic_features=['windspeed', 'temperature', 'pet', 'precipitation', 'discharge_vol']).unstack()
    >>> df.shape
    (16436, 5)
    # get names of available static features
    >>> dataset.static_features
    # get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape
    (164360, 10)  # remember this is multi-indexed DataFrame
    # when we get both static and dynamic data, the returned data is a dictionary
    # with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='97002', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
    ((1, 290), (164360, 1))
    """
    dynamic_features = ["precipitation", "pet", "temperature", "discharge_spec",
                        "discharge_vol", "peti",
                        "humidity", "shortwave_rad", "longwave_rad", "windspeed"]

    def __init__(self, path=None):
        """
        parameters
        ------------
        path : str
            If the data is alredy downloaded then provide the complete
            path to it. If None, then the data will be downloaded.
            The data is downloaded once and therefore susbsequent
            calls to this class will not download the data unless
            ``overwrite`` is set to True.
        """
        super().__init__(name="CAMELS_GB", path=path)

        self._maybe_to_netcdf('camels_gb_dyn')

    @property
    def path(self):
        """Directory where a particular dataset will be saved. """
        return self._path

    @path.setter
    def path(self, x):
        sanity_check('CAMELS-GB', x)
        self._path = x

    @property
    def static_attribute_categories(self) -> list:
        features = []
        path = os.path.join(self.path, 'data')
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)) and f.endswith('csv'):
                features.append(f.split('_')[2])

        return features

    @property
    def start(self):
        return pd.Timestamp("19701001")

    @property
    def end(self):
        return pd.Timestamp("20150930")

    @property
    def static_features(self):
        files = glob.glob(f"{os.path.join(self.path, 'data')}/*.csv")
        cols = []
        for f in files:
            if 'static_features.csv' not in f:
                df = pd.read_csv(f, nrows=1, index_col='gauge_id')
                cols += (list(df.columns))
        return cols

    def stations(self, to_exclude=None):
        # CAMELS_GB_hydromet_timeseries_StationID_number
        path = os.path.join(self.path, f'data{SEP}timeseries')
        gauge_ids = []
        for f in os.listdir(path):
            gauge_ids.append(f.split('_')[4])

        return gauge_ids

    def _mmd_feature_name(self) ->str:
        return 'discharge_spec'

    def area(
            self,
            stations: Union[str, List[str]] = None
    ) ->pd.Series:
        """
        Returns area (Km2) of all catchments as pandas series

        parameters
        ----------
        stations : str/list
            name/names of stations. Default is None, which will return
            area of all stations

        Returns
        --------
        pd.Series
            a pandas series whose indices are catchment ids and values
            are areas of corresponding catchments.

        Examples
        ---------
        >>> from ai4water.datasets import CAMELS_GB
        >>> dataset = CAMELS_GB()
        >>> dataset.area()  # returns area of all stations
        >>> dataset.stn_coords('101005')  # returns area of station whose id is 912101A
        >>> dataset.stn_coords(['101005', '54002'])  # returns area of two stations
        """
        stations = check_attributes(stations, self.stations())

        static_fpath = os.path.join(self.path,
                                    'data',
                                    'CAMELS_GB_topographic_attributes.csv')

        df = pd.read_csv(static_fpath)

        df.index = df['gauge_id'].astype(str)

        s = df.loc[stations, 'area']

        return s

    def stn_coords(
            self,
            stations:Union[str, List[str]] = None
    ) ->pd.DataFrame:
        """
        returns coordinates of stations as DataFrame
        with ``long`` and ``lat`` as columns.

        Parameters
        ----------
        stations :
            name/names of stations. If not given, coordinates
            of all stations will be returned.

        Examples
        --------
        >>> dataset = CAMELS_GB()
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('101005')  # returns coordinates of station whose id is 912101A
        >>> dataset.stn_coords(['101005', '54002'])  # returns coordinates of two stations
        """
        stations = check_attributes(stations, self.stations())

        static_fpath = os.path.join(self.path,
                                    'data',
                                    'CAMELS_GB_topographic_attributes.csv')

        df = pd.read_csv(static_fpath)

        df.index = df['gauge_id'].astype(str)
        df = df.loc[stations, ['gauge_lat', 'gauge_lon']]
        df.columns = ['lat', 'long']
        return df.loc[stations, :]

    def _read_dynamic_from_csv(
            self,
            stations,
            features: Union[str, list] = 'all',
            st=None,
            en=None,
    ):
        """Fetches dynamic attribute/features of one or more station."""
        dyn = {}
        for stn_id in stations:
            # making one separate dataframe for one station
            path = os.path.join(self.path, f"data{SEP}timeseries")
            fname = None
            for f in os.listdir(path):
                if stn_id in f:
                    fname = f
                    break

            df = pd.read_csv(os.path.join(path, fname), index_col='date')
            df.index = pd.to_datetime(df.index)
            df.index.freq = pd.infer_freq(df.index)

            dyn[stn_id] = df

        return dyn

    def fetch_static_features(
            self,
            stn_id: Union[str, List[str]] = "all",
            features:Union[str, List[str]]="all"
    ) -> pd.DataFrame:
        """
        Fetches static features of one or more stations for one or
        more category as dataframe.

        Parameters
        ----------
            stn_id : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Examples
        ---------
        >>> from ai4water.datasets import CAMELS_GB
        >>> dataset = CAMELS_GB(path="path/to/CAMELS_GB")
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            671
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (671, 290)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('85004')
        >>> static_data.shape
           (1, 290)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['area', 'elev_mean'])
        >>> static_data.shape
           (671, 2)
        """

        features = check_attributes(features, self.static_features)
        static_fname = 'static_features.csv'
        static_fpath = os.path.join(self.path, 'data', static_fname)
        if os.path.exists(static_fpath):
            static_df = pd.read_csv(static_fpath, index_col='gauge_id')
        else:
            files = glob.glob(f"{os.path.join(self.path, 'data')}/*.csv")
            static_df = pd.DataFrame()
            for f in files:
                _df = pd.read_csv(f, index_col='gauge_id')
                static_df = pd.concat([static_df, _df], axis=1)
            static_df.to_csv(static_fpath)

        if stn_id == "all":
            stn_id = self.stations()

        if isinstance(stn_id, str):
            station = [stn_id]
        elif isinstance(stn_id, int):
            station = [str(stn_id)]
        elif isinstance(stn_id, list):
            station = [str(stn) for stn in stn_id]
        else:
            raise ValueError

        static_df.index = static_df.index.astype(str)

        return static_df.loc[station][features]


class CAMELS_AUS(Camels):
    """
    This is a dataset of 222 Australian catchments with 161 static features
    and 26 dyanmic features for each catchment. The dyanmic features are
    timeseries from 1957-01-01 to 2018-12-31. This class Reads CAMELS-AUS dataset of
    `Fowler et al., 2020 <https://doi.org/10.5194/essd-13-3847-2021>`_
    dataset.

    Examples
    --------
    >>> from ai4water.datasets import CAMELS_AUS
    >>> dataset = CAMELS_AUS()
    >>> df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
       (21184, 26)
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       222
    ... # get data of 10 % of stations as dataframe
    >>> df = dataset.fetch(0.1, as_dataframe=True)
    >>> df.shape
       (550784, 22)
    ... # The returned dataframe is a multi-indexed data
    >>> df.index.names == ['time', 'dynamic_features']
        True
    ... # get data by station id
    >>> df = dataset.fetch(stations='224214A', as_dataframe=True).unstack()
    >>> df.shape
        (21184, 26)
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> data = dataset.fetch(1, as_dataframe=True,
    ...  dynamic_features=['tmax_AWAP', 'precipitation_AWAP', 'et_morton_actual_SILO', 'streamflow_MLd']).unstack()
    >>> data.shape
       (21184, 4)
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape  # remember this is a multiindexed dataframe
       (21184, 260)
    # when we get both static and dynamic data, the returned data is a dictionary
    # with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='224214A', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
    >>> ((1, 161), (550784, 1))
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

        'et_morton_actual_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries{SEP}et_morton_actual_SILO',
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

    def __init__(
            self,
            path: str = None,
            to_netcdf:bool = True
    ):
        """
        Arguments:
            path: path where the CAMELS-AUS dataset has been downloaded. This path
                must contain five zip files and one xlsx file. If None, then the
                data will be downloaded.
            to_netcdf :
        """
        if path is not None:
            assert isinstance(path, str), f'path must be string like but it is "{path}" of type {path.__class__.__name__}'
            if not os.path.exists(path) or len(os.listdir(path)) < 2:
                raise FileNotFoundError(f"The path {path} does not exist")
        self.path = path

        super().__init__(path=path)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for _file, url in self.urls.items():
            fpath = os.path.join(self.path, _file)
            if not os.path.exists(fpath):
                download(url + _file, fpath)

        _unzip(self.path)

        if to_netcdf:
            self._maybe_to_netcdf('camels_aus_dyn')

    @property
    def start(self):
        return "19570101"

    @property
    def end(self):
        return "20181231"

    @property
    def location(self):
        return "Australia"

    def stations(self, as_list=True) -> list:
        fname = os.path.join(self.path, f"01_id_name_metadata{SEP}01_id_name_metadata{SEP}id_name_metadata.csv")
        df = pd.read_csv(fname)
        if as_list:
            return df['station_id'].to_list()
        else:
            return df

    @property
    def static_attribute_categories(self):
        features = []
        path = os.path.join(self.path, f'04_attributes{SEP}04_attributes')
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)) and f.endswith('csv'):
                f = str(f.split('.csv')[0])
                features.append(''.join(f.split('_')[2:]))
        return features

    @property
    def static_features(self) -> list:
        static_fpath = os.path.join(self.path, 'CAMELS_AUS_Attributes-Indices_MasterTable.csv')

        df = pd.read_csv(static_fpath, index_col='station_id', nrows=1)
        cols = list(df.columns)

        return cols

    @property
    def dynamic_features(self) -> list:
        return list(self.folders.keys())

    def q_mmd(
            self,
            stations: Union[str, List[str]] = None
    )->pd.DataFrame:
        """
        returns streamflow in the units of milimeter per day. This is obtained
        by diving q_cms/area

        parameters
        ----------
        stations : str/list
            name/names of stations. Default is None, which will return
            area of all stations

        Returns
        --------
        pd.DataFrame
            a pandas DataFrame whose indices are time-steps and columns
            are catchment/station ids.

        """
        stations = check_attributes(stations, self.stations())
        q = self.fetch_stations_features(stations,
                                           dynamic_features='streamflow_MLd',
                                           as_dataframe=True)
        q.index = q.index.get_level_values(0)
        q = q * 0.01157  # mega liter per day to cms
        area_m2 = self.area(stations) * 1e6  # area in m2
        q = (q / area_m2) * 86400  # to m/day
        return q * 1e3  # to mm/day

    def area(
            self,
            stations: Union[str, List[str]] = None
    ) ->pd.Series:
        """
        Returns area of all catchments as pandas series

        parameters
        ----------
        stations : str/list
            name/names of stations. Default is None, which will return
            area of all stations

        Returns
        --------
        pd.Series
            a pandas series whose indices are catchment ids and values
            are areas of corresponding catchments.

        Examples
        ---------
        >>> from ai4water.datasets import CAMELS_AUS
        >>> dataset = CAMELS_AUS()
        >>> dataset.area()  # returns area of all stations
        >>> dataset.stn_coords('912101A')  # returns area of station whose id is 912101A
        >>> dataset.stn_coords(['G0050115', '912101A'])  # returns area of two stations
        """
        stations = check_attributes(stations, self.stations())

        static_fpath = os.path.join(self.path,
                                    'CAMELS_AUS_Attributes-Indices_MasterTable.csv')

        df = pd.read_csv(static_fpath, index_col='station_id')
        s = df.loc[stations, 'catchment_area']
        s.name = 'area'
        return s

    def stn_coords(
            self,
            stations:Union[str, List[str]] = None
    ) ->pd.DataFrame:
        """
        returns coordinates of stations as DataFrame
        with ``long`` and ``lat`` as columns.

        Parameters
        ----------
        stations :
            name/names of stations. If not given, coordinates
            of all stations will be returned.

        Returns
        -------
        coords :
            pandas DataFrame with ``long`` and ``lat`` columns.
            The length of dataframe will be equal to number of stations
            wholse coordinates are to be fetched.

        Examples
        --------
        >>> dataset = CAMELS_AUS()
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('912101A')  # returns coordinates of station whose id is 912101A
        >>> dataset.stn_coords(['G0050115', '912101A'])  # returns coordinates of two stations
        """
        stations = check_attributes(stations, self.stations())

        static_fpath = os.path.join(self.path,
                                    'CAMELS_AUS_Attributes-Indices_MasterTable.csv')

        df = pd.read_csv(static_fpath,
                         index_col='station_id')
        df = df.loc[stations, ['lat_outlet', 'long_outlet']]
        df.columns = ['lat', 'long']
        return df.loc[stations, :]

    def _read_static(self, stations, features,
                     st=None, en=None):

        features = check_attributes(features, self.static_features)
        static_fname = 'CAMELS_AUS_Attributes-Indices_MasterTable.csv'
        static_fpath = os.path.join(self.path, static_fname)
        static_df = pd.read_csv(static_fpath, index_col='station_id')

        static_df.index = static_df.index.astype(str)
        df = static_df.loc[stations][features]
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df).transpose()

        return self.to_ts(df, st, en)

    def _read_dynamic_from_csv(self, stations, dynamic_features, **kwargs):

        dyn_attrs = {}
        dyn = {}
        for _attr in dynamic_features:
            _path = os.path.join(self.path, f'{self.folders[_attr]}.csv')
            _df = pd.read_csv(_path, na_values=['-99.99'])
            _df.index = pd.to_datetime(_df[['year', 'month', 'day']])
            [_df.pop(col) for col in ['year', 'month', 'day']]

            dyn_attrs[_attr] = _df

        # making one separate dataframe for one station
        for stn in stations:
            stn_df = pd.DataFrame()
            for attr, attr_df in dyn_attrs.items():
                if attr in dynamic_features:
                    stn_df[attr] = attr_df[stn]
            dyn[stn] = stn_df

        return dyn

    def fetch_static_features(
            self,
            stn_id: Union[str, List[str]],
            features:Union[str, List[str]]="all",
            **kwargs
    ) -> pd.DataFrame:
        """Fetches static features of one or more stations as dataframe.

        Parameters
        ----------
            stn_id : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Examples
        ---------
        >>> from ai4water.datasets import CAMELS_AUS
        >>> dataset = CAMELS_AUS()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            222
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (222, 161)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('305202')
        >>> static_data.shape
           (1, 161)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['catchment_di', 'elev_mean'])
        >>> static_data.shape
           (222, 2)

        """

        if stn_id == "all":
            stn_id = self.stations()

        return self._read_static(stn_id, features)

    def plot(self, what, stations=None, **kwargs):
        assert what in ['outlets', 'boundaries']
        f1 = os.path.join(self.path,
                          f'02_location_boundary_area{SEP}02_location_boundary_area{SEP}shp{SEP}CAMELS_AUS_BasinOutlets_adopted.shp')
        f2 = os.path.join(self.path,
                          f'02_location_boundary_area{SEP}02_location_boundary_area{SEP}shp{SEP}bonus data{SEP}Australia_boundaries.shp')

        if plot_shapefile is not None:
            return plot_shapefile(f1, bbox_shp=f2, recs=stations, rec_idx=0, **kwargs)
        else:
            raise ModuleNotFoundError("Shapely must be installed in order to plot the datasets.")


class CAMELS_CL(Camels):
    """
    This is a dataset of 516 catchments with
    104 static features and 12 dyanmic features for each catchment.
    The dyanmic features are timeseries from 1913-02-15 to 2018-03-09.
    This class downloads and processes CAMELS dataset of Chile following the work of
    `Alvarez-Garreton et al., 2018 <https://doi.org/10.5194/hess-22-5817-2018>`_ .

    Examples
    ---------
    >>> from ai4water.datasets import CAMELS_CL
    >>> dataset = CAMELS_CL()
    >>> df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
        (38374, 12)
    # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    516
    # we can get data of 10% catchments as below
    >>> data = dataset.fetch(0.1, as_dataframe=True)
    >>> data.shape
    (460488, 51)
    # the data is multi-index with ``time`` and ``dynamic_features`` as indices
    >>> df.index.names == ['time', 'dynamic_features']
     True
    # get data by station id
    >>> df = dataset.fetch(stations='8350001', as_dataframe=True).unstack()
    >>> df.shape
    (38374, 12)
    # get names of available dynamic features
    >>> dataset.dynamic_features
    # get only selected dynamic features
    >>> df = dataset.fetch(1, as_dataframe=True,
    ... dynamic_features=['pet_hargreaves', 'precip_tmpa', 'tmean_cr2met', 'streamflow_m3s']).unstack()
    >>> df.shape
    (38374, 4)
    # get names of available static features
    >>> dataset.static_features
    # get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape
    (460488, 10)
    # when we get both static and dynamic data, the returned data is a dictionary
    # with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='8350001', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
    >>> ((1, 104), (460488, 1))

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

    dynamic_features = ['streamflow_m3s', 'streamflow_mm',
                        'precip_cr2met', 'precip_chirps', 'precip_mswep', 'precip_tmpa',
                        'tmin_cr2met', 'tmax_cr2met', 'tmean_cr2met',
                        'pet_8d_modis', 'pet_hargreaves',
                        'swe'
                        ]

    def __init__(self,
                 path: str = None
                 ):
        """
        Arguments:
            path: path where the CAMELS-CL dataset has been downloaded. This path must
                  contain five zip files and one xlsx file.
        """

        super().__init__(path=path)
        self.path = path

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for _file, url in self.urls.items():
            fpath = os.path.join(self.path, _file)
            if not os.path.exists(fpath):
                download(url + _file, fpath)
        _unzip(self.path)

        self.dyn_fname = os.path.join(self.path, 'camels_cl_dyn.nc')
        self._maybe_to_netcdf('camels_cl_dyn')

    @property
    def _all_dirs(self):
        """All the folders in the dataset_directory"""
        return [f for f in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, f))]

    @property
    def start(self):
        return "19130215"

    @property
    def end(self):
        return "20180309"

    @property
    def location(self):
        return "Chile"

    @property
    def static_features(self) -> list:
        path = os.path.join(self.path, f"1_CAMELScl_attributes{SEP}1_CAMELScl_attributes.txt")
        df = pd.read_csv(path, sep='\t', index_col='gauge_id')
        return df.index.to_list()

    def _mmd_feature_name(self) ->str:
        return 'streamflow_mm'

    def area(
            self,
            stations: Union[str, List[str]] = None
    ) ->pd.Series:
        """
        Returns area (Km2) of all catchments as pandas series

        parameters
        ----------
        stations : str/list
            name/names of stations. Default is None, which will return
            area of all stations

        Returns
        --------
        pd.Series
            a pandas series whose indices are catchment ids and values
            are areas of corresponding catchments.

        Examples
        ---------
        >>> from ai4water.datasets import CAMELS_CL
        >>> dataset = CAMELS_CL()
        >>> dataset.area()  # returns area of all stations
        >>> dataset.stn_coords('12872001')  # returns area of station whose id is 912101A
        >>> dataset.stn_coords(['12872001', '12876004'])  # returns area of two stations
        """
        stations = check_attributes(stations, self.stations())

        fpath = os.path.join(self.path,
                             '1_CAMELScl_attributes',
                             '1_CAMELScl_attributes.txt')
        df = pd.read_csv(fpath, sep='\t', index_col='gauge_id')
        df.columns = [column.strip() for column in df.columns]
        s = df.loc['area', stations]
        return s.astype(float)

    def stn_coords(
            self,
            stations:Union[str, List[str]] = None
    ) ->pd.DataFrame:
        """
        returns coordinates of stations as DataFrame
        with ``long`` and ``lat`` as columns.

        Parameters
        ----------
        stations :
            name/names of stations. If not given, coordinates
            of all stations will be returned.

        Returns
        -------
        coords :
            pandas DataFrame with ``long`` and ``lat`` columns.
            The length of dataframe will be equal to number of stations
            wholse coordinates are to be fetched.

        Examples
        --------
        >>> dataset = CAMELS_CL()
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('12872001')  # returns coordinates of station whose id is 912101A
        >>> dataset.stn_coords(['12872001', '12876004'])  # returns coordinates of two stations
        """
        fpath = os.path.join(self.path,
                             '1_CAMELScl_attributes',
                             '1_CAMELScl_attributes.txt')
        df = pd.read_csv(fpath, sep='\t', index_col='gauge_id')
        df = df.loc[['gauge_lat', 'gauge_lon'], :].transpose()
        df.columns = ['lat', 'long']
        stations = check_attributes(stations, self.stations())
        df.index  = [index.strip() for index in df.index]
        return df.loc[stations, :]

    def stations(self) -> list:
        """
        Tells all station ids for which a data of a specific attribute is available.
        """
        stn_fname = os.path.join(self.path, 'stations.json')
        if not os.path.exists(stn_fname):
            _stations = {}
            for dyn_attr in self.dynamic_features:
                for _dir in self._all_dirs:
                    if dyn_attr in _dir:
                        fname = os.path.join(self.path, f"{_dir}{SEP}{_dir}.txt")
                        df = pd.read_csv(fname, sep='\t', nrows=2, index_col='gauge_id')
                        _stations[dyn_attr] = list(df.columns)

            stns = list(set.intersection(*map(set, list(_stations.values()))))
            with open(stn_fname, 'w') as fp:
                json.dump(stns, fp)
        else:
            with open(stn_fname, 'r') as fp:
                stns = json.load(fp)
        return stns

    def _read_dynamic_from_csv(self, stations, dynamic_features, st=None, en=None):

        dyn = {}
        st, en = self._check_length(st, en)

        assert all(stn in self.stations() for stn in stations)

        dynamic_features = check_attributes(dynamic_features, self.dynamic_features)

        # reading all dynnamic features
        dyn_attrs = {}
        for attr in dynamic_features:
            fname = [f for f in self._all_dirs if '_' + attr in f][0]
            fname = os.path.join(self.path, f'{fname}{SEP}{fname}.txt')
            _df = pd.read_csv(fname, sep='\t', index_col=['gauge_id'], na_values=" ")
            _df.index = pd.to_datetime(_df.index)
            dyn_attrs[attr] = _df[st:en]

        # making one separate dataframe for one station
        for stn in stations:
            stn_df = pd.DataFrame()
            for attr, attr_df in dyn_attrs.items():
                if attr in dynamic_features:
                    stn_df[attr] = attr_df[stn]
            dyn[stn] = stn_df[st:en]

        return dyn

    def _read_static(self, stations: list, features: list) -> pd.DataFrame:
        # overwritten for speed
        path = os.path.join(self.path, f"1_CAMELScl_attributes{SEP}1_CAMELScl_attributes.txt")
        _df = pd.read_csv(path, sep='\t', index_col='gauge_id')

        stns_df = []
        for stn in stations:
            df = pd.DataFrame()
            if stn in _df:
                df[stn] = _df[stn]
            elif ' ' + stn in _df:
                df[stn] = _df[' ' + stn]

            stns_df.append(df.transpose()[features])

        stns_df = pd.concat(stns_df)
        return stns_df

    def fetch_static_features(
            self,
            stn_id: Union[str, List[str]]= "all",
            features:Union[str, List[str]]=None
    ):
        """
        Returns static features of one or more stations.

        Parameters
        ----------
            stn_id : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Examples
        ---------
        >>> from ai4water.datasets import CAMELS_CL
        >>> dataset = CAMELS_CL()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            516
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (516, 104)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('11315001')
        >>> static_data.shape
           (1, 104)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['slope_mean', 'area'])
        >>> static_data.shape
           (516, 2)
        >>> data = dataset.fetch_static_features('2110002', features=['slope_mean', 'area'])
        >>> data.shape
           (1, 2)

        """
        features = check_attributes(features, self.static_features)

        if stn_id == "all":
            stn_id = self.stations()

        if isinstance(stn_id, str):
            stn_id = [stn_id]

        return self._read_static(stn_id, features)


class CAMELS_CH(Camels):
    """
    Rainfall runoff dataset of Swiss catchments. It consists of 331 catchments.

    Examples
    ---------
    >>> from ai4water.datasets import CAMELS_CH
    >>> dataset = CAMELS_CH()
    >>> data = dataset.fetch(0.1, as_dataframe=True)
    >>> data.shape
    (128560, 10)
    >>> data.index.names == ['time', 'dynamic_features']
    True
    >>> df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
    (8036, 9)
    # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    331
    # get data by station id
    >>> df = dataset.fetch(stations='2004', as_dataframe=True).unstack()
    >>> df.shape
    (8036, 9)
    # get names of available dynamic features
    >>> dataset.dynamic_features
    # get only selected dynamic features
    >>> df = dataset.fetch(1, as_dataframe=True, dynamic_features=['precipitation(mm/d)', 'temperature_mean(°C)', 'discharge_vol(m3/s)']).unstack()
    >>> df.shape
    (8036, 3)
    # get names of available static features
    >>> dataset.static_features
    # get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape
    (72324, 10)  # remember this is multi-indexed DataFrame
    # when we get both static and dynamic data, the returned data is a dictionary
    # with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='2004', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
    ((1, 209), (72324, 1))


    """
    url = "https://zenodo.org/record/7957061"

    def __init__(
            self,
            path=None,
            prefix=None,
            overwrite:bool = False,
            to_netcdf: bool = True,
            **kwargs
    ):
        """

        Parameters
        ----------
        path : str
            If the data is alredy downloaded then provide the complete
            path to it. If None, then the data will be downloaded.
            The data is downloaded once and therefore susbsequent
            calls to this class will not download the data unless
            ``overwrite`` is set to True.
        overwrite : bool
            If the data is already down then you can set it to True,
            to make a fresh download.
        to_netcdf : bool
            whether to convert all the data into one netcdf file or not.
            This will fasten repeated calls to fetch etc. but will
            require netcdf5 package as well as xarry.
        """
        super().__init__(path=path, prefix=prefix, **kwargs)

        self._download(overwrite=overwrite)

        if to_netcdf:
            self._maybe_to_netcdf('camels_ch_dyn')

    @property
    def camels_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.path, 'camels_ch', 'camels_ch')

    @property
    def static_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.camels_path, 'static_attributes')

    @property
    def dynamic_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.camels_path, 'time_series', 'observation_based')

    @property
    def glacier_attr_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_glacier_attributes.csv')

    @property
    def clim_attr_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_climate_attributes_obs.csv')

    @property
    def geol_attr_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_geology_attributes.csv')

    @property
    def supp_geol_attr_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_geology_attributes_supplement.csv')

    @property
    def hum_inf_attr_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_humaninfluence_attributes.csv')

    @property
    def hydrogeol_attr_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_hydrogeology_attributes.csv')

    @property
    def hydrol_attr_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_hydrology_attributes_obs.csv')

    @property
    def lc_attr_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_landcover_attributes.csv')

    @property
    def soil_attr_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_soil_attributes.csv')

    @property
    def topo_attr_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_topographic_attributes.csv')

    @property
    def static_features(self):
        return self.fetch_static_features().columns.tolist()

    @property
    def dynamic_features(self) -> List[str]:
        return ['discharge_vol(m3/s)', 'discharge_spec(mm/d)', 'waterlevel(m)',
       'precipitation(mm/d)', 'temperature_min(°C)', 'temperature_mean(°C)',
       'temperature_max(°C)', 'rel_sun_dur(%)', 'swe(mm)']

    @property
    def start(self):  # start of data
        return pd.Timestamp('1981-01-01')

    @property
    def end(self):  # end of data
        return pd.Timestamp('2020-12-31')

    def stations(self)->List[str]:
        """Returns station ids for catchments"""
        stns =  pd.read_csv(
            self.glacier_attr_path,
            sep=';',
            skiprows=1
        )['gauge_id'].values.tolist()
        return [str(stn) for stn in stns]

    def glacier_attrs(self)->pd.DataFrame:
        """
        returns a dataframe with four columns
            - 'glac_area'
            - 'glac_vol'
            - 'glac_mass'
            - 'glac_area_neighbours'
        """
        df = pd.read_csv(
            self.glacier_attr_path,
            sep=';',
            skiprows=1,
            index_col='gauge_id',
            dtype=np.float32
        )
        df.index = df.index.astype(int).astype(str)
        return df

    def climate_attrs(self)->pd.DataFrame:
        """returns 14 climate attributes of catchments.
        """
        df = pd.read_csv(
            self.clim_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype={
                'gauge_id': str,
                'p_mean': float,
                'aridity': float,
                'pet_mean': float,
                'p_seasonality': float,
                'frac_snow': float,
                'high_prec_freq': float,
                'high_prec_dur': float,
                'high_prec_timing': str,
                'low_prec_timing': str
                         }
)
        return df

    def geol_attrs(self)->pd.DataFrame:
        """15 geological features"""
        df = pd.read_csv(
            self.geol_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype=np.float32
        )
        df.index = df.index.astype(int).astype(str)
        return df

    def supp_geol_attrs(self)->pd.DataFrame:
        """supplimentary geological features"""
        df = pd.read_csv(
            self.supp_geol_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype=np.float32
        )

        df.index = df.index.astype(int).astype(str)
        return df

    def human_inf_attrs(self)->pd.DataFrame:
        """
        14 athropogenic factors
        """
        df = pd.read_csv(
    self.hum_inf_attr_path,
    skiprows=1,
    sep=';',
    index_col='gauge_id',
    dtype={
        'gauge_id': str,
        'n_inhabitants': int,
        'dens_inhabitants': float,
        'hp_count': int,
        'hp_qturb': float,
        'hp_inst_turb': float,
        'hp_max_power': float,
        'num_reservoir': int,
        'reservoir_cap': float,
        'reservoir_he': float,
        'reservoir_fs': float,
        'reservoir_irr': float,
        'reservoir_nousedata': float,
        #'reservoir_year_first': int,
        #'reservoir_year_last': int
    }
)
        return df

    def hydrogeol_attrs(self)->pd.DataFrame:
        """10 hydrogeological factors"""
        df = pd.read_csv(
            self.hydrogeol_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype=float
        )
        df.index = df.index.astype(int).astype(str)
        return df

    def hydrol_attrs(self)->pd.DataFrame:
        """14 hydrological parameters + 2 useful infos"""
        df = pd.read_csv(
    self.hydrol_attr_path,
    skiprows=1,
    sep=';',
    index_col='gauge_id',
    dtype={
        'gauge_id': str,
        'sign_number_of_years': int,
        'q_mean': float,
        'runoff_ratio': float, 'stream_elas': float, 'slope_fdc': float,
        'baseflow_index_landson': float,
        'hfd_mean': float,
        'Q5': float, 'Q95': float, 'high_q_freq': float, 'high_q_dur': float,
        'low_q_freq': float
    }
)
        return df

    def landcolover_attrs(self)->pd.DataFrame:
        """13 landcover parameters"""
        return pd.read_csv(
            self.lc_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype={
                'gauge_id': str,
                'crop_perc': float,
                'grass_perc': float,
                'scrub_perc': float,
                'dwood_perc': float,
                'mixed_wood_perc': float,
                'ewood_perc': float,
                'wetlands_perc': float,
                'inwater_perc': float,
                'ice_perc': float,
                'loose_rock_perc': float,
                'rock_perc': float,
                'urban_perc': float,
            'dom_land_cover': str
            }
        )

    def soil_attrs(self)->pd.DataFrame:
        """80 soil parameters"""
        df = pd.read_csv(
            self.soil_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id'
        )
        df.index = df.index.astype(int).astype(str)
        return df

    def topo_attrs(self)->pd.DataFrame:
        """topographic parameters"""
        df = pd.read_csv(
            self.topo_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            encoding="unicode_escape"
        )

        df.index = df.index.astype(int).astype(str)
        return df

    def fetch_static_features(
            self,
            stn_id: Union[str, list] = None,
            features: Union[str, list] = None
    )->pd.DataFrame:
        """

        Returns static features of one or more stations.

        Parameters
        ----------
            stn_id : str
                name/id of station/stations of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe of shape (stations, features)

        Examples
        ---------
        >>> from ai4water.datasets import CAMELS_CH
        >>> dataset = CAMELS_CH()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            331
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (331, 209)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('2004')
        >>> static_data.shape
           (1, 209)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['gauge_lon', 'gauge_lat', 'area'])
        >>> static_data.shape
           (331, 3)
        >>> data = dataset.fetch_static_features('2004', features=['gauge_lon', 'gauge_lat', 'area'])
        >>> data.shape
           (1, 3)
        """
        stations = check_attributes(stn_id, self.stations())

        df = pd.concat(
    [
        self.climate_attrs(),
        self.geol_attrs(),
        self.supp_geol_attrs(),
        self.glacier_attrs(),
        self.human_inf_attrs(),
        self.hydrogeol_attrs(),
        self.hydrol_attrs(),
        self.landcolover_attrs(),
        self.soil_attrs(),
        self.topo_attrs(),
     ],
    axis=1)
        df.index = df.index.astype(str)

        features = check_attributes(features, df.columns.tolist(),
                                    "static features")
        return df.loc[stations, features]

    def _read_dynamic_from_csv(
            self,
            stations,
            dynamic_features,
            st=None,
            en=None
    ) ->dict:
        """
        reads dynamic data of one or more catchments
        """

        attributes = check_attributes(dynamic_features, self.dynamic_features)
        stations = check_attributes(stations, self.stations())

        dyn = {
            stn: self._read_dynamic_for_stn(stn).loc["19990101": "20201231", attributes] for stn in stations
        }

        return dyn

    def _read_dynamic_for_stn(self, stn_id)->pd.DataFrame:
        """
        Reads daily dynamic (meteorological + streamflow) data for one catchment
        and returns as DataFrame
        """

        return pd.read_csv(
            os.path.join(self.dynamic_path, f"CAMELS_CH_obs_based_{stn_id}.csv"),
            sep=';',
            index_col='date',
            parse_dates=True,
            dtype=np.float32
        )

    @property
    def _area_name(self) ->str:
        return 'area'

    @property
    def _coords_name(self)->List[str]:
        return ['gauge_lat', 'gauge_lon']

    @property
    def _mmd_feature_name(self)->str:
        return 'discharge_spec(mm/d)'
