
import json
import glob
from typing import Union, List

from ai4water.utils.utils import dateandtime_now
from ai4water.backend import os, random, np, pd, xr

from .._datasets import Datasets
from ..utils import check_attributes, download, sanity_check, _unzip

try:  # shapely may not be installed, as it may be difficult to isntall and is only needed for plotting data.
    from ai4water.preprocessing.spatial_utils import plot_shapefile
except ModuleNotFoundError:
    plot_shapefile = None

# directory separator
SEP = os.sep


def gb_message():
    link = "https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9"
    raise ValueError(f"Dwonlaoad the data from {link} and provide the directory "
                     f"path as dataset=Camels(data=data)")


class Camels(Datasets):
    """
    Get CAMELS dataset.
    This class first downloads the CAMELS dataset if it is not already downloaded.
    Then the selected attribute for a selected id are fetched and provided to the
    user using the method `fetch`.

    Attributes
    -----------
    - path str/path: diretory of the dataset
    - dynamic_features list: tells which dynamic attributes are available in
      this dataset
    - static_features list: a list of static attributes.
    - static_attribute_categories list: tells which kinds of static attributes
      are present in this category.

    Methods
    ---------
    - stations : returns name/id of stations for which the data (dynamic attributes)
        exists as list of strings.
    - fetch : fetches all attributes (both static and dynamic type) of all
            station/gauge_ids or a speficified station. It can also be used to
            fetch all attributes of a number of stations ids either by providing
            their guage_id or  by just saying that we need data of 20 stations
            which will then be chosen randomly.
    - fetch_dynamic_features :
            fetches speficied dynamic attributes of one specified station. If the
            dynamic attribute is not specified, all dynamic attributes will be
            fetched for the specified station. If station is not specified, the
            specified dynamic attributes will be fetched for all stations.
    - fetch_static_features :
            works same as `fetch_dynamic_features` but for `static` attributes.
            Here if the `category` is not specified then static attributes of
            the specified station for all categories are returned.
        stations : returns list of stations
    """

    DATASETS = {
        'CAMELS_BR': {'url': "https://zenodo.org/record/3964745#.YA6rUxZS-Uk",
                      },
        'CAMELS-GB': {'url': gb_message},
    }

    def __init__(self, path=None, **kwargs):
        super(Camels, self).__init__(path=path, **kwargs)
        self.path = path

    def stations(self):
        raise NotImplementedError

    def _read_dynamic_from_csv(self, stations, dynamic_features, st=None,
                               en=None)->dict:
        raise NotImplementedError

    def fetch_static_features(
            self,
            stn_id: Union[str, list],
            features: Union[str, list] = None
    ):
        """Fetches all or selected static attributes of one or more stations.

        Parameters
        ----------
            stn_id : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Examples
        --------
            >>> from ai4water.datasets import CAMELS_AUS
            >>> camels = CAMELS_AUS()
            >>> camels.fetch_static_features('224214A')
            >>> camels.static_features
            >>> camels.fetch_static_features('224214A',
            ... features=['elev_mean', 'relief', 'ksat', 'pop_mean'])
        """
        raise NotImplementedError

    @property
    def start(self):  # start of data
        raise NotImplementedError

    @property
    def end(self):  # end of data
        raise NotImplementedError

    @property
    def dynamic_features(self) -> list:
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
            static = pd.DataFrame(np.repeat(static.values, len(idx), axis=0), index=idx,
                                  columns=static.columns)
            return static
        else:
            return static

    @property
    def camels_dir(self):
        """Directory where all camels datasets will be saved. This will under
         datasets directory"""
        return os.path.join(self.base_ds_dir, "CAMELS")

    @property
    def path(self):
        """Directory where a particular dataset will be saved. """
        return self._path

    @path.setter
    def path(self, x):
        if x is None:
            x = os.path.join(self.camels_dir, self.__class__.__name__)

            if not os.path.exists(x):
                os.makedirs(x)
        else:
            assert os.path.exists(x), f"No data exist at {x}"
        # sanity_check(self.name, x)
        self._path = x

    def fetch(self,
              stations: Union[str, list, int, float, None] = None,
              dynamic_features: Union[list, str, None] = 'all',
              static_features: Union[str, list, None] = None,
              st: Union[None, str] = None,
              en: Union[None, str] = None,
              as_dataframe: bool = False,
              **kwargs
              ) -> Union[dict, pd.DataFrame]:
        """
        Fetches the attributes of one or more stations.

        Arguments:
            stations : if string, it is supposed to be a station name/gauge_id.
                If list, it will be a list of station/gauge_ids. If int, it will
                be supposed that the user want data for this number of
                stations/gauge_ids. If None (default), then attributes of all
                available stations. If float, it will be supposed that the user
                wants data of this fraction of stations.
            dynamic_features : If not None, then it is the attributes to be
                fetched. If None, then all available attributes are fetched
            static_features : list of static attributes to be fetches. None
                means no static attribute will be fetched.
            st : starting date of data to be returned. If None, the data will be
                returned from where it is available.
            en : end date of data to be returned. If None, then the data will be
                returned till the date data is available.
            as_dataframe : whether to return dynamic attributes as pandas
                dataframe or as xarray dataset.
            kwargs : keyword arguments to read the files

        returns:
            If both static  and dynamic features are obtained then it returns a
            dictionary whose keys are station/gauge_ids and values are the
            attributes and dataframes.
            Otherwise either dynamic or static features are returned.

        Examples
        --------
        >>> dataset = CAMELS_AUS()
        >>> # get data of 10% of stations
        >>> df = dataset.fetch(stations=0.1, as_dataframe=True)  # returns a multiindex dataframe
        ...  # fetch data of 5 (randomly selected) stations
        >>> df = dataset.fetch(stations=5, as_dataframe=True)
        ... # fetch data of 3 selected stations
        >>> df = dataset.fetch(stations=['912101A','912105A','915011A'], as_dataframe=True)
        ... # fetch data of a single stations
        >>> df = dataset.fetch(stations='318076', as_dataframe=True)
        ... # get both static and dynamic features as dictionary
        >>> data = dataset.fetch(1, static_features="all", as_dataframe=True)  # -> dict
        >>> data['dynamic']
        ... # get only selected dynamic features
        >>> df = dataset.fetch(stations='318076',
        ...     dynamic_features=['streamflow_MLd', 'solarrad_AWAP'], as_dataframe=True)
        ... # fetch data between selected periods
        >>> df = dataset.fetch(stations='318076', st="20010101", en="20101231", as_dataframe=True)

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

        if xr is None:
            raise ModuleNotFoundError("modeule xarray must be installed to use `datasets` module")

        return self.fetch_stations_attributes(
            stations,
            dynamic_features,
            static_features,
            st=st,
            en=en,
            as_dataframe=as_dataframe,
            **kwargs
        )

    def _maybe_to_netcdf(self, fname: str):
        self.dyn_fname = os.path.join(self.path, f'{fname}.nc')
        if not os.path.exists(self.dyn_fname):
            # saving all the data in netCDF file using xarray
            print(f'converting data to netcdf format for faster io operations')
            data = self.fetch(static_features=None)
            data_vars = {}
            coords = {}
            for k, v in data.items():
                data_vars[k] = (['time', 'dynamic_features'], v)
                index = v.index
                index.name = 'time'
                coords = {
                    'dynamic_features': list(v.columns),
                    'time': index
                }
            xds = xr.Dataset(
                data_vars=data_vars,
                coords=coords,
                attrs={'date': f"create on {dateandtime_now()}"}
            )
            xds.to_netcdf(self.dyn_fname)
        return

    def fetch_stations_attributes(
            self,
            stations: list,
            dynamic_features='all',
            static_features=None,
            st=None,
            en=None,
            as_dataframe: bool = False,
            **kwargs
    ):
        """Reads attributes of more than one stations.

        Arguments:
            stations : list of stations for which data is to be fetched.
            dynamic_features : list of dynamic attributes to be fetched.
                if 'all', then all dynamic attributes will be fetched.
            static_features : list of static attributes to be fetched.
                If `all`, then all static attributes will be fetched. If None,
                then no static attribute will be fetched.
            st : start of data to be fetched.
            en : end of data to be fetched.
            as_dataframe : whether to return the data as pandas dataframe. default
                is xr.dataset object
            kwargs dict: additional keyword arguments

        Returns:
            Dynamic and static features of multiple stations. Dynamic features
            are by default returned as xr.Dataset unless `as_dataframe` is True, in
            such a case, it is a pandas dataframe with multiindex. If xr.Dataset,
            it consists of `data_vars` equal to number of stations and for each
            station, the `DataArray` is of dimensions (time, dynamic_features).
            where `time` is defined by `st` and `en` i.e length of `DataArray`.
            In case, when the returned object is pandas DataFrame, the first index
            is `time` and second index is `dyanamic_features`. Static attributes
            are always returned as pandas DataFrame and have following shape
            `(stations, static_features). If `dynamic_features` is None,
            then they are not returned and the returned value only consists of
            static features. Same holds true for `static_features`.
            If both are not None, then the returned type is a dictionary with
            `static` and `dynamic` keys.

        Raises:
            ValueError, if both dynamic_features and static_features are None

        Examples
        --------
            >>> from ai4water.datasets import CAMELS_AUS
            >>> dataset = CAMELS_AUS()
            ... # find out station ids
            >>> dataset.stations()
            ... # get data of selected stations
            >>> dataset.fetch_stations_attributes(['912101A', '912105A', '915011A'],
            ...  as_dataframe=True)
        """
        st, en = self._check_length(st, en)

        if dynamic_features is not None:

            dynamic_features = check_attributes(dynamic_features, self.dynamic_features)

            if not os.path.exists(self.dyn_fname):
                # read from csv files
                # following code will run only once when fetch is called inside init method
                dyn = self._read_dynamic_from_csv(stations, dynamic_features, st=st, en=en)

            else:
                dyn = xr.load_dataset(self.dyn_fname)  # daataset
                dyn = dyn[stations].sel(dynamic_features=dynamic_features, time=slice(st, en))
                if as_dataframe:
                    dyn = dyn.to_dataframe(['time', 'dynamic_features'])

            if static_features is not None:
                static = self.fetch_static_features(stations, static_features)
                stns = {'dynamic': dyn, 'static': static}
            else:
                # if the dyn is a dictionary of key, DataFames, we will return a MultiIndex
                # dataframe instead of a dictionary
                if as_dataframe and isinstance(dyn, dict) and isinstance(list(dyn.values())[0], pd.DataFrame):
                    dyn = pd.concat(dyn, axis=0, keys=dyn.keys())
                stns = dyn

        elif static_features is not None:

            return self.fetch_static_features(stations, static_features)

        else:
            raise ValueError

        return stns

    def fetch_dynamic_features(
            self,
            stn_id: str,
            features='all',
            st=None,
            en=None,
            as_dataframe=False
    ):
        """Fetches all or selected dynamic attributes of one station.

        Parameters
        ----------
            stn_id : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                dynamic features are returned.
            st : Optional (default=None)
                start time from where to fetch the data.
            en : Optional (default=None)
                end time untill where to fetch the data
            as_dataframe : bool, optional (default=False)
                if true, the returned data is pandas DataFrame otherwise it
                is xarray dataset

        Examples
        --------
            >>> from ai4water.datasets import CAMELS_AUS
            >>> camels = CAMELS_AUS()
            >>> camels.fetch_dynamic_features('224214A', as_dataframe=True).unstack()
            >>> camels.dynamic_features
            >>> camels.fetch_dynamic_features('224214A',
            ... attributes=['tmax_AWAP', 'vprp_AWAP', 'streamflow_mmd'],
            ... as_dataframe=True).unstack()
        """

        assert isinstance(stn_id, str), f"station id must be string is is of type {type(stn_id)}"
        station = [stn_id]
        return self.fetch_stations_attributes(
            station,
            features,
            None,
            st=st,
            en=en,
            as_dataframe=as_dataframe
        )

    def fetch_station_attributes(
            self,
            station: str,
            dynamic_features: Union[str, list, None] = 'all',
            static_features: Union[str, list, None] = None,
            as_ts: bool = False,
            st: Union[str, None] = None,
            en: Union[str, None] = None,
            **kwargs
    ) -> pd.DataFrame:
        """
        Fetches attributes for one station.

        Parameters
        -----------
            station :
                station id/gauge id for which the data is to be fetched.
            dynamic_features : str/list, optional
                names of dynamic features/attributes to fetch
            static_features :
                names of static features/attributes to be fetches
            as_ts : bool
                whether static attributes are to be converted into a time
                series or not. If yes then the returned time series will be of
                same length as that of dynamic attribtues.
            st : str,optional
                starting point from which the data to be fetched. By default
                the data will be fetched from where it is available.
            en : str, optional
                end point of data to be fetched. By default the dat will be fetched

        Returns
        -------
        pd.DataFrame
            dataframe if as_ts is True else it returns a dictionary of static and
            dynamic attributes for a station/gauge_id

        Examples
        --------
            >>> from ai4water.datasets import CAMELS_AUS
            >>> dataset = CAMELS_AUS()
            >>> dataset.fetch_station_attributes('912101A')

        """
        st, en = self._check_length(st, en)

        station_df = pd.DataFrame()
        if dynamic_features:
            dynamic = self.fetch_dynamic_features(station, dynamic_features, st=st,
                                                  en=en, **kwargs)
            station_df = pd.concat([station_df, dynamic])

            if static_features is not None:
                static = self.fetch_static_features(station, static_features)

                if as_ts:
                    station_df = pd.concat([station_df, static], axis=1)
                else:
                    station_df = {'dynamic': station_df, 'static': static}

        elif static_features is not None:
            station_df = self.fetch_static_features(station, static_features)

        return station_df


class LamaH(Camels):
    """
    Large-Sample Data for Hydrology and Environmental Sciences for Central Europe
    (mainly Austria). The dataset is downloaded from
    from  `zenodo <https://zenodo.org/record/4609826#.YFNp59zt02w>`_
    following the work of
    `Klingler et al., 2021 <https://essd.copernicus.org/preprints/essd-2021-72/>`_ .
    For ``total_upstrm`` data, there are 859 stations with 61 static features
    and 17 dynamic features. The temporal extent of data is from 1981-01-01
    to 2019-12-31.
    """
    url = "https://zenodo.org/record/4609826#.YFNp59zt02w"
    _data_types = ['total_upstrm', 'diff_upstrm_all', 'diff_upstrm_lowimp']
    time_steps = ['daily', 'hourly']

    static_attribute_categories = ['']

    def __init__(self, *,
                 time_step: str,
                 data_type: str,
                 path=None,
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
            time_step :
                possible values are ``daily`` or ``hourly``
            data_type :
                possible values are ``total_upstrm``, ``diff_upstrm_all``
                or ``diff_upstrm_lowimp``

        Examples
        --------
        >>> from ai4water.datasets import LamaH
        >>> dataset = LamaH(time_step='daily', data_type='total_upstrm')
        # The daily dataset is from 859 with 61 static and 22 dynamic features
        >>> len(dataset.stations()), len(dataset.static_features), len(dataset.dynamic_features)
        (859, 61, 22)
        >>> df = dataset.fetch(3, as_dataframe=True)
        >>> df.shape
        (313368, 3)
        >>> dataset = LamaH(time_step='hourly', data_type='total_upstrm')
        >>> len(dataset.stations()), len(dataset.static_features), len(dataset.dynamic_features)
        (859, 61, 17)
        """

        assert time_step in self.time_steps, f"invalid time_step {time_step} given"
        assert data_type in self._data_types, f"invalid data_type {data_type} given."

        self.time_step = time_step
        self.data_type = data_type

        super().__init__(path=path, **kwargs)

        fpath = os.path.join(self.path, 'lamah_diff_upstrm_lowimp_hourly_dyn.nc')

        _data_types = self._data_types if self.time_step == 'daily' else ['total_upstrm']

        if not os.path.exists(fpath):
            for dt in _data_types:
                for ts in self.time_steps:
                    self.time_step = ts
                    self.data_type = dt
                    fname = f"lamah_{dt}_{ts}_dyn"
                    self._maybe_to_netcdf(fname)

            self.time_step = time_step
            self.data_type = data_type

        self.dyn_fname = os.path.join(self.path,
                                      f'lamah_{data_type}_{time_step}_dyn.nc')

    @property
    def dynamic_features(self):
        station = self.stations()[0]
        df = self.read_ts_of_station(station)
        return df.columns.to_list()

    @property
    def static_features(self) -> list:
        fname = os.path.join(self.data_type_dir,
                             f'1_attributes{SEP}Catchment_attributes.csv')
        df = pd.read_csv(fname, sep=';', index_col='ID')
        return df.columns.to_list()

    @property
    def data_type_dir(self):
        directory = 'CAMELS_AT'
        if self.time_step == 'hourly':
            directory = 'CAMELS_AT1'  # todo, use it only for hourly, daily is causing errors
        # self.path/CAMELS_AT/data_type_dir
        f = [f for f in os.listdir(os.path.join(self.path, directory)) if self.data_type in f][0]
        return os.path.join(self.path, f'{directory}{SEP}{f}')

    def stations(self) -> list:
        # assuming file_names of the format ID_{stn_id}.csv
        _dirs = os.listdir(os.path.join(self.data_type_dir,
                                        f'2_timeseries{SEP}{self.time_step}'))
        s = [f.split('_')[1].split('.csv')[0] for f in _dirs]
        return s

    def _read_dynamic_from_csv(self,
                               stations,
                               dynamic_features: Union[str, list] = 'all',
                               st=None,
                               en=None,
                               ):
        """Reads attributes of one or more station"""

        stations_attributes = {}

        for station in stations:

            station_df = pd.DataFrame()

            if dynamic_features is not None:
                dynamic_df = self.read_ts_of_station(station)

                station_df = pd.concat([station_df, dynamic_df])

            stations_attributes[station] = station_df

        return stations_attributes

    def fetch_static_features(
            self,
            stn_id: Union[str, List[str]],
            features:Union[str, List[str]]=None
    ) -> pd.DataFrame:
        """
        static features of LamaH

        Parameters
        ----------
            stn_id : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Examples
        --------
            >>> from ai4water.datasets import LamaH
            >>> dataset = LamaH(time_step='daily', data_type='total_upstrm')
            >>> df = dataset.fetch_static_features('99')  # (1, 61)
            ...  # get list of all static features
            >>> dataset.static_features
            >>> dataset.fetch_static_features('99',
            >>> features=['area_calc', 'elev_mean', 'agr_fra', 'sand_fra'])  # (1, 4)
        """
        fname = os.path.join(self.data_type_dir,
                             f'1_attributes{SEP}Catchment_attributes.csv')

        df = pd.read_csv(fname, sep=';', index_col='ID')

        # if features is not None:
        static_features = check_attributes(features, self.static_features)

        df = df[static_features]

        if isinstance(stn_id, list):
            stations = [str(i) for i in stn_id]
        elif isinstance(stn_id, int):
            stations = str(stn_id)
        else:
            stations = stn_id

        df.index = df.index.astype(str)
        df = df.loc[stations]
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df).transpose()

        return df

    def read_ts_of_station(self, station) -> pd.DataFrame:
        # read a file containing timeseries data for one station
        fname = os.path.join(self.data_type_dir,
                             f'2_timeseries{SEP}{self.time_step}{SEP}ID_{station}.csv')

        df = pd.read_csv(fname, sep=';')

        if self.time_step == 'daily':
            periods = pd.PeriodIndex(year=df["YYYY"], month=df["MM"], day=df["DD"],
                                     freq="D")
            df.index = periods.to_timestamp()
        else:
            periods = pd.PeriodIndex(year=df["YYYY"],
                                     month=df["MM"], day=df["DD"], hour=df["hh"],
                                     minute=df["mm"], freq="H")
            df.index = periods.to_timestamp()

        # remove the cols specifying index
        [df.pop(item) for item in ['YYYY', 'MM', 'DD', 'hh', 'mm'] if item in df]
        return df

    @property
    def start(self):
        return "19810101"

    @property
    def end(self):
        return "20191231"


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
            stn_id: Union[str, List[str]],
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
        attributes = check_attributes(features, self.static_features)

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
        df = static_df.loc[stn_id][attributes]
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
        super().__init__(name="CAMELS-GB", path=path)

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
        attributes = []
        path = os.path.join(self.path, 'data')
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)) and f.endswith('csv'):
                attributes.append(f.split('_')[2])

        return attributes

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

    def _read_dynamic_from_csv(self,
                               stations,
                               attributes: Union[str, list] = 'all',
                               st=None,
                               en=None,
                               ):
        """Fetches dynamic attribute/attributes of one or more station."""
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
            stn_id: Union[str, List[str]],
            features:Union[str, List[str]]="all"
    ) -> pd.DataFrame:
        """
        Fetches static attributes of one or more stations for one or
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

        attributes = check_attributes(features, self.static_features)
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

        if isinstance(stn_id, str):
            station = [stn_id]
        elif isinstance(stn_id, int):
            station = [str(stn_id)]
        elif isinstance(stn_id, list):
            station = [str(stn) for stn in stn_id]
        else:
            raise ValueError

        static_df.index = static_df.index.astype(str)

        return static_df.loc[station][attributes]


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
                data will downloaded.
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
        attributes = []
        path = os.path.join(self.path, f'04_attributes{SEP}04_attributes')
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)) and f.endswith('csv'):
                f = str(f.split('.csv')[0])
                attributes.append(''.join(f.split('_')[2:]))
        return attributes

    @property
    def static_features(self) -> list:
        static_fpath = os.path.join(self.path, 'CAMELS_AUS_Attributes-Indices_MasterTable.csv')

        df = pd.read_csv(static_fpath, index_col='station_id', nrows=1)
        cols = list(df.columns)

        return cols

    @property
    def dynamic_features(self) -> list:
        return list(self.folders.keys())

    def _read_static(self, stations, attributes,
                     st=None, en=None):

        attributes = check_attributes(attributes, self.static_features)
        static_fname = 'CAMELS_AUS_Attributes-Indices_MasterTable.csv'
        static_fpath = os.path.join(self.path, static_fname)
        static_df = pd.read_csv(static_fpath, index_col='station_id')

        static_df.index = static_df.index.astype(str)
        df = static_df.loc[stations][attributes]
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
        """Fetches static attribuets of one or more stations as dataframe.

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
    Alvarez-Garreton_ et al., 2018 .

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

    .. _Alvarez-Garreton: https://doi.org/10.5194/hess-22-5817-2018
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
    def static_features(self) -> list:
        path = os.path.join(self.path, f"1_CAMELScl_attributes{SEP}1_CAMELScl_attributes.txt")
        df = pd.read_csv(path, sep='\t', index_col='gauge_id')
        return df.index.to_list()

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

        # reading all dynnamic attributes
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

    def _read_static(self, stations: list, attributes: list) -> pd.DataFrame:
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

            stns_df.append(df.transpose()[attributes])

        stns_df = pd.concat(stns_df)
        return stns_df

    def fetch_static_features(
            self,
            stn_id: Union[str, List[str]],
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
        attributes = check_attributes(features, self.static_features)

        if isinstance(stn_id, str):
            stn_id = [stn_id]

        return self._read_static(stn_id, attributes)
