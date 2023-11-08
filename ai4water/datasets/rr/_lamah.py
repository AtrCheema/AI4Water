
import gc
import os
from datetime import datetime
from typing import Union, List
import concurrent.futures as cf
from multiprocessing import cpu_count


from ai4water.utils.utils import dateandtime_now
from ai4water.backend import pd, xr, np
from ..utils import check_attributes

from .camels import Camels


SEP = os.sep

class LamaH(Camels):
    """
    Large-Sample Data for Hydrology and Environmental Sciences for Central Europe
    (mainly Austria). The dataset is downloaded from
    `zenodo <https://zenodo.org/record/4609826#.YFNp59zt02w>`_
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

        _data_types = self._data_types if self.time_step == 'daily' else ['total_upstrm']


        if not self.all_ncs_exist:
            self._maybe_to_netcdf(fdir = f"{data_type}_{time_step}")

        self.dyn_fname = os.path.join(self.path,
                                      f'lamah_{data_type}_{time_step}_dyn.nc')
        return

    def _maybe_to_netcdf(self, fdir: str):
        # since data is very large, saving all the data in one file
        # consumes a lot of memory, which is impractical for most of the personal
        # computers! Therefore, saving each feature separately

        fdir = os.path.join(self.path, fdir)
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        if not self.all_ncs_exist:
            print(f'converting data to netcdf format for faster io operations')

            for feature in self.dynamic_features:

                # we must specify class level dyn_fname feature
                dyn_fname = os.path.join(fdir, f"{feature}.nc")

                if not os.path.exists(dyn_fname):
                    print(f'Saving {feature} to disk')
                    data = self.fetch(static_features=None, dynamic_features=feature)

                    data_vars = {}
                    coords = {}
                    for k, v in data.items():
                        data_vars[k] = (['time', 'dynamic_features'], v)
                        index = v.index
                        index.name = 'time'
                        coords = {
                            'dynamic_features': [feature],
                            'time': index
                        }
                    xds = xr.Dataset(
                        data_vars=data_vars,
                        coords=coords,
                        attrs={'date': f"create on {dateandtime_now()}"}
                    )

                    xds.to_netcdf(self.dyn_fname)

                    gc.collect()
        return

    @property
    def dynamic_fnames(self):
        return [f"{feature}.nc" for feature in self.dynamic_features]

    @property
    def all_ncs_exist(self):
        fdir = os.path.join(self.path, f"{self.data_type}_{self.time_step}")
        return all(os.path.exists(os.path.join(fdir, fname_)) for fname_ in self.dynamic_fnames)

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

    @property
    def q_dir(self):
        directory = 'CAMELS_AT'
        if self.time_step == 'hourly':
            directory = 'CAMELS_AT1'
        # self.path/CAMELS_AT/data_type_dir
        return os.path.join(self.path, f'{directory}', 'D_gauges', '2_timeseries')

    def stations(self) -> list:
        # assuming file_names of the format ID_{stn_id}.csv
        _dirs = os.listdir(os.path.join(self.data_type_dir,
                                        f'2_timeseries{SEP}{self.time_step}'))
        s = [f.split('_')[1].split('.csv')[0] for f in _dirs]
        return s

    def fetch_stations_features(
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

        This function checks of .nc files exist, then they are not prepared
        and saved otherwise first nc files are prepared and then the data is
        read again from nc files. Upon subsequent calls, the nc files are used
        for reading the data.

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
            are by default returned as xr.Dataset unless ``as_dataframe`` is True, in
            such a case, it is a pandas dataframe with multiindex. If xr.Dataset,
            it consists of ``data_vars`` equal to number of stations and for each
            station, the ``DataArray`` is of dimensions (time, dynamic_features).
            where `time` is defined by ``st`` and ``en`` i.e length of ``DataArray``.
            In case, when the returned object is pandas DataFrame, the first index
            is `time` and second index is `dyanamic_features`. Static attributes
            are always returned as pandas DataFrame and have the shape:
            ``(stations, static_features)``. If ``dynamic_features`` is None,
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
            >>> dataset.fetch_stations_features(['912101A', '912105A', '915011A'],
            ...  as_dataframe=True)
        """
        st, en = self._check_length(st, en)

        if dynamic_features is not None:

            dynamic_features = check_attributes(dynamic_features, self.dynamic_features)

            if not self.all_ncs_exist:
                # read from csv files
                # following code will run only once when fetch is called inside init method
                dyn = self._read_dynamic_from_csv(stations, dynamic_features, st=st, en=en)
            else:
                dyn = self._make_ds_from_ncs(dynamic_features, stations, st, en)

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
                                           dynamic_features='q_cms',
                                           as_dataframe=True)
        q.index = q.index.get_level_values(0)
        area_m2 = self.area(stations) * 1e6  # area in m2
        q = (q / area_m2) * 86400  # cms to m/day
        return q * 1e3  # to mm/day

    def area(
            self,
            stations: Union[str, List[str]] = None
    ) ->pd.Series:
        """
        Returns area_gov (Km2) of all catchments as pandas series
        ``area_gov`` is Catchment area obtained from the administration.

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
        >>> from ai4water.datasets import LamaH
        >>> dataset = LamaH(time_step="daily")
        >>> dataset.area()  # returns area of all stations
        >>> dataset.area('1')  # returns area of station whose id is 912101A
        >>> dataset.area(['1', '2'])  # returns area of two stations
        """
        stations = check_attributes(stations, self.stations())

        fname = os.path.join(self.path,
                             'CAMELS_AT1',
                             'D_gauges',
                             '1_attributes',
                             'Gauge_attributes.csv')
        df = pd.read_csv(fname, sep=';', index_col='ID')

        df.index = df.index.astype(str)

        s = df.loc[stations, 'area_gov']
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
         >>> dataset = LamaH(time_step="daily")
         >>> dataset.stn_coords() # returns coordinates of all stations
         >>> dataset.stn_coords('1')  # returns coordinates of station whose id is 912101A
         >>> dataset.stn_coords(['1', '2'])  # returns coordinates of two stations

         """
        fname = os.path.join(self.path,
                             'CAMELS_AT1',
                             'D_gauges',
                             '1_attributes', 'Gauge_attributes.csv')
        df = pd.read_csv(fname, sep=';', index_col='ID')

        df.index = df.index.astype(str)
        df = df[['lon', 'lat']]
        df.columns = ['long', 'lat']
        stations = check_attributes(stations, self.stations())

        return df.loc[stations, :]

    def _read_dynamic_from_csv1(
            self,
            stations,
            dynamic_features: Union[str, list] = 'all',
            st=None,
            en=None,
    ):
        """Reads features of one or more station"""

        stations_features = {}

        for station in stations:

            if dynamic_features is not None:
                station_df = self.read_ts_of_station(station, dynamic_features)
            else:
                station_df = pd.DataFrame()
            print(station_df.index[0], station_df.index[-1])
            stations_features[station] = station_df[dynamic_features]

        return stations_features

    def _read_dynamic_from_csv(
            self,
            stations,
            dynamic_features: Union[str, list] = 'all',
            st=None,
            en=None,
    ):
        """Reads features of one or more station"""

        cpus = self.processes or cpu_count()

        dynamic_features = [dynamic_features for _ in range(len(stations))]

        with  cf.ProcessPoolExecutor(max_workers=cpus) as executor:
            results = executor.map(
                self.read_ts_of_station,
                stations,
                dynamic_features
            )

        results = {stn:data[dynamic_features[0]] for stn, data in zip(stations, results)}
        return results

    def _make_ds_from_ncs(self, dynamic_features, stations, st, en):
        """makes xarray Dataset by reading multiple .nc files"""

        dyns = []
        for f in dynamic_features:
            dyn_fpath = os.path.join(self.path, f"{self.data_type}_{self.time_step}", f'{f}.nc')
            dyn = xr.load_dataset(dyn_fpath)  # daataset
            dyns.append(dyn[stations].sel(time=slice(st, en)))

        return xr.concat(dyns, dim='dynamic_features')  # dataset

    def fetch_static_features(
            self,
            stn_id: Union[str, List[str]] = "all",
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

        if stn_id == "all":
            stn_id = self.stations()

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

    def read_ts_of_station(
            self,
            station,
            features=None
    ) -> pd.DataFrame:
        # read a file containing timeseries data for one station

        q_df = pd.DataFrame()
        if features is None:
            q_df = self._read_q_for_station(station)
        elif features in ["q_cms", 'checked']:
            return self._read_q_for_station(station)
        if isinstance(features, list):
            if len(features)==1 and features[0] in ['q_cms', 'checked']:
                return self._read_q_for_station(station)
            elif 'q_cms' in features or 'checked' in features:
                q_df = self._read_q_for_station(station)


        met_df = self._read_met_for_station(station, features)

        if features:

            return pd.concat([met_df, q_df], axis=1).loc[:, features]

        return pd.concat([met_df, q_df], axis=1)

    def _read_met_for_station(self, station, features):
        if isinstance(features, list):
            features = features.copy()
            [features.remove(itm)for itm in ['q_cms', 'checked'] if itm in features]

        met_fname = os.path.join(
            self.data_type_dir,
            f'2_timeseries{SEP}{self.time_step}{SEP}ID_{station}.csv')

        usecols = None
        met_dtype = {
            'YYYY': np.int32,
            'MM': np.int32,
            'DD': np.int32,
            'DOY': np.int32,
            '2m_temp_max': np.float32,
            '2m_temp_mean': np.float32,
            '2m_temp_min': np.float32,
            '2m_dp_temp_max': np.float32,
            '2m_dp_temp_mean': np.float32,
            '2m_dp_temp_min': np.float32,
            '10m_wind_u': np.float32,
            '10m_wind_v': np.float32,
            'fcst_alb': np.float32,
            'lai_high_veg': np.float32,
            'lai_low_veg': np.float32,
            'swe': np.float32,
            'surf_net_solar_rad_max': np.float32,
            'surf_net_solar_rad_mean': np.float32,
            'surf_net_therm_rad_max': np.float32,
            'surf_net_therm_rad_mean': np.float32,
            'surf_press': np.float32,
            'total_et': np.float32,
            'prec': np.float32,
            'volsw_123': np.float32,
            'volsw_4': np.float32
        }

        if self.time_step == 'daily':
            if features:
                if not isinstance(features, list):
                    features = [features]

                usecols = ['YYYY', 'MM', 'DD'] + features

            met_df = pd.read_csv(met_fname, sep=';', dtype=met_dtype,
                                 usecols=usecols)

            periods = pd.PeriodIndex(year=met_df["YYYY"],
                                     month=met_df["MM"], day=met_df["DD"],
                                     freq="D")
            met_df.index = periods.to_timestamp()

        else:
            if features:
                if not isinstance(features, list):
                    features = [features]

                usecols = ['YYYY', 'MM', 'DD', 'hh', 'mm'] + features

            met_dtype.update({
                'hh': np.int32,
                'mm': np.int32,
                'HOD': np.int32,
                '2m_temp': np.float32,
                '2m_dp_temp': np.float32,
                'surf_net_solar_rad': np.float32,
                'surf_net_therm_rad': np.float32
            })

            met_df = pd.read_csv(met_fname, sep=';', dtype=met_dtype, usecols=usecols)

            periods = pd.PeriodIndex(year=met_df["YYYY"],
                                     month=met_df["MM"], day=met_df["DD"], hour=met_df["hh"],
                                     minute=met_df["mm"], freq="H")
            met_df.index = periods.to_timestamp()

        # remove the cols specifying index
        [met_df.pop(item) for item in ['YYYY', 'MM', 'DD', 'hh', 'mm'] if item in met_df]
        return met_df

    def _read_q_for_station(self, station):

        q_fname = os.path.join(self.q_dir,
                             f'{self.time_step}{SEP}ID_{station}.csv')

        q_dtype = {
            'YYYY': np.int32,
            'MM': np.int32,
            'DD': np.int32,
            'qobs': np.float32,
            'checked': np.bool_
        }

        if self.time_step == 'daily':
            q_df = pd.read_csv(q_fname, sep=';', dtype=q_dtype)
            periods = pd.PeriodIndex(year=q_df["YYYY"],
                                     month=q_df["MM"], day=q_df["DD"],
                                     freq="D")
            q_df.index = periods.to_timestamp()
            index = pd.date_range("1981-01-01", "2017-12-31", freq="D")
            q_df = q_df.reindex(index=index)
        else:
            q_dtype.update({
                'hh': np.int32,
                'mm': np.int32
            })

            q_df = pd.read_csv(q_fname, sep=';', dtype=q_dtype)

            periods = pd.PeriodIndex(year=q_df["YYYY"],
                                     month=q_df["MM"], day=q_df["DD"], hour=q_df["hh"],
                                     minute=q_df["mm"], freq="H")
            q_df.index = periods.to_timestamp()
            index = pd.date_range("1981-01-01", "2017-12-31", freq="H")
            q_df = q_df.reindex(index=index)

        [q_df.pop(item) for item in ['YYYY', 'MM', 'DD', 'hh', 'mm'] if item in q_df]
        q_df.rename(columns={'qobs': 'q_cms'}, inplace=True)

        return q_df

    @property
    def start(self):
        return "19810101"

    @property
    def end(self):  # todo, is it untill 2017 or 2019?
        return "20191231"


class LamaHIce(Camels):
    """
    Daily and hourly hydro-meteorological time series data of 107 river basins
    of Iceland. The total period of dataset is from 1950 to 2021. The average
    length of daily data is 33 years while for that of hourly it is 11 years.
    The dataset is available on hydroshare at
    https://www.hydroshare.org/resource/86117a5f36cc4b7c90a5d54e18161c91/ .
    The paper : https://doi.org/10.5194/essd-2023-349

    """

    url = {
'Caravan_extension_lamahice.zip':
'https://www.hydroshare.org/resource/86117a5f36cc4b7c90a5d54e18161c91/data/contents/Caravan_extension_lamahice.zip',
'lamah_ice.zip':
'https://www.hydroshare.org/resource/86117a5f36cc4b7c90a5d54e18161c91/data/contents/lamah_ice.zip',
'lamah_ice_hourly.zip':
'https://www.hydroshare.org/resource/86117a5f36cc4b7c90a5d54e18161c91/data/contents/lamah_ice_hourly.zip'
    }
    _data_types = ['total_upstrm', 'diff_upstrm_all', 'diff_upstrm_lowimp']
    time_steps = ['daily', 'hourly']
    DTYPES = {
        'total_upstrm': 'A_basins_total_upstrm',
        'diff_upstrm_all': 'B_basins_intermediate_all',
        'diff_upstrm_lowimp': 'C_basins_intermediate_lowimp'
    }
    def __init__(
            self,
            path=None,
            overwrite=False,
            *,
            time_step:str = "daily",
            data_type:str = "total_upstrm",
            **kwargs):
        super().__init__(path=path, **kwargs)
        self.path = path

        self._download(overwrite=overwrite)

        assert time_step in self.time_steps, f"invalid time_step {time_step} given"
        assert data_type in self._data_types, f"invalid data_type {data_type} given."

        self.time_step = time_step
        self.data_type = data_type

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
    """
    @property
    def start(self):
        return "19510101"

    @property
    def end(self):  # todo, is it untill 2017 or 2019?
        return "20211231"

    @property
    def gauges_path(self):
        """returns the path where gauge data files are located"""
        if self.time_step == "hourly":
            return os.path.join(self.path, "lamah_ice_hourly", "lamah_ice_hourly", "D_gauges")
        return os.path.join(self.path, "lamah_ice", "lamah_ice", "D_gauges")

    @property
    def q_path(self):
        """path where all q files are located"""
        if self.time_step == "hourly":
            return os.path.join(self.gauges_path, "2_timeseries", "hourly")
        return os.path.join(self.gauges_path, "2_timeseries", "daily")

    def stations(self)->List[str]:
        """
        returns names of stations as a list
        """
        return [fname.split('.')[0].split('_')[1] for fname in os.listdir(self.q_path)]

    def gauge_attributes(self)->pd.DataFrame:
        """
        returns gauge attributes from following two files

            - Gauge_attributes.csv
            - hydro_indices_filtered_obs.csv

        Returns
        -------
        pd.DataFrame
            a dataframe of shape (111, 29)
        """
        g_attr_fpath = os.path.join(self.gauges_path, "1_attributes", "Gauge_attributes.csv")

        df_gattr = pd.read_csv(g_attr_fpath)
        df_gattr.index = df_gattr.pop('id').astype(str)

        hydro_idx_fpath = os.path.join(self.gauges_path, "1_attributes", "hydro_indices_filtered_obs.csv")

        df_hidx = pd.read_csv(hydro_idx_fpath)
        df_hidx.index = df_hidx.pop('id').astype(str)

        df = pd.concat([df_gattr, df_hidx], axis=1)

        return df

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
         >>> dataset = LamaHIce(time_step="daily")
         >>> dataset.stn_coords() # returns coordinates of all stations
         >>> dataset.stn_coords('1')  # returns coordinates of station whose id is 912101A
         >>> dataset.stn_coords(['1', '2'])  # returns coordinates of two stations

         """
        g_attr_fpath = os.path.join(self.gauges_path, "1_attributes", "Gauge_attributes.csv")

        df = pd.read_csv(g_attr_fpath)

        df.index = df.pop('id').astype(str)
        df = df[['lon', 'lat']]
        df.columns = ['long', 'lat']
        stations = check_attributes(stations, self.stations())

        return df.loc[stations, :]

    def area(
            self,
            stations: Union[str, List[str]] = None
    ) ->pd.Series:
        """
        Returns area_gov (Km2) of all catchments as pandas series
        ``area_gov`` is Catchment area obtained from the administration.

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
        >>> from ai4water.datasets import LamaHIce
        >>> dataset = LamaHIce(time_step="daily")
        >>> dataset.area()  # returns area of all stations
        >>> dataset.area('1')  # returns area of station whose id is 912101A
        >>> dataset.area(['1', '2'])  # returns area of two stations
        """
        stations = check_attributes(stations, self.stations())

        df = self.catchment_attributes()
        return df.loc[stations, 'area_calc']

    def _catch_attr_path(self)->str:
        p = "lamah_ice"
        if self.time_step == "hourly":
            p = "lamah_ice_hourly"

        path = os.path.join(self.path, p,
                             p,
                             self.DTYPES[self.data_type],
                             "1_attributes")
        return path

    def _clim_ts_path(self)->str:
        p0 = "lamah_ice"
        p1 = "2_timeseries"
        p2 = "daily"

        if self.time_step == "hourly":
            p0 = "lamah_ice_hourly"
            p1 = "2_timeseries"
            p2 = "hourly"

        path = os.path.join(self.path, p0, p0,
                             self.DTYPES[self.data_type],
                            p1, p2, "meteorological_data")
        return path

    def catchment_attributes(self)->pd.DataFrame:
        """returns catchment attributes as DataFrame with 90 columns
        """

        fpath = os.path.join(self._catch_attr_path(), "Catchment_attributes.csv")

        df = pd.read_csv(fpath)
        df.index = df.pop('id').astype(str)
        return df

    def wat_bal_attrs(self)->pd.DataFrame:
        """water balance attributes"""
        fpath = os.path.join(self._catch_attr_path(),
                             "Water_balance_using_all_streamflow_data.csv")

        df = pd.read_csv(fpath)
        df.index = df.pop('Unnamed: 0').astype(str)
        df.columns = [col + "_all" for col in df.columns]
        return df

    def wat_bal_filt_attrs(self)->pd.DataFrame:
        """water balance attributes from filtered q"""
        fpath = os.path.join(self._catch_attr_path(),
                             "Water_balance_using_filtered_streamflow_data.csv")

        df = pd.read_csv(fpath)
        df.index = df.pop('Unnamed: 0').astype(str)
        df.columns = [col + "_filtered" for col in df.columns]
        return df

    def basin_attributes(self)->pd.DataFrame:
        """returns basin attributes which are catchment attributes, water
        balance all attributes and water balance filtered attributes

        Returns
        -------
        pd.DataFrame
            a dataframe of shape (111, 104) where 104 are the static
            catchment/basin attributes
        """
        cat = self.catchment_attributes()
        wat_bal_all = self.wat_bal_attrs()
        wat_bal_filt = self.wat_bal_filt_attrs()

        df = pd.concat([cat, wat_bal_all, wat_bal_filt], axis=1)
        return df

    def fetch_static_features(
            self,
            stn_id: Union[str, list] = None,
            features: Union[str, list] = None
    )->pd.DataFrame:

        basin = self.basin_attributes()
        gauge = self.gauge_attributes()

        df = pd.concat([basin, gauge], axis=1)
        return df

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
        q = self.fetch_q(stations)
        area_m2 = self.area(stations) * 1e6  # area in m2
        q = (q / area_m2) * 86400  # cms to m/day
        return q * 1e3  # to mm/day

    def fetch_q(
            self,
            stations:Union[str, List[str]] = None,
    ):
        """
        returns streamflow for one or more stations

        parameters
        -----------
        stations : str/List[str]
            name or names of stations for which streamflow is to be fetched

        Returns
        --------
        pd.DataFrame
            a pandas dataframe whose index is the time and columns are names of stations
            For daily timestep, the dataframe has shape of 32630 rows and 111 columns

        """
        stations = check_attributes(stations, self.stations())

        qs = []
        for stn in stations:
            qs.append(self.fetch_stn_q(stn))

        return pd.concat(qs, axis=1)

    def fetch_stn_q(self, stn:str)->pd.Series:
        """returns streamflow for single station"""

        fpath = os.path.join(self.q_path, f"ID_{stn}.csv")

        df = pd.read_csv(fpath, sep=';',
                         dtype={'YYYY': int,
                                'MM': int,
                                'DD': int,
                                'qobs': np.float32,
                                'qc_flag': np.float32
                                })

        index = df.apply(
            lambda x:datetime.strptime("{0} {1} {2}".format(
                x['YYYY'].astype(int),x['MM'].astype(int), x['DD'].astype(int)),"%Y %m %d"),
            axis=1)
        df.index = pd.to_datetime(index)

        return df['qobs']

    def fetch_clim_features(
            self,
            stations:Union[str, List[str]] = None
    ):
        """Returns climate time series data for one or more stations

        Returns
        -------
        pd.DataFrame
        """
        stations = check_attributes(stations, self.stations())
        raise NotImplementedError

    def fetch_stn_clim(self, stn)->pd.DataFrame:
        """returns climate time series data for one station

        Returns
        -------
        pd.DataFrame
            a pandas dataframe with 23 columns
        """
        fpath = os.path.join(self._clim_ts_path(), f"ID_{stn}.csv")

        dtypes = {
            "YYYY": np.int32,
            "DD": np.int32,
            "MM": np.int32,
            "2m_temp_max": np.float32,
            "2m_temp_mean": np.float32,
            "2m_temp_min": np.float32,
            "2m_dp_temp_max": np.float32,
            "2m_dp_temp_mean": np.float32,
            "2m_dp_temp_min": np.float32,
            "10m_wind_u": np.float32,
            "10m_wind_v": np.float32,
            "fcst_alb": np.float32,
            "lai_high_veg": np.float32,
            "lai_low_veg": np.float32,
            "swe": np.float32,
            "surf_net_solar_rad_max": np.int32,
            "surf_net_solar_rad_mean": np.int32,
            "surf_net_therm_rad_max": np.int32,
            "surf_net_therm_rad_mean": np.int32,
            "surf_press": np.float32,
            "total_et": np.float32,
            "prec": np.float32,
            "volsw_123": np.float32,
            "volsw_4": np.float32,
            "prec_rav": np.float32,
            "prec_carra": np.float32,
        }
        df = pd.read_csv(fpath, sep=';', dtype=dtypes)

        index = df.apply(
            lambda x: datetime.strptime("{0} {1} {2}".format(
                x['YYYY'].astype(int), x['MM'].astype(int), x['DD'].astype(int)), "%Y %m %d"),
            axis=1)
        df.index = pd.to_datetime(index)
        for col in ['YYYY', 'MM', 'DD', 'DOY']:
            df.pop(col)

        return df
