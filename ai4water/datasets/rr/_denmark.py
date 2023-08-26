
from typing import Union, List

from ai4water.backend import os, pd
from .camels import Camels
from ..utils import check_attributes


class CAMELS_DK(Camels):
    """
    Reads Caravan extension Denmark - Danish dataset for large-sample hydrology.
    The dataset is downloaded from https://zenodo.org/record/7962379 . This dataset
    consists of static and dynamic features from 308 danish catchments. There are 38
    dynamic (time series) features from 1981-01-02 to 2020-12-31 with daily timestep
    and 211 static features for each of 308 catchments.

    Examples
    ---------
    >>> from ai4water.datasets import CAMELS_DK
    >>> dataset = CAMELS_DK()
    >>> data = dataset.fetch(0.1, as_dataframe=True)
    >>> data.shape
    (569751, 30)  # 30 represents number of stations
    >>> data.index.names == ['time', 'dynamic_features']
    True
    >>> df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
    (14609, 39)
    # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    308
    # get data by station id
    >>> df = dataset.fetch(stations='80001', as_dataframe=True).unstack()
    >>> df.shape
    (14609, 39)
    # get names of available dynamic features
    >>> dataset.dynamic_features
    # get only selected dynamic features
    >>> df = dataset.fetch(1, as_dataframe=True,
    ... dynamic_features=['snow_depth_water_equivalent_mean', 'temperature_2m_mean',
    ... 'potential_evaporation_sum', 'total_precipitation_sum', 'streamflow']).unstack()
    >>> df.shape
    (14609, 5)
    # get names of available static features
    >>> dataset.static_features
    # get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape
    (569751, 10)  # remember this is multi-indexed DataFrame
    # when we get both static and dynamic data, the returned data is a dictionary
    # with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='80001', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
    ((1, 211), (569751, 1))
    """

    url = "https://zenodo.org/record/7962379"

    def __init__(self,
                 path=None,
                 overwrite=False,
                 to_netcdf:bool = True,
                 **kwargs):
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
            This will fasten repeated calls to fetch etc but will
            require netcdf5 package as well as xarry.
        """
        super(CAMELS_DK, self).__init__(path=path, **kwargs)
        self.path = path
        self._download(overwrite=overwrite)

        self.dyn_fname = os.path.join(self.path, 'camelsdk_dyn.nc')

        if to_netcdf:
            self._maybe_to_netcdf('camelsdk_dyn')

    @property
    def csv_path(self):
        return os.path.join(self.path, "Caravan_extension_DK",
                        "Caravan_extension_DK", "Caravan_extension_DK",
                        "timeseries", "csv", "camelsdk")

    @property
    def nc_path(self):
        return os.path.join(self.path, "Caravan_extension_DK",
                        "Caravan_extension_DK", "Caravan_extension_DK",
                        "timeseries", "netcdf", "camelsdk")

    @property
    def other_attr_fpath(self):
        """returns path to attributes_other_camelsdk.csv file
        """
        return os.path.join(self.path, "Caravan_extension_DK",
                        "Caravan_extension_DK", "Caravan_extension_DK",
                        "attributes", "camelsdk", "attributes_other_camelsdk.csv")

    @property
    def caravan_attr_fpath(self):
        """returns path to attributes_caravan_camelsdk.csv file
        """
        return os.path.join(self.path, "Caravan_extension_DK",
                        "Caravan_extension_DK", "Caravan_extension_DK",
                        "attributes", "camelsdk",
                            "attributes_caravan_camelsdk.csv")
    def stations(self)->List[str]:
        return [fname.split(".csv")[0][9:] for fname in os.listdir(self.csv_path)]

    def _read_csv(self, stn:str)->pd.DataFrame:
        fpath = os.path.join(self.csv_path, f"camelsdk_{stn}.csv")
        df = pd.read_csv(os.path.join(fpath))
        df.index = pd.to_datetime(df.pop('date'))
        return df

    @property
    def dynamic_features(self)->List[str]:
        """returns names of dynamic features"""
        return self._read_csv('100006').columns.to_list()

    @property
    def static_features(self)->List[str]:
        """returns static features for Denmark catchments"""
        caravan = pd.read_csv(self.caravan_attr_fpath)
        _ = caravan.pop('gauge_id')
        other = pd.read_csv(self.other_attr_fpath)
        _ = other.pop('gauge_id')
        atlas = pd.read_csv(self.hyd_atlas_fpath)
        _ = atlas.pop('gauge_id')

        return pd.concat([caravan, other, atlas], axis=1).columns.to_list()

    @property
    def start(self):  # start of data
        return pd.Timestamp('1981-01-02 00:00:00')

    @property
    def end(self)->pd.Timestamp:  # end of data
        return pd.Timestamp('2020-12-31 00:00:00')


    def q_mmd(
            self,
            stations: Union[str, List[str]] = None
    )->pd.DataFrame:
        """
        returns streamflow in the units of milimeter per day. This is obtained
        by diving ``streamflow``/area

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
                                           dynamic_features='streamflow',
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
        >>> from ai4water.datasets import CAMELS_DK
        >>> dataset = CAMELS_DK()
        >>> dataset.area()  # returns area of all stations
        >>> dataset.stn_coords('100010')  # returns area of station whose id is 912101A
        >>> dataset.stn_coords(['100010', '210062'])  # returns area of two stations
        """

        stations = check_attributes(stations, self.stations())

        df = pd.read_csv(self.other_attr_fpath,
                         dtype={"gauge_id": str,
                                'gauge_lat': float,
                                'gauge_lon': float,
                                'area': float,
                                'gauge_name': str,
                                'country': str
                                })
        df.index = [name.split('camelsdk_')[1] for name in df['gauge_id']]

        return df.loc[stations, 'area']

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
        >>> dataset = CAMELS_DK()
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('100010')  # returns coordinates of station whose id is 912101A
        >>> dataset.stn_coords(['100010', '210062'])  # returns coordinates of two stations

        """
        df = pd.read_csv(self.other_attr_fpath,
                         dtype={"gauge_id": str,
                                'gauge_lat': float,
                                'gauge_lon': float,
                                'area': float,
                                'gauge_name': str,
                                'country': str
                                })
        df.index = [name.split('camelsdk_')[1] for name in df['gauge_id']]
        df = df[['gauge_lat', 'gauge_lon']]
        df.columns = ['lat', 'long']

        stations = check_attributes(stations, self.stations())

        return df.loc[stations, :]

    def _read_dynamic_from_csv(
            self,
            stations,
            dynamic_features,
            st=None,
            en=None)->dict:

        features = check_attributes(dynamic_features, self.dynamic_features)

        dyn = {stn: self._read_csv(stn)[features] for stn in stations}

        return dyn

    def fetch_static_features(
            self,
            stn_id: Union[str, List[str]] = None,
            features:Union[str, List[str]]=None
    ) -> pd.DataFrame:
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
        >>> from ai4water.datasets import CAMELS_DK
        >>> dataset = CAMELS_DK()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            308
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (308, 211)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('80001')
        >>> static_data.shape
           (1, 211)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['gauge_lat', 'area'])
        >>> static_data.shape
           (308, 2)
        >>> data = dataset.fetch_static_features('80001', features=['gauge_lat', 'area'])
        >>> data.shape
           (1, 2)

        """
        stations = check_attributes(stn_id, self.stations())
        features = check_attributes(features, self.static_features)
        df = pd.concat([self.hyd_atlas_attributes(),
                   self.other_static_attributes(),
                   self.caravan_static_attributes()], axis=1)
        return df.loc[stations, features]

    @property
    def hyd_atlas_fpath(self):
        return os.path.join(self.path,
                            "Caravan_extension_DK",
                            "Caravan_extension_DK",
                            "Caravan_extension_DK",
                            "attributes", "camelsdk",
                            "attributes_hydroatlas_camelsdk.csv")

    def hyd_atlas_attributes(self, stations=None)->pd.DataFrame:
        """
        Returns
        --------
            a pandas DataFrame of shape (308, 196)
        """
        stations = check_attributes(stations, self.stations())
        df = pd.read_csv(self.hyd_atlas_fpath)

        indices = df.pop('gauge_id')
        df.index = [idx[9:] for idx in indices]
        return df

    def other_static_attributes(self, stations=None) -> pd.DataFrame:
        """
        Returns
        --------
            a pandas DataFrame of shape (308, 5)
        """
        stations = check_attributes(stations, self.stations())
        df = pd.read_csv(self.other_attr_fpath)
        indices = df.pop('gauge_id')
        df.index = [idx[9:] for idx in indices]
        return df

    def caravan_static_attributes(self, stations=None) -> pd.DataFrame:
        """
        Returns
        --------
            a pandas DataFrame of shape (308, 10)
        """
        stations = check_attributes(stations, self.stations())
        df = pd.read_csv(self.caravan_attr_fpath)
        indices = df.pop('gauge_id')
        df.index = [idx[9:] for idx in indices]
        return df
