
import json
from typing import Union, List

from ai4water.utils.utils import dateandtime_now
from ai4water.backend import os, pd, xr, np

from .camels import Camels
from ..utils import check_attributes


class CCAM(Camels):
    """
    Dataset for chinese catchments. The CCAM dataset was published by
    `Hao et al., 2021 <https://doi.org/10.5194/essd-13-5591-2021>`_ has two sets.
    One set consists of catchment attributes, meteorological data, catchment boundaries
    of over 4000 catchments. However this data does not have streamflow data. The second
    set consists of streamflow, catchment attributes, catchment boundaries and meteorological
    data for 102 catchments of Yellow River. Since this second set conforms to the norms
    of CAMELS, this class uses this second set. Therefore, the ``fetch``, ``stations`` and other
    methods/attributes of this class return data of only Yellow River catchments
    and not for whole china. However, the first set of data is can
    also be fetched using `fetch_meteo` method of this class. The temporal extent of both
    sets is from 1999 to 2020. However, the streamflow time series in first set has very
    large number of missing values. The data of Yellow river consists fo 16 dynamic
    features (time series) and 124 static features (catchment attributes).

    Examples
    ---------
    >>> from ai4water.datasets import CCAM
    >>> dataset = CCAM()
    >>> data = dataset.fetch(0.1, as_dataframe=True)
    >>> data.shape
    (128560, 10)
    >>> df.index.names == ['time', 'dynamic_features']
    True
    >>> df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
    (8035, 16)
    # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    102
    # get data by station id
    >>> df = dataset.fetch(stations='0010', as_dataframe=True).unstack()
    >>> df.shape
    (8035, 16)
    # get names of available dynamic features
    >>> dataset.dynamic_features
    # get only selected dynamic features
    >>> df = dataset.fetch(1, as_dataframe=True, dynamic_features=['pre', 'tem_mean', 'evp', 'rhu', 'q']).unstack()
    >>> df.shape
    (8035, 5)
    # get names of available static features
    >>> dataset.static_features
    # get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape
    (128560, 10)  # remember this is multi-indexed DataFrame
    # when we get both static and dynamic data, the returned data is a dictionary
    # with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='0010', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
    ((1, 124), (128560, 1))

    """
    url = "https://zenodo.org/record/5729444"

    def __init__(self,
                 path=None,
                 overwrite:bool=False,
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
        super(CCAM, self).__init__(path=path, **kwargs)
        self.path = path
        self._download(overwrite=overwrite)

        self.dyn_fname = os.path.join(self.path, 'ccam_dyn.nc')

        if to_netcdf:
            self._maybe_to_netcdf('ccam_dyn')
            self._maybe_meteo_to_nc()

    @property
    def meteo_path(self):
        """path where daily meteorological data of stations is present"""
        return os.path.join(self.path, "1_meteorological", '1_meteorological')

    @property
    def meteo_nc_path(self):
        return os.path.join(self.path, "meteo_data.nc")

    @property
    def meteo_stations(self)->List[str]:
        stations = [fpath.split('.')[0] for fpath in os.listdir(self.meteo_path)]
        stations.remove('35616')
        return stations

    @property
    def yr_data_path(self):
        return os.path.join(self.path, "7_HydroMLYR", "7_HydroMLYR", '1_data')

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
        >>> from ai4water.datasets import CCAM
        >>> dataset = CCAM()
        >>> dataset.area()  # returns area of all stations
        >>> dataset.stn_coords('92')  # returns area of station whose id is 912101A
        >>> dataset.stn_coords(['92', '142'])  # returns area of two stations
        """

        stations = check_attributes(stations, self.stations())

        df = self.fetch_static_features(features=['area'])
        df.columns = ['area']

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
        >>> from ai4water.datasets import CCAM
        >>> dataset = CCAM()
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('92')  # returns coordinates of station whose id is 912101A
        >>> dataset.stn_coords(['92', '142'])  # returns coordinates of two stations

        """
        df = self.fetch_static_features(features=['lat', 'lon'])
        df.columns = ['lat', 'long']
        stations = check_attributes(stations, self.stations())

        return df.loc[stations, :]

    def stations(self):
        """Returns station ids for catchments on Yellow River"""
        return os.listdir(self.yr_data_path)

    @property
    def dynamic_features(self)->List[str]:
        """names of hydro-meteorological time series data for Yellow River catchments"""

        return ['pre', 'evp', 'gst_mean', 'prs_mean', 'tem_mean', 'rhu', 'win_mean',
       'gst_min', 'prs_min', 'tem_min', 'gst_max', 'prs_max', 'tem_max', 'ssd',
       'win_max', 'q']

    @property
    def static_features(self)->List[str]:
        """names of static features for Yellow River catchments"""
        attr_fpath = os.path.join(self.yr_data_path, self.stations()[0], 'attributes.json')
        with open(attr_fpath, 'r') as fp:
            data = json.load(fp)
        return list(data.keys())

    @property
    def start(self):  # start of data
        return pd.Timestamp('1999-01-02 00:00:00')

    @property
    def end(self):  # end of data
        return pd.Timestamp('2020-12-31 00:00:00')

    def _read_meteo_from_csv(
            self,
            stn_id:str
    )->pd.DataFrame:
        """returns daily meteorological data of one station as DataFrame after reading it
        from csv file. This data is from 1990-01-01 to 2021-03-31. The returned
        dataframe has following columns
            - 'PRE'
            - 'TEM': temperature
            - 'PRS': pressure
            - 'RHU',
            - 'EVP',
            - 'WIN',
            - 'SSD': sunshine duration
            - 'GST': ground surface temperature
            - 'PET'

        """
        fpath = os.path.join(self.meteo_path, f"{stn_id}.txt")

        df = pd.read_csv(fpath)
        df.index = pd.to_datetime(df.pop("Date"))

        if 'PET' not in df:
            df['PET'] = None

        # following two stations have multiple enteries
        if stn_id in ['17456', '18161']:
            df = drop_duplicate_indices(df)

        return df

    def _maybe_meteo_to_nc(self):
        if os.path.exists(self.meteo_nc_path):
            return
        stations = os.listdir(self.meteo_path)
        dyn = {}
        for idx, stn in enumerate(stations):

            if stn not in ['35616.txt']:
                stn_id = stn.split('.')[0]

                dyn[stn_id] = self._read_meteo_from_csv(stn_id).astype(np.float32)

        data_vars = {}
        coords = {}
        for k, v in dyn.items():
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

        xds.to_netcdf(self.meteo_nc_path)
        return

    def fetch_meteo(
            self,
            stn_id:Union[str, List[str]]="all",
            features:Union[str, List[str]] = "all",
            st = '1990-01-01',
            en = '2021-03-31',
            as_dataframe:bool = True
    ):
        """
        fetches meteorological data of 4902 chinese catchments

        >>> from ai4water.datasets import CCAM
        >>> dataset = CCAM()
        >>> dynamic_features = ['PRE', 'TEM', 'PRS', 'RHU', 'EVP', 'WIN', 'PET']
        >>> st = '1999-01-01'
        >>> en = '2020-03-31'
        >>> xds = dataset.fetch_meteo(features=features, st=st, en=en)
        """
        def_features = ['PRE', 'TEM', 'PRS', 'RHU', 'EVP', 'WIN', 'SSD', 'GST', 'PET']
        features = check_attributes(features, def_features)
        stations = check_attributes(stn_id, self.meteo_stations)
        if xr is None:
            pass
        else:

            dyn = xr.load_dataset(self.meteo_nc_path)
            dyn = dyn[stations].sel(dynamic_features=features, time=slice(st, en))
            if as_dataframe:
                dyn = dyn.to_dataframe(['time', 'dynamic_features'])

        return dyn

    def _read_yr_dynamic_from_csv(
            self,
            stn_id:str
        )->pd.DataFrame:
        """
        Reads daily dynamic (meteorological + streamflow) data for one catchment of
        yellow river and returns as DataFrame
        """
        meteo_fpath = os.path.join(self.yr_data_path, stn_id, 'meteorological.txt')
        q_fpath = os.path.join(self.yr_data_path, stn_id, 'streamflow_raw.txt')

        meteo = pd.read_csv(meteo_fpath)
        meteo.index = pd.to_datetime(meteo.pop('date'))
        q = pd.read_csv(q_fpath)
        q.index = pd.to_datetime(q.pop('date'))

        return pd.concat([meteo, q], axis=1).astype(np.float32)

    def _read_dynamic_from_csv(
            self,
            stations,
            dynamic_features,
            st=None,
            en=None)->dict:
        """reads dynamic data of one or more catchments located along Yellow River basin
        """
        attributes = check_attributes(dynamic_features, self.dynamic_features)

        dyn = {stn: self._read_yr_dynamic_from_csv(stn).loc["19990101": "20201231", attributes] for stn in stations}

        # making sure that data for all stations has same dimensions by inserting nans
        # and removign duplicates
        dyn = {stn:drop_duplicate_indices(data) for stn, data in dyn.items()}
        dummy = pd.DataFrame(index=pd.date_range("19990101", "20201231", freq="D"))
        dyn = {stn: pd.concat([v, dummy], axis=1) for stn, v in dyn.items()}
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
            102
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (102, 124)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('0140')
        >>> static_data.shape
           (1, 124)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['lon', 'lat', 'area'])
        >>> static_data.shape
           (102, 3)
        >>> data = dataset.fetch_static_features('0140', features=['lon', 'lat', 'area'])
        >>> data.shape
           (1, 3)

        """
        stations = check_attributes(stn_id, self.stations())
        features = check_attributes(features, self.static_features)
        ds = []
        for stn in stations:
            d = self._read_yr_static(stn)
            ds.append(d)
        return pd.concat(ds, axis=1).transpose().loc[:, features]

    def _read_yr_static(
            self,
            stn_id:str
        )->pd.Series:
        """
        Reads catchment attributes data for Yellow River catchments
        """
        fpath = os.path.join(self.yr_data_path, stn_id, 'attributes.json')

        with open(fpath, 'r') as fp:
            data = json.load(fp)

        return pd.Series(data, name=stn_id)


def drop_duplicate_indices(df):
    return df[~df.index.duplicated(keep='first')]