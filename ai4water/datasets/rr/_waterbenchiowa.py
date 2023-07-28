
from typing import List, Union

from ai4water.backend import os, pd

from .camels import Camels
from ..utils import check_attributes


class WaterBenchIowa(Camels):
    """
    Rainfall run-off dataset for Iowa (US) following the work of
    `Demir et al., 2022 <https://doi.org/10.5194/essd-14-5605-2022>`_
    This is hourly dataset of 125 catchments with
    7 static features and 3 dyanmic features (pcp, et, discharge) for each catchment.
    The dyanmic features are timeseries from 2011-10-01 12:00 to 2018-09-30 11:00.

    Examples
    --------
    >>> from ai4water.datasets import WaterBenchIowa
    >>> ds = WaterBenchIowa()
    ... # fetch static and dynamic features of 5 stations
    >>> data = ds.fetch(5, as_dataframe=True)
    >>> data.shape  # it is a multi-indexed DataFrame
    (184032, 5)
    ... # fetch both static and dynamic features of 5 stations
    >>> data = ds.fetch(5, static_features="all", as_dataframe=True)
    >>> data.keys()
    dict_keys(['dynamic', 'static'])
    >>> data['static'].shape
    (5, 7)
    >>> data['dynamic']  # returns a xarray DataSet
    ... # using another method
    >>> data = ds.fetch_dynamic_features('644', as_dataframe=True)
    >>> data.unstack().shape
    (61344, 3)
    # when we get both static and dynamic data, the returned data is a dictionary
    # with ``static`` and ``dyanic`` keys.
    >>> data = ds.fetch(stations='644', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
    >>> ((1, 7), (184032, 1))
    """
    url = "https://zenodo.org/record/7087806#.Y6rW-BVByUk"

    def __init__(self, path=None):
        super(WaterBenchIowa, self).__init__(path=path)

        self._download()

        self._maybe_to_netcdf('WaterBenchIowa.nc')

    def stations(self)->List[str]:
        return [fname.split('_')[0] for fname in os.listdir(self.ts_path) if fname.endswith('.csv')]

    @property
    def ts_path(self)->str:
        return os.path.join(self.path, 'data_time_series', 'data_time_series')

    @property
    def dynamic_features(self) -> List[str]:
        return ['precipitation', 'et', 'discharge']

    @property
    def static_features(self)->List[str]:
        return ['travel_time', 'area', 'slope', 'loam', 'silt',
                'sandy_clay_loam', 'silty_clay_loam']

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

        Examples
        --------
            >>> from ai4water.datasets import WaterBenchIowa
            >>> dataset = WaterBenchIowa()
            >>> data = dataset.fetch_station_attributes('666')
        """
        check_attributes(dynamic_features, self.dynamic_features)
        fname = os.path.join(self.ts_path, f"{station}_data.csv")
        df = pd.read_csv(fname)
        df.index = pd.to_datetime(df.pop('datetime'))

        return df

    def fetch_static_features(
            self,
            stn_id: Union[str, List[str]],
            features:Union[str, List[str]]=None
    )->pd.DataFrame:
        """

        Parameters
        ----------
            stn_id : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Examples
        ---------
        >>> from ai4water.datasets import WaterBenchIowa
        >>> dataset = WaterBenchIowa()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            125
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (125, 7)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('592')
        >>> static_data.shape
           (1, 7)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['slope', 'area'])
        >>> static_data.shape
           (125, 2)
        >>> data = dataset.fetch_static_features('592', features=['slope', 'area'])
        >>> data.shape
           (1, 2)

        """
        if not isinstance(stn_id, list):
            stn_id = [stn_id]

        features = check_attributes(features, self.static_features)

        dfs = []
        for stn in stn_id:
            fname = os.path.join(self.ts_path, f"{stn}_data.csv")
            df = pd.read_csv(fname, nrows=1)
            dfs.append(df[features])

        return pd.concat(dfs)

    def _read_dynamic_from_csv(
            self,
            stations,
            dynamic_features,
            st=None,
            en=None)->dict:

        dyn = dict()
        for stn in stations:
            fname = os.path.join(self.ts_path, f"{stn}_data.csv")
            df = pd.read_csv(fname)
            df.index = pd.to_datetime(df.pop('datetime'))
            dyn[stn] = df[self.dynamic_features]
        return dyn

    @property
    def start(self):
        return "20111001 12:00"

    @property
    def end(self):
        return "20180930 11:00"
