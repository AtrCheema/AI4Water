
import json
from typing import Union, List

from ai4water.backend import os, pd, np
from .camels import Camels
from ..utils import check_attributes


class HYPE(Camels):
    """
    Downloads and preprocesses HYPE [1]_ dataset from Lindstroem et al., 2010 [2]_ .
    This is a rainfall-runoff dataset of Costa Rica of 564 stations from 1985 to
    2019 at daily, monthly and yearly time steps.

    Examples
    --------
        >>> from ai4water.datasets import HYPE
        >>> dataset = HYPE()
        ... # get data of 5% of stations
        >>> df = dataset.fetch(stations=0.05, as_dataframe=True)  # returns a multiindex dataframe
        >>> df.shape
          (115047, 28)
        ... # fetch data of 5 (randomly selected) stations
        >>> df = dataset.fetch(stations=5, as_dataframe=True)
        >>> df.shape
           (115047, 5)
        fetch data of 3 selected stations
        >>> df = dataset.fetch(stations=['564','563','562'], as_dataframe=True)
        >>> df.shape
           (115047, 3)
        ... # fetch data of a single stations
        >>> df = dataset.fetch(stations='500', as_dataframe=True)
           (115047, 1)
        # get only selected dynamic features
        >>> df = dataset.fetch(stations='501',
        ...    dynamic_features=['AET_mm', 'Prec_mm',  'Streamflow_mm'], as_dataframe=True)
        # fetch data between selected periods
        >>> df = dataset.fetch(stations='225', st="20010101", en="20101231", as_dataframe=True)
        >>> df.shape
           (32868, 1)
        ... # get data at monthly time step
        >>> dataset = HYPE(time_step="month")
        >>> df = dataset.fetch(stations='500', as_dataframe=True)
        >>> df.shape
           (3780, 1)

    .. [1] https://zenodo.org/record/4029572

    .. [2] https://doi.org/10.2166/nh.2010.007

    """
    url = [
        "https://zenodo.org/record/581435",
        "https://zenodo.org/record/4029572"
    ]
    dynamic_features = [
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

    def __init__(self,
                 time_step: str = 'daily',
                 path = None,
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
        time_step : str
            one of ``daily``, ``month`` or ``year``
        **kwargs
            key word arguments
        """
        assert time_step in ['daily', 'month', 'year']
        self.time_step = time_step
        self.path = path
        super().__init__(path=path, **kwargs)

        self._download()

        fpath = os.path.join(self.path, 'hype_year_dyn.nc')
        if not os.path.exists(fpath):
            self.time_step = 'daily'
            self._maybe_to_netcdf('hype_daily_dyn')
            self.time_step = 'month'
            self._maybe_to_netcdf('hype_month_dyn')
            self.time_step = 'year'
            self._maybe_to_netcdf('hype_year_dyn')
            self.time_step = time_step

        self.dyn_fname = os.path.join(self.path, f'hype_{time_step}_dyn.nc')

    def stations(self) -> list:
        _stations = np.arange(1, 565).astype(str)
        return list(_stations)

    @property
    def static_features(self):
        return []

    def _read_dynamic_from_csv(self,
                               stations: list,
                               features: Union[str, list] = 'all',
                               st=None,
                               en=None,
                               ):

        dynamic_features = check_attributes(features, self.dynamic_features)

        _dynamic_features = []
        for dyn_attr in dynamic_features:
            pref, suff = dyn_attr.split('_')[0], dyn_attr.split('_')[-1]
            _dyn_attr = f"{pref}_{self.time_step}_{suff}"
            _dynamic_features.append(_dyn_attr)

        df_attrs = {}
        for dyn_attr in _dynamic_features:
            fname = f"{dyn_attr}.csv"
            fpath = os.path.join(self.path, fname)
            index_col_name = 'DATE'
            if fname in ['SM_month_mm.csv', 'SM_year_mm.csv']:
                index_col_name = 'Date'
            _df = pd.read_csv(fpath, index_col=index_col_name)
            _df.index = pd.to_datetime(_df.index)
            # todo, some stations have wider range than self.st/self.en
            df_attrs[dyn_attr] = _df.loc[self.start:self.end]

        stns_dfs = {}
        for st in stations:
            stn_dfs = []
            cols = []
            for dyn_attr, dyn_df in df_attrs.items():
                stn_dfs.append(dyn_df[st])
                col_name = f"{dyn_attr.split('_')[0]}_{dyn_attr.split('_')[-1]}"  # get original name without time_step
                cols.append(col_name)
            stn_df = pd.concat(stn_dfs, axis=1)
            stn_df.columns = cols
            stns_dfs[st] = stn_df

        return stns_dfs

    def q_mmd(
            self,
            stations: Union[str, List[str]] = None
    )->pd.DataFrame:
        """
        returns streamflow in the units of milimeter per day. The name of
        original timeseries is ``Streamflow_mm``.

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
                                           dynamic_features='Streamflow_mm',
                                           as_dataframe=True)
        q.index = q.index.get_level_values(0)
        return q

    def fetch_static_features(self, stn_id, features=None):
        """static data for HYPE is not available."""
        raise ValueError(f'No static feature for {self.name}')

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
        >>> from ai4water.datasets import HYPE
        >>> dataset = HYPE()
        >>> dataset.area()  # returns area of all stations
        >>> dataset.stn_coords('2')  # returns area of station whose id is 912101A
        >>> dataset.stn_coords(['2', '605'])  # returns area of two stations
        """
        stations = check_attributes(stations, self.stations())

        fpath = os.path.join(self.path, 'Catchments_CostaRica.geojson')

        with open(fpath, 'r') as fp:
            data = json.load(fp)

        areas = []
        indices = []
        indices = []
        for idx, feature in enumerate(data['features']):
            area_m2 = feature['properties']['Area m2']

            areas.append(area_m2/1e6)
            indices.append(str(feature['properties']['subid']))

        s = pd.Series(
            np.array(areas),
            name="area",
            index=indices)

        return s.loc[stations]

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
        >>> dataset = HYPE()
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('2')  # returns coordinates of station whose id is 912101A
        >>> dataset.stn_coords(['2', '605'])  # returns coordinates of two stations
        """

        stations = check_attributes(stations, self.stations())
        fpath = os.path.join(self.path, 'Catchments_CostaRica.geojson')

        with open(fpath, 'r') as fp:
            data = json.load(fp)

        lats = []
        longs = []
        indices = []
        for idx, feature in enumerate(data['features']):
            coord = feature['geometry']['coordinates']
            lat = feature['properties']['Latitude']
            if len(coord) == 1:
                xy = np.array(coord)[0]
            else:
                xy = np.array(coord[0])

            long = xy[:, 0].min()
            longs.append(long)
            lats.append(lat)
            indices.append(str(feature['properties']['subid']))

        df = pd.DataFrame(
            np.vstack([np.array(lats), np.array(longs)]).transpose(),
            columns=['lat', 'long'], index=indices)

        return df.loc[stations, :]

    @property
    def start(self):
        return '19850101'

    @property
    def end(self):
        return '20191231'
