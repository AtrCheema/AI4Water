
from typing import Union, List

from ai4water.backend import os, pd, xr, np
from .camels import Camels
from ..utils import check_attributes, sanity_check


class HYSETS(Camels):
    """
    database for hydrometeorological modeling of 14,425 North American watersheds
    from 1950-2018 following the work of `Arsenault et al., 2020 <https://doi.org/10.1038/s41597-020-00583-2>`_
    The user must manually download the files, unpack them and provide
    the `path` where these files are saved.

    This data comes with multiple sources. Each source having one or more dynamic_features
    Following data_source are available.

    +---------------+------------------------------+
    |sources        | dynamic_features             |
    |===============|==============================|
    |SNODAS_SWE     | dscharge, swe                |
    +---------------+------------------------------+
    |SCDNA          | discharge, pr, tasmin, tasmax|
    +---------------+------------------------------+
    |nonQC_stations | discharge, pr, tasmin, tasmax|
    +---------------+------------------------------+
    |Livneh         | discharge, pr, tasmin, tasmax|
    +---------------+------------------------------+
    |ERA5           | discharge, pr, tasmax, tasmin|
    +---------------+------------------------------+
    |ERAS5Land_SWE  | discharge, swe               |
    +---------------+------------------------------+
    |ERA5Land       | discharge, pr, tasmax, tasmin|
    +---------------+------------------------------+

    all sources contain one or more following dynamic_features
    with following shapes

    +----------------------------+------------------+
    |dynamic_features            |      shape       |
    |============================|==================|
    |time                        |   (25202,)       |
    +----------------------------+------------------+
    |watershedID                 |   (14425,)       |
    +----------------------------+------------------+
    |drainage_area               |   (14425,)       |
    +----------------------------+------------------+
    |drainage_area_GSIM          |   (14425,)       |
    +----------------------------+------------------+
    |flag_GSIM_boundaries        |   (14425,)       |
    +----------------------------+------------------+
    |flag_artificial_boundaries  |   (14425,)       |
    +----------------------------+------------------+
    |centroid_lat                |   (14425,)       |
    +----------------------------+------------------+
    |centroid_lon                |   (14425,)       |
    +----------------------------+------------------+
    |elevation                   |   (14425,)       |
    +----------------------------+------------------+
    |slope                       |   (14425,)       |
    +----------------------------+------------------+
    |discharge                   |   (14425, 25202) |
    +----------------------------+------------------+
    |pr                          |   (14425, 25202) |
    +----------------------------+------------------+
    |tasmax                      |   (14425, 25202) |
    +----------------------------+------------------+
    |tasmin                      |   (14425, 25202) |
    +----------------------------+------------------+

    Examples
    --------
    >>> from ai4water.datasets import HYSETS
    >>> dataset = HYSETS(path="path/to/HYSETS")
    ... # fetch data of a random station
    >>> df = dataset.fetch(1, as_dataframe=True)
    >>> df.shape
    (25202, 5)
    >>> stations = dataset.stations()
    >>> len(stations)
    14425
    >>> df = dataset.fetch('999', as_dataframe=True)
    >>> df.unstack().shape
    (25202, 5)

    """
    doi = "https://doi.org/10.1038/s41597-020-00583-2"
    url = "https://osf.io/rpc3w/"
    Q_SRC = ['ERA5', 'ERA5Land', 'ERA5Land_SWE', 'Livneh', 'nonQC_stations', 'SCDNA', 'SNODAS_SWE']
    SWE_SRC = ['ERA5Land_SWE', 'SNODAS_SWE']
    OTHER_SRC = [src for src in Q_SRC if src not in ['ERA5Land_SWE', 'SNODAS_SWE']]
    dynamic_features = ['discharge', 'swe', 'tasmin', 'tasmax', 'pr']

    def __init__(self,
                 path: str,
                 swe_source: str = "SNODAS_SWE",
                 discharge_source: str = "ERA5",
                 tasmin_source: str = "ERA5",
                 tasmax_source: str = "ERA5",
                 pr_source: str = "ERA5",
                 **kwargs
                 ):
        """
        parameters
        --------------
            path : str
                If the data is alredy downloaded then provide the complete
                path to it. If None, then the data will be downloaded.
                The data is downloaded once and therefore susbsequent
                calls to this class will not download the data unless
                ``overwrite`` is set to True.
            swe_source : str
                source of swe data.
            discharge_source :
                source of discharge data
            tasmin_source :
                source of tasmin data
            tasmax_source :
                source of tasmax data
            pr_source :
                source of pr data
            kwargs :
                arguments for `Camels` base class

        """

        assert swe_source in self.SWE_SRC, f'source must be one of {self.SWE_SRC}'
        assert discharge_source in self.Q_SRC, f'source must be one of {self.Q_SRC}'
        assert tasmin_source in self.OTHER_SRC, f'source must be one of {self.OTHER_SRC}'
        assert tasmax_source in self.OTHER_SRC, f'source must be one of {self.OTHER_SRC}'
        assert pr_source in self.OTHER_SRC, f'source must be one of {self.OTHER_SRC}'

        self.sources = {
            'swe': swe_source,
            'discharge': discharge_source,
            'tasmin': tasmin_source,
            'tasmax': tasmax_source,
            'pr': pr_source
        }

        super().__init__(**kwargs)

        self.path = path

        fpath = os.path.join(self.path, 'hysets_dyn.nc')
        if not os.path.exists(fpath):
            self._maybe_to_netcdf('hysets_dyn')

    def _maybe_to_netcdf(self, fname: str):
        # todo saving as one file takes very long time
        oneD_vars = []
        twoD_vars = []

        for src in self.Q_SRC:
            xds = xr.open_dataset(os.path.join(self.path, f'HYSETS_2020_{src}.nc'))

            for var in xds.variables:
                print(f'getting {var} from source {src} ')

                if len(xds[var].data.shape) > 1:
                    xar = xds[var]
                    xar.name = f"{xar.name}_{src}"
                    twoD_vars.append(xar)
                else:
                    xar = xds[var]
                    xar.name = f"{xar.name}_{src}"
                    oneD_vars.append(xar)

        oneD_xds = xr.merge(oneD_vars)
        twoD_xds = xr.merge(twoD_vars)
        oneD_xds.to_netcdf(os.path.join(self.path, "hysets_static.nc"))
        twoD_xds.to_netcdf(os.path.join(self.path, "hysets_dyn.nc"))

        return

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, x):
        sanity_check('HYSETS', x)
        self._path = x

    @property
    def static_features(self)->list:
        df = self.read_static_data()
        return df.columns.to_list()

    def stations(self) -> List[str]:
        """
        Returns
        -------
        list
            a list of ids of stations

        Examples
        --------
        >>> dataset = HYSETS()
        ... # get name of all stations as list
        >>> dataset.stations()

        """
        return self.read_static_data().index.to_list()

    @property
    def start(self)->str:
        return "19500101"

    @property
    def end(self)->str:
        return "20181231"

    def fetch_stations_attributes(
            self,
            stations: list,
            dynamic_features: Union[str, list, None] = 'all',
            static_features: Union[str, list, None] = None,
            st=None,
            en=None,
            as_dataframe: bool = False,
            **kwargs
    ):
        """returns attributes of multiple stations
        Examples
        --------
        >>> from ai4water.datasets import HYSETS
        >>> dataset = HYSETS()
        >>> stations = dataset.stations()[0:3]
        >>> attributes = dataset.fetch_stations_attributes(stations)
        """
        stations = check_attributes(stations, self.stations())
        stations = [int(stn) for stn in stations]

        if dynamic_features is not None:

            dyn = self._fetch_dynamic_features(stations=stations,
                                               dynamic_features=dynamic_features,
                                               as_dataframe=as_dataframe,
                                               st=st,
                                               en=en,
                                               **kwargs
                                               )

            if static_features is not None:  # we want both static and dynamic
                to_return = {}
                static = self._fetch_static_features(station=stations,
                                                     static_features=static_features,
                                                     st=st,
                                                     en=en,
                                                     **kwargs
                                                     )
                to_return['static'] = static
                to_return['dynamic'] = dyn
            else:
                to_return = dyn

        elif static_features is not None:
            # we want only static
            to_return = self._fetch_static_features(
                station=stations,
                static_features=static_features,
                **kwargs
            )
        else:
            raise ValueError

        return to_return

    def fetch_dynamic_features(
            self,
            stn_id,
            features='all',
            st=None,
            en=None,
            as_dataframe=False
    ):
        """Fetches dynamic attributes of one station.

        Examples
        --------
        >>> from ai4water.datasets import HYSETS
        >>> dataset = HYSETS()
        >>> dyn_features = dataset.fetch_dynamic_features('station_name')
        """
        station = [int(stn_id)]
        return self._fetch_dynamic_features(
            stations=station,
            dynamic_features=features,
            st=st,
            en=en,
            as_dataframe=as_dataframe
        )

    def _fetch_dynamic_features(
            self,
            stations: list,
            dynamic_features='all',
            st=None,
            en=None,
            as_dataframe=False,
            as_ts=False
    ):
        """Fetches dynamic attributes of station."""
        st, en = self._check_length(st, en)
        attrs = check_attributes(dynamic_features, self.dynamic_features)

        stations = np.subtract(stations, 1).tolist()
        # maybe we don't need to read all variables
        sources = {k: v for k, v in self.sources.items() if k in attrs}

        # original .nc file contains datasets with dynamic and static features as data_vars
        # however, for uniformity of this API and easy usage, we want a Dataset to have
        # station names/gauge_ids as data_vars and each data_var has
        # dimension (time, dynamic_variables)
        # Therefore, first read all data for each station from .nc file
        # then rearrange it.
        # todo, this operation is slower because of `to_dataframe`
        # also doing this removes all the metadata
        x = {}
        f = os.path.join(self.path, "hysets_dyn.nc")
        xds = xr.open_dataset(f)
        for stn in stations:
            xds1 = xds[[f'{k}_{v}' for k, v in sources.items()]].sel(watershed=stn, time=slice(st, en))
            xds1 = xds1.rename_vars({f'{k}_{v}': k for k, v in sources.items()})
            x[stn] = xds1.to_dataframe(['time'])
        xds = xr.Dataset(x)
        xds = xds.rename_dims({'dim_1': 'dynamic_features'})
        xds = xds.rename_vars({'dim_1': 'dynamic_features'})

        if as_dataframe:
            return xds.to_dataframe(['time', 'dynamic_features'])

        return xds

    def _fetch_static_features(
            self,
            station,
            static_features: Union[str, list] = 'all',
            st=None,
            en=None,
            as_ts=False
    ):

        df = self.read_static_data()

        static_features = check_attributes(static_features, self.static_features)

        if isinstance(station, str):
            station = [station]
        elif isinstance(station, int):
            station = [str(station)]
        elif isinstance(station, list):
            station = [str(stn) for stn in station]
        else:
            raise ValueError

        return self.to_ts(df.loc[station][static_features], st=st, en=en, as_ts=as_ts)

    def fetch_static_features(
            self,
            stn_id: Union[str, List[str]],
            features:Union[str, List[str]]="all",
            st=None,
            en=None,
            as_ts=False
    ) -> pd.DataFrame:
        """returns static atttributes of one or multiple stations

        Parameters
        ----------
            stn_id : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Examples
        ---------
        >>> from ai4water.datasets import HYSETS
        >>> dataset = HYSETS()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            14425
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (14425, 28)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('991')
        >>> static_data.shape
           (1, 28)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['Drainage_Area_km2', 'Elevation_m'])
        >>> static_data.shape
           (14425, 2)
        """
        return self._fetch_static_features(stn_id, features, st, en, as_ts)

    def read_static_data(self):
        fname = os.path.join(self.path, 'HYSETS_watershed_properties.txt')
        static_df = pd.read_csv(fname, index_col='Watershed_ID', sep=';')
        static_df.index = static_df.index.astype(str)
        return static_df
