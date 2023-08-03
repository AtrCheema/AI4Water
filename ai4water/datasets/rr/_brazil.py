
import glob
from typing import Union, List

from ai4water.backend import pd, os, np
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
        """
        parameters
        ----------
        path : str
            If the data is alredy downloaded then provide the complete
            path to it. If None, then the data will be downloaded.
            The data is downloaded once and therefore susbsequent
            calls to this class will not download the data unless
            ``overwrite`` is set to True.
        """
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
        >>> dataset = CAMELS_BR()
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('65100000')  # returns coordinates of station whose id is 912101A
        >>> dataset.stn_coords(['65100000', '64075000'])  # returns coordinates of two stations
        """
        fpath = os.path.join(self.path, '01_CAMELS_BR_attributes',
                             '01_CAMELS_BR_attributes',
                             'camels_br_location.txt')
        df = pd.read_csv(fpath, sep=' ')
        df.index = df['gauge_id'].astype(str)
        df = df[['gauge_lat', 'gauge_lon']]
        df.columns = ['lat', 'long']
        stations = check_attributes(stations, self.stations())

        return df.loc[stations, :]

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
            stn_id: Union[str, List[str]] = "all",
            features:Union[str, List[str]]=None
    ) -> pd.DataFrame:
        """

        fetches static feature/features of one or mroe stations

        Parameters
        ----------
            stn_id : int/list
                station id whose attribute to fetch.
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

        if stn_id == "all":
            stn_id = self.stations()

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


class CABra(Camels):
    """
    Reads and fetches CABra dataset which is catchment attribute dataset
    following the work of `Almagro et al., 2021 <https://doi.org/10.5194/hess-25-3105-2021>`_
    This dataset consists of 97 static and 12 dynamic features of 735 Brazilian
    catchments. The temporal extent is from 1980 to 2020. The dyanmic features
    consist of daily hydro-meteorological time series

    Examples
    ---------
    >>> from ai4water.datasets import CABra
    >>> dataset = CABra()
    >>> data = dataset.fetch(0.1, as_dataframe=True)
    >>> data.shape
    (131472, 73)  # 73 represents number of stations
    >>> data.index.names == ['time', 'dynamic_features']
    True
    >>> df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
    (10956, 12)
    # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    735
    # get data by station id
    >>> df = dataset.fetch(stations='92', as_dataframe=True).unstack()
    >>> df.shape
    (10956, 12)
    # get names of available dynamic features
    >>> dataset.dynamic_features
    # get only selected dynamic features
    >>> df = dataset.fetch(1, as_dataframe=True,
    ... dynamic_features=['p_ens', 'tmax_ens', 'pet_pm', 'rh_ens', 'Streamflow']).unstack()
    >>> df.shape
    (10956, 5)
    # get names of available static features
    >>> dataset.static_features
    # get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape
    (131472, 10)  # remember this is multi-indexed DataFrame
    # when we get both static and dynamic data, the returned data is a dictionary
    # with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='92', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
    ((1, 97), (131472, 1))

    """
    url = 'https://zenodo.org/record/7612350'

    def __init__(self,
                 path=None,
                 overwrite=False,
                 to_netcdf:bool = True,
                 met_src:str = 'ens',
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
        met_src : str
            source of meteorological data, must be one of
            ``ens``, ``era5`` or ``ref``.

        """
        super(CABra, self).__init__(path=path,
                                    **kwargs)
        self.path = path
        self.met_src = met_src
        self._download(overwrite=overwrite)

        self.dyn_fname = os.path.join(self.path,
                                      f'cabra_{met_src}_dyn.nc')

        if to_netcdf:
            self._maybe_to_netcdf(f'cabra_{met_src}_dyn.nc')

    @property
    def q_path(self):
        return os.path.join(self.path, "CABra_streamflow_daily_series",
                      "CABra_daily_streamflow")

    @property
    def attr_path(self):
        return os.path.join(self.path, 'CABra_attributes', 'CABra_attributes')

    @property
    def dynamic_features(self) -> list:
        return ["p_ens",
                  "tmin_ens",
                  "tmax_ens",
                  "rh_ens",
                  "wnd_ens",
                  "srad_ens",
                  "et_ens",
                  "pet_pm",
                  "pet_pt",
                  "pet_hg", 'Quality', 'Streamflow']

    @property
    def static_features(self)->List[str]:
        """names of static features"""
        return pd.concat(
            [
                self.climate_attrs(),
               self.general_attrs(),
               self.geology_attrs(),
               self.gw_attrs(),
               self.hydro_distrub_attrs(),
               self.lc_attrs(),
               self.soil_attrs(),
               self.q_attrs(),
               self.topology_attrs()], axis=1).columns.to_list()

    def stations(self)->List[str]:
        return self.add_attrs().index.astype(str).to_list()

    @property
    def start(self)->pd.Timestamp:
        return pd.Timestamp("1980-10-02")

    @property
    def end(self)->pd.Timestamp:
        return pd.Timestamp("2010-09-30")

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
        >>> dataset = CABra()
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('92')  # returns coordinates of station whose id is 912101A
        >>> dataset.stn_coords(['92', '142'])  # returns coordinates of two stations

        """
        df = self.general_attrs()
        df.index = df.index.astype(str)
        df = df[['latitude', 'longitude']]
        df.columns = ['lat', 'long']
        stations = check_attributes(stations, self.stations())
        return df.loc[stations, :]

    def add_attrs(self)->pd.DataFrame:
        """
        Returns additional catchment attributes
        """
        fpath = os.path.join(self.attr_path, "CABra_additional_attributes.txt")

        dtypes = {"CABra_ID": int,  # todo shouldn't it be str?
                  "ANA_ID": int,
                  "longitude_centroid": np.float32,
                  "latitude_centroid": np.float32,
                  "dist_coast": np.float32}

        add_attributes = pd.read_csv(fpath, sep='\t',
                                     names=list(dtypes.keys()),
                                     dtype=dtypes,
                                     header=4)
        add_attributes.index = add_attributes.pop('CABra_ID')
        return add_attributes

    def climate_attrs(self)->pd.DataFrame:
        """
        returns climate attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_climate_attributes.txt")

        dtypes = {"CABra_ID": int,  # todo shouldn't it be str?
                  "ANA_ID": int,
                  "clim_p": np.float32,
                  "clim_tmin": np.float32,
                  "clim_tmax": np.float32,
                  "clim_rh": np.float32,
                  "clim_wind": np.float32,
                  "clim_srad": np.float32,
                  "clim_et": np.float32,
                  "clim_pet": np.float32,
                  "aridity_index": np.float32,
                  "p_seasonality": np.float32,
                  "clim_quality": int,
                  }

        clim_attrs = pd.read_csv(fpath, sep='\t',
                                     names=list(dtypes.keys()),
                                     dtype=dtypes,
                                     encoding_errors='ignore',
                                     header=6)
        clim_attrs.index = clim_attrs.pop('CABra_ID')
        return clim_attrs

    def general_attrs(self)->pd.DataFrame:
        """
        returns general attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_general_attributes.txt")


        dtypes = {"CABra_ID": int,  # todo shouldn't it be str?
                  "ANA_ID": int,
                  "longitude": np.float32,
                  "latitude": np.float32,
                  "gauge_hreg": str,
                  "gauge_biome": str,
                  "gauge_state": str,
                  "missing_data": np.float32,
                  "series_length": np.float32,
                  "quality_index": np.float32,
                  }

        gen_attrs = pd.read_csv(fpath,
                                sep='\t',
                                     names=list(dtypes.keys()),
                                     dtype=dtypes,
                                     encoding_errors='ignore',
                                     header=6)
        gen_attrs.index = gen_attrs.pop('CABra_ID')
        return gen_attrs


    def geology_attrs(self)->pd.DataFrame:
        """
        returns geological attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_geology_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "catch_lith": str,
                  "sub_porosity": np.float32,
                  "sub_permeability": np.float32,
                  "sub_hconduc": np.float32,
                  }

        gen_attrs = pd.read_csv(fpath,
                             sep='\t',
                             names=list(dtypes.keys()),
                             dtype=dtypes,
                             encoding_errors='ignore',
                             header=6)
        gen_attrs.index = gen_attrs.pop('CABra_ID')
        return gen_attrs

    def gw_attrs(self)->pd.DataFrame:
        """
        returns groundwater attributes for all catchments


        """
        fpath = os.path.join(self.attr_path,
                             "CABra_groundwater_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "aquif_name": str,
                  "aquif_type": str,
                  "catch_wtd": np.float32,
                  "catch_hand": np.float32,
                  "hand_class": str,
                  "well_number": int,
                  "well_static": str,
                  "well_dynamic": str,
                  }

        gen_attrs = pd.read_csv(fpath,
                             sep='\t',
                             names=list(dtypes.keys()),
                             dtype=dtypes,
                             encoding_errors='ignore',
                             header=7)
        gen_attrs.index = gen_attrs.pop('CABra_ID')
        return gen_attrs

    def hydro_distrub_attrs(self)->pd.DataFrame:
        """
        returns geological attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_hydrologic_disturbance_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "dist_urban": int,
                  "cover_urban": np.float32,
                  "cover_crops": np.float32,
                  "res_number": int,
                  "res_area": np.float32,
                  "res_volume": np.float32,
                  "res_regulation": np.float32,
                  "water_demand": int,
                  "hdisturb_index": np.float32,
                  }

        gen_attrs = pd.read_csv(fpath,
                             sep='\t',
                             names=list(dtypes.keys()),
                             dtype=dtypes,
                             encoding_errors='ignore',
                             header=8)
        gen_attrs.index = gen_attrs.pop('CABra_ID')
        return gen_attrs

    def lc_attrs(self)->pd.DataFrame:
        """
        returns land cover attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_land-cover_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "cover_main": str,
                  "cover_bare": np.float32,
                  "cover_forest": np.float32,
                  "cover_crops": np.float32,
                  "cover_grass": np.float32,
                  "cover_moss": np.float32,
                  "cover_shrub": np.float32,
                  "cover_urban": np.float32,
                  "cover_snow": np.float32,
                  "cover_waterp": np.float32,
                  "cover_waters": np.float32,
                  "ndvi_djf": np.float32,
                  "ndvi_mam": np.float32,
                  "ndvi_jja": np.float32,
                  "ndvi_son": np.float32,
                  }

        lc_attrs = pd.read_csv(fpath,
                             sep='\t',
                             names=list(dtypes.keys()),
                             dtype=dtypes,
                             encoding_errors='ignore',
                             header=6)
        lc_attrs.index = lc_attrs.pop('CABra_ID')
        return lc_attrs

    def soil_attrs(self)->pd.DataFrame:
        """
        returns soil attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_soil_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "soil_type": str,
                  "soil_textclass": str,
                  "soil_sand": np.float32,
                  "soil_silt": np.float32,
                  "soil_clay": np.float32,
                  "soil_carbon": np.float32,
                  "soil_bulk": np.float32,
                  "soil_depth": np.float32,
                  }

        soil_attrs = pd.read_csv(fpath,
                             sep='\t',
                             names=list(dtypes.keys()),
                            dtype=dtypes,
                             encoding_errors='ignore',
                             header=7)
        soil_attrs.index = soil_attrs.pop('CABra_ID')
        return soil_attrs

    def q_attrs(self)->pd.DataFrame:
        """
        returns streamflow attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_streamflow_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "q_mean": np.float32,
                  "q_1": np.float32,
                  "q_5": np.float32,
                  "q_95": np.float32,
                  "q_99": np.float32,
                  "q_lf": np.float32,
                  "q_ld": np.float32,
                  "q_hf": np.float32,
                  "q_hd": np.float32,
                  "q_hfd": np.float32,
                  "q_zero": int,
                  "q_cv": np.float32,
                  "q_lcv": np.float32,
                  "q_hcv": np.float32,
                  "q_elasticity": np.float32,
                  "fdc_slope": np.float32,
                  "baseflow_index": np.float32,
                  'runoff_coef': np.float32
                  }
        names = list(dtypes.keys())
        dtypes.pop('q_cv')
        dtypes.pop('q_mean')
        dtypes.pop('q_lcv')
        dtypes.pop('fdc_slope')
        q_attrs = pd.read_csv(fpath,
                                sep='\t',
                                names=names,
                                dtype=dtypes,
                                encoding_errors='ignore',
                                header=7)

        q_attrs.index = q_attrs.pop('CABra_ID')
        return q_attrs


    def topology_attrs(self)->pd.DataFrame:
        """
        returns topology attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_topography_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "catch_area": np.float32,
                  "elev_mean": np.float32,
                  "elev_min": np.float32,
                  "elev_max": np.float32,
                  "elev_gauge": np.float32,
                  "catch_slope": np.float32,
                  "catch_order": int,
                  }

        gen_attrs = pd.read_csv(fpath,
                             sep='\t',
                             names=list(dtypes.keys()),
                             dtype=dtypes,
                             encoding_errors='ignore',
                             header=7)
        gen_attrs.index = gen_attrs.pop('CABra_ID')
        return gen_attrs

    def _read_q_from_csv(self, stn_id:str)->pd.DataFrame:
        q_fpath = os.path.join(self.q_path, f"CABra_{stn_id}_streamflow.txt")

        df = pd.read_csv(q_fpath, sep='\t',
                         header=9,
                         names=['Year', 'Month', 'Day', 'Streamflow', 'Quality'],
                         dtype={'Year': int,
                                'Month': int,
                                'Day': int,
                                #'Streamflow': np.float32,
                                'Quality': int}
                         )
        return df

    def _read_meteo_from_csv(
            self,
            stn_id:str,
            source="ens")->pd.DataFrame:

        meteo_path = os.path.join(self.path,
                                  'CABra_climate_daily_series',
                                  'climate_daily',
                                  source
                                  )
        meteo_fpath = os.path.join(meteo_path,
                                   f"CABra_{stn_id}_climate_{source.upper()}.txt")

        dtypes = {"Year": int,
                  "Month": int,
                  "Day": int,
                  "p_ens": np.float32,
                  "tmin_ens": np.float32,
                  "tmax_ens": np.float32,
                  "rh_ens": np.float32,
                  "wnd_ens": np.float32,
                  "srad_ens": np.float32,
                  "et_ens": np.float32,
                  "pet_pm": np.float32,
                  "pet_pt": np.float32,
                  "pet_hg": np.float32}

        if source == "ref" and stn_id in [
            '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '15', '17', '18', '19', '27', '28', '34', '526',
            '564', '567', '569'
        ]:
            df = pd.DataFrame(columns=list(dtypes.keys()))
        else:
            df = pd.read_csv(meteo_fpath,
                             sep="\t",
                             names=list(dtypes.keys()),
                             dtype=dtypes,
                             header=12)
        return df

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
        >>> from ai4water.datasets import CABra
        >>> dataset = CABra()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            735
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (735, 97)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('92')
        >>> static_data.shape
           (1, 97)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['gauge_lat', 'area'])
        >>> static_data.shape
           (735, 2)
        >>> data = dataset.fetch_static_features('92', features=['gauge_lat', 'area'])
        >>> data.shape
           (1, 2)

        """

        stations = check_attributes(stn_id, self.stations())
        features = check_attributes(features, self.static_features)

        df = pd.concat([self.climate_attrs(),
                   self.general_attrs(),
                   self.geology_attrs(),
                   self.gw_attrs(),
                   self.hydro_distrub_attrs(),
                   self.lc_attrs(),
                   self.soil_attrs(),
                   self.q_attrs(),
                   self.topology_attrs()], axis=1)

        df.index = df.index.astype(str)
        # drop duplicate columns
        df = df.loc[:, ~df.columns.duplicated()].copy()
        return df.loc[stations, features]

    def _read_dynamic_from_csv(
            self,
            stations,
            dynamic_features,
            st=None,
            en=None
    )->dict:

        attributes = check_attributes(dynamic_features, self.dynamic_features)

        # qs and meteo data has different index

        qs = [self._read_q_from_csv(stn_id=stn_id) for stn_id in stations]
        q_idx = pd.to_datetime(
            qs[0]['Year'].astype(str) + '-' + qs[0]['Month'].astype(str)+'-' + qs[0]['Day'].astype(str))

        meteos = [
            self._read_meteo_from_csv(stn_id=stn_id, source=self.met_src) for stn_id in stations]
        # 10 because first 10 stations don't have data for "ref" source
        met_idx = pd.to_datetime(
            meteos[10]['Year'].astype(str) + '-' + meteos[10]['Month'].astype(str)+'-' + meteos[10]['Day'].astype(str))

        met_cols = ["p_ens",
                  "tmin_ens",
                  "tmax_ens",
                  "rh_ens",
                  "wnd_ens",
                  "srad_ens",
                  "et_ens",
                  "pet_pm",
                  "pet_pt",
                  "pet_hg"]

        dyn = {}

        for stn, q, meteo in zip(self.stations(), qs, meteos):

            if len(meteo)==0:
                meteo = pd.DataFrame(meteo, index=met_idx)
            else:
                meteo.index = met_idx
            q.index = q_idx

            dyn[stn] = pd.concat([meteo[met_cols], q[['Quality', 'Streamflow']]], axis=1)[attributes]

        return dyn
