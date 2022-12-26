
__all__ = ["Quadica"]

from typing import Union, List, Tuple

from ai4water.backend import pd, os, np

from ._datasets import Datasets
from .utils import check_attributes, sanity_check, check_st_en


class Quadica(Datasets):
    """
    This is dataset of water quality parameters of Germany from 828 stations
    from 1950 to 2018 following the work of Ebeling_ et al., 2022. The time-step
    is monthly and annual but the monthly timeseries data is not continuous.

    .. _Ebeling:
        https://doi.org/10.5194/essd-2022-6

    """
    url = {
        "quadica.zip":
            "https://www.hydroshare.org/resource/26e8238f0be14fa1a49641cd8a455e29/data/contents/QUADICA.zip",
        "metadata.pdf":
            "https://www.hydroshare.org/resource/26e8238f0be14fa1a49641cd8a455e29/data/contents/Metadata_QUADICA.pdf",
        "catchment_attributes.csv":
            "https://www.hydroshare.org/resource/88254bd930d1466c85992a7dea6947a4/data/contents/catchment_attributes.csv"
    }

    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self.ds_dir = path
        self._download()

    @property
    def features(self)->list:
        """names of water quality parameters available in this dataset"""
        return ['Q', 'NO3', 'NO3N', 'NMin', 'TN', 'PO4', 'PO4P', 'TP', 'DOC', 'TOC']

    @property
    def stattions(self)->list:
        """IDs of stations for which data is available"""
        return self.metadata()['OBJECTID'].tolist()

    @property
    def station_names(self):
        """names of stations"""
        return self.metadata()[['OBJECTID', 'Station']]

    def wrtds_monthly(
            self,
            features:Union[str, list] = None,
            stations:Union[List[int], int] = None,
            st: Union[str, int, pd.DatetimeIndex] = None,
            en: Union[str, int, pd.DatetimeIndex] = None,
    )->pd.DataFrame:
        """
        Monthly median concentrations, flow-normalized concentrations and mean
        fluxes of water chemistry parameters. These are estimated using Weighted
        Regressions on Time, Discharge, and Season (WRTDS)
        for stations with enough data availability. This data is available for total
        140 stations. The data from all stations does not start and end at the same period.
        Therefore, some stations have more datapoints while others have less. The maximum
        datapoints for a station are 576 while smallest datapoints are 244.

        Parameters
        ----------
            features : str/list, optional
            stations : int/list optional (default=None)
                name/names of satations whose data is to be retrieved.
            st : optional
                starting point of data. By default, the data starts from 1992-09
            en : optional
                end point of data. By default, the data ends at 2013-12

        Returns
        -------
        pd.DataFrame
            a dataframe of shape (50186, 47)

        Examples
        --------
            >>> from ai4water.datasets import Quadica
            >>> dataset = Quadica()
            >>> df = dataset.wrtds_monthly()

        """
        fname = os.path.join(self.ds_dir, "quadica", "wrtds_monthly.csv")
        wrtds = pd.read_csv(fname)
        wrtds.index = pd.to_datetime(wrtds['Year'].astype(str) + ' ' + wrtds['Month'].astype(str))

        if features is None:
            features = wrtds.columns.tolist()
        elif isinstance(features, str):
            features = [features]

        assert isinstance(features, list)

        wrtds = wrtds[features]

        return check_st_en(wrtds, st, en)

    def wrtds_annual(
            self,
            features:Union[str, list] = None,
            st: Union[str, int, pd.DatetimeIndex] = None,
            en: Union[str, int, pd.DatetimeIndex] = None,
    )->pd.DataFrame:
        """Annual median concentrations, flow-normalized concentrations, and mean
        fluxes estimated using Weighted Regressions on Time, Discharge, and Season (WRTDS)
        for stations with enough data availability.

        Parameters
        ----------
            features : optional
            st : optional
                starting point of data. By default, the data starts from 1992
            en : optional
                end point of data. By default, the data ends at 2013

        Returns
        -------
        pd.DataFrame
            a dataframe of shape (4213, 46)

        Examples
        --------
            >>> from ai4water.datasets import Quadica
            >>> dataset = Quadica()
            >>> df = dataset.wrtds_annual()

        """
        fname = os.path.join(self.ds_dir, "quadica", "wrtds_annual.csv")
        wrtds = pd.read_csv(fname)
        wrtds.index = pd.to_datetime(wrtds['Year'].astype(str))

        if features is None:
            features = wrtds.columns.tolist()
        elif isinstance(features, str):
            features = [features]

        assert isinstance(features, list)

        wrtds = wrtds[features]

        return check_st_en(wrtds, st, en)

    def metadata(self)->pd.DataFrame:
        """
        fetches the metadata about the stations as pandas' dataframe.
        Each row represents metadata about one station and each column
        represents one feature. The R2 and pbias are regression coefficients
        and percent bias of WRTDS models for each parameter.

        Returns
        -------
        pd.DataFrame
            a dataframe of shape (1386, 60)
        """
        fname = os.path.join(self.ds_dir, "quadica", "metadata.csv")
        return pd.read_csv(fname,encoding='cp1252')

    def pet(
            self,
            stations: Union[List[int], int] = None,
            st: Union[str, int, pd.DatetimeIndex] = None,
            en: Union[str, int, pd.DatetimeIndex] = None,
    )->pd.DataFrame:
        """
        average monthly  potential evapotranspiration starting from
        1950-01 to 2018-09

        Examples
        --------
            >>> from ai4water.datasets import Quadica
            >>> dataset = Quadica()
            >>> df = dataset.pet() # -> (828, 1386)
        """
        fname = os.path.join(self.ds_dir, "quadica", "pet_monthly.csv")
        pet = pd.read_csv(fname, parse_dates=[['Year', 'Month']], index_col='Year_Month')

        if stations is not None:
            stations = [str(stn) for stn in stations]
            pet = pet[stations]
        return check_st_en(pet, st, en)

    def avg_temp(
            self,
            stations: Union[List[int], int] = None,
            st: Union[str, int, pd.DatetimeIndex] = None,
            en: Union[str, int, pd.DatetimeIndex] = None,
    )->pd.DataFrame:
        """
        monthly median average temperatures starting from 1950-01 to 2018-09

        parameters
        -----------
            stations :
                name of stations for which data is to be retrieved. By default, data
                for all stations is retrieved.
            st : optional
                starting point of data. By default, the data starts from 1950-01
            en : optional
                end point of data. By default, the data ends at 2018-09

        Returns
        -------
        pd.DataFrame
            a pandas dataframe of shape (time_steps, stations). With default input
            arguments, the shape is (828, 1386)

        Examples
        --------
            >>> from ai4water.datasets import Quadica
            >>> dataset = Quadica()
            >>> df = dataset.avg_temp() # -> (828, 1388)
        """

        fname = os.path.join(self.ds_dir, "quadica", "tavg_monthly.csv")
        temp = pd.read_csv(fname, parse_dates=[['Year', 'Month']], index_col='Year_Month')

        if stations is not None:
            stations = [str(stn) for stn in stations]
            temp = temp[stations]
        return check_st_en(temp, st, en)

    def precipitation(
            self,
            stations: Union[List[int], int] = None,
            st: Union[str, int, pd.DatetimeIndex] = None,
            en: Union[str, int, pd.DatetimeIndex] = None,
    )->pd.DataFrame:
        """ sums of precipitation starting from 1950-01 to 2018-09

        parameters
        -----------
            stations :
                name of stations for which data is to be retrieved. By default, data
                for all stations is retrieved.
            st : optional
                starting point of data. By default, the data starts from 1950-01
            en : optional
                end point of data. By default, the data ends at 2018-09

        Returns
        -------
        pd.DataFrame
            a dataframe of shape (828, 1388)

        Examples
        --------
            >>> from ai4water.datasets import Quadica
            >>> dataset = Quadica()
            >>> df = dataset.precipitation() # -> (828, 1388)
        """

        fname = os.path.join(self.ds_dir, "quadica", "pre_monthly.csv")
        pcp = pd.read_csv(fname, parse_dates=[['Year', 'Month']], index_col='Year_Month')

        if stations is not None:
            stations = [str(stn) for stn in stations]
            pcp = pcp[stations]

        return check_st_en(pcp, st, en)

    def monthly_medians(
            self,
            features:Union[List[str], str] = None,
            stations: Union[List[int], int] = None,
    )->pd.DataFrame:
        """
        This function reads the `c_months.csv` file which contains the monthly
        medians over the whole time series of water quality variables
        and discharge

        parameters
        ----------
        features : list/str, optional, (default=None)
            name/names of features
        stations : list/int, optional (default=None)
            stations for which

        Returns
        -------
        pd.DataFrame
            a dataframe of shape (16629, 18). 15 of the 18 columns represent a
            water chemistry parameter. 16629 comes from 1386*12 where 1386 is stations
            and 12 is months.
        """
        fname = os.path.join(self.ds_dir, "quadica", "c_months.csv")
        df = pd.read_csv(fname)

        if features is not None:
            df = df[features]

        if stations is not None:
            df = df.loc[df['OBJECTID'].isin(stations)]

        return df

    def annual_medians(
            self,
    )->pd.DataFrame:
        """Annual medians over the whole time series of water quality variables
        and discharge

        Returns
        -------
        pd.DataFrame
            a dataframe of shape (24393, 18)
        """
        fname = os.path.join(self.ds_dir, "quadica", "c_annual.csv")
        return pd.read_csv(fname)

    def fetch_annual(self):
        raise NotImplementedError

    def catchment_attributes(
            self,
            features:Union[List[str], str] = None,
            stations: Union[List[int], int] = None,
    )->pd.DataFrame:
        """
        Returns static physical catchment attributes in the form of dataframe.

        parameters
        ----------
            features : list/str, optional, (default=None)
                name/names of static attributes to fetch
            stations : list/int, optional (default=None)
                name/names of stations whose static/physical features are to be read

        Returns
        --------
        pd.DataFrame
            a pandas dataframe of shape (stations, features). With default input arguments,
            shape is (1386, 113)

        Examples
        ---------
        >>> from ai4water.datasets import Quadica
        >>> dataset = Quadica()
        >>> cat_features = dataset.catchment_attributes()
        ... # get attributes of only selected stations
        >>> dataset.catchment_attributes(stations=[1,2,3])

        """
        fname = os.path.join(self.ds_dir, "catchment_attributes.csv")
        df = pd.read_csv(fname, encoding='unicode_escape')

        if features:
            assert isinstance(features, list)
            df = df[features]

        if stations is not None:
            assert isinstance(stations, (list, np.ndarray))
            df = df.loc[df['OBJECTID'].isin(stations)]

        return df

    def fetch_monthly(
            self,
            features:Union[List[str], str] = None,
            stations:Union[List[int], int] = None,
            median:bool = True,
            fnc:bool = True,
            fluxes:bool = True,
            precipitation:bool = True,
            avg_temp:bool = True,
            pet:bool = True,
            only_continuous:bool = True,
            cat_features:bool = True,
            max_nan_tol:Union[int, None] = 0,
    )->Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetches monthly concentrations of water quality parameters.

        +----------+----------------------------------------------------+
        | median_Q | Median discharge |
        +----------+----------------------------------------------------+
        | median_COMPOUND | Median concentration from grab sampling data |
        +----------+----------------------------------------------------+
        | median_C | Median concentration from WRTDS |
        +----------+----------------------------------------------------+
        | median_FNC | Median flow-normalized concentration from WRTDS |
        +----------+----------------------------------------------------+
        | mean_Flux | Mean flux from WRTDS |
        +----------+----------------------------------------------------+
        | mean_FNFlux | Mean flow-normalized flux from WRTDS |
        +----------+----------------------------------------------------+

        parameters
        ----------
        features : str/list, optional (default=None)
            name or names of water quality parameters to fetch. By default
            following parameters are considered
                - ``NO3``
                - ``NO3N``
                - ``TN``
                - ``Nmin``
                - ``PO4``
                - ``PO4P``
                - ``TP``
                - ``DOC``
                - ``TOC``

        stations : int/list, optional (default=None)
            name or names of stations whose data is to be fetched
        median : bool, optional (default=True)
            whether to fetch median concentration values or not
        fnc : bool, optional (default=True)
            whether to fetch flow normalized concentrations or not
        fluxes : bool, optional (default=True)
            Setting this to true will add two features i.e. mean_Flux_FEATURE
            and mean_FNFlux_FEATURE
        precipitation : bool, optional (default=True)
            whether to fetch average monthly precipitation or not
        avg_temp : bool, optional (default=True)
            whether to fetch average monthly temperature or not
        pet : bool, optional (default=True)
            whether to fether potential evapotranspiration data or not
        only_continuous : bool, optional (default=True)
            If true, will return data for only those stations who have continuos
            monthly timeseries data from 1993-01-01 to 2013-01-01.
        cat_features : bool, optional (default=True)
            whether to fetch catchment features or not.
        max_nan_tol : int, optional (default=0)
            setting this value to 0 will remove the whole time-series with any
            missing values. If None, no time-series with NaNs values will be removed.

        Returns
        --------
        tuple
            two dataframes whose length is same but the columns are different
                - a pandas dataframe of timeseries of parameters (stations*timesteps, dynamic_features)
                - a pandas dataframe of static features (stations*timesteps, catchment_features)

        Examples
        --------
        >>> from ai4water.datasets import Quadica
        >>> dataset = Quadica()
        >>> mon_dyn, mon_cat = dataset.fetch_monthly(max_nan_tol=None)
        ... # However, mon_dyn contains data for all parameters and many of which have
        ... # large number of nans. If we want to fetch data only related to TN without any
        ... # missing value, we can do as below
        >>> mon_dyn_tn, mon_cat_tn = dataset.fetch_monthly(features="TN", max_nan_tol=0)
        ... # if we want to find out how many catchments are included in mon_dyn_tn
        >>> len(mon_dyn_tn['OBJECTID'].unique())
        ... # 25
        """

        if features is None:
            features = self.features

        if isinstance(features, str):
            features = [features]

        assert isinstance(features, list)

        _wrtd_features = ['median_Q']
        for feat in features:
            if fluxes:
                _wrtd_features += self._consider_fluxes(feat)
            if median:
                _wrtd_features += self._consider_median(feat)
            if fnc:
                _wrtd_features += self._consider_fnc(feat)

        _wrtd_features = list(set(_wrtd_features))
        _features = _wrtd_features.copy()

        df = self.wrtds_monthly(features=_wrtd_features + ['OBJECTID'], stations=stations)

        if only_continuous:
            groups = []
            for idx, grp in df.groupby('OBJECTID'):
                # there are 252 months from 1993 to 2013
                if len(grp.loc["19930101": "20131201"]) == 252:
                    groups.append(grp.loc["19930101": "20131201"])
            df = pd.concat(groups)

        #df[_med_features] = self.monthly_medians(features=_features, stations=stations)

        if max_nan_tol is not None:
            groups = []
            for idx, grp in df.groupby('OBJECTID'):
                if grp.isna().sum().sum() <= max_nan_tol:
                    groups.append(grp)
            if len(groups) == 0:
                raise ValueError(f"""
                No data with nans less or equal to {max_nan_tol} is found.
                Please increase the value of "max_nan_tol" or choose a different parameter.
                """)
            df = pd.concat(groups)

        if avg_temp:
            temp = self.avg_temp(df['OBJECTID'].unique(), "19930101", "20131201")
            stns = np.array([np.repeat(int(val), len(temp)) for val in temp.columns]).reshape(-1, )
            temp = np.concatenate([temp[col] for col in temp.columns])
            assert np.allclose(stns, df['OBJECTID'].values)
            df['avg_temp'] = temp

        if precipitation:
            pcp = self.precipitation(df['OBJECTID'].unique(), "19930101", "20131201")
            stns = np.array([np.repeat(int(val), len(pcp)) for val in pcp.columns]).reshape(-1, )
            pcp = np.concatenate([pcp[col] for col in pcp.columns])
            assert np.allclose(stns, df['OBJECTID'].values)
            df['precip'] = pcp

        if pet:
            pet = self.pet(df['OBJECTID'].unique(), "19930101", "20131201")
            stns = np.array([np.repeat(int(val), len(pet)) for val in pet.columns]).reshape(-1, )
            pet = np.concatenate([pet[col] for col in pet.columns])
            assert np.allclose(stns, df['OBJECTID'].values)
            df['pet'] = pet

        if cat_features:
            cat_features = self.catchment_attributes(stations=df['OBJECTID'].unique())
            n = len(df) / len(df['OBJECTID'].unique())
            # repeat each row of cat_features n times
            cat_features = cat_features.loc[cat_features.index.repeat(n)]
            assert np.allclose(cat_features['OBJECTID'].values, df['OBJECTID'].values)

        return df, cat_features

    def _consider_median(self, feature):
        d = {
            'Q': ['median_Q'],
            'DOC': ['median_C_DOC'],
            'TOC': ['median_C_TOC'],
            'TN': ['median_C_TN'],
            'TP': ['median_C_TP'],
            'PO4': ['median_C_PO4'],
            'PO4P': [],
            'NMin': ['median_C_NMin'],
            'NO3': ['median_C_NO3'],
            'NO3N': [],
        }
        return d[feature]

    def _consider_fnc(self, feature):
        d = {
            'Q': ['median_Q'],
            'DOC': ['median_FNC_DOC'],
            'TOC': ['median_FNC_TOC'],
            'TN': ['median_FNC_TN'],
            'TP': ['median_FNC_TP'],
            'PO4': ['median_FNC_PO4'],
            'PO4P': [],
            'NMin': ['median_FNC_NMin'],
            'NO3': ['median_FNC_NO3'],
            'NO3N': [],
        }
        return d[feature]

    def _consider_fluxes(self, feature):
        d = {
            'Q': ['median_Q'],
            'DOC': ['mean_Flux_DOC', 'mean_FNFlux_DOC'],
            'TOC': ['mean_Flux_TOC', 'mean_FNFlux_TOC'],
            'TN': ['mean_Flux_TN', 'mean_FNFlux_TN'],
            'TP': ['mean_Flux_TP', 'mean_FNFlux_TP'],
            'PO4': ['mean_Flux_PO4', 'mean_FNFlux_PO4'],
            'PO4P': [],
            'NMin': ['mean_Flux_NMin', 'mean_FNFlux_NMin'],
            'NO3': ['mean_Flux_NO3', 'mean_FNFlux_NO3'],
            'NO3N': [],
        }
        return d[feature]

    def to_DataSet(
            self,
            target:str = "TP",
            input_features:list = None,
            split:str = "temporal",
            lookback:int = 24,
            **ds_args
    ):
        """
        This function prepares data for machine learning prediction problem. It
        returns an instance of ai4water.preprocessing.DataSetPipeline which can be
        given to model.fit or model.predict

        parameters
        ----------
        target : str, optional (default="TN")
            parameter to consider as target
        input_features : list, optional
            names of input features
        split : str, optional (default="temporal")
            if ``temporal``, validation and test sets are taken from the data of
            each station and then concatenated. If ``spatial``, training
            validation and test is decided based upon stations.
        lookback : int
        **ds_args :
            key word arguments

        Returns
        -------
        ai4water.preprocessing.DataSet
            an instance of DataSetPipeline

        Example
        --------
        >>> from ai4water.datasets import Quadica
        ... # initialize the Quadica class
        >>> dataset = Quadica()
        ... # define the input features
        >>> inputs = ['median_Q', 'OBJECTID', 'avg_temp', 'precip', 'pet']
        ... # prepare data for TN as target
        >>> dsp = dataset.to_DataSet("TN", inputs, lookback=24)

        """

        assert split in ("temporal", "spatial")

        from ai4water.preprocessing import DataSet, DataSetPipeline

        dyn, cat = self.fetch_monthly(features=target, max_nan_tol=0)

        if input_features is None:
            input_features = ['median_Q', 'OBJECTID', 'avg_temp', 'precip', 'pet']

        output_features = [f'median_C_{target}']

        _ds_args = {
            'val_fraction': 0.2,
            'train_fraction': 0.7
        }

        if ds_args is None:
            ds_args = dict()

        _ds_args.update(ds_args)

        dsets = []
        for idx, grp in dyn.groupby("OBJECTID"):
            ds = DataSet(data=grp,
                         ts_args={'lookback': lookback},
                         input_features=input_features,
                         output_features=output_features,
                         **_ds_args)
            dsets.append(ds)

        return DataSetPipeline(*dsets)
