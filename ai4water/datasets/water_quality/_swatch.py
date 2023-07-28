
__all__ = ["Swatch"]

from typing import Union

from ai4water.backend import pd, os, np
from ai4water.datasets._datasets import Datasets


class Swatch(Datasets):
    """
    The Surface Water Chemistry (SWatCh) database as introduced in Franz_ and Lobke, 2022.

    .. _Franz:
        https://essd.copernicus.org/preprints/essd-2021-43/
    """

    url = "https://zenodo.org/record/6484939"
    def __init__(self,
                 remove_csv_after_download=False,
                 path=None,
                 **kwargs):
        """
        parameters
        ----------
        remove_csv_after_download : bool (default=False)
            if True, the csv will be removed after downloading and processing.
        """
        super().__init__(path=path, **kwargs)
        self.path = path

        self._download(tolerate_error=True)
        self._maybe_to_binary()

        if remove_csv_after_download:
            if os.path.exists(self.csv_name):
                os.remove(self.csv_name)

    @property
    def parameters(self)->list:
        """list of water quality parameters available"""
        return list(self.names.values())

    @property
    def sites(self)->list:
        """list of site names"""
        all_sites = np.load(os.path.join(self.path, 'loc_id.npy'), allow_pickle=True)
        # numpy's unique is much slower
        return list(np.sort(pd.unique(all_sites)))

    @property
    def site_names(self)->list:
        """list of site names"""
        all_sites = np.load(os.path.join(self.path, 'location.npy'), allow_pickle=True)
        # numpy's unique is much slower
        return list(np.sort(pd.unique(all_sites)))

    @property
    def csv_name(self)->str:
        return os.path.join(self.path, "SWatCh_v2.csv")

    @property
    def npy_files(self)->list:
        return [fname for f in os.walk(self.path) for fname in f[2] if fname.endswith('.npy')]

    def _maybe_to_binary(self):
        """reads the csv file and saves each columns in binary format using numpy.
        The csv file is 1.5 GB which takes lot of time for loading most the columns
        are not required most of the times.
        """
        if len(self.npy_files) == 28:
            return

        df = pd.read_csv(self.csv_name)
        h = {col: "category" for col in cats}
        dates = pd.to_datetime(df.pop("ActivityStartDate") + " " + df.pop("ActivityStartTime"))
        df.index = dates

        maybe_reduce_memory(df, hints=h)

        strings = ["ResultComment", "ResultAnalyticalMethodID", "MonitoringLocationID",
                   "MonitoringLocationName"]
        for col in strings:
            df[col] = df[col].astype(str)

        df.rename(columns=self.names, inplace=True)

        for col in df.columns:
            np.save(os.path.join(self.path, col), df[col].values)

        np.save(os.path.join(self.path, "index"), df.index)
        return

    def _load_as_df(self, parameters)->pd.DataFrame:
        paras = []
        for para in parameters:
            paras.append(np.load(os.path.join(self.path, f"{para}.npy"), allow_pickle=True))

        index = np.load(os.path.join(self.path, "index.npy"), allow_pickle=True)

        return pd.DataFrame(np.column_stack(paras),
                            columns=parameters,
                            index=pd.to_datetime(index))

    def fetch(
            self,
            parameters: Union[list, str] = None,
            station_id: Union[list, str] = None,
            station_names: Union[list, str] = None,
    )->pd.DataFrame:
        """
        parameters
        ----------
        parameters : str/list (default=None)
            Names of parameters to fetch. By default, ``name``, ``value``, ``val_unit``, ``location``,
             ``lat``, and ``long`` are read.
        station_id : str/list (default=None)
            name/names of station id for which the data is to be fetched.
            By default, the data for all stations is fetched. If given, then
            ``station_names`` should not be given.
        station_names : str/list (default=None)
            name/names of station id for which the data is to be fetched.
            By default, the data for all stations is fetched. If given, then ``station_id``
            should not be given.
        Returns
        -------
        pd.DataFrame

        Examples
        --------
        >>> from ai4water.datasets import Swatch
        >>> ds = Swatch()
        >>> df = ds.fetch()
        """
        def_paras = ["name", "value", "val_unit", "lat", "long"]

        if station_id is not None and station_names is not None:
            raise ValueError(f"Either station_id or station_names should be given. Not both.")

        if station_id is not None:
            loc = "loc_id"
        else:
            loc = "location"

        def_paras.append(loc)

        if parameters is None:
            parameters = def_paras

        if isinstance(parameters, str):
            parameters = [parameters]

        assert isinstance(parameters, list)

        df = self._load_as_df(parameters)

        return df

    def num_samples(
            self,
            parameter,
            station_id = None,
    )->int:
        """

        parameters
        ----------
        parameter : str
            name of the water quality parameter whose samples are to be quantified.
        station_id :
            if given, samples of parameter will be returned for only this site/sites
            otherwise for all sites
        """
        raise NotImplementedError

    @property
    def names(self)->dict:
        """tells the names of parameters in this class and their original names
        in SWatCh dataset in the form of a python dictionary
        """
        return {
            "LaboratoryName": "lab_name",
            'ActivityDepthHeightMeasure': "depth_height",
            'ActivityDepthHeightUnit': "depth_height_unit",
            "ActivityMediaName": "act_name",
            "ActivityType": "ActivityType",
            "MonitoringLocationHorizontalCoordinateReferenceSystem": "coord_system",
            'MonitoringLocationLongitude': "long",
            'MonitoringLocationLatitude': "lat",
            "CharacteristicName": "name",
            "ResultValue": "value",
            "ResultValueType": "val_type",
            "MonitoringLocationName": "location",
            "MonitoringLocationID": "loc_id",
            "MonitoringLocationType": "loc_type",
            "ResultDetectionQuantitationLimitType": "detect_limit",
            "ResultDetectionQuantitationLimitUnit": "detect_limit_type",
            "ResultDetectionQuantitationLimitMeasure": "detect_limit_measure",
            "ResultDetectionCondition": "detect_cond",
            "ResultAnalyticalMethodName": "method_name",
            "ResultAnalyticalMethodContext": "method_context",
            "ResultAnalyticalMethodID": "method_id",
            "ResultUnit": "val_unit",

        }

cats = ['ActivityDepthHeightUnit', 'ActivityMediaName', 'ActivityType', 'CharacteristicName',
        'DatasetName', 'LaboratoryName', 'MethodSpeciation', 'MonitoringLocationHorizontalCoordinateReferenceSystem',
        'MonitoringLocationType', 'ResultAnalyticalMethodContext', 'ResultDetectionCondition',
        'ResultDetectionQuantitationLimitType', 'ResultDetectionQuantitationLimitUnit',
        'ResultSampleFraction', 'ResultStatusID', 'ResultUnit', 'ResultValueType'
        ]


def int8(array:Union[np.ndarray, pd.Series])->bool:
    return array.min() > np.iinfo(np.int8).min and array.max() < np.iinfo(np.int8).max

def int16(array:Union[np.ndarray, pd.Series])->bool:
    return array.min() > np.iinfo(np.int16).min and array.max() < np.iinfo(np.int16).max

def int32(array:Union[np.ndarray, pd.Series])->bool:
    return array.min() > np.iinfo(np.int32).min and array.max() < np.iinfo(np.int32).max

def int64(array:Union[np.ndarray, pd.Series])->bool:
    return array.min() > np.iinfo(np.int64).min and array.max() < np.iinfo(np.int64).max

def float16(array:Union[np.ndarray, pd.Series])->bool:
    return array.min() > np.finfo(np.float16).min and array.max() < np.finfo(np.float16).max

def float32(array:Union[np.ndarray, pd.Series])->bool:
    return array.min() > np.finfo(np.float32).min and array.max() < np.finfo(np.float32).max


def maybe_convert_int(series:pd.Series)->pd.Series:
    if int8(series):
        return series.astype(np.int8)
    if int16(series):
        return series.astype(np.int16)
    if int32(series):
        return series.astype(np.int32)
    if int64(series):
        return series.astype(np.int64)
    return series


def maybe_convert_float(series:pd.Series)->pd.Series:

    if float16(series):
        return series.astype(np.float16)
    if float32(series):
        return series.astype(np.float32)

    return series


def memory_usage(dataframe):
    return round(dataframe.memory_usage().sum() / 1024**2, 4)


def maybe_reduce_memory(dataframe:pd.DataFrame, hints=None)->pd.DataFrame:

    init_memory = memory_usage(dataframe)

    _hints = {col:dataframe[col].dtype.name for col in dataframe.columns}
    if hints:
        _hints.update(hints)

    for col in dataframe.columns:
        col_dtype = dataframe[col].dtype.name

        if 'int' in  _hints[col]:
            dataframe[col] = maybe_convert_int(dataframe[col])
        elif 'float' in _hints[col]:
            dataframe[col] = maybe_convert_float(dataframe[col])
        elif 'int' in col_dtype:
            dataframe[col] = maybe_convert_int(dataframe[col])
        elif 'float' in col_dtype or 'float' in  _hints[col]:
            dataframe[col] = maybe_convert_float(dataframe[col])

        elif col_dtype in ['object'] and 'cat' in _hints[col]:
            dataframe[col] = dataframe[col].astype('category')

    print(f"memory reduced from {init_memory} to {memory_usage(dataframe)}")
    return dataframe
