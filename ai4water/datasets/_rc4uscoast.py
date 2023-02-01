
__all__ = ["RC4USCoast"]

from typing import Union, List

import numpy as np

from ai4water.backend import pd, os, xr

from ._datasets import Datasets
from .utils import check_st_en

class RC4USCoast(Datasets):
    """
    Monthly river water chemistry (N, P, SIO2, DO, ... etc), discharge and temperature of 140
    monitoring sites of US coasts from 1950 to 2020 following the work of
    `Gomez et al., 2022 <https://doi.org/10.5194/essd-2022-341>`_.

    Examples
    --------
    >>> from ai4water.datasets import RC4USCoast
    >>> dataset = RC4USCoast()

    """
    url = {
        'RC4USCoast.zip':
            'https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0260455/RC4USCoast.zip',
        'info.xlsx':
'https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0260455/supplemental/dataset_info.xlsx'
    }

    def __init__(self, path=None, *args, **kwargs):
        """

        Parameters
        ----------
        path :
            path where the data is already downloaded. If None, the data will
            be downloaded into the disk.
        """
        super(RC4USCoast, self).__init__(path=path, *args, **kwargs)
        self.ds_dir = path
        self._download()

    @property
    def chem_fname(self)->str:
        return os.path.join(self.ds_dir, "RC4USCoast", "series_chem.nc")

    @property
    def q_fname(self) -> str:
        return os.path.join(self.ds_dir, "RC4USCoast", "series_disc.nc")

    @property
    def info_fname(self) -> str:
        return os.path.join(self.ds_dir, "RC4USCoast", "info.xlsx")

    @property
    def stations(self)->np.ndarray:
        """
        >>> from ai4water.datasets import RC4USCoast
        >>> ds = RC4USCoast(path=r'F:\data\RC4USCoast')
        >>> len(ds.stations)
        140
        """
        return xr.load_dataset(self.q_fname).RC4USCoast_ID.data

    @property
    def parameters(self)->List[str]:
        """
        >>> from ai4water.datasets import RC4USCoast
        >>> ds = RC4USCoast()
        >>> len(ds.parameters)
        27
        """
        df = xr.load_dataset(self.chem_fname)
        return list(df.data_vars.keys())

    @property
    def start(self)->pd.Timestamp:
        return pd.Timestamp(xr.load_dataset(self.q_fname).time.data[0])

    @property
    def end(self)->pd.Timestamp:
        return pd.Timestamp(xr.load_dataset(self.q_fname).time.data[-1])

    def fetch_chem(
            self,
            parameter,
            stations: Union[List[int], int, str] = "all",
            as_dataframe:bool = False,
            st: Union[int, str, pd.DatetimeIndex] = None,
            en: Union[int, str, pd.DatetimeIndex] = None,
    ):
        """
        Returns water chemistry parameters from one or more stations.

        parameters
        ----------
        parameter : list, str
            name/names of parameters to fetch
        stations : list, str
            name/names of stations from which the parameters are to be fetched
        as_dataframe : bool (default=False)
            whether to return data as pandas.DataFrame or xarray.Dataset
        st :
            start time of data to be fetched. The default starting
            date is 19500101
        en :
            end time of data to be fetched. The default end date is
            20201201

        Returns
        -------
        pandas DataFrame or xarray Dataset

        Examples
        --------
        >>> from ai4water.datasets import RC4USCoast
        >>> ds = RC4USCoast()
        >>> data = ds.fetch_chem(['temp', 'do'])
        >>> data
        >>> data = ds.fetch_chem(['temp', 'do'], as_dataframe=True)
        >>> data.shape  # this is a multi-indexed dataframe
        (119280, 4)
        >>> data = ds.fetch_chem(['temp', 'do'], st="19800101", en="20181230")
        """
        if isinstance(parameter, str):
            parameter = [parameter]

        ds = xr.load_dataset(self.chem_fname)[parameter]
        if stations == "all":
            pass
        elif not isinstance(stations, list):
            stations = [stations]
            ds = ds.sel(RC4USCoast_ID=stations)
        elif isinstance(stations, list):
            ds = ds.sel(RC4USCoast_ID = stations)
        else:
            assert stations is None

        ds = ds.sel(time=slice(st or self.start, en or self.end))

        if as_dataframe:
            return ds.to_dataframe()
        return ds

    def fetch_q(
            self,
            stations:Union[int, List[int], str, np.ndarray] = "all",
            as_dataframe:bool=True,
            nv=0,
            st: Union[int, str, pd.DatetimeIndex] = None,
            en: Union[int, str, pd.DatetimeIndex] = None,
    ):
        """returns discharge data

        parameters
        -----------
        stations :
            stations for which q is to be fetched
        as_dataframe : bool (default=True)
            whether to return the data as pd.DataFrame or as xarray.Dataset
        nv : int (default=0)
        st :
            start time of data to be fetched. The default starting
            date is 19500101
        en :
            end time of data to be fetched. The default end date is
            20201201

        Examples
        --------
        >>> from ai4water.datasets import RC4USCoast
        >>> ds = RC4USCoast()
        # get data of all stations as DataFrame
        >>> q = ds.fetch_q("all")
        >>> q.shape
        (852, 140)  # where 140 is the number of stations
        # get data of only two stations
        >>> q = ds.fetch_q([1,10])
        >>> q.shape
        (852, 2)
        # get data as xarray Dataset
        >>> q = ds.fetch_q("all", as_dataframe=False)
        >>> type(q)
        xarray.core.dataset.Dataset
        # getting data between specific periods
        >>> data = ds.fetch_q("all", st="20000101", en="20181230")
        """
        q = xr.load_dataset(self.q_fname)
        if stations:
            if stations == "all":
                q = q.sel(nv=nv)
            elif not isinstance(stations, list):
                stations = [stations]
                q = q.sel(RC4USCoast_ID=stations, nv=nv)
            elif isinstance(stations, list):
                q = q.sel(RC4USCoast_ID=stations, nv=nv)
            else:
                raise ValueError(f"invalid {stations}")

        q = q.sel(time=slice(st or self.start, en or self.end))

        if as_dataframe:
            return q.to_dataframe()['disc'].unstack()
        return q
