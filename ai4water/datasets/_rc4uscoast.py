
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
    ` Gomez_ et al., 2022 <https://doi.org/10.5194/essd-2022-341>`_.

    Examples
    --------
    >>> from ai4water.datasets import RC4USCoast
    >>> dataset = RC4USCoast()

    """
    url = {
        'RC4USCoast.zip': 'https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0260455/RC4USCoast.zip',
        'info.xlsx': 'https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0260455/supplemental/dataset_info.xlsx'
    }

    def __init__(self, path=None, *args, **kwargs):
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

        parameters
        ----------
        parameter : list, str
        stations : list, str
        as_dataframe : bool (default=False)
        st :
        en :

        Returns
        -------
        pd.DataFrame

        Examples
        --------
        >>> from ai4water.datasets import RC4USCoast
        >>> ds = RC4USCoast()
        >>> data = ds.fetch_chem(['temp', 'do'])
        >>> data
        >>> data = ds.fetch_chem(['temp', 'do'], as_dataframe=True)
        >>> data.shape  # this is a multi-indexed dataframe
        (119280, 4)
        """
        if isinstance(parameter, str):
            parameter = [parameter]

        ds = xr.load_dataset(self.chem_fname)[parameter]
        if stations == "all":
            pass
        elif not isinstance(stations, list):
            stations = [stations]
            ds = ds.sel(RC4USCoast_ID=stations)
        else:
            assert stations is None

        if as_dataframe:
            return ds.to_dataframe()
        return ds

    def fetch_q(
            self,
            stations:Union[int, List[int], str, np.ndarray] = "all",
            as_dataframe:bool=True,
            nv=0,
    ):
        """returns discharge data

        parameters
        -----------
        stations :
        as_dataframe : bool (default=True)
        nv : int (default=0)

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
        """
        q = xr.load_dataset(self.q_fname)
        if stations:
            if stations == "all":
                q = q.sel(nv=nv)
            elif not isinstance(stations, list):
                stations = [stations]
                q = q.sel(RC4USCoast_ID=stations, nv=nv)
            else:
                raise ValueError(f"invalid {stations}")

        if as_dataframe:
            return q.to_dataframe()['disc'].unstack()
        return q

