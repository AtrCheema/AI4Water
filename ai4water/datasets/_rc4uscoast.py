
__all__ = ["RC4USCoast"]

from typing import Union, List

from ai4water.backend import pd, os, xr

from ._datasets import Datasets
from .utils import check_st_en

class RC4USCoast(Datasets):
    """
    River water chemistry (N, P, SIO2, DO), discharge and temperature of 140
    monitoring sites of US coasts from 1950 to 2020 following the work of
    Gomez_ et al., 2022.

    Examples
    --------
    >>> from ai4water.datasets import RC4USCoast
    >>> dataset = RC4USCoast()

    .. _Gomez https://doi.org/10.5194/essd-2022-341
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
    def parameters(self)->List[str]:
        df = xr.load_dataset(self.chem_fname)
        return list(df.data_vars.keys())

    def fetch(
            self,
            parameter,
            station: Union[List[str], str] = None,
            st: Union[int, str, pd.DatetimeIndex] = None,
            en: Union[int, str, pd.DatetimeIndex] = None,
    ):
        """

        parameters
        ----------
        parameter : list, str
        station : list, str
        st :
        en :

        Returns
        -------
        pd.DataFrame

        Examples
        --------
        """
        if isinstance(parameter, str):
            parameter = [parameter]

        df = xr.load_dataset(self.chem_fname)
        return df[parameter]

