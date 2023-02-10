__all__ = ["GRiMeDB"]

from typing import Union, List

from ai4water.backend import pd, os

from ._datasets import Datasets
from .utils import check_st_en


class GRiMeDB(Datasets):
    """
    global river database of methan concentrations and fluxes following
    `Stanley et al., 2022 <https://doi.org/10.5194/essd-2022-346>`_
    """
    url = {
        "concentrations.csv": "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-ntl.420.1&entityid=ba3e270bcab8ace5d157c995e4b791e4",
        "fluxes.csv": "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-ntl.420.1&entityid=1a559f00566ed9f9f33ccb0daab0bef5",
        "sites.csv": "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-ntl.420.1&entityid=3faa64303d5f5bcd043bb88f6768e603",
        "sources.csv": "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-ntl.420.1&entityid=3615386d27a2d148be09e70ac22799e4"
    }

    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self.ds_dir = path
        self._download()
