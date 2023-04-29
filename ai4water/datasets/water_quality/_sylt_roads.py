__all__ = ["SyltRoads"]

from typing import Union, List, Tuple

from ai4water.backend import pd, os, np

from .._datasets import Datasets
from ..utils import check_attributes, sanity_check, check_st_en


class SyltRoads(Datasets):
    """
    Dataset of physico-hydro-chemical time series data at Sylt Roads following
    `Rick et al., 2023 <https://doi.org/10.5194/essd-15-1037-2023>`_ .
    Following parameters are available
        - SST (sea surface temperature)
        - salinity
        - ammonium
        - nitrite
        - nitrate
        - srp (soluble reactive phosphorus)
        - si (reactive silicate)
        - chla (Chlorophyll-a)

    Examples
    --------
    >>> from ai4water.datasets import SyltRoads
    >>> ds = SyltRoads()

    """
    url = {
        "list_entrance_2014.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.873545?format=textfile",
        "list_reede_2014.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.873549?format=textfile",
        "list_ferry_2014.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.873547?format=textfile",
        "list_reede_2015.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918018?format=textfile",
        "list_entrance_2015.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918032?format=textfile",
        "list_ferry_2015.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918027?format=textfile",
        "list_reede_2016.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918023?format=textfile",
        "list_entrance_2016.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918033?format=textfile",
        "list_ferry_2016.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918028?format=textfile",
        "list_reede_2017.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918024?format=textfile",
        "list_entrace_2017.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918034?format=textfile",
        "list_ferry_2017.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918029?format=textfile",
        "list_reede_2018.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918025?format=textfile",
        "list_entrace_2018.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918035?format=textfile",
        "list_ferry_2018.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918030?format=textfile",
        "list_reede_2019.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918026?format=textfile",
        "list_entrace_2019.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918036?format=textfile",
        "list_ferry_2019.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918031?format=textfile",
        "1973_2013.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.150032?format=textfile"
    }

    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self.ds_dir = path
        self._download()