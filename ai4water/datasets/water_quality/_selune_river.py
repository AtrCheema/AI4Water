
__all__ = ["SeluneRiver"]

from typing import Union, List, Tuple

from ai4water.backend import pd, os, np

from .._datasets import Datasets
from ..utils import check_attributes, sanity_check, check_st_en


class SeluneRiver(Datasets):
    """
    Dataset of physico-chemical variables for a full-year time series characterization
    of Hyporheic zone of Selune River, Manche, Normandie, France following
    `Moustapha Ba et al., 2023 <https://doi.org/10.1016/j.dib.2022.108837>`_ .
    """
    url = {
        "data.zip":
    "https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/SBXWUC"
    }

    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self.ds_dir = path
        self._download()