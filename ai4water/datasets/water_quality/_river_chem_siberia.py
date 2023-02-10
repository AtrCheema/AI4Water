
__all__ = ["RiverChemSiberia"]

from typing import Union, List, Tuple

from ai4water.backend import pd, os, np

from .._datasets import Datasets
from ..utils import check_attributes, sanity_check, check_st_en


class RiverChemSiberia(Datasets):
    """
    A database of water chemistry in eastern Siberian rivers following
    `Liu et al., 2022 <https://doi.org/10.1038/s41597-022-01844-y>` .
    """
    url = "https://doi.org/10.6084/m9.figshare.c.5831975.v1"

    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self.ds_dir = path
        self._download()