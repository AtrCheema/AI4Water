
__all__ = ["SoilPhosphorus"]

from typing import Union, List, Tuple

from ai4water.backend import pd, os, np

from ._datasets import Datasets
from .utils import check_attributes, sanity_check, check_st_en


class SoilPhosphorus(Datasets):
    """
    Dataset for the determination of phosphorus in soil through the analysis of hyperspectral images
    following `Rivadeneira-Bolaños et al., 2023 <https://doi.org/10.1016/j.dib.2022.108789>`_
    """
    url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/fvgswvt5ws-3.zip"

    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self.ds_dir = path
        self._download()
