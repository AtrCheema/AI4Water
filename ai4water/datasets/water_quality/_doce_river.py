
__all__ = ["DoceRiver"]

from typing import Union, List, Tuple

from ai4water.backend import pd, os, np

from .._datasets import Datasets
from ..utils import check_attributes, sanity_check, check_st_en


class DoceRiver(Datasets):
    """
    Chemical data of contaminants in water and sediments from the Doce River four
    years after the mining dam collapse disaster following
    `Yamamoto et al., 2022a <https://doi.org/10.1016/j.dib.2022.108715>`_ and
    `Yamamoto et al., 2022b <https://doi.org/10.1016/j.scitotenv.2022.157332>`_
    """
    url = {
        "data.zip":
            "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/sf2h83t8v3-4.zip"}

    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self.ds_dir = path
        self._download()