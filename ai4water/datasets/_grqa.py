
__all__ = ["GRQA"]

from typing import Union, List, Tuple

from ai4water.backend import pd, os, np

from .datasets import Datasets
from .utils import check_attributes, sanity_check, check_st_en


class GRQA(Datasets):
    url = 'https://zenodo.org/record/7056647#.YzBzDHZByUk'

    """
    Global River Water Quality Archive following the work of Virro et al., 2021 [21]_.

    .. [21] https://essd.copernicus.org/articles/13/5483/2021/
    """

    def __init__(self, download_source=False, **kwargs):
        super().__init__(**kwargs)

        files = ['GRQA_data_v1.3.zip', 'GRQA_meta.zip']
        if download_source:
            files += ['GRQA_source_data.zip']
        self._download(include=files)

    def fetch(
            self,
            parameter: Union[List[str], str],
            location: Union[List[str], str],
            st=None,
            en=None,
    ):
        """
        parameters
        ----------
        parameter : str/list, optional
            name of parameter
        location : str/list, optional
            location for which data is to be fetched.
        st : str
            starting date
        en : str
            end date

        Returns
        -------
        pd.DataFrame
            a multiindex dataframe of shape

        Example
        --------
        >>> from ai4water.datasets import GRQA
        >>> dataset = GRQA()
        >>> df = dataset.fetch()
        """
        raise NotImplementedError
