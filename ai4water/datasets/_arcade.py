
__all__ = ["ARCADE"]

from ai4water.datasets._datasets import Datasets


class ARCADE(Datasets):
    """
    pan-ARctic CAtchments summary DatabasE (ARCADE) of >40,000 catchments,
    with watersheds delineated at high resolution of 90 m and 103 geospatial,
    environmental, climatic, and physiographic catchment properties following
    the work of `Speetjens et al., 2023 <https://doi.org/10.5194/essd-15-541-2023>`_

    """

    url = {
        "ARCADE_v1_261_1km.zip": "https://dataverse.nl/api/access/datafile/353354",
        "ARCADE_v1_37_1km.zip": "https://dataverse.nl/api/access/datafile/353359",
        "S1_ARCADE_v1_36_1km.zip": "https://dataverse.nl/api/access/datafile/353357",
        "S1_ARCADE_v1_37_1km.zip": "https://dataverse.nl/api/access/datafile/353358",
    }
    def __init__(self,
                 path=None,
                 **kwargs):

        super().__init__(path=path, **kwargs)
        self.ds_dir = path

        self._download(tolerate_error=True)
