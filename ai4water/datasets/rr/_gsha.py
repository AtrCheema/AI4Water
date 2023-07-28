
from .camels import Camels


class GSHA(Camels):
    """
    Global streamflow characteristics, hydrometeorology and catchment
    attributes following Peirong et al., 2023 <https://doi.org/10.5194/essd-2023-256>_`


    """
    url = "https://zenodo.org/record/8090704"

    def __init__(self,
                 path=None,
                 overwrite=False,
                 to_netcdf:bool = True,
                 **kwargs):
        """
        Parameters
        ----------
        to_netcdf : bool
            whether to convert all the data into one netcdf file or not.
            This will fasten repeated calls to fetch etc but will
            require netcdf5 package as well as xarry.
        """
        super(GSHA, self).__init__(path=path, **kwargs)
        self.ds_dir = path

        files = ['Global_files.zip',
                 'GSHAreadme.docx',
                 'LAI.zip',
                 'Landcover.zip',
                 'Meteorology_PartI_arcticnet_AFD_GRDC_IWRIS_MLIT.zip',
                 'Meteorology_ PartII_ANA_BOM_CCRR_HYDAT.zip',
                 'Meteorology_PartIII_China_CHP_RID_USGS.zip',
                 'Reservoir.zip',
                 'Storage.zip',
                 'StreamflowIndices.zip',
                 'WatershedPolygons.zip',
                 'WatershedsAll.csv'
                 ]
        self._download(overwrite=overwrite, files_to_check=files)