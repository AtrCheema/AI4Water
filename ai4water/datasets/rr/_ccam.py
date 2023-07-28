
from ai4water.backend import os

from .camels import Camels


class CCAM(Camels):
    url = "https://zenodo.org/record/5729444"

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
        super(CCAM, self).__init__(path=path, **kwargs)
        self.path = path
        self._download(overwrite=overwrite)

        # self.dyn_fname = os.path.join(self.path, 'ccam_dyn.nc')
        #
        # if to_netcdf:
        #     self._maybe_to_netcdf('ccam_dyn')