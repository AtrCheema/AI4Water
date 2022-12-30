
__all__ = ["NPCTRCatchments"]

from typing import Union, List, Tuple

from ai4water.backend import pd, os, np

from ._datasets import Datasets
from .utils import check_attributes, sanity_check, check_st_en


class NPCTRCatchments(Datasets):
    """
    High-resolution streamflow and weather data (2013–2019) for seven small coastal
    watersheds in the northeast Pacific coastal temperate rainforest, Canada following
    `Korver et al., 2022 <https://doi.org/10.5194/essd-14-4231-2022>`_

    """
    url = {
        "2013-2019_Discharge1015_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Discharge1015_5min.csv",
        "2013-2019_Discharge626_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Discharge626_5min.csv",
        "2013-2019_Discharge693_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Discharge693_5min.csv",
        "2013-2019_Discharge703_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Discharge703_5min.csv",
        "2013-2019_Discharge708_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Discharge708_5min.csv",
        "2013-2019_Discharge819_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Discharge819_5min.csv",
        "2013-2019_Discharge844_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Discharge844_5min.csv",
        "2013-2019_Discharge_Hourly.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Discharge_Hourly.csv",
        "2013-2019_RH_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_RH_5min.csv",
        "2013-2019_RH_Hourly.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_RH_Hourly.csv",
        "2013-2019_Rad_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Rad_5min.csv",
        "2013-2019_Rad_Hourly.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Rad_Hourly.csv",
        "2013-2019_Rain_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Rain_5min.csv",
        "2013-2019_Rain_Hourly.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Rain_Hourly.csv",
        "2013-2019_SnowDepth_Hourly.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_SnowDepth_Hourly.csv",
        "2013-2019_Ta_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Ta_5min.csv",
        "2013-2019_Ta_Hourly.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Ta_Hourly.csv",
        "2013-2019_WindDir_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_WindDir_5min.csv",
        "2013-2019_WindDir_Hourly.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_WindDir_Hourly.csv",
        "2013-2019_WindSpd_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_WindSpd_5min.csv",
        "2013-2019_WindSpd_Hourly.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_WindSpd_Hourly.csv",
        "Data-Dictonary.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/Data-Dictonary.csv",
    }

    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self.ds_dir = path
        self._download()

    @property
    def stations(self)->List[str]:
        return ["626", "693", "703", "708", "819", "844", "1015"]

    def fetch_wind_speed(
            self,
            station,
            timestep,
    ):
        _verify_timestep(timestep)

        return

    @property
    def q_attributes(self):
        return ["Qrate", "Qrate_min", "Qrate_max", "Qvol", "Qvol_min", "Qvol_max",
                "Qmm", "Qmm_min", "Qmm_max"]

    def fetch_q(
            self,
            station:Union[str, List[str]],
            timestep:str,
    ):
        """

        parameters
        ----------
        station :
        timestep :

        Examples
        ---------
        >>> from ai4water.datasets import NPCTRCatchments
        >>> dataset = NPCTRCatchments()
        """
        _verify_timestep(timestep)

        stations = check_attributes(station, self.stations)

        if timestep == "Hourly":
            fname = f"2013-2019_Discharge_{timestep}.csv"
            df = pd.read_csv(os.path.join(self.ds_dir, fname))
            dfs = []
            for gname, grp in df.groupby('Watershed'):
                grp.index = pd.to_datetime(grp.pop('Datetime'))
                grp = grp.resample('H').interpolate(method='linear')
                if gname[3:] in stations:
                    dfs.append(grp)
        else:
            for station in stations:

                fname = f"2013-2019_Discharge{station}_{timestep}.csv"
                df = pd.read_csv(fname)

        return 


    def fetch_pcp(
            self,
            station,
            time_step,
    ):
        return

    def fetch_temp(
            self,
            station,
            time_step,
    ):
        return

    def fetch_rel_hum(
            self,
            station,
            time_step,
    ):
        return


def _verify_timestep(timestep):
    assert timestep in ["Hourly", "5min"], f"""
    timestep must be either Hourly or 5min but it is {timestep}
    """
    return