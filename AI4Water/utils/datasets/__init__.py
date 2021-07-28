from typing import Union

import os
import pandas as pd

from AI4Water.utils.datasets.camels import Camels
from AI4Water.utils.datasets.camels import CAMELS_AUS
from AI4Water.utils.datasets.camels import CAMELS_CL
from AI4Water.utils.datasets.camels import CAMELS_BR
from AI4Water.utils.datasets.camels import CAMELS_GB
from AI4Water.utils.datasets.camels import CAMELS_US
from AI4Water.utils.datasets.camels import LamaH
from AI4Water.utils.datasets.camels import HYSETS
from AI4Water.utils.datasets.camels import HYPE

from AI4Water.utils.datasets.datasets import Weisssee
from AI4Water.utils.datasets.datasets import WaterChemEcuador
from AI4Water.utils.datasets.datasets import WaterChemVictoriaLakes
from AI4Water.utils.datasets.datasets import WeatherJena
from AI4Water.utils.datasets.datasets import WQCantareira
from AI4Water.utils.datasets.datasets import WQJordan
from AI4Water.utils.datasets.datasets import FlowSamoylov
from AI4Water.utils.datasets.datasets import FlowSedDenmark
from AI4Water.utils.datasets.datasets import StreamTempSpain
from AI4Water.utils.datasets.datasets import RiverTempEroo
from AI4Water.utils.datasets.datasets import HoloceneTemp
from AI4Water.utils.datasets.datasets import FlowTetRiver
from AI4Water.utils.datasets.datasets import SedimentAmersee
from AI4Water.utils.datasets.datasets import HydrocarbonsGabes
from AI4Water.utils.datasets.datasets import HydroChemJava
from AI4Water.utils.datasets.datasets import PrecipBerlin
from AI4Water.utils.datasets.datasets import GeoChemMatane
from AI4Water.utils.datasets.datasets import WQJordan2
from AI4Water.utils.datasets.datasets import YamaguchiClimateJp
from AI4Water.utils.datasets.datasets import FlowBenin
from AI4Water.utils.datasets.datasets import HydrometricParana
from AI4Water.utils.datasets.datasets import RiverTempSpain
from AI4Water.utils.datasets.datasets import RiverIsotope
from AI4Water.utils.datasets.datasets import EtpPcpSamoylov
from AI4Water.utils.datasets.datasets import SWECanada
from AI4Water.utils.datasets.datasets import MtropicsLaos
from AI4Water.utils.datasets.datasets import MtropcsThailand
from AI4Water.utils.datasets.datasets import MtropicsVietnam


def arg_beach(inputs: list = None, target: Union[list, str] = 'tetx_coppml') -> pd.DataFrame:
    """
    Loads the Antibiotic resitance genes (ARG) data from a recreational beach
    in Korea along with environment variables. The data is in the form of
    mutlivariate time series and was collected over the period of 2 years during
    several precipitation events. The frequency of environmental data is 30 mins
    while the ARG is discontinuous. The data and its pre-processing is described
    in detail in [Jang et al., 2021](https://doi.org/10.1016/j.watres.2021.117001)
    Arguments:
        inputs list: features to use as input. By default all environmental data
            is used which consists of following parameters
                - tide_cm
                - wat_temp_c
                - sal_psu
                - air_temp_c
                - pcp_mm
                - pcp3_mm
                - pcp6_mm
                - pcp12_mm
                - wind_dir_deg
                - wind_speed_mps
                - air_p_hpa
                - mslp_hpa
                - rel_hum

        target list/str: feature/features to use as target/output. By default
            `tetx_coppml` is used as target.
            Logically one or more from following can be considered as target
                - ecoli
                - 16s
                - inti1
                - Total_args
                - tetx_coppml
                - sul1_coppml
                - blaTEM_coppml
                - aac_coppml
                - Total_otus
                - otu_5575
                - otu_273
                - otu_94
    Returns:
        a pandas dataframe with inputs and target and indexed
            with pandas.DateTimeIndex

    Examples
    --------
    ```python
    >>>from AI4Water.utils.datasets import arg_beach
    >>>df = arg_beach()
    ```
    """
    fpath = os.path.join(os.path.dirname(__file__), "arg_busan.csv")
    df = pd.read_csv(fpath, index_col="index")
    df.index = pd.to_datetime(df.index)

    default_inputs = ['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm', 'pcp6_mm',
                      'pcp12_mm', 'wind_dir_deg', 'wind_speed_mps', 'air_p_hpa', 'mslp_hpa', 'rel_hum'
                      ]
    default_targets = [col for col in df.columns if col not in default_inputs]

    if inputs is None:
        inputs = default_inputs

    if not isinstance(target, list):
        if isinstance(target, str):
            target = [target]
    elif isinstance(target, list):
        pass
    else:
        target = default_targets

    assert isinstance(target, list)

    df = df[inputs + target]

    return df


def load_u1(target: Union[str, list] = 'target') -> pd.DataFrame:
    """loads 1d data that can be used fo regression and classification"""
    fpath = os.path.join(os.path.dirname(__file__), "input_target_u1.csv")
    df = pd.read_csv(fpath, index_col='index')

    inputs = [col for col in df.columns if col not in ['target', 'target_by_group']]

    if isinstance(target, str):
        target = [target]

    df = df[inputs + target]

    return df


def load_nasdaq(inputs: Union[str, list, None] = None, target: str = 'NDX'):
    """loads Nasdaq100 by downloading it if it is not already downloaded."""
    fname = os.path.join(os.path.dirname(__file__), "nasdaq100_padding.csv")

    if not os.path.exists(fname):
        print(f"downloading file to {fname}")
        df = pd.read_csv("https://raw.githubusercontent.com/KurochkinAlexey/DA-RNN/master/nasdaq100_padding.csv")
        df.to_csv(fname)

    df = pd.read_csv(fname)
    in_cols = list(df.columns)
    in_cols.remove(target)
    if inputs is None:
        inputs = in_cols
    target = [target]

    return df[inputs + target]

all_datasets = ['CAMELS_AUS', 'CAMELS_CL', 'CAMELS_US', 'CAMELS_GB', 'CAMELS_BR', 'CAMELS_CL', 'LamaH', 'HYPE',
                'HYSETS']
