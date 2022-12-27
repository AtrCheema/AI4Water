from typing import Union

import os
import pandas as pd

from .camels import Camels
from .camels import CAMELS_AUS
from .camels import CAMELS_CL
from .camels import CAMELS_BR
from .camels import CAMELS_GB
from .camels import CAMELS_US
from .camels import LamaH
from .camels import HYSETS
from .camels import HYPE
from .camels import WaterBenchIowa

from ._datasets import Weisssee
from ._datasets import WaterChemEcuador
from ._datasets import WaterChemVictoriaLakes
from ._datasets import WeatherJena
from ._datasets import WQCantareira
from ._datasets import WQJordan
from ._datasets import FlowSamoylov
from ._datasets import FlowSedDenmark
from ._datasets import StreamTempSpain
from ._datasets import RiverTempEroo
from ._datasets import HoloceneTemp
from ._datasets import FlowTetRiver
from ._datasets import SedimentAmersee
from ._datasets import HydrocarbonsGabes
from ._datasets import HydroChemJava
from ._datasets import PrecipBerlin
from ._datasets import GeoChemMatane
from ._datasets import WQJordan2
from ._datasets import YamaguchiClimateJp
from ._datasets import FlowBenin
from ._datasets import HydrometricParana
from ._datasets import RiverTempSpain
from ._datasets import RiverIsotope
from ._datasets import EtpPcpSamoylov
from ._datasets import SWECanada
from .mtropics import MtropicsLaos
from .mtropics import MtropcsThailand
from .mtropics import MtropicsVietnam
from .mtropics import ecoli_mekong_laos
from .mtropics import ecoli_houay_pano
from .mtropics import ecoli_mekong_2016
from .mtropics import ecoli_mekong
from ._quadica import Quadica
from ._datasets import RRAlpileCatchments
from ._datasets import RRLuleaSweden
from ._datasets import mg_photodegradation
from ._grqa import GRQA
from ._swatch import Swatch
from ._datasets import gw_punjab
from ._rc4uscoast import RC4USCoast


def busan_beach(
        inputs: list = None,
        target: Union[list, str] = 'tetx_coppml'
) -> pd.DataFrame:
    """
    Loads the Antibiotic resitance genes (ARG) data from a recreational beach
    in Busan, South Korea along with environment variables.

    The data is in the form of
    mutlivariate time series and was collected over the period of 2 years during
    several precipitation events. The frequency of environmental data is 30 mins
    while that of ARG is discontinuous. The data and its pre-processing is described
    in detail in Jang_ et al., 2021

    Arguments
    ---------
        inputs :
            features to use as input. By default all environmental data
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

        target :
            feature/features to use as target/output. By default
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

    Returns
    -------
        a pandas dataframe with inputs and target and indexed
            with pandas.DateTimeIndex

    Example:
        >>> from ai4water.datasets import busan_beach
        >>> dataframe = busan_beach()

    .. _Jang:
        https://doi.org/10.1016/j.watres.2021.117001
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


def load_nasdaq(inputs: Union[str, list, None] = None, target: str = 'NDX'):
    """Loads Nasdaq100 by downloading it if it is not already downloaded."""
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


all_datasets = ['CAMELS_AUS', 'CAMELS_CL', 'CAMELS_US', 'CAMELS_GB', 'CAMELS_BR',
                'CAMELS_CL', 'LamaH', 'HYPE',
                'HYSETS']
