from typing import Union

import os
import pandas as pd

from AI4Water.utils.datasets.datasets import CAMELS_AUS
from AI4Water.utils.datasets.datasets import CAMELS_CL
from AI4Water.utils.datasets.datasets import CAMELS_BR
from AI4Water.utils.datasets.datasets import CAMELS_GB
from AI4Water.utils.datasets.datasets import CAMELS_US
from AI4Water.utils.datasets.datasets import LamaH
from AI4Water.utils.datasets.datasets import HYSETS
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


def load_30min(target:Union[list, str]='target5')->pd.DataFrame:
    """Loads a  multi-variate time series data which has mutliple targets with missing values."""
    fpath = os.path.join(os.path.dirname(__file__), "mts_30min.csv")
    df = pd.read_csv(fpath, index_col="Date_Time2")
    df.index=pd.to_datetime(df.index)

    default_targets = [col for col in df.columns if col.startswith('target')]

    inputs = [col for col in df.columns if col.startswith('input')]

    if not isinstance(target, list):
        if isinstance(target, str):
            target = [target]
    else:
        target = default_targets

    assert isinstance(target, list)

    df = df[inputs + target]

    return df


def load_u1(target:Union[str, list]='target')->pd.DataFrame:
    """loads 1d data that can be used fo regression and classification"""
    fpath = os.path.join(os.path.dirname(__file__), "input_target_u1.csv")
    df = pd.read_csv(fpath, index_col='index')

    inputs = [col for col in df.columns if col not in ['target', 'target_by_group']]

    if isinstance(target, str):
        target = [target]

    df = df[inputs + target]

    return df


def load_nasdaq(inputs:Union[str, list, None]=None, target:str='NDX'):
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