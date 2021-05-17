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


def load_30min(target:Union[None, list]=None)->pd.DataFrame:
    """Loads a  multi-variate time series data which has mutliple targets with missing values."""
    fpath = os.path.join(os.path.dirname(__file__), "mts_30min.csv")
    df = pd.read_csv(fpath, index_col="Date_Time2")
    df.index=pd.to_datetime(df.index)

    default_targets = [col for col in df.columns if col.startswith('target')]

    inputs = [col for col in df.columns if col.startswith('input')]

    if target is not None:
        if isinstance(target, str):
            target = [target]
        assert isinstance(target, list)
    else:
        target = default_targets

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