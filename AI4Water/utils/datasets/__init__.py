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


def load_30min():
    fpath = os.path.join(os.path.dirname(__file__), "data_30min.csv")
    df = pd.read_csv(fpath, index_col="Date_Time2")
    df.index=pd.to_datetime(df.index)
    return df

def load_u1():
    """loads 1d data that can be used fo regression and classification"""
    fpath = os.path.join(os.path.dirname(__file__), "input_target_u1.csv")
    df = pd.read_csv(fpath)
    return df