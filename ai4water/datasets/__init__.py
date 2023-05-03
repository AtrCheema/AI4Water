from typing import Union

from ai4water.backend import pd, os

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
from ._datasets import gw_punjab
from ._datasets import RRAlpineCatchments
from ._datasets import RRLuleaSweden
from ._datasets import mg_photodegradation
from ._datasets import ec_removal_biochar

from .mtropics import MtropicsLaos
from .mtropics import MtropcsThailand
from .mtropics import MtropicsVietnam
from .mtropics import ecoli_mekong_laos
from .mtropics import ecoli_houay_pano
from .mtropics import ecoli_mekong_2016
from .mtropics import ecoli_mekong

from ._quadica import Quadica
from ._grqa import GRQA
from ._swatch import Swatch
from ._rc4uscoast import RC4USCoast
from ._grimedb import GRiMeDB
from ._npctr import NPCTRCatchments
from ._hyperspectral import SoilPhosphorus

from .water_quality import DoceRiver
from .water_quality import SeluneRiver
from .water_quality import busan_beach
from .water_quality import RiverChemSiberia
from .water_quality import SyltRoads


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
