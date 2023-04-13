
from ai4water.backend import pd

class ETBase(object):
    def __init__(self,
                 input_df: pd.DataFrame,
                 units:dict,
                 constants:dict,
                 **kwargs
                 ):
        raise NotImplementedError(f"""
        module `et` has been moved to a separate package.
        Please install it using `pip install ETUtil` or visit https://github.com/AtrCheema/ETUtil
        """)

class Abtew(ETBase): pass


class Albrecht(ETBase): pass


class BlaneyCriddle(ETBase): pass


class BrutsaertStrickler(ETBase): pass


class Camargo(ETBase): pass


class Caprio(ETBase): pass


class ChapmanAustralia(ETBase): pass


class Copais(ETBase): pass


class Dalton(ETBase): pass


class DeBruinKeijman(ETBase): pass


class DoorenbosPruitt(ETBase): pass


class GrangerGray(ETBase): pass


class Hamon(ETBase): pass


class HargreavesSamani(ETBase): pass


class Haude(ETBase): pass


class JensenHaiseBasins(ETBase): pass


class Kharrufa(ETBase): pass


class Linacre(ETBase): pass


class Makkink(ETBase): pass


class Irmak(ETBase): pass


class Mahringer(ETBase): pass


class Mather(ETBase): pass


class MattShuttleworth(ETBase): pass


class McGuinnessBordne(ETBase): pass


class Penman(ETBase): pass


class PenPan(ETBase): pass


class PenmanMonteith(ETBase): pass


class PriestleyTaylor(ETBase): pass


class Romanenko(ETBase): pass


class SzilagyiJozsa(ETBase): pass


class Thornthwait(ETBase): pass


class MortonCRAE(ETBase): pass


class Papadakis(ETBase): pass


class Ritchie(ETBase): pass


class Turc(ETBase): pass


class Valiantzas(ETBase): pass

class Oudin(ETBase): pass

class RengerWessolek(ETBase): pass

class Black(ETBase): pass

class McNaughtonBlack(ETBase): pass
