
import unittest

import pandas as pd
from ai4water.datasets import GRQA

ds = GRQA()


class TestGRQA(unittest.TestCase):

    def test_basic(self):

        assert len(ds.parameters) == 42
        return

    def test_BOD5(self):
        df = ds.fetch_parameter("BOD5")
        assert df.shape == (219320, 31), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("BOD5", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_BOD7(self):
        df = ds.fetch_parameter("BOD7")
        assert df.shape == (5282, 29), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("BOD7", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_BOD(self):
        df = ds.fetch_parameter("BOD")
        assert df.shape == (163531, 32), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("BOD", country="Pakistan")
        assert len(df) == 1326, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_CODCr(self):
        df = ds.fetch_parameter("CODCr")
        assert df.shape == (7350, 29), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("CODCr", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_CODMn(self):
        df = ds.fetch_parameter("CODMn")
        assert df.shape == (2310, 29), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("CODMn", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_COD(self):
        df = ds.fetch_parameter("COD")
        assert df.shape == (126372, 32), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("COD", country="Pakistan")
        assert len(df) == 1317
        df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        assert df.shape == (345, 34), df.shape
        return

    def test_DC(self):
        df = ds.fetch_parameter("DC")
        assert df.shape == (9, 32), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("DC", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_DIC(self):
        df = ds.fetch_parameter("DIC")
        assert df.shape == (30633, 36), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("DIC", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_DIN(self):
        df = ds.fetch_parameter("DIN")
        assert df.shape == (7822, 31), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("DIN", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_DIP(self):
        df = ds.fetch_parameter("DIP")
        assert df.shape == (612922, 34), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("DIP", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_DKN(self):
        df = ds.fetch_parameter("DKN")
        assert df.shape == (80732, 34), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("DKN", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_DOC(self):
        df = ds.fetch_parameter("DOC")
        assert df.shape == (524542, 38), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("DOC", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df["CODMn"].shape == (1317, 34)
        return

    def test_DON(self):
        df = ds.fetch_parameter("DON")
        assert df.shape == (163630, 36), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("DON", country="Pakistan")
        assert len(df) == 123, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_DOP(self):
        df = ds.fetch_parameter("DOP")
        assert df.shape == (899, 29), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("DOP", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_DOSAT(self):
        df = ds.fetch_parameter("DOSAT")
        assert df.shape == (953274, 38), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("DOSAT", country="Pakistan")
        assert len(df) == 221, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_DO(self):
        df = ds.fetch_parameter("DO")
        assert df.shape == (1487724, 38)
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("DO", country="Pakistan")
        assert len(df) == 1327, df.shape
        df = ds.fetch_parameter("DO", site_name="Indus River - at Kotri")
        assert len(df) == 354
        return

    def test_NH4N(self):
        df = ds.fetch_parameter("NH4N")
        assert df.shape == (651850, 36), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("NH4N", country="Pakistan")
        assert len(df) == 28, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_NO2N(self):
        df = ds.fetch_parameter("NO2N")
        assert df.shape == (720944, 38)
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("NO2N", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_NO3N(self):
        df = ds.fetch_parameter("NO3N")
        assert df.shape == (1229584, 38), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("NO3N", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_PC(self):
        df = ds.fetch_parameter("PC")
        assert df.shape == (51049, 29), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("PC", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_pH(self):
        df = ds.fetch_parameter("pH")
        assert df.shape == (2723605, 38), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("pH", country="Pakistan")
        assert len(df) == 1285, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_PIC(self):
        df = ds.fetch_parameter("PIC")
        assert df.shape == (9196, 31), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("PIC", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_PN(self):
        df = ds.fetch_parameter("PN")
        assert df.shape == (56125, 36), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("PN", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_POC(self):
        df = ds.fetch_parameter("POC")
        assert df.shape == (124450, 36), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("POC", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_PON(self):
        df = ds.fetch_parameter("PON")
        assert df.shape == (1111, 36), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("PON", country="Pakistan")
        assert len(df) == 1, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_POP(self):
        df = ds.fetch_parameter("POP")
        assert df.shape == (13, 29), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("POP", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_TAN(self):
        df = ds.fetch_parameter("TAN")
        assert df.shape == (717776, 29), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("TAN", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_TC(self):
        df = ds.fetch_parameter("TC")
        assert df.shape == (12338, 36), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("TC", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_TDN(self):
        df = ds.fetch_parameter("TDN")
        assert df.shape == (216926, 36), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("TDN", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_TDP(self):
        df = ds.fetch_parameter("TDP")
        assert df.shape == (470473, 36), df.shape
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("TDP", country="Pakistan")
        assert len(df) == 92, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_TEMP(self):
        df = ds.fetch_parameter("TEMP")
        assert df.shape == (2991485, 38)
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("TEMP", country="Pakistan")
        assert len(df) == 1324, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_TIC(self):
        df = ds.fetch_parameter("TIC")
        assert df.shape == (23024, 36)
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("TIC", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_TIN(self):
        df = ds.fetch_parameter("TIN")
        assert df.shape == (12951, 33)
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("TIN", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_TIP(self):
        df = ds.fetch_parameter("TIP")
        assert df.shape == (42495, 34)
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("TIP", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_TKN(self):
        df = ds.fetch_parameter("TKN")
        assert df.shape == (425595, 36)
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("TKN", country="Pakistan")
        assert len(df) == 58, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_TN(self):
        df = ds.fetch_parameter("TN")
        assert df.shape == (987294, 38)
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("TN", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_TOC(self):
        df = ds.fetch_parameter("TOC")
        assert df.shape == (481066, 38)
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("TOC", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_TON(self):
        df = ds.fetch_parameter("TON")
        assert df.shape == (592654, 38)
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("TON", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_TOP(self):
        df = ds.fetch_parameter("TOP")
        assert df.shape == (1811, 29)
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("TOP", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_TPP(self):
        df = ds.fetch_parameter("TPP")
        assert df.shape == (12018, 36)
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("TPP", country="Pakistan")
        assert len(df) == 0, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_TP(self):
        df = ds.fetch_parameter("TP")
        assert df.shape == (1607092, 38)
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("TP", country="Pakistan")
        assert len(df) == 249, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return

    def test_TSS(self):
        df = ds.fetch_parameter("TSS")
        assert df.shape == (754336, 38)
        assert isinstance(df.index, pd.DatetimeIndex)
        df = ds.fetch_parameter("TSS", country="Pakistan")
        assert len(df) == 1293, df.shape
        # df = ds.fetch_parameter(site_name="Indus River - at Kotri")
        # assert df.shape == (1317, 34)
        return


if __name__=="__main__":
    unittest.main()
