
import os

import unittest
import site
site.addsitedir(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ai4water import Model
from ai4water.datasets import busan_beach

data = busan_beach()
input_features=data.columns.tolist()[0:3]
output_features = data.columns.tolist()[-1:]

model=Model(model="RandomForestRegressor",
            verbosity=0,
            input_features=input_features,
            output_features=output_features)

model.fit(data=data)


def test_basic(**kwargs):
    si = model.sensitivity_analysis(
        data=data[input_features].values,
        **kwargs
    )
    return si


class TestMLRegression(unittest.TestCase):

    def test_morris_hdmr(self):
        test_basic(
            sampler="morris",
            analyzer="hdmr",
            sampler_kwds={'N': 100}
        )
        return

    def test_morris_sobol(self):
        test_basic(
            sampler="morris",
            analyzer="sobol",
            sampler_kwds={'N': 100}
        )
        return

    def test_morris_morris(self):
        test_basic(
            sampler="morris",
            analyzer="morris",
            sampler_kwds={'N': 100}
        )
        return

    def test_morris_pawn(self):
        test_basic(
            sampler="morris",
            analyzer="pawn",
            sampler_kwds={'N': 100}
        )
        return

    def test_morris_ff(self):
        test_basic(
            sampler="morris",
            analyzer="morris",
            sampler_kwds={'N': 100}
        )
        return

    def test_morris_dgsm(self):
        test_basic(
            sampler="morris",
            analyzer="dgsm",
            sampler_kwds={'N': 100}
        )
        return

    def test_saltelli_hdmr(self):
        test_basic(
            sampler="saltelli",
            analyzer="hdmr",
            sampler_kwds={'N': 100}
        )
        return

    def test_saltelli_sobol(self):
        test_basic(
            sampler="saltelli",
            analyzer="sobol",
            sampler_kwds={'N': 100}
        )
        return

    def test_saltelli_morris(self):
        test_basic(
            sampler="saltelli",
            analyzer="morris",
            sampler_kwds={'N': 100}
        )
        return

    def test_saltelli_pawn(self):
        test_basic(
            sampler="saltelli",
            analyzer="pawn",
            sampler_kwds={'N': 100}
        )
        return

    # def test_saltelli_ff(self):
    #     test_basic(
    #         sampler="saltelli",
    #         analyzer="ff",
    #         sampler_kwds={'N': 100}
    #     )
    #     return

    def test_saltelli_dgsm(self):
        test_basic(
            sampler="saltelli",
            analyzer="dgsm",
            sampler_kwds={'N': 100}
        )
        return

    def test_multiple_analyzers(self):
        test_basic(
            sampler="morris",
            analyzer=["dgsm", 'morris'],
            sampler_kwds={'N': 100}
        )
        return

    # def test_latin_hdmr(self):
    #     test_basic(
    #         sampler="latin",
    #         analyzer="hdmr",
    #         sampler_kwds={'N': 100}
    #     )
    #     return
    #
    # def test_latin_sobol(self):
    #     test_basic(
    #         sampler="latin",
    #         analyzer="sobol",
    #         sampler_kwds={'N': 100}
    #     )
    #     return
    #
    # def test_latin_morris(self):
    #     test_basic(
    #         sampler="latin",
    #         analyzer="morris",
    #         sampler_kwds={'N': 100}
    #     )
    #     return


if __name__ == "__main__":

    unittest.main()