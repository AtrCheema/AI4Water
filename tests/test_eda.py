
from ai4water.eda import EDA
from ai4water.datasets import MtropicsLaos, arg_beach

laos = MtropicsLaos()

pcp = laos.fetch_pcp(
    st="20110101", en="20110104"
)


eda = EDA(data=pcp, save=True)
eda()

eda = EDA(data=arg_beach(), dpi=50, save=True)
eda()

eda = EDA(data=arg_beach(), in_cols=arg_beach().columns.to_list()[0:-1],
          dpi=50,
          save=True)
eda()
