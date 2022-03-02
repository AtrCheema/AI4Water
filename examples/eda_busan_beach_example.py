""
===================
eda_arg_beach
==================
"""

from ai4water.datasets import busan_beach
from ai4water.eda import EDA

data = busan_beach(target=['ecoli', 'sul1_coppml', 'aac_coppml', 'tetx_coppml', 'blaTEM_coppml'])
data.shape

eda = EDA(data)

eda.heatmap()