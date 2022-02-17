"""SeqMetrics, the module to calculate performance related to tabular/structured and sequential data.
The values in a sequence are not necessarily related.
"""

from ._main import Metrics
from ._rgr import RegressionMetrics
from ._cls import ClassificationMetrics
from .utils import plot_metrics
