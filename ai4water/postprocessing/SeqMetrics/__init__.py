"""
SeqMetrics has been moved to its own dedicated repository. This will save us
many sleepless nights later.
"""

from ._main import Metrics
from ._rgr import RegressionMetrics
from ._cls import ClassificationMetrics
from .utils import plot_metrics
