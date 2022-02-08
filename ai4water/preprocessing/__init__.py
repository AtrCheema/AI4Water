
import warnings

from .dataset import DataSet, DataSetUnion, DataSetPipeline
from .transformations import Transformation, Transformations
from .imputation import Imputation
from .spatial_processing import MakeHRUs

class DataHandler(DataSet):
    def __init__(self, *args, **kwargs):
        warnings.warn("""
        DataHandler is deprecated and will be removed in future versions.
        Use `DataSet` class instead as follows
        from ai4water.preprocessing import DataSet
        """)
        super().__init__(*args, **kwargs)