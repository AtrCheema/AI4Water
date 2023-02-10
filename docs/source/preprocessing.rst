preprocessing
*************

.. toctree::
   :maxdepth: 2

   preprocessing/dataset

   preprocessing/imputation

   preprocessing/featurization

   preprocessing/make_hrus

   preprocessing/transformation


The preprocessing sub-module contains classes which handles preparation of input data.
The fundamental class is the `DataSet` class which prepares data from a single data
source. If you hvae multiple data sources then you can either use `DataSetUnion`
or `DataSetPipeline` class.  The DataSet can take a data in a variety of commonly
found formats such as csv, xlsx and prepares the data so that it can be fed to
`Model` for training. This class works with modules in conjunction with `Imputation` class.

It should be noted that transformations applied in ai4water are part of **Model**.
This means transformations are applied, everytime a call to the model is made using
`fit`, `predict`, `evaluate`, `score` or `predict_prob` methods.

