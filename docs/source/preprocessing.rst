preprocessing
*************

.. toctree::
   :maxdepth: 2

   preprocessing/dataset

   preprocessing/imputation

   preprocessing/featurization

   preprocessing/make_hrus

   preprocessing/transformation


The preprocessing sub-module contains classes which handlers preparation of input data.
The fundamental class is the `DataSet` class which prepares data from a single data
source. If you hvae multiple data sources then you can either use `DataSetUnion`
or `DataSetPipeline` class.
It can take a data in a variety of commonly found formats such as csv, xlsx and 
prepares the data so that it can be fed to `Model` for training. This class
works with modules such as `Imputation`, `Transformation` etc.

preprocessing/dataset

preprocessing/imputation
This sub-module is for handling/imputing missing data.

preprocessing/featurization

preprocessing/make_hrus
Create Hydrologic Response Unit using various HRU definitions.

preprocessing/transformation
It contains various preprocessing, feature-engineering methods.

It should be noted that transformations applied in ai4water are part of **Model**.
This means trnasformations are applied, everytime a call to the model is made using
`fit`, `predict`, `evaluate`, `score` or `predict_prob` methods.

