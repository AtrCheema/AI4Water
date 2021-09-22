
The pre-processing contains modules which handlers preparation of input data.
The fundamental class is the `DataHandler` class.
It can take a data in a variety of commonly found formats such as csv, xlsx and 
prepares the data so that it can be fed to `Model` for training. This class
works with modules such as `Imputation`, `Transformation` etc.

[DataHandler](preprocessing/datahandler.md)

[Imputation](preprocessing/imputation.md)

[Featurization](preprocessing/featurization.md)

[HRU Discretization](preprocessing/make_hrus.md)

[Data Transformations](preprocessing/transformation.md)