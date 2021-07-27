#Model Building
The core of `AI4Water` is the `Model` class which builds and trains the machine learning model. This class also perform the pre-processing and post-processing of data. 

[Model](model.md)


The `Model` class uses a python dictionary to build layers of neural networks.

To build Tensorflow based models using python dictionary see following guide

[Declarative Model definition using Tensorflow](build_dl_models.md)

To build pytorch based NN models using python dicitonary see following guide

[Declarative Model definition using pytorch](declarative_torch.md)

# `DataHandler`
The pre-processing and preparation of input data is handled by `DataHandler` class.
It can take a data in a variety of commonly found formats such as csv, xlsx and 
prepares the data so that it can be fed to `Model` for training. This class
works with modules such as `Imputation`, `Transformation` etc.
The detailed documentation of `DataHandler` class can be found [here](datahandler.md)

# `Learner` module for pytorch based models
For pytorch based models, the `Learner` module comes handy.
[Learner](pt_learner.md)