# Model Building
The core of `AI4Water` is the `Model` class which builds and trains the machine learning model. 
This class interacts with pre-processing and post-processing modules. 

[Model](model.md)
The `Model` class uses a python dictionary to build layers of neural networks.

To build Tensorflow based models using python dictionary see the guide 
for [Declarative Model definition using Tensorflow](build_dl_models.md)

To build pytorch based NN models using python dicitonary see the guide 
for [Declarative Model definition using pytorch.](declarative_torch.md)

# [Learner](pt_learner.md)
This module is for pytorch based models
This module performs building, training and prediction of model.
