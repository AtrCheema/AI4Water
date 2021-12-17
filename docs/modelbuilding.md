# Model Building
The core of `AI4Water` is the `Model` class which builds and trains the machine learning model. 
This class interacts with pre-processing and post-processing modules. 

[Model](model.md)

The `Model` class uses a python dictionary to build layers of neural networks.

To build Tensorflow based models using python dictionary see the guide 
for [declarative Model definition using Tensorflow](build_dl_models.md)

To build pytorch based NN models using python dicitonary see the guide 
for [declarative Model definition using pytorch.](declarative_torch.md)

# [Learner](pt_learner.md)
This module is for pytorch based models
This module performs building, training and prediction of model.


**`DualAttentionModel`**
::: ai4water.tf_models.DualAttentionModel
    handler: python
    selection:
        members:
            - __init__
            - build
            - plot_act_along_inputs
    rendering:
        show_root_heading: true

**`TemporalFusionTransformer`**
::: ai4water.models.tensorflow.TemporalFusionTransformer
    handler: python
    selection:
        members:
            - __init__
            - __call__
    rendering:
        show_root_heading: true

**`NBeats`**
::: ai4water.models.tensorflow.NBeats
    handler: python
    selection:
        members:
            - __init__
            - __call__
    rendering:
        show_root_heading: true

**`HARHNModel`**
::: ai4water.pytorch_models.HARHNModel
    handler: python
    selection:
        members:
            - __init__
            - initialize_layers
    rendering:
        show_root_heading: true

**`IMVModel`**
::: ai4water.pytorch_models.IMVModel
    handler: python
    selection:
        members:
            - __init__
            - initialize_layers
            - interpret
    rendering:
        show_root_heading: true
