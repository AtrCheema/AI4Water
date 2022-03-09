models
******


BaseModel
=========

The core of `AI4Water` is the `Model` class which builds and trains the machine learning model.
This class interacts with pre-processing and post-processing modules.

The `Model` class uses a python dictionary to build layers of neural networks.

To build Tensorflow based models using python dictionary see the guide
for :doc:`declarative_def_tf`. To build pytorch based NN models using python dicitonary see the guide
for  :doc:`declarative_def_torch` .

.. autoclass:: ai4water._main.BaseModel
    :members:
        __init__,
        fit,
        evaluate,
        predict,
        predict_proba,
        predict_log_proba,
        interpret,
        view,
        eda,
        score,
        from_config,
        from_config_file,
        update_weights,
        activations,
        cross_val_score,
        explain,
        optimize_transformations,
        optimize_hyperparameters,
        permutation_importance,
        sensitivity_analysis,


Model subclassing
======================

.. automodule:: ai4water.main.Model
        :members:
            __init__,
            initialize_layers,
            call,
            forward,
            fit_pytorch,

Model for functional API
============================
.. autoclass:: ai4water.functional.Model
        :members:
            __init__,
            add_layers,
            from_config


Pytorch Learner
===============

This module can be used to train models which are built outside `AI4Water`'s model class.
Thus, this module does not do any pre-processing, model building and post-processing of results.

This module is inspired from fastai's Learner_ and keras's Model_ class.

.. autoclass:: ai4water.models.torch.Learner
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__



DualAttentionModel
==================
.. autoclass:: ai4water.tf_models.DualAttentionModel
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

TemporalFusionTransformer
=========================
.. autoclass:: ai4water.models.tensorflow.TemporalFusionTransformer
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

NBeats
======
.. autoclass:: ai4water.models.tensorflow.NBeats
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

HARHNModel
==========
.. autoclass:: ai4water.pytorch_models.HARHNModel
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

IMVModel
========
.. autoclass:: ai4water.pytorch_models.IMVModel
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

.. _Learner:
    https://docs.fast.ai/learner.html#Learner

.. _Model:
    https://www.tensorflow.org/api_docs/python/tf/keras/Model