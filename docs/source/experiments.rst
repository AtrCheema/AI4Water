Experiments
***********

The purpose of this module is to compare more than one models. Furthermore,
this module can also optimize the hyper-parameters of these models and compare
them. The Experiments class provides the basic building block for conducting
experiments. The MLRegressionExperiments and MLClassificationExperiments compare
several classical machine learning regression and classification models respectively.
The DLRegressionExperiments class compares some common basic deep learning algorithms
for a given data.

Experiments
===========
.. autoclass:: ai4water.experiments.Experiments
   :members:
   :show-inheritance:

   .. automethod:: __init__

RegressionExperiments
=====================
.. autoclass:: ai4water.experiments.MLRegressionExperiments
   :members:
   :show-inheritance:

   .. automethod:: __init__

.. autoclass:: ai4water.experiments.DLRegressionExperiments
   :members:
   :show-inheritance:

   .. automethod:: __init__


ClassificationExperiments
=========================
.. autoclass:: ai4water.experiments.MLClassificationExperiments
   :members:
   :show-inheritance:

   .. automethod:: __init__


DLRegressionExperiments
=========================
.. autoclass:: ai4water.experiments.DLRegressionExperiments
   :members:
   :show-inheritance:

   .. automethod:: __init__,
                    input_shape,
                    model_MLP,
                    model_LSTM,
                    model_CNN,
                    model_CNNLSTM,
                    model_LSTMAutoEncoder,
                    model_TCN,
                    model_TemporalFusionTransformer,


DLClassificationExperiments
============================
.. autoclass:: ai4water.experiments.DLClassificationExperiments
   :members:
   :show-inheritance:

