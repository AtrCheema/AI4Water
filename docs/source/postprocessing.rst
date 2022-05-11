postprocessing
**************

This consists of modules which handles the output of `Model` after
the model has been trained i.e. after `.fit` method has been called on it.

Please note that the `SeqMetrics` sub-module has been deprecated. 
Please use `SeqMetrics <https://seqmetrics.readthedocs.io/en/latest/>`_ library instead.


.. toctree::
   :maxdepth: 2

   postprocessing/explain

   postprocessing/interpret

   postprocessing/seqmetrics

   postprocessing/visualize


ProcessPredictions
==================
.. autoclass:: ai4water.postprocessing.ProcessPredictions
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__, __call__