HyperParameter Optimization
***************************

This module is for optimization of hyper-parameters. The `HyperOpt` class performs 
optimization by minimizing the objective which is defined by a user defined
objective function. The space of hyperparameters can be defined by 
using `Categorical`, `Integer` and `Real` classes.

For tutorial on using this class, see this `hpo_tutorial`_ notebook

Categorical
===========
.. autoclass:: ai4water.hyperopt.Categorical
   :members:

   .. automethod:: __init__

Real
====
.. autoclass:: ai4water.hyperopt.Real
   :members:

   .. automethod:: __init__

Integer
=======
.. autoclass:: ai4water.hyperopt.Integer
   :members:

   .. automethod:: __init__

HyperOpt
=========
.. autoclass:: ai4water.hyperopt.HyperOpt
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: __getattr__

.. _hpo_tutorial:
    https://github.com/AtrCheema/AI4Water/blob/master/examples/hyper_para_opt.ipynb