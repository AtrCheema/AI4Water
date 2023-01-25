.. _dec_def_ml:

declarative model definition for classical ML models
*****************************************************

sklearn based models
========================


catboost, lgbm and xgboost based models
=========================================

Specifying hyperparameters of models
=====================================


custom models
===============

custom model as initialized

.. code-block:: python

    >>> class MyRF(RandomForestRegressor):
    >>>     pass


.. code-block:: python

    >>> model = Model(model=MyRF, mode="regression")
    >>> model.fit(data=data)


custom model with hyperparameters

.. code-block:: python

    >>>  model = Model(model={MyRF: {"n_estimators": 10}},
    >>>               ts_args={'lookback': 1},
    >>>               verbosity=0,
    >>>               mode="regression")
    >>>  model.fit(data=data)


initialized model

.. code-block:: python

    >>> model = Model(model=MyRF(), mode="regression", verbosity=0)
    >>> model.fit(data=data)