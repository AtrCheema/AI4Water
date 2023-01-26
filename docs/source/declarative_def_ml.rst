.. _dec_def_ml:

declarative model definition for classical ML models
*****************************************************

sklearn based models
========================
All the sklearn_ based `models <https://scikit-learn.org/stable/modules/classes.html>`_
are available by default and can be used by giving their names to the `model` argument in `Model` class.

.. code-block:: python

    >>> from ai4water import Model
    >>> from ai4water.datasets import busan_beach
    >>> data = busan_beach()
    >>> model = Model(model="RandomForestRegressor")
    >>> model.fit(data=data)


catboost, lgbm and xgboost based models
=========================================
If catboost_, lgbm_ and xgboost_ libraries are installed, then the models from these
libraries can also be used seamlessly. This is true for both regressors and classifiers
from these libraries.

.. code-block:: python

    >>> from ai4water import Model
    >>> from ai4water.datasets import busan_beach
    >>> data = busan_beach()
    >>> model = Model(model="CatBoostRegressor")
    >>> model.fit(data=data)

Following example shows XGBRegressor from xgboost_ library

.. code-block:: python

    >>> from ai4water import Model
    >>> from ai4water.datasets import busan_beach
    >>> data = busan_beach()
    >>> model = Model(model="XGBRegressor")
    >>> model.fit(data=data)


Following example shows LGBMRegressor from lgbm_ library

.. code-block:: python

    >>> from ai4water import Model
    >>> from ai4water.datasets import busan_beach
    >>> data = busan_beach()
    >>> model = Model(model="LGBMRegressor")
    >>> model.fit(data=data)

Specifying hyperparameters of models
=====================================
If we want to specify hyperparameters of these models, they can be
put together in a dictionary where model name is the key and the hyperparameters
are are given in a dictionary. This is true for all sklearn based models
as well as for catboost, lgbm and xgboost based models.

.. code-block:: python

    >>> from ai4water import Model
    >>> from ai4water.datasets import busan_beach
    >>> data = busan_beach()
    >>> model = Model(model={"RandomForestRegressor": {"n_estimators": 10}})
    >>> model.fit(data=data)


custom models
===============

We can also have our own model as far as it implements `.fit` , `.predict` and `.evaluate`
methods.

.. code-block:: python

    >>> class MyRF(RandomForestRegressor):
    >>>     pass


However, we must specify the `mode` either as `regression` or as `classification` when
we are using our own custom models.

.. code-block:: python

    >>> model = Model(model=MyRF, mode="regression")
    >>> model.fit(data=data)


The hyperparameters of the custom models can also be defined in a similar way as
defined for the sklearn based models.

.. code-block:: python

    >>>  model = Model(model={MyRF: {"n_estimators": 10}},
    >>>               ts_args={'lookback': 1},
    >>>               verbosity=0,
    >>>               mode="regression")
    >>>  model.fit(data=data)


We can also use the initialized model

.. code-block:: python

    >>> model = Model(model=MyRF(), mode="regression", verbosity=0)
    >>> model.fit(data=data)


.. _sklearn:
    https://scikit-learn.org/stable/modules/classes.html

.. _xgboost:
    https://xgboost.readthedocs.io/en/stable/python/index.html

.. _catboost:
    https://catboost.ai/en/docs/concepts/python-quickstart

.. _lightgbm:
    https://lightgbm.readthedocs.io/en/v3.3.2/Python-API.html#scikit-learn-api

