quick start
***********


Build a `Model` by providing all the arguments to initiate it.

.. code-block:: python

    >>> from ai4water import Model
    >>> from ai4water.datasets import busan_beach
    >>> data = busan_beach()
    >>> model = Model(
    ...         model = {'layers': {"LSTM": 64,
    ...                             'Dense': 1}},
    ...         input_features=['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm'],   # columns in csv file to be used as input
    ...         output_features = ['tetx_coppml'],     # columns in csv file to be used as output
    ...         ts_args={'lookback': 12}  # how much historical data we want to feed to model
    >>> )


Train the model by calling the `fit()` method

.. code-block:: python

    >>> history = model.fit(data=data)


Make predictions from it

.. code-block:: python

    >>> predicted = model.predict()


The model object returned from initiating AI4Wwater's `Model` is same as that of Keras' `Model`
We can verify it by checking its type

.. code-block:: python

    >>> import tensorflow as tf
    >>> isinstance(model, tf.keras.Model)  # True



Using your own pre-processed data
=================================
You can use your own pre-processed data without using any of pre-processing tools of AI4Water. You will need to provide
input output paris to `data` argument to `fit` and/or `predict` methods.

.. code-block:: python

    >>> import numpy as np
    >>> from ai4water import Model  # import any of the above model
    ...
    >>> batch_size = 16
    >>> lookback = 15
    >>> inputs = ['dummy1', 'dummy2', 'dummy3', 'dumm4', 'dummy5']  # just dummy names for plotting and saving results.
    >>> outputs=['DummyTarget']
    ...
    >>> model = Model(
    ...             model = {'layers': {"LSTM": 64,
    ...                                 'Dense': 1}},
    ...             batch_size=batch_size,
    ...             ts_args={'lookback':lookback},
    ...             input_features=inputs,
    ...             output_features=outputs,
    ...             lr=0.001
    ...               )
    >>> x = np.random.random((batch_size*10, lookback, len(inputs)))
    >>> y = np.random.random((batch_size*10, len(outputs)))
    ...
    >>> history = model.fit(x=x,y=y)



using for `scikit-learn`/`xgboost`/`lgbm`/`catboost` based models
===================================================================
The repository can also be used for machine learning based models such as scikit-learn/xgboost based models for both
classification and regression problems by making use of `model` keyword arguments in `Model` function.
However, integration of ML based models is not complete yet.

.. code-block:: python

    >>> from ai4water import Model
    >>> from ai4water.datasets import busan_beach
    ...
    >>> data = busan_beach()  # path for data file
    ...
    >>> model = Model(
    ...         input_features=['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm'],   # columns in csv file to be used as input
    ...         output_features = ['tetx_coppml'],
    ...         val_fraction=0.0,
    ...         #  any regressor from https://scikit-learn.org/stable/modules/classes.html
    ...         model={"RandomForestRegressor": {"n_estimators":1000}},  # set any of regressor's parameters. e.g. for RandomForestRegressor above used,
    ...     # some of the paramters are https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
    ...               )
    ...
    >>> history = model.fit(data=data)
    ...
    >>> preds = model.predict()


Hyperparameter optimization
=============================
For hyperparameter optimization, replace the actual values of hyperparameters
with the space.

.. code-block:: python

    >>> from ai4water import Model
    >>> from ai4water.datasets import busan_beach
    >>> from ai4water.hyperopt import Integer, Real
    >>> data = busan_beach()
    >>> model = Model(
    ...         model = {'layers': {"LSTM": Integer(low=30, high=100,name="units"),
    ...                             'Dense': 1}},
    ...         input_features=['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm'],   # columns in csv file to be used as input
    ...         output_features = ['tetx_coppml'],     # columns in csv file to be used as output
    ...         ts_args={'lookback': Integer(low=5, high=15, name="lookback")},
    ...         lr=Real(low=0.00001, high=0.001, name="lr")
    >>> )
    >>> model.optimize_hyperparameters(data=data,
    ...                                algorithm="bayes",  # choose between 'random', 'grid' or 'atpe'
    ...                                num_iterations=30
    ...                                )


