quick start
***********


Build a `Model` by providing all the arguments to initiate it.
For building deep learning models, we can use higher level functions such as :py:class:`ai4water.models.LSTM`.


.. code-block:: python

    >>> from ai4water import Model
    >>> from ai4water.models import LSTM
    >>> from ai4water.datasets import busan_beach
    >>> data = busan_beach()
    >>> model = Model(
    ...         model = LSTM(64),
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


The model object returned from initiating AI4Water's `Model` is same as that of Keras' `Model`
We can verify it by checking its type

.. code-block:: python

    >>> import tensorflow as tf
    >>> isinstance(model, tf.keras.Model)  # True


Defining layers of neural networks
==================================
Above we had used LSTM model. Other available deep learning models are MLP (:py:class:`ai4water.models.MLP`),
CNN (:py:class:`ai4water.models.CNN`) CNNLSTM (:py:class:`ai4water.models.CNNLSTM`),
TCN (:py:class:`ai4water.models.TCN`) and TFT (:py:class:`ai4water.models.TFT`). On the other hand
if we wish to define the layers of neural networks ourselves, we can also do so using :ref:`dec_def_tf`


.. code-block:: python

    >>> from ai4water import Model
    >>> from ai4water.datasets import busan_beach
    >>> data = busan_beach()
    >>> model = Model(
    ...         model = {'layers': {"LSTM": 64,
    ...                             'Dense': 1}},
    ...         input_features=['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm'],
    ...         output_features = ['tetx_coppml'],
    ...         ts_args={'lookback': 12}
    >>> )


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
    >>> inputs = ['dummy1', 'dummy2', 'dummy3', 'dummy4', 'dummy5']  # just dummy names for plotting and saving results.
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



using `scikit-learn`/`xgboost`/`lgbm`/`catboost` based models
=================================================================
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
    ...     # some of the parameters are https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
    ...               )
    ...
    >>> history = model.fit(data=data)
    ...
    >>> preds = model.predict()


Using your own (custom) model
=============================
If you don't want to use sklearn/xgboost/catboost/lgbm's Models and you
have your own model. You can use this model seamlessly as far as this
model has .fit, .evaluate and .predict methods.

.. code-block:: python

    >>> from ai4water import Model
    >>> from ai4water.datasets import busan_beach
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> class MyRF(RandomForestRegressor):
    >>>     pass  # your own customized random forest model
    >>> data = busan_beach()
    >>> model = Model(model=MyRF, mode="regression")
    >>> model.fit(data=data)

    you can initialize your Model with arguments as well
    >>> model = Model(model={MyRF: {"n_estimators": 10}},
    >>>               mode="regression")
    >>> model.fit(data=data)


Hyperparameter optimization
===========================
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


Experiments
===========
The experiments module can be used to compare a large range of  regression
and classification algorithms. For example, to compare performance of
regression algorithms on your data

.. code-block:: python

    >>> from ai4water.datasets import busan_beach
    >>> from ai4water.experiments import MLRegressionExperiments
    # first compare the performance of all available models without optimizing their parameters
    >>> data = busan_beach()  # read data file, in this case load the default data
    >>> inputs = list(data.columns)[0:-1]  # define input and output columns in data
    >>> outputs = list(data.columns)[-1]
    >>> comparisons = MLRegressionExperiments(
    >>>       input_features=inputs, output_features=outputs)
    >>> comparisons.fit(data=data,run_type="dry_run")
    >>> comparisons.compare_errors('r2')
