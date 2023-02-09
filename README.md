# AI4Water


[![Build Status](https://github.com/AtrCheema/AI4Water/workflows/tf/badge.svg)](https://github.com/AtrCheema/AI4Water/actions)
[![Documentation Status](https://readthedocs.org/projects/ai4water/badge/?version=latest)](https://ai4water.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5194/gmd-2021-139.svg)](https://doi.org/10.5194/gmd-15-3021-2022)
[![Downloads](https://pepy.tech/badge/ai4water)](https://pepy.tech/project/ai4water)
[![PyPI version](https://badge.fury.io/py/AI4Water.svg)](https://badge.fury.io/py/AI4Water)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/AtrCheema/AI4Water)
![GitHub last commit (branch)](https://img.shields.io/github/last-commit/AtrCheema/AI4Water/master)


A uniform and simplified framework for rapid experimentation with deep leaning and machine learning based models
for time series and tabular data. To put into Andrej Karapathy's [words](https://twitter.com/karpathy/status/1350503355299205120)

`Because deep learning is so empirical, success in it is to a large extent proportional to raw experimental throughput,
 the ability to babysit a large number of experiments at once, staring at plots and tweaking/re-launching what works. 
 This is necessary, but not sufficient.` 

The specific purposes of the repository are

-    compliment the functionality of `keras`/`pytorch`/`sklearn` by making pre and 
 post-processing easier for time-series prediction/classification problems (also holds
 true for any tabular data).
 
-    save, load/reload or build models from readable json file. This repository 
 provides a framework to build layered models using python dictionary and with 
 several helper tools which fasten the process of  modeling time-series forecasting.

-    provide a uniform interface for optimizing hyper-parameters for 
 [skopt](https://scikit-optimize.github.io/stable/index.html);
 [sklearn](https://scikit-learn.org/stable/modules/classes.html#hyper-parameter-optimizers) 
 based [grid](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 
 and [random](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html);
  [hyperopt](http://hyperopt.github.io/hyperopt/) based 
  [tpe](https://papers.nips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf), 
  [atpe](https://www.electricbrain.io/blog/learning-to-optimize) or 
  [optuna](https://optuna.readthedocs.io/en/stable/) based 
  [tpe](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html), 
  [cmaes](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.CmaEsSampler.html) etc. 
  See [example](https://github.com/AtrCheema/AI4Water/blob/master/examples/hyper_para_opt.ipynb)  
  using its application.
 
-    cut short the time to write boilerplate code in developing machine learning 
 based models.

-    It should be possible to overwrite/customize any of the functionality of the AI4Water's `Model` 
 by subclassing the
 `Model`. So at the highest level you just need to initiate the `Model`, and then need `fit`, `predict` and 
 `view_model` methods of `Model` class, but you can go as low as you could go with tensorflow/keras. 

-    All the above functionalities should be available without complicating keras 
 implementation.


## Installation

An easy way to install ai4water is using pip

    pip install ai4water

You can also use GitHub link

	python -m pip install git+https://github.com/AtrCheema/AI4Water.git

or using setup file, go to folder where repo is downloaded

    python setup.py install

The latest code however (possibly with fewer bugs and more features) can be installed from `dev` branch instead

    python -m pip install git+https://github.com/AtrCheema/AI4Water.git@dev

To install the latest branch (`dev`) with all requirements use the following command

    python -m pip install "AI4Water[all] @ git+https://github.com/AtrCheema/AI4Water.git@dev"

### installation options
`all` keyword will install all the dependencies. You can choose the dependencies of particular sub-module
by using the specific keyword. Following keywords are available

 - `hpo` if you want hyperparameter optimization
 - `post_process` if you want postprocessing
 - `exp` for experiments sub-module


## Sub-modules
AI4Water consists of several submodules, each of wich responsible for a specific tasks.
The modules are also liked with each other. For understanding sub-module structure of
ai4water, [see this article](https://ai4water.readthedocs.io/en/dev/understanding.html)
<p float="left">
  <img src="/docs/source/imgs/architecture.png" width="500" />
</p>

## How to use

Build a `Model` by providing all the arguments to initiate it.

```python
from ai4water import Model
from ai4water.models import MLP
from ai4water.datasets import mg_photodegradation
data, *_ = mg_photodegradation(encoding="le")

model = Model(
    # define the model/algorithm
    model=MLP(units=24, activation="relu", dropout=0.2),
    # columns in data file to be used as input
    input_features=data.columns.tolist()[0:-1],
    # columns in csv file to be used as output
    output_features=data.columns.tolist()[-1:],
    lr=0.001,  # learning rate
    batch_size=8,  # batch size
    epochs=500,  # number of epochs to train the neural network
    patience=50,  # used for early stopping
)
```

Train the model by calling the `fit()` method
```python
history = model.fit(data=data)
```

<p float="left">
  <img src="/docs/source/imgs/mlp_loss.png" width="500" />
</p>

After training, we can make predictions from it on test/training data
```python
prediction = model.predict_on_test_data(data=data)
```

<p float="left">
  <img src="/docs/source/imgs/mlp_reg.png" width="500" />
  <img src="/docs/source/imgs/mlp_residue.png" width="500" />
</p>

<p float="left">
  <img src="/docs/source/imgs/mlp_line.png" width="500" />
  <img src="/docs/source/imgs/mlp_edf.png" width="500" />
</p>

The model object returned from initiating AI4Water's `Model` is same as that of Keras' `Model`
We can verify it by checking its type
```python
import tensorflow as tf
isinstance(model, tf.keras.Model)  # True
``` 


## Using your own pre-processed data
You can use your own pre-processed data without using any of pre-processing tools of AI4Water. You will need to provide
input output paris to `data` argument to `fit` and/or `predict` methods.
```python
import numpy as np
from ai4water import Model  # import any of the above model
from ai4water.models import LSTM

batch_size = 16
lookback = 15
inputs = ['dummy1', 'dummy2', 'dummy3', 'dummy4', 'dummy5']  # just dummy names for plotting and saving results.
outputs=['DummyTarget']

model = Model(
            model = LSTM(units=64),
            batch_size=batch_size,
            ts_args={'lookback':lookback},
            input_features=inputs,
            output_features=outputs,
            lr=0.001
              )
x = np.random.random((batch_size*10, lookback, len(inputs)))
y = np.random.random((batch_size*10, len(outputs)))

model.fit(x=x,y=y)

```

## using for `scikit-learn`/`xgboost`/`lgbm`/`catboost` based models
The repository can also be used for machine learning based models such as scikit-learn/xgboost based models for both
classification and regression problems by making use of `model` keyword arguments in `Model` function.
However, integration of ML based models is not complete yet.
```python
from ai4water import Model
from ai4water.datasets import busan_beach

data = busan_beach()  # path for data file

model = Model(
    # columns in data to be used as input
    input_features=['tide_cm', 'wat_temp_c', 'sal_psu', 'rel_hum', 'pcp_mm'],
    output_features = ['tetx_coppml'], # columns in data file to be used as input
    seed=1872,
    val_fraction=0.0,
    split_random=True,
        #  any regressor from https://scikit-learn.org/stable/modules/classes.html
        model={"RandomForestRegressor": {}},  # set any of regressor's parameters. e.g. for RandomForestRegressor above used,
    # some of the paramters are https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
              )

history = model.fit(data=data)

model.predict_on_test_data(data=data)
```

# Hyperparameter optimization
For hyperparameter optimization, replace the actual values of hyperparameters
with the space.
```python

from ai4water.functional import Model
from ai4water.datasets import MtropicsLaos
from ai4water.hyperopt import Real, Integer

data = MtropicsLaos().make_regression(lookback_steps=1)

model = Model(
    model = {"RandomForestRegressor": {
        "n_estimators": Integer(low=5, high=30, name='n_estimators', num_samples=10),
       "max_leaf_nodes": Integer(low=2, high=30, prior='log', name='max_leaf_nodes', num_samples=10),
        "min_weight_fraction_leaf": Real(low=0.0, high=0.5, name='min_weight_fraction_leaf', num_samples=10),
        "max_depth": Integer(low=2, high=10, name='max_depth', num_samples=10),
        "min_samples_split": Integer(low=2, high=10, name='min_samples_split', num_samples=10),
        "min_samples_leaf": Integer(low=1, high=5, name='min_samples_leaf', num_samples=10),
    }},
    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1:],
    cross_validator = {"KFold": {"n_splits": 5}},
    x_transformation="zscore",
    y_transformation="log",
)

# First check the performance on test data with default parameters
model.fit_on_all_training_data(data=data)
print(model.evaluate_on_test_data(data=data, metrics=["r2_score", "r2"]))

# optimize the hyperparameters
optimizer = model.optimize_hyperparameters(
   algorithm = "bayes",  # you can choose between `random`, `grid` or `tpe`
    data=data,
    num_iterations=60,
)

# Now check the performance on test data with default parameters
print(model.evaluate_on_test_data(data=data, metrics=["r2_score", "r2"]))
```

Running the above code will optimize the hyperparameters and generate
following figures

<p float="left">
  <img src="/docs/source/imgs/hpo_ml_convergence.png" width="500" />
  <img src="/docs/source/imgs/hpo_fanova_importance_hist.png" width="500" />
</p>

<p float="left">
  <img src="/docs/source/imgs/hpo_objective.png" width="500" />
  <img src="/docs/source/imgs/hpo_evaluations.png" width="500" />
</p>

<p float="left"> 
  <img src="/docs/source/imgs/hpo_parallel_coordinates.png" width="500" />
</p>


# Experiments
The experiments module is for comparison of multiple models on a single data
or for comparison of one model under different conditions.

```python
from ai4water.datasets import busan_beach
from ai4water.experiments import MLRegressionExperiments

data = busan_beach()

comparisons = MLRegressionExperiments(
    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1:],
    split_random=True
)
# train all the available machine learning models
comparisons.fit(data=data)
# Compare R2 of models 
best_models = comparisons.compare_errors(
    'r2',
    data=data,
    cutoff_type='greater',
    cutoff_val=0.1,
    figsize=(8, 9),
    colors=['salmon', 'cadetblue']
)
# Compare model performance using Taylor diagram
_ = comparisons.taylor_plot(
    data=data,
    figsize=(5, 9),
    exclude=["DummyRegressor", "XGBRFRegressor",
             "SGDRegressor", "KernelRidge", "PoissonRegressor"],
    leg_kws={'facecolor': 'white',
             'edgecolor': 'black','bbox_to_anchor':(2.0, 0.9),
             'fontsize': 10, 'labelspacing': 1.0, 'ncol': 2
            },
)
```

<p float="left">
  <img src="/docs/source/imgs/exp_r2.png" width="500" />
  <img src="/docs/source/imgs/exp_taylor.png" width="500" />
</p>

For more comprehensive and detailed examples see [![Documentation Status](https://readthedocs.org/projects/ai4water-examples/badge/?version=latest)](https://ai4water.readthedocs.io/projects/Examples/en/latest/?badge=latest)

## Disclaimer
The library is still under development. Fundamental changes are expected without prior notice or
without regard of backward compatability.

#### Related

[sktime: A Unified Interface for Machine Learning with Time Series](https://github.com/alan-turing-institute/sktime)

[Seglearn: A Python Package for Learning Sequences and Time Series](https://github.com/dmbee/seglearn)

[Pastas: Open Source Software for the Analysis of Groundwater Time Series](https://github.com/pastas/pastas)

[Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh -- A Python package)](https://github.com/blue-yonder/tsfresh)

[MLAir](https://gmd.copernicus.org/preprints/gmd-2020-332/)

[pyts: A Python Package for Time Series Classification](https://github.com/johannfaouzi/pyts)

[Tslearn, A Machine Learning Toolkit for Time Series Data](https://github.com/tslearn-team/tslearn)

[TSFEL: Time Series Feature Extraction Library](https://doi.org/10.1016/j.softx.2020.100456)

[catch22](https://github.com/chlubba/catch22)

[vest](https://github.com/vcerqueira/vest-python)

[pyunicorn (Unified Complex Network and RecurreNce analysis toolbox](https://github.com/pik-copan/pyunicorn)

[TSFuse Python package for automatically constructing features from multi-view time series data](https://github.com/arnedb/tsfuse)

[Catalyst](https://github.com/catalyst-team/catalyst)

[tsai - A state-of-the-art deep learning library for time series and sequential data](https://github.com/timeseriesAI/tsai)
