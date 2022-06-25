.. _sub_vs_func:

Model subclassing vs functional API
***********************************

In ai4water there are two implementations of Model class. One implementation is
called functional API and the other is called model-subclassing API. As far as
you are using machine learning algorithms from sklearn, xgboost, catboost or lightgbm libraries,
there is literally no difference which API you are using. However, when you are
building model using neural networks with Tensorflow as backend, it is better
to know the difference, especially if you want to extract more from the ``Model``
class.

The choice of the API decides, how a model/neural network is constructed. When you
import the model class using ``from ai4water import Model``, then you are using
model-subclassing API. This means the default behavior in ai4water is to use
model-subclassing. In model-subclassing API, the Model class is inherited from
tensorflow.keras Model class. This means, all the functionalities i.e. attributes
and methods of tensorflow Model are directly available from ai4water's Model class.
Therefore, the user does not lose any advantage that he/she may have when using tensorflow's
Model class directly. We will illustrate in following example

.. code-block:: python

    >>> from ai4water import Model
    >>> import tensorflow as tf
    >>> from ai4water.models import MLP

    >>> model = Model(model=MLP())
    >>> isinstance(model, tf.keras.Model)  # -> True

So in above example, we see that the model that we built using ai4water's Model class
was (also) an instance of tensorflow's Model. This is due to inheritance mechanism of
object oriented programming.

When you are building neural networks using pytorch, then the same model is an instance of
pytorch's nn.Module. This is because, in those cases, the Model class inherits from nn.Module
of pytorch.

In functional API, the model class does not inherit from tf.keras.Model or nn.Module. In
such a case, the model class has an attribute ``_model``. This ``_model`` is the object
from underlying library, whatever that is. This means when you are using tensorflow based
neural networks, then ``model._model`` will be ``tf.keras.Model``. In order to use functional
API in ai4water, you can do as following

.. code-block:: python

    >>> from ai4water.functional import Model
    >>> import tensorflow as tf
    >>> from ai4water.models import MLP
    ... # build a neural network based model
    >>> model = Model(model=MLP())
    >>> isinstance(model, tf.keras.Model)  # -> False
    .. # However, model._mdoel is tensorflow Model
    >>> isinstance(model._model, tf.keras.Model)  # -> True

It should be noted that the front-end/user-end is almost exactly same whether you are using functional API
or model-subclassing.

As has been mentioned earlier that for sklearn/xgboost/catboost/lightgbm based models, whether you
use functional API or model-sublcassing API, there is no difference. In these cases, ``model._model``
object will point out the object from underlying library. The following example will make it clearn

.. code-block:: python

    >>> from ai4water import Model
    ... # buil XGBRegressor model which will use xgboost library under the hood
    >>> model = Model(model="XGBRegressor")
    >>> model._model  # -> is an instance of xgboost.XGBRegressor
    ... # buil RandomForestRegressor model which will use sklearn library under the hood
    >>> model = Model(model="RandomForestRegressor")
    >>> model._model  # -> is an instance of sklearn.RandomForestRegressor