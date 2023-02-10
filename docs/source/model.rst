Model
******


BaseModel
=========

The core of `AI4Water` is the `Model` class which builds and trains the machine learning model.
This class interacts with pre-processing and post-processing modules.

The `Model` class uses a python dictionary to build layers of neural networks.

To build Tensorflow based models using python dictionary see the guide
for :doc:`declarative_def_tf`. To build pytorch based NN models using python dictionary see the guide
for  :doc:`declarative_def_torch` .

.. autoclass:: ai4water._main.BaseModel
    :members:
        __init__,
        training_data,
        validation_data,
        test_data,
        all_data,
        fit,
        fit_on_all_training_data,
        evaluate,
        evaluate_on_training_data,
        evaluate_on_validation_data,
        evaluate_on_test_data,
        evaluate_on_all_data,
        predict,
        predict_on_training_data,
        predict_on_validation_data,
        predict_on_test_data,
        predict_on_all_data,
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
        explain_example,
        shap_values,
        prediction_analysis,
        partial_dependence_plot,
        optimize_transformations,
        optimize_hyperparameters,
        permutation_importance,
        sensitivity_analysis,
        seed_everything


Model subclassing
======================
Model subclassing is different from functional API in the way the model (neural network)
is constructed. To understand the difference between model-subclassing API and functional
API see :ref:`sub_vs_func`

.. automodule:: ai4water.main.Model
        :members:
            __init__,
            initialize_layers,
            build_from_config,
            forward,
            fit_pytorch,

Model for functional API
============================
.. autoclass:: ai4water.functional.Model
        :members:
            __init__,
            add_layers,
            compile,
            build,


Pytorch Learner
===============

This module can be used to train models which are built outside `AI4Water`'s model class.
Thus, this module does not do any pre-processing, model building and post-processing of results.

This module is inspired from fastai's Learner_ and keras's Model_ class.

.. autoclass:: ai4water.models._torch.Learner
   :undoc-members:
   :show-inheritance:
   :members:
        __init__,
        fit,
        evaluate,
        predict,
        update_metrics,
        update_weights,
        plot_model,



.. _Learner:
    https://docs.fast.ai/learner.html#Learner

.. _Model:
    https://www.tensorflow.org/api_docs/python/tf/keras/Model