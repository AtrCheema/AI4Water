
# Pytorch Learner

This module can be used to train models which are built outside `AI4Water`'s model class.
Thus, this module does not do any pre-processing, model building and post-processing of results.

This module is inspired from [fastai's Learner](https://docs.fast.ai/learner.html#Learner) and 
[keras's Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) class.

::: ai4water.models.torch.Learner
    handler: python
    selection:
        members:
            - __init__
            - fit
            - evaluate
            - predict
            - update_weights
            - plot_model
            - plot_model_using_tensorboard
    rendering:
        show_root_heading: true