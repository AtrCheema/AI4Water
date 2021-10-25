The post-processing part consists of modules which handles the output of `Model` after
the model has been trained i.e. after `.fit` method has been called on it.

# [Model Explaination](postprocessing/explain.md)
This module applies post-hoc methods such as LIME or SHAP to interpret/explain the
behaviour of model. These are all model-agnostic methods to interpret machine learning
models. 

# [Model Interpretation](postprocessing/interpret.md)
This module is useful for models which are inherently interpretable
such as NBeats, TFT, DA-LSTM, IA-LSTM or models which contain attention mechanism.

# [Model Visualization](postprocessing/visualize.md)
This module helps in visualization of model such as 
- decision trees 
- output/activations of individual layers of neural networks
- weights and biases of layers of neural networks
- gradients of weights and biases
- gradients of outputs/activations of layers


# [Performance Metrics](postprocessing/seqmetrics.md)
This module is for calaculation of performance matrix of sequential data. 
It consists of Metrics class and some utils.