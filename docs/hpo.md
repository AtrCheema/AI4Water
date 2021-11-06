# HyperParameter Optimization 
This module is for optimization of hyper-parameters. The `HyperOpt` class performs 
optimization by minimizing the objective which is defined by a user defined
objective function. The space of hyperparameters can be defined by 
using `Categorical`, `Integer` and `Real` classes.

For tutorial on using this class, see 
[this](https://github.com/AtrCheema/AI4Water/blob/master/examples/hyper_para_opt.ipynb) notebook

**`Categorical`**
::: ai4water.hyperopt.utils.Categorical 
    rendering:
        show_root_heading: true


**`Real`**
::: ai4water.hyperopt.utils.Real 
    rendering:
        show_root_heading: true


**`Integer`**
::: ai4water.hyperopt.utils.Integer 
    rendering:
        show_root_heading: true


**`HyperOpt`**
::: ai4water.hyperopt.hyper_opt.HyperOpt 
    handler: python
    rendering:
        show_root_heading: true
    selection:
        members:
            - __init__
            - fit
            - eval_with_best
            - space
            - skopt_sapce
            - hp_space
            - optuna_study
