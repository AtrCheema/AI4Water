The purpose of this module is to compare more than one models. Furthermore, 
this module can also optimize the hyper-parameters of these models and compare
them. 

**`Experiments`**
::: ai4water.experiments.Experiments
    handler: python
    rendering:
        show_root_heading: true
    selection:
        members:
            - __init__
            - fit
            - fit_with_tpot
            - compare_errors
            - plot_convergence
            - taylor_plot
            - plot_losses
            - plot_improvement
            - plot_cv_scores
            - from_config

**`RegressionExperiments`**
::: ai4water.experiments.MLRegressionExperiments
    handler: python
    rendering:
        show_root_heading: true

**`ClassificationExperiments`**
::: ai4water.experiments.MLClassificationExperiments
    handler: python
    rendering:
        show_root_heading: true
        