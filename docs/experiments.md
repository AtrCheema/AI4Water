**`Experiments`**
::: ai4water.experiments.Experiments
    handler: python
    rendering:
        show_root_heading: true
    selection:
        members:
            - __init__
            - fit
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
        