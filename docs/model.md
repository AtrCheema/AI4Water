**`BaseModel`**
::: ai4water._main.BaseModel
    handler: python
    selection:
        members:
            - __init__
            - fit
            - evaluate
            - predict
            - predict_proba
            - predict_log_proba
            - interpret
            - view
            - eda
            - score
            - from_config
            - from_config_file
            - update_weights
            - activations
            - cross_val_score
            - explain
            - optimize_transformations
            - optimize_hyperparameters
            - permutation_importance
    rendering:
        show_root_heading: true

**`Model subclassing`**
::: ai4water.main.Model
    handler: python
    selection:
        members:
            - __init__
            - initialize_layers
            - call
            - forward
            - fit_pytorch
    rendering:
        show_root_heading: true

**`Model for functional API`**
::: ai4water.functional.Model
    handler: python
    selection:
        members:
            - __init__
            - add_layers
            - from_config
    rendering:
        show_root_heading: true
