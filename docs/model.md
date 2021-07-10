# BaseModel
::: AI4Water._main.BaseModel
    handler: python
    selection:
        members:
            - __init__
            - fit
            - evaluate
            - predict
            - interpret
            - view_model
            - eda
            - from_config
            - update_weights
    rendering:
        show_root_heading: true

# Model subclassing
::: AI4Water.main.Model
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

# Model for functional API
::: AI4Water.functional.Model
    handler: python
    selection:
        members:
            - __init__
            - add_layers
    rendering:
        show_root_heading: true