**`DataHanlder`**
::: ai4water.preprocessing.DataSet
    handler: python
    selection:
        members:
            - __init__
            - training_data
            - validation_data
            - test_data
            - inverse_transform
            - KFold_splits
            - LeaveOneOut_splits
            - TimeSeriesSplit_splits
            - plot_KFold_splits
            - plot_LeaveOneOut_splits
            - plot_TimeSeriesSplit_splits
            - from_h5
            - to_disk
    rendering:
        show_root_heading: true

**`DataSetUnion`**
::: ai4water.preprocessing.DataSetUnion
    handler: python
    selection:
        members:
            - __init__
            - training_data
            - validation_data
            - test_data
    rendering:
        show_root_heading: true

**`DataSetPipeline`**
::: ai4water.preprocessing.DataSetPipeline
    handler: python
    selection:
        members:
            - __init__
            - training_data
            - validation_data
            - test_data
    rendering:
        show_root_heading: true