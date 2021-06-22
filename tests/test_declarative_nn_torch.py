from AI4Water import Model


from AI4Water.utils.datasets import arg_beach


df = arg_beach(inputs=['tide_cm', 'wat_temp_c', 'sal_psu', 'air_temp_c', 'pcp_mm', 'pcp3_mm',
       'pcp6_mm', 'pcp12_mm'])

def build_and_fit(nn_model, parameters):

    model = Model(
        model=nn_model,
        data=df,
        lookback=1,
        epochs=50,
        transformation='minmax',
        lr=0.0001,
        batch_size=4,
        test_fraction=0.0,
        val_fraction=0.0,
        patience=10,
        #val_data="same",
    )

    assert model.trainable_parameters() == parameters
    model.fit()
    return model

default_model = {'layers':{
    "Linear_0": {"in_features": 8, "out_features": 64},
    "ReLU_0": {},
    "Dropout_0": 0.3,
    "Linear_1": {"in_features": 64, "out_features": 32},
    "ReLU_1": {},
    "Dropout_1": 0.3,
    "Linear_2": {"in_features": 32, "out_features": 16},
    "Linear_3": {"in_features": 16, "out_features": 1},
}}

build_and_fit(default_model, 3201)

default_model = {'layers':{
    'LSTM_0': {"config": {'input_size': 8, 'hidden_size': 64, "batch_first": True},
               "outputs": ['lstm0_output', 'states_0']},
    'LSTM_1': {"config": {'input_size': 64, 'hidden_size': 32, "batch_first": True, "dropout": 0.3},
               "outputs": ["lstm1_output", 'states_1'],
               "inputs": "lstm0_output"},
    'slice': {"config": lambda x: x[:, -1, :],
              "inputs": "lstm1_output"},
    "Linear": {"in_features": 32, "out_features": 1},
}}

build_and_fit(default_model, 31521)




