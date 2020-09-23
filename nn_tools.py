from models.global_variables import keras, ACTIVATION_FNS, ACTIVATION_LAYERS, LAYERS

from weakref import WeakKeyDictionary


layers = keras.layers


class AttributeNotSetYet:
    def __init__(self, func_name):
        self.data = WeakKeyDictionary()
        self.func_name = func_name

    def __get__(self, instance, owner):
        raise AttributeError("run the function {} first to get {}".format(self.func_name, self.name))

    def __set_name__(self, owner, name):
        self.name = name


class AttributeStore(object):
    """ a class which will just make sure that attributes are set at its childs class level and not here.
    It's purpose is just to avoid cluttering of __init__ method of its child classes. """
    k_model = AttributeNotSetYet("`build_nn` to build neural network")
    method = None
    ins = None
    outs = None
    en_densor_We = None
    en_LSTM_cell = None
    auto_enc_composite = None
    de_LSTM_cell = None
    de_densor_We = None
    test_indices = None
    train_indices = None
    training = False  # by default the model is not in training mode
    scalers = {}
    cnn_counter = 0
    lstm_counter = 0
    act_counter = 0
    time_dist_counter = 0
    conv2d_lstm_counter = 0
    dense_counter = 0
    run_paras = AttributeNotSetYet("You must define the `run_paras` method first")


class NN(AttributeStore):

    def __init__(self, data_config: dict,
                 nn_config: dict
                 ):
        self.data_config = data_config
        self.nn_config = nn_config

        super(NN, self).__init__()

    @property
    def lookback(self):
        return self.data_config['lookback']

    def conv_lstm_model(self):
        """
         basic structure of a ConvLSTM based encoder decoder model.
        """
        sub_seq = self.nn_config['subsequences']
        sub_seq_lens = int(self.lookback / sub_seq)

        inputs = keras.layers.Input(shape=(sub_seq, 1, sub_seq_lens, self.ins))

        predictions = self.add_layers(inputs, self.nn_config['layers'])

        return inputs, predictions

    def cnn_lstm(self):
        """ blue print for developing a CNN-LSTM model.
        """
        timesteps = self.lookback // self.nn_config['subsequences']

        inputs = layers.Input(shape=(None, timesteps, self.ins))

        predictions = self.add_layers(inputs, self.nn_config['layers'])

        return inputs, predictions

    def lstm_autoencoder(self):
        """
        basic structure of an LSTM autoencoder
        """
        inputs = layers.Input(shape=(self.lookback, self.ins))

        outputs = self.add_layers(inputs, self.nn_config['layers'])

        return inputs, outputs

    def add_layers(self, inputs, layers_config):

        layer_inputs = inputs
        layer_outputs = inputs
        td_layer = None
        for lyr, lyr_args in layers_config.items():
            lyr = get_layer_name(lyr)
            if lyr.upper() in ACTIVATION_LAYERS:
                layer_outputs = ACTIVATION_LAYERS[lyr.upper()](layer_inputs)
            elif lyr.upper() in LAYERS:
                if lyr.upper() == 'TIMEDISTRIBUTED':
                    td_layer = LAYERS[lyr.upper()]
                    continue

                lyr_args, activation = check_act_fn(lyr_args)
                if td_layer is not None:
                    layer_outputs = td_layer(LAYERS[lyr.upper()](**lyr_args))(layer_inputs)
                    td_layer = None
                else:
                    layer_outputs = LAYERS[lyr.upper()](**lyr_args)(layer_inputs)
                    if activation is not None:  # put the string back to dictionary to be saved in config file
                        lyr_args['activation'] = activation
            else:
                raise ValueError("unknown layer {}".format(lyr))

            layer_inputs = layer_outputs

        return layer_outputs


def check_act_fn(config: dict):
    """ it is possible that the config file does not have activation argument or activation is None"""
    activation = None
    if 'activation' in config:
        activation = config['activation']
    if activation is not None:
        assert isinstance(activation, str)
        config['activation'] = ACTIVATION_FNS[activation.upper()]

    return config, activation


def get_layer_name(lyr:str)->str:

    layer_name = lyr.split('_')[0]
    if layer_name.upper() not in list(LAYERS.keys()) + list(ACTIVATION_LAYERS.keys()):
        raise ValueError("unknown layer {}".format(lyr))

    return layer_name