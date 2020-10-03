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
    quantiles = None  # when predicted quantiles, this will not be None, and post-processing will be different
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
        self.lookback = self.data_config['lookback']

        super(NN, self).__init__()

    @property
    def lookback(self):
        return self._lookback

    @lookback.setter
    def lookback(self, x):
        self._lookback = x


    def add_layers(self, inputs, layers_config):
        """
        Every layer must contain initializing arguments as `config` dictionary and calling arguments as `inputs`
        dictionary. If `inputs` dictionay is missing for a layer, it will be supposed that either this is an Input layer
        or it uses previous outputs as inputs.
        The `config` dictionary for every layer must contain `name` key and its value must be `str` type.
        """
        lyr_cache = {}
        td_layer = None
        first_layer = True

        for lyr, lyr_args in layers_config.items():

            lyr_config, lyr_inputs = lyr_args['config'], lyr_args['inputs'] if 'inputs' in lyr_args else None

            lyr_name, lyr_config, activation = self.check_lyr_config(lyr, lyr_config)

            # may be user has defined layers without input layer, in this case add Input layer as first layer
            if first_layer:
                if lyr_name.upper() != "INPUT":
                    # for simple dense layer based models, lookback will not be used
                    def_shape = (self.ins,) if self.lookback == 1 else (self.lookback, self.ins)
                    layer_outputs = LAYERS["INPUT"](shape=def_shape)
                    # first layer is built so next iterations will not be for first layer
                    first_layer = False
                    # put the first layer in memory to be used for model compilation
                    lyr_cache["INPUT"] = layer_outputs
                    # add th layer which the user had specified as first layer

            if lyr_inputs is None:
                if lyr_name.upper() == "INPUT":
                    # it is an Input layer, hence should not be called
                    layer_outputs = LAYERS[lyr_name.upper()](**lyr_config)
                else:
                    # it is executable and uses previous outputs as inputs
                    if lyr_name.upper() in ACTIVATION_LAYERS:
                        layer_outputs = ACTIVATION_LAYERS[lyr_name.upper()](layer_outputs)
                    elif lyr_name.upper() == 'TIMEDISTRIBUTED':
                        td_layer = LAYERS[lyr_name.upper()]
                        continue
                    else:
                        if td_layer is not None:
                            layer_outputs = td_layer(LAYERS[lyr_name.upper()](**lyr_config))(layer_outputs)
                            td_layer = None
                        else:
                            layer_outputs = LAYERS[lyr_name.upper()](**lyr_config)(layer_outputs)
            else:
                # it is an executable
                if lyr_name.upper() in ACTIVATION_LAYERS:
                    layer_outputs = ACTIVATION_LAYERS[lyr_name.upper()](get_call_args(lyr_inputs, lyr_cache))
                elif lyr_name.upper() == 'TIMEDISTRIBUTED':
                    td_layer = LAYERS[lyr_name.upper()]
                    continue
                else:
                    if td_layer is not None:
                        layer_outputs = td_layer(LAYERS[lyr_name.upper()](**lyr_args))(get_call_args(lyr_inputs, lyr_cache))
                        td_layer = None
                    else:
                        layer_outputs = LAYERS[lyr_name.upper()](**lyr_config)(get_call_args(lyr_inputs, lyr_cache))

            if activation is not None:  # put the string back to dictionary to be saved in config file
                lyr_config['activation'] = activation

            lyr_cache[lyr_config['name']] = layer_outputs
            first_layer = False


        inputs = []
        for k,v in lyr_cache.items():
            if 'INPUT' in k.upper():
                inputs.append(v)

        return inputs, layer_outputs


    def check_lyr_config(self,lyr_name:str, config:dict):

        if 'name' not in config:
            config['name'] = lyr_name

        config, activation = self.check_act_fn(config)

        # get keras/tensorflow layer compatible layer name
        lyr_name = self.get_layer_name(lyr_name)

        return lyr_name, config, activation

    def check_act_fn(self, config: dict):
        """ it is possible that the config file does not have activation argument or activation is None"""
        activation = None
        if 'activation' in config:
            activation = config['activation']
        if activation is not None:
            assert isinstance(activation, str)
            config['activation'] = ACTIVATION_FNS[activation.upper()]

        return config, activation


    def get_layer_name(self, lyr:str)->str:

        layer_name = lyr.split('_')[0]
        if layer_name.upper() not in list(LAYERS.keys()) + list(ACTIVATION_LAYERS.keys()):
            raise ValueError("unknown layer {}".format(lyr))

        return layer_name

def get_call_args(lyr_inputs, lyr_cache):
    if isinstance(lyr_inputs, list):
        call_args = []
        for lyr_ins in lyr_inputs:
            call_args.append(lyr_cache[lyr_ins])
    else:
        call_args = lyr_cache[lyr_inputs]

    return call_args