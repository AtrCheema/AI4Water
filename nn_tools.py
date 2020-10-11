from models.global_variables import ACTIVATION_FNS, ACTIVATION_LAYERS, LAYERS, tf

from weakref import WeakKeyDictionary


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
    layers = None


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

    def add_layers(self, layers_config:dict, inputs=None):
        """
        @param layers_config: `dict`
        Every layer must contain initializing arguments as `config` dictionary and calling arguments as `inputs`. If
        `inputs` key is missing for a layer, it will be supposed that either this is an Input layer or it uses previous
        outputs as inputs. The `config` dictionary for every layer can contain `name` key and its value must be `str`
        type. If `name` key  is not provided in the config, the provided layer name will be used as its name e.g in following case
        layers = {'LSTM': {'config': {'units': 16}}}
        the name of `LSTM` layer will be `LSTM` while in follwoing case
        layers = {'LSTM': {'config': {'units': 16, 'name': 'MyLSTM'}}}
        the name of the lstm will be `MyLSTM`.
        We can specifity the outputs from a layer by using the `outputs` key. The value to `outputs` must be a string or
        list of strings specifying the name of outputs from current layer which can be used later in the mdoel.
        We can also specify additional call arguments by `call_args` key. The value to `call_args` must be a string or
        a list of strings.

        @param inputs: if None, it will be supposed the the `Input` layer either exists in `layers_config` or an Input
        layer will be created withing this method before adding any other layer. If not None, then it must be in `Input`
        layer and the remaining NN architecture will be built as defined in `layers_config`. This can be handy when we
        want to use this method several times to build a complex or parallel NN structure.
        """
        lyr_cache = {}
        td_layer = None
        first_layer = True

        for lyr, lyr_args in layers_config.items():

            lyr_config, lyr_inputs, named_outs, call_args = self.deconstruct_lyr_args(lyr_args)

            lyr_name, lyr_config, activation = self.check_lyr_config(lyr, lyr_config)

            # may be user has defined layers without input layer, in this case add Input layer as first layer
            if first_layer:
                if inputs is not None: # This method was called by providing it inputs.
                    assert isinstance(inputs, tf.Tensor)
                    lyr_cache["INPUT"] = inputs
                    first_layer = False # since inputs have been defined, all the layers that will be added will be next to first layer
                    layer_outputs = inputs

                elif lyr_name.upper() != "INPUT":
                    # for simple dense layer based models, lookback will not be used
                    def_shape = (self.ins,) if self.lookback == 1 else (self.lookback, self.ins)
                    layer_outputs = get_layer(lyr_name)(shape=def_shape)
                    # first layer is built so next iterations will not be for first layer
                    first_layer = False
                    # put the first layer in memory to be used for model compilation
                    lyr_cache["INPUT"] = layer_outputs
                    # add th layer which the user had specified as first layer

            if lyr_inputs is None:  # The inputs to the layer have not been specified, so either it is an Input layer
                # or it uses the previous outputs as inputs
                if lyr_name.upper() == "INPUT":
                    # it is an Input layer, hence should not be called
                    layer_outputs = get_layer(lyr_name)(**lyr_config)
                else:
                    # it is executable and uses previous outputs as inputs
                    if lyr_name.upper() in ACTIVATION_LAYERS:
                        layer_outputs = ACTIVATION_LAYERS[lyr_name.upper()](layer_outputs)
                    elif lyr_name.upper() == 'TIMEDISTRIBUTED':
                        td_layer = get_layer(lyr_name)
                        lyr_cache[lyr_name] = td_layer
                        continue
                    else:
                        if td_layer is not None:
                            layer_outputs = td_layer(get_layer(lyr_name)(**lyr_config))(layer_outputs)
                            td_layer = None
                        else:
                            layer_outputs = get_layer(lyr_name)(**lyr_config)(layer_outputs)

            else:  # The inputs to this layer have been specified so they must exist in lyr_cache.
                # it is an executable
                if lyr_name.upper() in ACTIVATION_LAYERS:
                    call_args, add_args = get_call_args(lyr_inputs, lyr_cache, call_args, lyr_config['name'])
                    layer_outputs = ACTIVATION_LAYERS[lyr_name.upper()](call_args, **add_args)
                elif lyr_name.upper() == 'TIMEDISTRIBUTED':
                    td_layer = get_layer(lyr_name)
                    lyr_cache[lyr_name] = td_layer
                    continue
                else:
                    if td_layer is not None:
                        call_args, add_args = get_call_args(lyr_inputs, lyr_cache, call_args, lyr_config['name'])
                        layer_outputs = td_layer(get_layer(lyr_name)(**lyr_config))(call_args, **add_args)
                        td_layer = None
                    else:
                        call_args, add_args = get_call_args(lyr_inputs, lyr_cache, call_args, lyr_config['name'])
                        layer_outputs = get_layer(lyr_name)(**lyr_config)(call_args, **add_args)

            if activation is not None:  # put the string back to dictionary to be saved in config file
                lyr_config['activation'] = activation

            if named_outs is not None:

                if isinstance(named_outs, list):
                    # this layer is returning more than one output
                    assert len(named_outs) == len(layer_outputs), "Layer {} is expected to return {} outputs but it actually returns {}".format(lyr_name, named_outs, layer_outputs)
                    for idx, out_name in enumerate(named_outs):
                        self.update_cache(lyr_cache, out_name, layer_outputs[idx])
                else:
                    # this layer returns just one output, TODO, this might be re
                    self.update_cache(lyr_cache, named_outs, layer_outputs)

            self.update_cache(lyr_cache, lyr_config['name'], layer_outputs)
            first_layer = False

        inputs = []
        for k,v in lyr_cache.items():
            if 'INPUT' in k.upper():
                inputs.append(v)
        setattr(self, 'layers', lyr_cache)
        return inputs, layer_outputs

    def update_cache(self, cache:dict, key, value):
        if key in cache:
            raise ValueError("Duplicate input/output name found. The name {} already exists as input/output for another layer"
                             .format(key))
        cache[key] = value
        return

    def deconstruct_lyr_args(self, lyr_args:dict) ->tuple:

        inputs = lyr_args['inputs'] if 'inputs' in lyr_args else None
        outputs = lyr_args['outputs'] if 'outputs' in lyr_args else None
        call_args = lyr_args['call_args'] if 'call_args' in lyr_args else None

        return lyr_args['config'], inputs, outputs, call_args

    def check_lyr_config(self, lyr_name: str, config: dict):

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

    def get_layer_name(self, lyr: str) -> str:

        layer_name = lyr.split('_')[0]
        if layer_name.upper() not in list(LAYERS.keys()) + list(ACTIVATION_LAYERS.keys()):
            if not hasattr(tf.keras.layers, layer_name):
                raise ValueError("unknown layer {}".format(lyr))

        return layer_name


def get_call_args(lyr_inputs, lyr_cache, add_args, lyr_name):
    if isinstance(lyr_inputs, list):
        call_args = []
        for lyr_ins in lyr_inputs:
            if lyr_ins not in lyr_cache:
                raise ValueError("No layer named '{}' currently exists in the model which can be fed as input to '{}' layer.".format(lyr_ins, lyr_name))
            call_args.append(lyr_cache[lyr_ins])
    else:
        if lyr_inputs not in lyr_cache:
            raise ValueError("No layer named '{}' currently exists in the model which can be fed as input to '{}' layer.".format(lyr_inputs, lyr_name))
        call_args = lyr_cache[lyr_inputs]

    additional_args = {}
    if add_args is not None:
        assert isinstance(add_args, dict), "call_args to layer '{}' must be provided as dictionary".format(lyr_name)
        for arg_name, arg_val in add_args.items():
            if isinstance(arg_val, str):
                if arg_val not in lyr_cache:
                    raise NotImplementedError("The value {} for additional call argument {} to '{}' layer not understood".format(arg_val, arg_name, lyr_name))
                additional_args[arg_name] = lyr_cache[arg_val]

            elif isinstance(arg_val, list):
                # the additional argument is a list of tensors, get all of them from lyr_cache
                add_arg_val_list = []
                for arg in arg_val:
                    assert isinstance(arg, str)
                    add_arg_val_list.append(lyr_cache[arg])

                additional_args[arg_name] = add_arg_val_list
            else:
                raise NotImplementedError("The value {} for additional call argument {} to '{}' layer not understood".format(arg_val, arg_name, lyr_name))

    return call_args, additional_args

def get_layer(lyr_name:str):
    if hasattr(tf.keras.layers, lyr_name):
        lyr =  getattr(tf.keras.layers, lyr_name)
    elif lyr_name.upper() in LAYERS:
        lyr = LAYERS[lyr_name.upper()]
    else:
        raise ValueError(f"The layer name '{lyr_name}' you specified, does not exist")
    return lyr
