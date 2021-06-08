__all__ = ["NN", "check_act_fn"]


try:
    import tensorflow as tf
    from AI4Water.tf_attributes import ACTIVATION_LAYERS, ACTIVATION_FNS, LAYERS
except ModuleNotFoundError:
    tf = None

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
    def __init__(self):
        self._model = AttributeNotSetYet("`build` to build neural network")
        self.method = None
        self.en_densor_We = None
        self.en_LSTM_cell = None
        self.auto_enc_composite = None
        self.de_LSTM_cell = None
        self.de_densor_We = None
        self.scalers = {}
        self.layers = None
        self.is_training = False


class NN(AttributeStore):

    def __init__(self,
                 config: dict,
                 ):
        self.config = config
        self.lookback = self.config['lookback']

        super(NN, self).__init__()

    @property
    def lookback(self):
        return self._lookback

    @lookback.setter
    def lookback(self, x):
        self._lookback = x

    def add_layers(self, layers_config:dict, inputs=None):
        """
        @param layers_config: `dict`, wholse keys can be one of the following:
            `config`: `dict`/lambda, Every layer must contain initializing arguments as `config` dictionary. The `config`
                                     dictionary for every layer can contain `name` key and its value must be `str`
                                     type. If `name` key  is not provided in the config, the provided layer name will be
                                     used as its name e.g in following case
                                       layers = {'LSTM': {'config': {'units': 16}}}
                                     the name of `LSTM` layer will be `LSTM` while in follwoing case
                                        layers = {'LSTM': {'config': {'units': 16, 'name': 'MyLSTM'}}}
                                     the name of the lstm will be `MyLSTM`.
            `inputs`: str/list,  The calling arguments for the list. If `inputs` key is missing for a layer, it will be
                                 supposed that either this is an Input layer or it uses previous outputs as inputs.
            `outputs`: str/list  We can specifity the outputs from a layer by using the `outputs` key. The value to `outputs` must be a string or
                                 list of strings specifying the name of outputs from current layer which can be used later in the mdoel.
            `call_args`: str/list  We can also specify additional call arguments by `call_args` key. The value to `call_args` must be a string or
                                   a list of strings.

        @param inputs: if None, it will be supposed the the `Input` layer either exists in `layers_config` or an Input
        layer will be created withing this method before adding any other layer. If not None, then it must be in `Input`
        layer and the remaining NN architecture will be built as defined in `layers_config`. This can be handy when we
        want to use this method several times to build a complex or parallel NN structure.
        avoid `Input` in layer names.
        """
        lyr_cache = {}
        wrp_layer = None  # indicator for wrapper layers
        first_layer = True
        idx = 0

        for lyr, lyr_args in layers_config.items():
            idx += 0

            lyr_config, lyr_inputs, named_outs, call_args = self.deconstruct_lyr_args(lyr, lyr_args)

            lyr_name, args, lyr_config, activation = self.check_lyr_config(lyr, lyr_config)

            # may be user has defined layers without input layer, in this case add Input layer as first layer
            if first_layer:
                if inputs is not None: # This method was called by providing it inputs.
                    assert isinstance(inputs, tf.Tensor)
                    lyr_cache["INPUT"] = inputs
                    first_layer = False # since inputs have been defined, all the layers that will be added will be next to first layer
                    layer_outputs = inputs

                elif lyr_name.upper() != "INPUT":
                    if 'input_shape' in lyr_config:  # input_shape is given in the first layer so make input layer
                        layer_outputs = LAYERS["INPUT"](shape=lyr_config['input_shape'])
                    else:
                        # for simple dense layer based models, lookback will not be used
                        def_shape = (self.ins,) if self.lookback == 1 else (self.lookback, self.ins)
                        layer_outputs = LAYERS["INPUT"](shape=def_shape)

                    # first layer is built so next iterations will not be for first layer
                    first_layer = False
                    # put the first layer in memory to be used for model compilation
                    lyr_cache["INPUT"] = layer_outputs
                    # add th layer which the user had specified as first layer

            if lyr_inputs is None:  # The inputs to the layer have not been specified, so either it is an Input layer
                # or it uses the previous outputs as inputs
                if lyr_name.upper() == "INPUT":
                    # it is an Input layer, hence should not be called
                    layer_outputs = LAYERS[lyr_name.upper()](*args, **lyr_config)
                else:
                    # it is executable and uses previous outputs as inputs
                    if lyr_name.upper() in ACTIVATION_LAYERS:
                        layer_outputs = ACTIVATION_LAYERS[lyr_name.upper()](name=lyr_config['name'])(layer_outputs)
                    elif lyr_name.upper() in ['TIMEDISTRIBUTED', 'BIDIRECTIONAL']:
                        wrp_layer = LAYERS[lyr_name.upper()]
                        lyr_cache[lyr_name] = wrp_layer
                        continue
                    elif  "LAMBDA" in lyr_name.upper():
                        # lyr_config is serialized lambda layer, which needs to be deserialized
                        layer_outputs = tf.keras.layers.deserialize(lyr_config)(layer_outputs)
                        # layers_config['lambda']['config'] still contails lambda, so we need to replace the python
                        # object (lambda) with the serialized version (lyr_config) so that it can be saved as json file.
                        layers_config[lyr]['config'] = lyr_config
                    else:
                        if wrp_layer is not None:
                            layer_outputs = wrp_layer(LAYERS[lyr_name.upper()](*args, **lyr_config))(layer_outputs)
                            wrp_layer = None
                        else:
                            add_args = get_add_call_args(call_args, lyr_cache, lyr_config['name'])
                            layer_initialized = LAYERS[lyr_name.upper()](*args, **lyr_config)
                            layer_outputs = layer_initialized(layer_outputs, **add_args)
                            self.get_and_set_attrs(layer_initialized)

            else:  # The inputs to this layer have been specified so they must exist in lyr_cache.
                # it is an executable
                if lyr_name.upper() in ACTIVATION_LAYERS:
                    call_args, add_args = get_call_args(lyr_inputs, lyr_cache, call_args, lyr_config['name'])
                    layer_outputs = ACTIVATION_LAYERS[lyr_name.upper()](name=lyr_config['name'])(call_args, **add_args)
                elif lyr_name.upper() in ['TIMEDISTRIBUTED', 'BIDIRECTIONAL']:
                    wrp_layer = LAYERS[lyr_name.upper()]
                    lyr_cache[lyr_name] = wrp_layer
                    continue
                elif "LAMBDA" in lyr_name.upper():
                    call_args, add_args = get_call_args(lyr_inputs, lyr_cache, call_args, lyr_config['name'])
                    layer_outputs = tf.keras.layers.deserialize(lyr_config)(call_args)
                    layers_config[lyr]['config'] = lyr_config
                else:
                    if wrp_layer is not None:
                        call_args, add_args = get_call_args(lyr_inputs, lyr_cache, call_args, lyr_config['name'])
                        layer_outputs = wrp_layer(LAYERS[lyr_name.upper()](*args, **lyr_config))(call_args, **add_args)
                        wrp_layer = None
                    else:
                        call_args, add_args = get_call_args(lyr_inputs, lyr_cache, call_args, lyr_config['name'])
                        layer_initialized = LAYERS[lyr_name.upper()](*args, **lyr_config)
                        layer_outputs = layer_initialized(call_args, **add_args)
                        self.get_and_set_attrs(layer_initialized)

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

        layer_outputs = self.maybe_add_output_layer(layer_outputs, lyr_cache)

        inputs = []
        for k,v in lyr_cache.items():
            # since the model is not build yet and we have access to only output tensors of each list, this is probably
            # the only way to know that how many `Input` layers were encountered during the run of this method. Each
            # tensor (except TimeDistributed) has .op.inputs attribute, which is empty if a tensor represents output of Input layer.
            if int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 240:
                if k.upper() != "TIMEDISTRIBUTED" and hasattr(v, 'op'):
                    if hasattr(v.op, 'inputs'):
                        _ins = v.op.inputs
                        if len(_ins) == 0:
                            inputs.append(v)
            else:  # not sure if this is the proper way of checking if a layer receives an input or not!
                if hasattr(v, '_keras_mask'):
                    inputs.append(v)

        setattr(self, 'layers', lyr_cache)

        # for case when {Input -> Dense, Input_1}, this method wrongly makes Input_1 as output so in such case use
        # {Input_1, Input -> Dense }, thus it makes Dense as output and first 2 as inputs, so throwing warning
        if int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 240:
            if len(layer_outputs.op.inputs) < 1:
                print("Warning: the output is of Input tensor class type")
        else:
            if 'op' not in dir(layer_outputs):  # layer_outputs does not have `op`, which means it has no incoming node
                print("Warning: the output is of Input tensor class type")

        return inputs, layer_outputs

    def maybe_add_output_layer(self, current_outputs, lyr_cache:dict):
        """
        If the shape of the last layer in the NN structure does not
        matches with the number of outputs/self.outs, then this layer
        automatically a dense layer matching the number of outputs and
        then reshapes it to match the following dimention (outs, forecast_len/horizons)
        """
        new_outputs = current_outputs
        shape = current_outputs.shape
        quantiles = 1 if getattr(self, 'quantiles', None) is None else len(self.quantiles)
        change_shape = True
        if isinstance(self.outs, int):
            outs = self.outs
        elif isinstance(self.outs, dict):
            if len(self.outs) == 1:
                outs = list(self.outs.values())[0]
            else:
                change_shape = False
        elif self.outs is None:  # outputs are not known yet
            change_shape = False
        else:
            raise NotImplementedError

        if len(shape) < 3 and change_shape:  # the output is not of shape (?, num_outputs, horizons)
            num_outs = outs * quantiles
            if shape[-1] != num_outs:  # we add a Dense layer followed by reshaping it

                dense = tf.keras.layers.Dense(num_outs)
                dense_out = dense(current_outputs)
                self.update_cache(lyr_cache, dense.name, dense)
                reshape = tf.keras.layers.Reshape(target_shape=(num_outs, self.forecast_len))
                new_outputs = reshape(dense_out)
                self.update_cache(lyr_cache, reshape.name, reshape)

            else:  # just reshape the output to match (?, num_outputs, horizons)
                reshape = tf.keras.layers.Reshape(target_shape=(num_outs, self.forecast_len))
                new_outputs = reshape(current_outputs)
                self.update_cache(lyr_cache, reshape.name, reshape)
        elif len(shape) > 3:
            raise NotImplementedError  # let's raise the error now and see where it should not
        else:
            pass # currently not checking 3d output and supposing it is of currect shape

        return new_outputs

    def update_cache(self, cache:dict, key, value):
        if key in cache:
            raise ValueError("Duplicate input/output name found. The name '{}' already exists as input/output for another layer"
                             .format(key))
        cache[key] = value
        return

    def deconstruct_lyr_args(self, lyr_name,  lyr_args) ->tuple:

        if not isinstance(lyr_args, dict):
            return lyr_args, None, None,None

        if 'config' not in lyr_args:
            if all([arg not in lyr_args for arg in ['inputs', 'outputs', 'call_args']]):
                config = lyr_args
                inputs = None
                outputs = None
                call_args = None
            else:
                raise ValueError(f"No config found for layer {lyr_name}")
        else:

            config = lyr_args['config']
            inputs = lyr_args['inputs'] if 'inputs' in lyr_args else None
            outputs = lyr_args['outputs'] if 'outputs' in lyr_args else None
            call_args = lyr_args['call_args'] if 'call_args' in lyr_args else None

        if isinstance(config, tf.keras.layers.Lambda):
            config = tf.keras.layers.serialize(config)

        return config, inputs, outputs, call_args

    def check_lyr_config(self, lyr_name: str, config: dict):

        if not isinstance(config, dict):
            args = [config]
            config = {}
        else:
            args = []

        if 'name' not in config:
            config['name'] = lyr_name

        activation = None
        if "LAMBDA" not in lyr_name.upper():
            # for lambda layers, we don't need to check activation functions and layer names.
            config, activation = check_act_fn(config)

            # get keras/tensorflow layer compatible layer name
            lyr_name = self.get_layer_name(lyr_name)

        return lyr_name, args, config, activation

    def get_layer_name(self, lyr: str) -> str:

        layer_name = lyr.split('_')[0]
        if layer_name.upper() not in list(LAYERS.keys()) + list(ACTIVATION_LAYERS.keys()):
            raise ValueError(f"""
                            The layer name '{lyr}' you specified, does not exist.
                            Is this a user defined layer? If so, make sure your
                            layer is being considered by AI4Water
                            """)

        return layer_name

    def get_and_set_attrs(self, layer):
        if layer.__class__.__name__ == "TemporalFusionTransformer":  # check the type without importing the layer
            # use layer name as there can be more than one layers from same class.
            setattr(self, f'{layer.name}_attentions', layer.attention_components)
        return


def check_act_fn(config: dict):
    """ it is possible that the config file does not have activation argument or activation is None"""
    activation = None
    if 'activation' in config:
        activation = config['activation']
    if activation is not None:
        assert isinstance(activation, str), f"unknown activation function {activation}"
        config['activation'] = ACTIVATION_FNS[activation.upper()]

    return config, activation


def get_call_args(lyr_inputs, lyr_cache, add_args, lyr_name):
    """ gets the additional call arguments for a layer. It is supposed that the call arguments are actually tensors/layers
    that have been created so far in the model including input layer. The call_args can be a list of inputs as well."""
    if isinstance(lyr_inputs, list):
        call_args = []
        for lyr_ins in lyr_inputs:
            if lyr_ins not in lyr_cache:
                raise ValueError("""No layer named '{}' currently exists in the model which can be fed
                                    as input to '{}' layer.""".format(lyr_ins, lyr_name))
            call_args.append(lyr_cache[lyr_ins])
    else:
        if lyr_inputs not in lyr_cache:
            raise ValueError(f"""
                                No layer named '{lyr_inputs}' currently exists in the model which
                                can be fed as input to '{lyr_name}' layer. Available layers are
                                {list(lyr_cache.keys())}
""")
        call_args = lyr_cache[lyr_inputs]

    return call_args, get_add_call_args(add_args, lyr_cache, lyr_name)

def get_add_call_args(add_args,lyr_cache, lyr_name):
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

            elif isinstance(arg_val, bool) or arg_val is None:
                additional_args[arg_name] = arg_val

            else:
                raise NotImplementedError("The value `{}` for additional call argument {} to '{}' layer not understood".format(arg_val, arg_name, lyr_name))

    return additional_args
