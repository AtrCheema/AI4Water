__all__ = ["NN", "check_act_fn"]

from weakref import WeakKeyDictionary

try:
    from AI4Water.tf_attributes import ACTIVATION_LAYERS, ACTIVATION_FNS, LAYERS, tf
except ModuleNotFoundError:
    tf = None

from .backend import BACKEND

if BACKEND == 'tensorflow':
    from AI4Water.tf_attributes import LAYERS, tf
else:
    from .pytorch_attributes import LAYERS


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
        self.is_training = False


class NN(AttributeStore):

    def __init__(self,
                 config: dict
                 ):

        self.config = config

        self.lookback = self.config['lookback']
        AttributeStore.__init__(self)

    @property
    def lookback(self):
        return self._lookback

    @lookback.setter
    def lookback(self, x):
        self._lookback = x

    def maybe_add_output_layer(self, current_outputs, lyr_cache:dict):
        """
        If the shape of the last layer in the NN structure does not
        matches with the number of outputs/self.num_outs, then this layer
        automatically a dense layer matching the number of outputs and
        then reshapes it to match the following dimention (outs, forecast_len/horizons)
        """
        if self.problem=='classification':
            return current_outputs
        new_outputs = current_outputs
        shape = current_outputs.shape
        quantiles = 1 if getattr(self, 'quantiles', None) is None else len(self.quantiles)
        change_shape = True
        if isinstance(self.num_outs, int):
            outs = self.num_outs
        elif isinstance(self.num_outs, dict):
            if len(self.num_outs) == 1:
                outs = list(self.num_outs.values())[0]
            else:
                change_shape = False
        elif self.num_outs is None:  # outputs are not known yet
            change_shape = False
        else:
            raise NotImplementedError

        if len(shape) < 3 and change_shape:  # the output is not of shape (?, num_outputs, horizons)
            num_outs = outs * quantiles
            if shape[-1] != num_outs:  # we add a Dense layer followed by reshaping it

                dense = LAYERS["DENSE"](num_outs, name="model_output")
                dense_out = dense(current_outputs)
                self.update_cache(lyr_cache, dense.name, dense)
                reshape = tf.keras.layers.Reshape(target_shape=(num_outs, self.forecast_len), name="output_reshaped")
                new_outputs = reshape(dense_out)
                self.update_cache(lyr_cache, reshape.name, reshape)

            else:  # just reshape the output to match (?, num_outputs, horizons)
                reshape = tf.keras.layers.Reshape(target_shape=(num_outs, self.forecast_len), name="output_reshaped")
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

        if tf is not None:
            if isinstance(config, tf.keras.layers.Lambda):
                config = tf.keras.layers.serialize(config)

        return config, inputs, outputs, call_args

    def check_lyr_config(self, lyr_name: str, config: dict):

        if callable(config):
            return lyr_name, [], config, None

        elif not isinstance(config, dict):
            args = [config]
            config = {}
        else:
            args = []

        if 'name' not in config and BACKEND != 'pytorch':
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
