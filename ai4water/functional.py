import numpy as np

from ._main import BaseModel
from ai4water.tf_attributes import ACTIVATION_LAYERS, LAYERS, tf, keras, tcn
from .nn_tools import get_add_call_args, get_call_args


class Model(BaseModel):

    """
    Model class with Functional API and inherits from `BaseModel`.

    For ML/non-Neural Network based models, there is no difference in functional
    or sub-clsasing api. For DL/NN-based models, this class implements functional
    api and differs from subclassing api in internal implementation of NN. This
    class is usefull, if you want to use the functional API of keras to build
    your own NN structure. In such as case you can construct your NN structure
    by overwriting `add_layers`. Another advantage of this class is that sometimes,
    model_subclsasing is not possible for example due to some bugs in tensorflow.
    In such a case this class can be used. Otherwise all the features of ai4water
    are available in this class as well.

    Example:
        >>>from ai4water.functional import Model
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes and builds the NN/ML model.
        """
        self._go_up = True
        super().__init__(*args, **kwargs)

        def_KModel = None
        if keras is not None:
            def_KModel = keras.models.Model
        self.KModel = kwargs.get('KModel', def_KModel)

        self.build()

    @property
    def api(self):
        return 'functional'

    @property
    def KModel(self):
        """sets k_model.

        In case when we want to customize the model such as for implementing custom
        `train_step`, we can provide the customized model as input the this Model
        class
        """
        return self._k_model

    @KModel.setter
    def KModel(self, x):
        self._k_model = x

    @property
    def weights(self):
        """Returns names of weights in model."""
        _ws = []
        for w in self._model.weights:
            _ws.append(w.name)
        return _ws

    @property
    def layer_names(self):
        _all_layers = []
        if self.category == "ML":
            return None
        for layer in self._model.layers:
            _all_layers.append(layer.name)
        return _all_layers

    @property
    def num_input_layers(self) -> int:
        if self.category != "DL":
            return np.inf
        else:
            return len(self._model.inputs)

    @property
    def input_layer_names(self) -> list:

        return [lyr.name.split(':')[0] for lyr in self._model.inputs]

    @property
    def layers_out_shapes(self) -> dict:
        """ returns shapes of outputs from all layers in model as dictionary"""
        shapes = {}

        for lyr in self._model.layers:
            shapes[lyr.name] = lyr.output_shape

        return shapes

    @property
    def layers_in_shapes(self) -> dict:
        """ returns the shapes of inputs to all layers"""
        shapes = {}

        for lyr in self._model.layers:
            shapes[lyr.name] = lyr.input_shape

        return shapes

    @property
    def fit_fn(self):
        return self._model.fit

    @property
    def evaluate_fn(self):
        return self._model.evaluate

    @property
    def predict_fn(self):
        return self._model.predict

    def first_layer_shape(self):
        """ instead of tuple, returning a list so that it can be moified if needed"""
        if self.num_input_layers > 1:
            shapes = {}
            for lyr in self._model.inputs:
                shapes[lyr.name] = lyr.shape
            return shapes
        shape = []
        for idx, d in enumerate(self._model.layers[0].input.shape):
            if int(tf.__version__[0]) == 1:
                if isinstance(d, tf.Dimension):  # for tf 1.x
                    d = d.value

            if idx == 0:  # the first dimension must remain undefined so that the user may define batch_size
                d = -1
            shape.append(d)
        return shape

    def add_layers(self, layers_config: dict, inputs=None):
        """
        Builds the NN from dictionary.

        Arguments:
            layers_config : wholse keys can be one of the following:
                `config`: `dict`/lambda, Every layer must contain initializing
                    arguments as `config` dictionary. The `config` dictionary
                    for every layer can contain `name` key and its value must be
                    `str` type. If `name` key  is not provided in the config,
                    the provided layer name will be used as its name e.g in following case
                        layers = {'LSTM': {'config': {'units': 16}}}
                    the name of `LSTM` layer will be `LSTM` while in follwoing case
                        layers = {'LSTM': {'config': {'units': 16, 'name': 'MyLSTM'}}}
                    the name of the lstm will be `MyLSTM`.
                `inputs`: str/list,  The calling arguments for the list. If `inputs`
                    key is missing for a layer, it will be supposed that either
                    this is an Input layer or it uses previous outputs as inputs.
                `outputs`: str/list  We can specifity the outputs from a layer
                    by using the `outputs` key. The value to `outputs` must be a
                    string or list of strings specifying the name of outputs from
                    current layer which can be used later in the mdoel.
                `call_args`: str/list  We can also specify additional call arguments
                    by `call_args` key. The value to `call_args` must be a string
                    or a list of strings.
            inputs : if None, it will be supposed the the `Input` layer either
                exists in `layers_config` or an Input layer will be created
                within this method before adding any other layer. If not None,
                then it must be in `Input` layer and the remaining NN architecture
                will be built as defined in `layers_config`. This can be handy
                when we want to use this method several times to build a complex
                or parallel NN structure. avoid `Input` in layer names.
        Returns:
            inputs :
            outputs :
        """
        lyr_cache = {}
        wrp_layer = None  # indicator for wrapper layers
        first_layer = True
        idx = 0

        for lyr, lyr_args in layers_config.items():
            idx += 0

            if callable(lyr) and hasattr(lyr, '__call__'):
                LAYERS[lyr.__name__] = lyr
                self.config['model']['layers'] = update_layers_config(layers_config, lyr)
                lyr = lyr.__name__

            lyr_config, lyr_inputs, named_outs, call_args = self.deconstruct_lyr_args(lyr, lyr_args)

            if callable(lyr) and not hasattr(lyr, '__call__'):
                lyr = "lambda"

            lyr_name, args, lyr_config, activation = self.check_lyr_config(lyr, lyr_config)

            # may be user has defined layers without input layer, in this case add Input layer as first layer
            if first_layer:
                if inputs is not None:  # This method was called by providing it inputs.
                    assert isinstance(inputs, tf.Tensor)
                    lyr_cache["Input"] = inputs
                    # since inputs have been defined, all the layers that will be added will be next to first layer
                    first_layer = False
                    layer_outputs = inputs
                    assign_dummy_name(layer_outputs, 'input')
                elif lyr_name != "Input":
                    if 'input_shape' in lyr_config:  # input_shape is given in the first layer so make input layer
                        layer_outputs = LAYERS["Input"](shape=lyr_config['input_shape'])
                        assign_dummy_name(layer_outputs, 'input')
                    else:
                        # for simple dense layer based models, lookback will not be used
                        def_shape = (self.ins,) if self.lookback == 1 else (self.lookback, self.ins)
                        layer_outputs = LAYERS["Input"](shape=def_shape)

                    # first layer is built so next iterations will not be for first layer
                    first_layer = False
                    # put the first layer in memory to be used for model compilation
                    lyr_cache["Input"] = layer_outputs
                    # add th layer which the user had specified as first layer

                    assign_dummy_name(layer_outputs, 'input')

            if lyr_inputs is None:  # The inputs to the layer have not been specified, so either it is an Input layer
                # or it uses the previous outputs as inputs
                if lyr_name == "Input":
                    # it is an Input layer, hence should not be called
                    layer_outputs = LAYERS[lyr_name](*args, **lyr_config)
                    assign_dummy_name(layer_outputs, 'input')
                else:
                    # it is executable and uses previous outputs as inputs
                    if lyr_name in ACTIVATION_LAYERS:
                        layer_outputs = ACTIVATION_LAYERS[lyr_name](name=lyr_config['name'])(layer_outputs)
                    elif lyr_name in ['TimeDistributed', 'Bidirectional']:
                        wrp_layer = LAYERS[lyr_name]
                        lyr_cache[lyr_name] = wrp_layer
                        continue
                    elif "LAMBDA" in lyr_name.upper():
                        # lyr_config is serialized lambda layer, which needs to be deserialized
                        # by default the lambda layer takes the previous output as input
                        # however when `call_args` are provided, they overwrite the layer_outputs
                        if call_args is not None:  # todo, add example in docs
                            layer_outputs = get_add_call_args(call_args, lyr_cache, lyr_config['name'])
                        layer_outputs = tf.keras.layers.deserialize(lyr_config)(layer_outputs)
                        # layers_config['lambda']['config'] still contails lambda, so we need to replace the python
                        # object (lambda) with the serialized version (lyr_config) so that it can be saved as json file.
                        layers_config[lyr]['config'] = lyr_config
                    else:
                        if wrp_layer is not None:
                            layer_outputs = wrp_layer(LAYERS[lyr_name](*args, **lyr_config))(layer_outputs)
                            wrp_layer = None
                        else:
                            add_args = get_add_call_args(call_args, lyr_cache, lyr_config['name'])
                            layer_initialized = LAYERS[lyr_name](*args, **lyr_config)
                            layer_outputs = layer_initialized(layer_outputs, **add_args)
                            self.get_and_set_attrs(layer_initialized)

            else:  # The inputs to this layer have been specified so they must exist in lyr_cache.
                # it is an executable
                if lyr_name in ACTIVATION_LAYERS:
                    call_args, add_args = get_call_args(lyr_inputs, lyr_cache, call_args, lyr_config['name'])
                    layer_outputs = ACTIVATION_LAYERS[lyr_name](name=lyr_config['name'])(call_args, **add_args)
                elif lyr_name in ['TimeDistributed', 'Bidirectional']:
                    wrp_layer = LAYERS[lyr_name]
                    lyr_cache[lyr_name] = wrp_layer
                    continue
                elif "LAMBDA" in lyr_name.upper():
                    call_args, add_args = get_call_args(lyr_inputs, lyr_cache, call_args, lyr_config['name'])
                    layer_outputs = tf.keras.layers.deserialize(lyr_config)(call_args)
                    layers_config[lyr]['config'] = lyr_config
                else:
                    if wrp_layer is not None:
                        call_args, add_args = get_call_args(lyr_inputs, lyr_cache, call_args, lyr_config['name'])
                        layer_outputs = wrp_layer(LAYERS[lyr_name](*args, **lyr_config))(call_args, **add_args)
                        wrp_layer = None
                    else:
                        call_args, add_args = get_call_args(lyr_inputs, lyr_cache, call_args, lyr_config['name'])
                        layer_initialized = LAYERS[lyr_name](*args, **lyr_config)
                        # todo, following conditioning is not good
                        # for concat layer inputs should be ([a,b,c]) instaed of (a,b,c)
                        if isinstance(lyr_inputs, list) and lyr_name != "Concatenate":
                            layer_outputs = layer_initialized(*call_args, **add_args)
                        else:
                            layer_outputs = layer_initialized(call_args, **add_args)
                        self.get_and_set_attrs(layer_initialized)

            if activation is not None:  # put the string back to dictionary to be saved in config file
                lyr_config['activation'] = activation

            if named_outs is not None:

                if isinstance(named_outs, list):
                    # this layer is returning more than one output
                    assert len(named_outs) == len(layer_outputs), "Layer {} is expected to return {} " \
                                                                  "outputs but it actually returns " \
                                                                  "{}".format(lyr_name, named_outs, layer_outputs)
                    for idx, out_name in enumerate(named_outs):
                        self.update_cache(lyr_cache, out_name, layer_outputs[idx])
                else:
                    # this layer returns just one output, TODO, this might be re
                    self.update_cache(lyr_cache, named_outs, layer_outputs)

            self.update_cache(lyr_cache, lyr_config['name'], layer_outputs)
            first_layer = False

        layer_outputs = self.maybe_add_output_layer(layer_outputs, lyr_cache)

        inputs = []
        for k, v in lyr_cache.items():
            # since the model is not build yet and we have access to only output tensors of each list, this is probably
            # the only way to know that how many `Input` layers were encountered during the run of this method. Each
            # tensor (except TimeDistributed) has .op.inputs attribute,
            # which is empty if a tensor represents output of Input layer.
            if int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 240:
                if k != "TimeDistributed" and hasattr(v, 'op'):
                    if hasattr(v.op, 'inputs'):
                        _ins = v.op.inputs
                        if len(_ins) == 0:
                            inputs.append(v)
            # not sure if this is the proper way of checking if a layer receives an input or not!
            else:
                if hasattr(v, '__dummy_name'):
                    inputs.append(v)

        # for case when {Input -> Dense, Input_1}, this method wrongly makes Input_1 as output so in such case use
        # {Input_1, Input -> Dense }, thus it makes Dense as output and first 2 as inputs, so throwing warning
        if int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 240:
            if len(layer_outputs.op.inputs) < 1:
                print("Warning: the output is of Input tensor class type")
        else:
            if 'op' not in dir(layer_outputs):  # layer_outputs does not have `op`, which means it has no incoming node
                print("Warning: the output is of Input tensor class type")

        return inputs, layer_outputs

    def compile(self, model_inputs, outputs, **compile_args):

        k_model = self.KModel(inputs=model_inputs, outputs=outputs)

        k_model.compile(loss=self.loss(), optimizer=self.get_optimizer(), metrics=self.get_metrics(), **compile_args)

        if self.verbosity > 0:
            k_model.summary()

        self.plot_model(k_model)
        return k_model

    def build(self):

        self.print_info()

        if self.category == "DL":
            if self.config.get('model', None) is None:
                lyrs = None
            else:
                lyrs = self.config['model']['layers']

            inputs, predictions = self.add_layers(lyrs)

            self._model = self.compile(inputs, predictions)

            self.info['model_parameters'] = int(self._model.count_params()) if self._model is not None else None

            if self.verbosity > 0 and self.config['model'] is not None:
                if 'tcn' in self.config['model']['layers']:
                    if int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) >= 250:
                        # tf >= 2.5 does not have _layers and tcn uses _layers
                        setattr(self._model, '_layers', self._model.layers)
                    tcn.tcn_full_summary(self._model, expand_residual_blocks=True)
        else:
            self.build_ml_model()

        if not getattr(self, 'from_check_point', False):
            # fit may fail so better to save config before as well. This will be overwritten once the fit is complete
            self.save_config()

        self.update_info()

        return

    def loss_name(self):
        if isinstance(self._model.loss, str):
            return self._model.loss
        elif hasattr(self._model.loss, 'name'):
            return self._model.loss.name
        else:
            return self._model.loss.__name__


def update_layers_config(layers_config, lyr):
    new_config = {}
    for k, v in layers_config.items():
        if k == lyr:
            new_config[lyr.__name__] = v
        else:
            new_config[k] = v
    return new_config


def assign_dummy_name(tensor, dummy_name):
    if isinstance(tensor, list):
        for t in tensor:
            setattr(t, '__dummy_name', dummy_name)
    else:
        setattr(tensor, '__dummy_name', dummy_name)
