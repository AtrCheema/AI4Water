
from typing import Any, List
from collections import OrderedDict


from ._main import BaseModel
from ai4water.tf_attributes import ACTIVATION_LAYERS, tcn
from .nn_tools import get_call_args
from .backend import tf, torch, np, os
import ai4water.backend as K

from .models.torch import LAYERS as TORCH_LAYERS
from .tf_attributes import LAYERS

if K.BACKEND == 'tensorflow' and tf is not None:
    MODEL = tf.keras.Model
elif K.BACKEND == 'pytorch' and torch is not None:
    MODEL = torch.nn.Module
else:
    class MODEL(object):
        pass


class Model(MODEL, BaseModel):
    """
    This class Inherits from `BaseModel`.
    This class is a subclass of keras.Model/torch.nn.Module depending upon the
    backend used. For scikit-learn/xgboost/catboost type models, this class only
    inherits from `BaseModel. For deep learning/neural network based models, this
    class directly exposes all the functionalities of underlying Model. Thus `self`
    is now a keras Model or torch.nn.Module. If the user wishes to create his/her
    own NN architecture, he/she should overwrite `initialize_layers` and `call`/`forward`
    methods.
    """

    def __init__(self,
                 verbosity=1,
                 model=None,
                 path=None,
                 prefix=None,
                 **kwargs):

        """
        Initializes the layers of NN model using `initialize_layers` method.
        All other input arguments goes to `BaseModel`.
        """
        if K.BACKEND == 'tensorflow' and tf is not None:
            min_version = tf.__version__.split(".")[1]
            maj_version = tf.__version__.split(".")[0]
            if maj_version in ["2"] and min_version in ["3", "4"]:
                raise NotImplementedError(f"""
            Not implemented due to a bug in tensorflow as shown here https://github.com/tensorflow/tensorflow/issues/44646
            You can use functional API instead by using
            from ai4water.functional import Model
            instead of 
            from ai4water import Model
            Or change the tensorflow version. Current version is {tf.__version__}. 
            """)

        tf_kwargs = {}
        for arg in ['inputs', 'outputs']:
            if arg in kwargs:
                tf_kwargs[arg] = kwargs[arg]

        self._go_up = False

        MODEL.__init__(self, **tf_kwargs)

        self._go_up = True
        BaseModel.__init__(self,
                           prefix=prefix, path=path, verbosity=verbosity, model=model,
                           **kwargs)

        self.config['backend'] = K.BACKEND

        if torch is not None:
            from .models.torch import Learner
            self.torch_learner = Learner(
                model=self,
                batch_size=self.config['batch_size'],
                num_epochs=self.config['epochs'],
                shuffle=self.config['shuffle'],
                to_monitor=self.config['monitor'],
                patience=self.config['patience'],
                path=self.path,
                use_cuda=False,
                wandb_config=self.config['wandb_config'],
                verbosity=self.verbosity
            )

        if self.category == "DL":
            self.initialize_layers(self.config['model']['layers'])

            if K.BACKEND == 'tensorflow':
                outs = self.call(self._input_lyrs(), run_call=False)
                setattr(self, 'output_lyrs', outs)
                self._go_up = False  # do not reinitiate BaseModel and other upper classes

                maj_ver = int(tf.__version__.split('.')[0])
                min_ver = int(tf.__version__.split('.')[1][0])
                # in tf versions >= 2.5, we don't need to specify inputs and outputs as keyword arguments
                if maj_ver>1 and min_ver>=5:
                    MODEL.__init__(self, self._input_lyrs(), self.output_lyrs)
                else:
                    MODEL.__init__(self, inputs=self._input_lyrs(), outputs=self.output_lyrs)

        self.build(self._get_dummy_input_shape())  # will initialize ML models or build NNs

    def _input_lyrs(self):
        """
        Input  layers of deep learning model.

        `input_lyrs` can be a ListWrapper so just extract the tensor from the
        list. if the length of the list ==1
        """
        input_lyrs = None
        if hasattr(self, 'input_lyrs'):
            _input_lyrs = self.input_lyrs
            if isinstance(_input_lyrs, list) and len(_input_lyrs) == 1:
                input_lyrs = _input_lyrs[0]
            elif _input_lyrs.__class__.__name__ == "ListWrapper" and len(_input_lyrs) == 1:
                input_lyrs = _input_lyrs[0]
            else:
                input_lyrs = _input_lyrs

        return input_lyrs

    @property
    def torch_learner(self):
        return self._torch_learner

    @torch_learner.setter
    def torch_learner(self, x):
        """So that learner can be changed."""
        self._torch_learner = x

    @property
    def layer_names(self) -> List[str]:
        """Returns a list of names of layers/nn.modules
        for deep learning model. For ML models, returns empty list"""
        _all_layers = []
        if self.category == "ML":
            pass
        elif self.config['backend'] == 'tensorflow':
            for layer in self.layers:
                _all_layers.append(layer.name)
        elif self.config['backend'] == 'pytorch':
            _all_layers = list(self._modules.keys())
        return _all_layers

    @property
    def layers_in_shapes(self) -> dict:
        """Returns the shapes of inputs to all layers"""
        shapes = {}

        for lyr in self.layers:
            shapes[lyr.name] = lyr.input_shape

        return shapes

    @property
    def layers_out_shapes(self) -> dict:
        """ returns shapes of outputs from all layers in model as dictionary"""
        shapes = {}

        for lyr in self.layers:
            shapes[lyr.name] = lyr.output_shape

        return shapes

    @property
    def num_input_layers(self) -> int:
        if self.category != "DL":
            return np.inf
        if K.BACKEND == 'pytorch':
            return 1
        else:
            return len(self.inputs)

    @property
    def input_layer_names(self) -> list:
        default = []
        if self.inputs:
            default =  [lyr.name.split(':')[0] for lyr in self.inputs]
        if len(default) == 0:
            sec_option_inputs = self._input_lyrs()
            if isinstance(sec_option_inputs, list):
                default = []
                for i in sec_option_inputs:
                    default.append(i.name)
            else:
                default = sec_option_inputs.name
        return default

    def _get_dummy_input_shape(self):
        shape = ()
        if K.BACKEND == 'tensorflow' and self.category == "DL":
            if isinstance(self.inputs, list):
                if len(self.inputs)==1:
                    shape = self.inputs[0].shape
                else:
                    shape = [inp.shape for inp in self.inputs]


        return shape

    @property
    def api(self):
        return 'subclassing'

    @property
    def fit_fn(self):
        if self.category == "DL":
            if K.BACKEND == 'tensorflow':
                return super().fit
            elif K.BACKEND == 'pytorch':
                return self.torch_learner.fit

        return self._model.fit  # e.g. for ML models

    @property
    def evaluate_fn(self):
        if self.category == "DL":
            if K.BACKEND == 'tensorflow':
                return super().evaluate
            elif K.BACKEND == 'pytorch':
                return self.torch_learner.evaluate
            else:
                raise ValueError
        elif self.category == "ML":
            return self.evalute_ml_models

        return self._model.evaluate

    @property
    def predict_fn(self):
        if self.category == "DL":
            if K.BACKEND == 'tensorflow':
                return super().predict
            elif K.BACKEND == 'pytorch':
                return self.torch_learner.predict
        return self._model.predict

    def initialize_layers(self, layers_config: dict, inputs=None):
        """
        Initializes the layers/weights/variables which are to be used in `forward`
        or `call` method.

        Parameters
        ---------
            layers_config : python dictionary to define neural network. For details
                [see](https://ai4water.readthedocs.io/en/latest/build_dl_models.html)

            inputs : if None, it will be supposed the the `Input` layer either
                exists in `layers_config` or an Input layer will be created
                withing this method before adding any other layer. If not None,
                then it must be in `Input` layer and the remaining NN architecture
                will be built as defined in `layers_config`. This can be handy
                when we want to use this method several times to build a complex
                or parallel NN structure. Avoid `Input` in layer names.
        """
        layers_config = layers_config.copy()
        input_lyrs = []
        initiated_layers = OrderedDict()
        wrp_layer = None  # indicator for wrapper layers
        first_layer = True

        for lyr, lyr_args in layers_config.items():

            lyr_config, lyr_inputs, named_outs, call_args = self.deconstruct_lyr_args(lyr, lyr_args)

            lyr_name, args, lyr_config, activation = self.check_lyr_config(lyr, lyr_config)

            if K.BACKEND == 'pytorch':

                if first_layer:
                    first_layer = False

                if callable(lyr_config):
                    lyr_initiated = lyr_config
                else:
                    lyr_initiated = TORCH_LAYERS[lyr_name](**lyr_config)
                setattr(self, lyr, lyr_initiated)
                initiated_layers[lyr] = {"layer": lyr_initiated, "named_outs": named_outs, 'call_args': call_args,
                                         'inputs': lyr_inputs}

            else:
                # may be user has defined layers without input layer, in this case add Input layer as first layer
                if first_layer:
                    if inputs is not None:  # This method was called by providing it inputs.
                        assert isinstance(inputs, tf.Tensor)
                        # since inputs have been defined, all the layers that will be added will be next to first layer
                        first_layer = False
                        layer_outputs = inputs
                        initiated_layers[layer_outputs.name] = {'layer': layer_outputs, 'tf_name': lyr_name}

                    elif lyr_name != "Input":
                        if 'input_shape' in lyr_config:  # input_shape is given in the first layer so make input layer
                            initialized_layer = LAYERS["Input"](shape=lyr_config['input_shape'])
                        else:
                            # for simple dense layer based models, lookback will not be used
                            def_shape = (self.num_ins,) if self.lookback == 1 else (self.lookback, self.num_ins)
                            initialized_layer = LAYERS["Input"](shape=def_shape)

                        # first layer is built so next iterations will not be for first layer
                        first_layer = False
                        # put the first layer in memory to be used for model compilation
                        # add th layer which the user had specified as first layer
                        initiated_layers[initialized_layer.name] = {'layer': initialized_layer,
                                                                    'tf_name': lyr_name}
                        input_lyrs.append(initialized_layer)

                # The inputs to the layer have not been specified, so either it is an Input layer
                if lyr_inputs is None:
                    # or it uses the previous outputs as inputs
                    if lyr_name == "Input":
                        # it is an Input layer, hence should not be called
                        initialized_layer = LAYERS[lyr_name](*args, **lyr_config)
                        initiated_layers[lyr_config['name']] = {'layer': initialized_layer,
                                                                'tf_name': lyr_name}
                        input_lyrs.append(initialized_layer)
                    else:
                        # it is executable and uses previous outputs as inputs
                        if lyr_name in ACTIVATION_LAYERS:
                            layer_outputs = ACTIVATION_LAYERS[lyr_name](name=lyr_config['name'])
                            initiated_layers[lyr_config['name']] = {'layer': layer_outputs,
                                                                    'named_outs': named_outs,
                                                                    'call_args': call_args,
                                                                    'inputs': lyr_inputs,
                                                                    'tf_name': lyr_name}
                        elif lyr_name in ['TimeDistributed', 'Bidirectional']:
                            wrp_layer = LAYERS[lyr_name]
                            # because wrapper layer name is property
                            initiated_layers[lyr_config['name']] = {'layer': wrp_layer,
                                                                    'tf_name': lyr_name}
                            continue
                        elif "LAMBDA" in lyr_name.upper():
                            # lyr_config is serialized lambda layer, which needs to be deserialized
                            initialized_layer = tf.keras.layers.deserialize(lyr_config)
                            # layers_config['lambda']['config'] still contails lambda, so we need to replace the python
                            # object (lambda) with the serialized version (lyr_config) so that it can be saved as json file.
                            layers_config[lyr]['config'] = lyr_config
                            initiated_layers[lyr_config['name']] = {'layer': initialized_layer,
                                                                    'named_outs': named_outs,
                                                                    'call_args': call_args,
                                                                    'inputs': lyr_inputs,
                                                                    'tf_name': lyr_name}
                        else:
                            if wrp_layer is not None:
                                initialized_layer = wrp_layer(LAYERS[lyr_name](*args, **lyr_config))
                                initiated_layers[lyr_config['name']] = {'layer': initialized_layer,
                                                                        'named_outs': named_outs,
                                                                        'call_args': call_args,
                                                                        'inputs': lyr_inputs,
                                                                        'tf_name': lyr_name}
                                wrp_layer = None
                            else:
                                if lyr_name == "TemporalFusionTransformer":
                                    lyr_config['return_attention_components'] = True
                                initialized_layer = LAYERS[lyr_name](*args, **lyr_config)
                                initiated_layers[lyr_config['name']] = {'layer': initialized_layer,
                                                                        'named_outs': named_outs,
                                                                        'call_args': call_args,
                                                                        'inputs': lyr_inputs,
                                                                        'tf_name': lyr_name}

                else:  # The inputs to this layer have been specified so they must exist in lyr_cache.
                    # it is an executable
                    if lyr_name in ACTIVATION_LAYERS:

                        layer_outputs = ACTIVATION_LAYERS[lyr_name](name=lyr_config['name'])
                        initiated_layers[lyr_config['name']] = {'layer': layer_outputs,
                                                                'named_outs': named_outs,
                                                                'call_args': call_args,
                                                                'inputs': lyr_inputs,
                                                                'tf_name': lyr_name}
                    elif lyr_name in ['TimeDistributed', 'Bidirectional']:
                        wrp_layer = LAYERS[lyr_name]
                        # because wrapper layer name is property
                        initiated_layers[lyr_config['name']] = {'layer': wrp_layer,
                                                                'tf_name': lyr_name}
                        continue
                    elif "LAMBDA" in lyr_name.upper():
                        initialized_layer = tf.keras.layers.deserialize(lyr_config)
                        layers_config[lyr]['config'] = lyr_config
                        initiated_layers[lyr_config['name']] = {'layer': initialized_layer,
                                                                'named_outs': named_outs,
                                                                'call_args': call_args,
                                                                'inputs': lyr_inputs,
                                                                'tf_name': lyr_name}
                    else:
                        if wrp_layer is not None:
                            initialized_layer = wrp_layer(LAYERS[lyr_name](*args, **lyr_config))
                            initiated_layers[lyr_config['name']] = {'layer': initialized_layer,
                                                                    'named_outs': named_outs,
                                                                    'call_args': call_args,
                                                                    'inputs': lyr_inputs,
                                                                    'tf_name': lyr_name}
                            wrp_layer = None
                        else:
                            layer_initialized = LAYERS[lyr_name](*args, **lyr_config)
                            initiated_layers[lyr_config['name']] = {'layer': layer_initialized,
                                                                    'named_outs': named_outs,
                                                                    'call_args': call_args,
                                                                    'inputs': lyr_inputs,
                                                                    'tf_name': lyr_name}

                if activation is not None:  # put the string back to dictionary to be saved in config file
                    lyr_config['activation'] = activation

                first_layer = False

        # inputs = [] todo, indentify input layers
        # for k,v in lyr_cache.items():
        #     since the model is not build yet and we have access to only output tensors of each list, this is probably
        #     # the only way to know that how many `Input` layers were encountered during the run of this method. Each
        # tensor (except TimeDistributed) has .op.inputs attribute, which is empty if a tensor represents output of Input layer.
        #     if int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 240:
        #         if k.upper() != "TIMEDISTRIBUTED" and hasattr(v, 'op'):
        #             if hasattr(v.op, 'inputs'):
        #                 _ins = v.op.inputs
        #                 if len(_ins) == 0:
        #                     inputs.append(v)
        #     else:  # not sure if this is the proper way of checking if a layer receives an input or not!
        #         if hasattr(v, '_keras_mask'):
        #             inputs.append(v)

        setattr(self, 'initiated_layers', initiated_layers)
        setattr(self, 'input_lyrs', input_lyrs)


        # todo,
        # # for case when {Input -> Dense, Input_1}, this method wrongly makes Input_1 as output so in such case use
        # # {Input_1, Input -> Dense }, thus it makes Dense as output and first 2 as inputs, so throwing warning
        # if int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 240:
        #     if len(layer_outputs.op.inputs) < 1:
        #         print("Warning: the output is of Input tensor class type")
        # else:
        #    if 'op' not in dir(layer_outputs):  # layer_outputs does not have `op`, which means it has no incoming node
        #         print("Warning: the output is of Input tensor class type")

        # outs = None
        #if BACKEND == 'tensorflow':
            # outs = self.call(input_lyrs)
            # setattr(self, 'output_lyrs', outs)
            # if BACKEND == 'tensorflow':
            #     ## Reinitial
            #     super(Model, self).__init__(
            #         inputs=input_lyrs,
            #         outputs=outs)
                #MODEL.__init__(self, inputs=inputs, outputs=outs)

        return input_lyrs  # , outs

    def call(self, inputs, training=None, mask=None, run_call=True):

        version = ''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')
        return getattr(self, f'call_{version}')(inputs, training, mask, run_call=run_call)

    def call_260(self, *args, **kwargs):
        return self.call_250(*args, **kwargs)

    def call_270(self, *args, **kwargs):
        return self.call_250(*args, **kwargs)

    def call_290(self, *args, **kwargs):
        return self.call_250(*args, **kwargs)

    def call_280(self, *args, **kwargs):
        return self.call_250(*args, **kwargs)

    def call_200(self, *args, **kwargs):
        return self.call_210(*args, **kwargs)

    def call_250(self, inputs, training=None, mask=None, run_call=True):

        self.treat_casted_inputs(inputs)

        outs = inputs

        # inputs can be a list of tensors
        if isinstance(inputs, list) or isinstance(inputs, tuple) or inputs.__class__.__name__ == "Listwrapper":
            cache = {getattr(i, '__dummy_name').split(':')[0]:i for i in inputs}

        # if inputs is a list, then just save it in cache
        elif isinstance(inputs, dict):
            cache = inputs

        elif isinstance(inputs, tuple):
            if len(inputs) == 1:
                inputs, = inputs
                cache = {inputs.name.split(':')[0] : inputs}
            else:
                cache = {i.name.split(':')[0]: i for i in inputs}

        # hopefully this is just one tensor
        else:
            cache = {getattr(inputs, '__dummy_name').split(':')[0]: inputs}

        # todo keep tensor chache and layers cache separate
        input_tensor = False
        for idx, (lyr, lyr_args) in enumerate(self.initiated_layers.items()):

            if isinstance(lyr_args['layer'], tf.Tensor) or idx == 0 or is_input(lyr_args['layer']):
                # this must be an input layer
                # assert is_input(lyr_args['layer'])
                if isinstance(inputs, list):
                    assert all([is_input(_input) for _input in inputs])
                if isinstance(inputs, tuple):
                    if not run_call:
                        assert all([is_input(_input) for _input in inputs])
                # else:
                #     assert is_input(inputs)
                input_tensor = True
                # don't use the tf.keras.Input from self.initiated_layers
                # outs = lyr_args['layer']

            elif lyr_args['tf_name'] in ['TimeDistributed', 'Bidirectional']:
                # no need to call wrapper layer so move to next iteration
                continue

            else:
                _inputs = lyr_args.get('inputs', None)

                # inputs have not been explicitly defined by the user so just use previous output
                if _inputs is None:
                    _inputs = prev_output_name

                if idx == 1 and _inputs not in cache:
                    call_args, add_args = inputs, {}
                else:
                    call_args, add_args = get_call_args(_inputs, cache, lyr_args['call_args'], lyr)

                # call the initiated layer
                outs = lyr_args['layer'](call_args, **add_args)

                # if the layer is TFT, we need to extract the attention components
                # so that they can be used during post-processign
                if lyr in ["TemporalFusionTransformer", "TFT"]:
                    outs, self.TemporalFusionTransformer_attentions = outs

                if lyr_args['named_outs'] is not None:
                    if isinstance(outs, list):
                        assert len(lyr_args['named_outs']) == len(outs)
                        for name, out_tensor in zip(lyr_args['named_outs'], outs):
                            cache[name] = out_tensor
                    else:
                        cache[lyr_args['named_outs']] = outs

            if input_tensor:
                input_tensor = False
            else:
                cache[lyr] = outs
            prev_output_name = lyr

        return outs

    def call_210(self, inputs, training=True, mask=None, run_call=True):

        if int(''.join(np.__version__.split('.')[0:2]).ljust(3, '0')) >= 120:
            raise NumpyVersionException("Decrease")

        self.treat_casted_inputs(inputs)

        outs = inputs

        # inputs can be a list of tensors
        if isinstance(inputs, list) or isinstance(inputs, tuple) or inputs.__class__.__name__ == "Listwrapper":
            cache = {getattr(i, '__dummy_name').split(':')[0]:i for i in inputs}

        # if inputs is a list, then just save it in cache
        elif isinstance(inputs, dict):
            cache = inputs

        # hopefully this is just one tensor
        else:
            cache = {getattr(inputs, '__dummy_name').split(':')[0]: inputs}

        is_input_tensor = False

        for idx, (lyr, lyr_args) in enumerate(self.initiated_layers.items()):

            lyr = lyr.split(':')[0]

            if isinstance(lyr_args['layer'], tf.Tensor) or idx == 0 or is_input(lyr_args['layer']):
                is_input_tensor = True
                # this must be an input layer
                # assert is_input(lyr_args['layer'])
                if isinstance(inputs, list):
                    if not run_call:
                        assert all([is_input(_input) for _input in inputs])
                if isinstance(inputs, tuple):
                    assert all([is_input(_input) for _input in inputs])

            elif lyr_args['tf_name'] in ['TimeDistributed', 'Bidirectional']:
                # no need to call wrapper layer so move to next iteration
                continue
            else:
                _inputs = lyr_args.get('inputs', None)

                # inputs have not been explicitly defined by the user so just use previous output
                if _inputs is None:
                    _inputs = prev_output_name

                if idx == 1 and _inputs not in cache:
                    call_args, add_args = inputs, {}
                else:
                    call_args, add_args = get_call_args(_inputs, cache, lyr_args['call_args'], lyr)

                # call the initiated layer
                outs = lyr_args['layer'](call_args, **add_args)

                # if the layer is TFT, we need to extract the attention components
                # so that they can be used during post-processign
                if lyr in ["TemporalFusionTransformer", "TFT"]:
                    outs, self.TemporalFusionTransformer_attentions = outs

                if lyr_args['named_outs'] is not None:
                    if isinstance(outs, list):
                        assert len(lyr_args['named_outs']) == len(outs)
                        for name, out_tensor in zip(lyr_args['named_outs'], outs):
                            cache[name] = out_tensor
                    else:
                        cache[lyr_args['named_outs']] = outs

            if is_input_tensor:
                is_input_tensor = False
            else:
                cache[lyr] = outs
            prev_output_name = lyr

        return outs

    def treat_casted_inputs(self, casted_inputs):
        if isinstance(casted_inputs, tuple) or isinstance(casted_inputs, list):
            for in_tensor, orig_in_name in zip(casted_inputs, self.input_layer_names):
                assign_dummy_name(in_tensor, orig_in_name)
        elif isinstance(casted_inputs, dict):
            names_to_assign = self.input_layer_names
            if isinstance(names_to_assign, list):
                assert len(names_to_assign) == len(casted_inputs)
                for new_name, (_input, in_tensor) in zip(names_to_assign, casted_inputs.items()):
                    assign_dummy_name(in_tensor, new_name)
            else:
                raise ValueError
        else:
            name_to_assign = self.input_layer_names
            if isinstance(name_to_assign, list):
                if len(name_to_assign) == 1:
                    name_to_assign = name_to_assign[0]
                else:
                    raise ValueError
            assign_dummy_name(casted_inputs, name_to_assign)

        return

    def call_115(self, inputs, training=None, mask=None, run_call=True):
        outs = inputs

        # inputs can be a list of tensors
        if isinstance(inputs, list):
            cache = {i.name.split(':')[0]: i for i in inputs}

        # if inputs is a list, then just save it in cache
        elif isinstance(inputs, dict):
            cache = inputs

        # inputs can be a list of tensors but as a ListWrapper
        elif inputs.__class__.__name__ == "Listwrapper":
            cache = {i.name.split(':')[0]: i for i in inputs}

        elif isinstance(inputs, tuple):
            cache = {i.name.split(':')[0]: i for i in inputs}

        # hopefully this is just one tensor
        else:
            cache = {inputs.name.split(':')[0]: inputs}

        # todo keep tensor chache and layers cache separate
        input_tensor = False
        for idx, (lyr, lyr_args) in enumerate(self.initiated_layers.items()):

            lyr = lyr.split(':')[0]  # todo, this should not have been added

            if isinstance(lyr_args['layer'], tf.Tensor) or idx == 0 or is_input(lyr_args['layer']):
                # this must be an input layer
                # assert is_input(lyr_args['layer'])
                if isinstance(inputs, list):
                    assert all([is_input(_input) for _input in inputs])
                if isinstance(inputs, tuple):
                    assert all([is_input(_input) for _input in inputs])
                # else:
                #     assert is_input(inputs)
                input_tensor = True
                # don't use the tf.keras.Input from self.initiated_layers

            elif lyr_args['tf_name'] in ['TimeDistributed', 'Bidirectional']:
                # no need to call wrapper layer so move to next iteration
                continue
            else:
                _inputs = lyr_args.get('inputs', None)

                # inputs have not been explicitly defined by the user so just use previous output
                if _inputs is None:
                    _inputs = prev_output_name

                call_args, add_args = get_call_args(_inputs, cache, lyr_args['call_args'], lyr)

                # call the initiated layer
                outs = lyr_args['layer'](call_args, **add_args)

                # if the layer is TFT, we need to extract the attention components
                # so that they can be used during post-processign
                if lyr in ["TemporalFusionTransformer", "TFT"]:
                    outs, self.TemporalFusionTransformer_attentions = outs

                if lyr_args['named_outs'] is not None:
                    if isinstance(outs, list):
                        assert len(lyr_args['named_outs']) == len(outs)
                        for name, out_tensor in zip(lyr_args['named_outs'], outs):
                            cache[name] = out_tensor
                    else:
                        cache[lyr_args['named_outs']] = outs

            if input_tensor:
                input_tensor = False  # cache[_tensor.name] = _tensor
            else:
                cache[lyr] = outs
            prev_output_name = lyr

        return outs

    def forward(self, *inputs: Any, **kwargs: Any):
        """implements forward pass implementation for pytorch based NN models."""
        outs = inputs

        # if inputs is a list, then just save it in cache
        if isinstance(inputs, dict):
            cache = inputs

        # inputs can be a list of tensors but as a ListWrapper
        elif inputs.__class__.__name__ == "Listwrapper":
            cache = {i.name.split(':')[0]: i for i in inputs}

        elif isinstance(inputs, tuple) or isinstance(inputs, list):
            cache = {}
            for idx, i in enumerate(inputs):
                _name = i.name if i.name is not None else f'input_{idx}'
                cache[_name] = i

        # hopefully this is just one tensor
        else:
            cache = {inputs.name: inputs}

        for idx, (lyr_name, lyr_args) in enumerate(self.initiated_layers.items()):
            lyr = lyr_args['layer']
            named_outs = lyr_args['named_outs']
            call_args = lyr_args['call_args']
            _inputs = lyr_args['inputs']

            if idx == 0:
                assert isinstance(inputs, tuple) and len(inputs) == 1
                _inputs = 'input_0'

            if _inputs is None:
                _inputs = prev_output_name

            call_args, add_args = get_call_args(_inputs, cache, call_args, lyr_name)

            # actuall call
            outs = lyr(call_args, **add_args)

            if named_outs is not None:
                if isinstance(outs, list):
                    assert len(named_outs) == len(outs)
                    for name, out_tensor in zip(named_outs, outs):
                        if name in cache:
                            raise ValueError(f"Duplicate layer found with name {name}")
                        cache[name] = out_tensor
                if isinstance(outs, tuple):
                    assert len(named_outs) == len(outs)
                    for name, out_tensor in zip(named_outs, outs):
                        if name in cache:
                            raise ValueError(f"Duplicate layer found with name {name}")
                        cache[name] = out_tensor
                else:
                    cache[named_outs] = outs

            cache[lyr_name] = outs
            inputs = outs
            prev_output_name = lyr_name

        return outs

    def build(self, input_shape):

        self.print_info()

        if self.category == "DL" and K.BACKEND == 'tensorflow':
            # Initialize the graph
            self._is_graph_network = True
            self._init_graph_network(
                inputs=self._input_lyrs(),
                outputs=self.output_lyrs
            )

            super().compile(
                loss=self.loss(), optimizer=self.get_optimizer(), metrics=self.get_metrics())

            self.info['model_parameters'] = self.trainable_parameters()

            if self.verbosity > 0:
                if 'tcn' in self.config['model']['layers']:
                    if not hasattr(self, '_layers'):
                        setattr(self, '_layers', self.layers)
                    tcn.tcn_full_summary(self, expand_residual_blocks=True)
                else:
                    self.summary()

            if self.verbosity >= 0:  # if verbosity is -ve then don't plot this
                self.plot_model(self)

        elif self.category == "ML":
            self.build_ml_model()

        if not getattr(self, 'from_check_point', False):
            # fit may fail so better to save config before as well. This will be overwritten once the fit is complete
            self.save_config()

        self.update_info()
        return

    def first_layer_shape(self):
        """ instead of tuple, returning a list so that it can be moified if needed"""
        if K.BACKEND == 'pytorch':
            if self.lookback == 1:
                return [-1, self.num_ins]
            else:
                return [-1, self.lookback, self.num_ins]

        if self.num_input_layers > 1:
            shapes = {}
            for lyr in self.inputs:
                shapes[lyr.name] = lyr.shape
            return shapes
        shape = []
        for idx, d in enumerate(self.nn_layers()[0].input.shape):
            if int(tf.__version__[0]) == 1:
                if isinstance(d, tf.Dimension):  # for tf 1.x
                    d = d.value

            if idx == 0:  # the first dimension must remain undefined so that the user may define batch_size
                d = -1
            shape.append(d)
        return shape

    def fit(self, *args, **kwargs):
        # this function is necessary here so that self.fit does not directly call keras.Model.fit
        # we need to pre-process the data before feeding it to keras.fit
        return self.call_fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.call_evaluate(*args, **kwargs)

    def predict(self,
                *args, **kwargs
                ):
        return self.call_predict(*args, **kwargs)

    def loss_name(self):
        return self.loss

    @classmethod
    def from_config(cls, *args, **kwargs):
        """This method primarily behaves like `from_config` of BaseModel. However,
        it can also be used like `from_config` of the underlying Model such as
        `from_config` of tf.keras.Model.
        # todo test from_config with keras
        """
        _config = args

        if isinstance(args, tuple):  # multiple non-keyword arguments were provided
            if len(args) > 0:
                _config = args[0]

            else:
                _config = kwargs['config_path']
                kwargs.pop('config_path')

        local = False
        if 'make_new_path' in kwargs:
            local = True
        elif isinstance(_config, str) and os.path.isfile(_config):
            local = True
        elif isinstance(_config, dict) and "category" in _config:
            local = True

        if local:
            config = None
            config_path = None

            # we need to build ai4water's Model class
            if isinstance(_config, dict):
                config = _config
            else:
                config_path = _config
            return BaseModel._get_config_and_path(
                cls,
                config=config,
                config_path=config_path,
                **kwargs
            )

        # tf1.15 has from_config so call it
        return super().from_config(*args, **kwargs)

    def fit_pytorch(self,  x, **kwargs):
        """Trains the pytorch model."""

        history = self.torch_learner.fit(x, **kwargs)

        setattr(self, 'history', history)
        return history

    def predict_pytorch(self, x, **kwargs):
        from .models.torch.utils import to_torch_dataset
        from torch.utils.data import DataLoader

        if isinstance(x, torch.utils.data.Dataset):
            dataset = x
        elif isinstance(x, np.ndarray):
            dataset = to_torch_dataset(x=x)
        elif isinstance(x, list) and len(x) == 1:
            dataset = to_torch_dataset(x[0])
        else:
            raise ValueError

        data_loader = DataLoader(dataset, batch_size=self.config['batch_size'])

        predictions = []
        for i, batch_x in enumerate(data_loader):

            y_pred_ = self(batch_x.float())
            predictions.append(y_pred_.detach().numpy())

        return np.concatenate(predictions, axis=0)


def is_input(tensor, name=''):
    _is_input = False
    if int(''.join(tf.__version__.split('.')[0:2]).ljust(3, '0')) < 240:
        # Each tensor (except TimeDistributed) has .op.inputs attribute, which is empty
        # if a tensor represents output of Input layer.

        if name != "TimeDistributed" and hasattr(tensor, 'op'):
            if hasattr(tensor.op, 'inputs'):
                _ins = tensor.op.inputs
                if len(_ins) == 0:
                    _is_input = True
    # # not sure if this is the proper way of checking if a layer receives an input or not!
    elif hasattr(tensor, '_keras_mask'):
        _is_input = True

    return _is_input


class NumpyVersionException(Exception):
    def __init__(self, action):
        self.action = action
        super().__init__(self.msg())

    def msg(self):
        return f"""
                version {np.__version__} of numpy is not compatible with tf version {tf.__version__}
                {self.action} numpy version."""


def assign_dummy_name(tensor, dummy_name):
    if tf.executing_eagerly():
        setattr(tensor, '__dummy_name', dummy_name)
    else:
        if "CAST" in tensor.name.upper() or "IteratorGetNext" in tensor.name:
            setattr(tensor, '__dummy_name', dummy_name)
            print(f"assigning name {dummy_name} to {tensor.name} with shape {getattr(tensor, 'shape', None)}")
        else:
            setattr(tensor, '__dummy_name', tensor.name)
