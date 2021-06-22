from typing import Any
from collections import OrderedDict

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from ._main import BaseModel
from AI4Water.tf_attributes import ACTIVATION_LAYERS, tcn
from .nn_tools import get_call_args

from .backend import BACKEND, tf, torch

if BACKEND =='tensorflow' and tf is not None:
    MODEL = tf.keras.Model
    from .tf_attributes import LAYERS, OPTIMIZERS

elif BACKEND == 'pytorch' and torch is not None:
    MODEL = torch.nn.Module
    from .pytorch_attributes import LAYERS, OPTIMIZERS
    from torch.utils.data import DataLoader
    from .utils.torch_utils import to_torch_dataset

else:
    class MODEL(object): pass


class Model(MODEL, BaseModel):
    """
    Inherits from `BaseModel`
    Model class which is a subclass of keras.Model/torch.nn.Module. This class
    directly exposes all the functionalities of underlying Model. Such as `self`
    is now a keras Model or torch.nn.Module. The custom NN can be constructed
    by overwrting `initialize_layers` and `call`/`forward` methods.
    """

    def __init__(self,
                 data=None,
                 verbosity=1,
                 outputs=None,
                 inputs=None,
                 model=None,
                 path=None,
                 prefix=None,
                 **kwargs):

        """
        Initializes the layers of NN model using `initialize_layers` method. All
        other input arguments goes to `BaseModel`.

        """
        if BACKEND == 'tensorflow' and tf is not None:
            if tf.__version__ in  ["2.3.0"]:
                raise NotImplementedError(f"""
            Not implemented due to a bug in tensorflow as shown here https://github.com/tensorflow/tensorflow/issues/44646
            You can use functional API instead by using
            from AI4Water.functional import Model
            instead of 
            from AI4Water import Model
            Or change the tensorflow version. Current version is {tf.__version__}. 
            """)

        tf_kwargs = {}
        for arg in ['inputs', 'outputs']:
            if arg in kwargs:
                tf_kwargs[arg] = kwargs[arg]

        MODEL.__init__(self, **tf_kwargs)

        BaseModel.__init__(self, data=data, prefix=prefix, path=path, verbosity=verbosity, model=model,
                           outputs=outputs, inputs=inputs,
                           **kwargs)

        self.config['backend'] = BACKEND

        if torch is not None:
            from .pytorch_training import Learner
            self.torch_learner = Learner

        if self.category == "DL":
            self.initialize_layers(self.config['model']['layers'])

            if BACKEND == 'tensorflow':
                outs = self.call(self.input_lyrs)
                setattr(self, 'output_lyrs', outs)
                MODEL.__init__(self, inputs=self.input_lyrs, outputs=self.output_lyrs)

        self.build(self._get_dummy_input_shape())  # will initialize ML models or build NNs

    @property
    def torch_learner(self):
        return self._torch_learner

    @torch_learner.setter
    def torch_learner(self, x):
        """So that learner can be changed."""
        self._torch_learner = x

    @property
    def layer_names(self):
        _all_layers = []
        if self.category == "ML":
            return None
        for layer in self.layers:
            _all_layers.append(layer.name)
        return _all_layers

    @property
    def layers_in_shapes(self) -> dict:
        """ returns the shapes of inputs to all layers"""
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
        if self.category.upper() != "DL":
            return np.inf
        if BACKEND == 'pytorch':
            return 1
        else:
            return len(self.inputs)

    @property
    def input_layer_names(self) -> list:

        return [lyr.name.split(':')[0] for lyr in self.inputs]

    def _get_dummy_input_shape(self):
        shape = ()
        if BACKEND == 'tensorflow' and self.category == "DL":
            if isinstance(self.inputs, list):
                shape = self.inputs[0].shape

        return shape

    @property
    def api(self):
        return 'subclassing'

    @property
    def fit_fn(self):
        if self.category == "DL":
            if BACKEND == 'tensorflow':
                return super().fit
            elif BACKEND == 'pytorch':
                return self.fit_pytorch

        return self._model.fit  # e.g. for ML models

    @property
    def evaluate_fn(self):
        if self.category == "DL" and BACKEND == 'tensorflow':
            return super().evaluate
        return self._model.evaluate

    @property
    def predict_fn(self):
        if self.category == "DL":
            if BACKEND == 'tensorflow':
                return super().predict
            elif BACKEND == 'pytorch':
                return self.predict_pytorch
        return self._model.predict

    def initialize_layers(self, layers_config:dict, inputs=None):
        """
        Initializes the layers/weights/variables which are to be used in `forward`
        or `call` method.
        Arguments:
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

            if BACKEND == 'pytorch':

                if first_layer:
                    first_layer = False

                if callable(lyr_config):
                    lyr_initiated = lyr_config
                else:
                    lyr_initiated = LAYERS[lyr_name.upper()](**lyr_config)
                setattr(self, lyr, lyr_initiated)
                initiated_layers[lyr] = {"layer": lyr_initiated, "named_outs": named_outs, 'call_args': call_args, 'inputs': lyr_inputs}

            else:
                # may be user has defined layers without input layer, in this case add Input layer as first layer
                if first_layer:
                    if inputs is not None: # This method was called by providing it inputs.
                        assert isinstance(inputs, tf.Tensor)
                        first_layer = False # since inputs have been defined, all the layers that will be added will be next to first layer
                        layer_outputs = inputs
                        initiated_layers[layer_outputs.name] = {'layer': layer_outputs}

                    elif lyr_name.upper() != "INPUT":
                        if 'input_shape' in lyr_config:  # input_shape is given in the first layer so make input layer
                            initialized_layer = LAYERS["INPUT"](shape=lyr_config['input_shape'])
                        else:
                            # for simple dense layer based models, lookback will not be used
                            def_shape = (self.num_ins,) if self.lookback == 1 else (self.lookback, self.num_ins)
                            initialized_layer = LAYERS["INPUT"](shape=def_shape)

                        # first layer is built so next iterations will not be for first layer
                        first_layer = False
                        # put the first layer in memory to be used for model compilation
                        # add th layer which the user had specified as first layer
                        initiated_layers[initialized_layer.name] = {'layer': initialized_layer}
                        input_lyrs.append(initialized_layer)

                if lyr_inputs is None:  # The inputs to the layer have not been specified, so either it is an Input layer
                    # or it uses the previous outputs as inputs
                    if lyr_name.upper() == "INPUT":
                        # it is an Input layer, hence should not be called
                        initialized_layer = LAYERS[lyr_name.upper()](*args, **lyr_config)
                        initiated_layers[lyr_config['name']] = {'layer': initialized_layer}
                        input_lyrs.append(initialized_layer)
                    else:
                        # it is executable and uses previous outputs as inputs
                        if lyr_name.upper() in ACTIVATION_LAYERS:
                            layer_outputs = ACTIVATION_LAYERS[lyr_name.upper()](name=lyr_config['name'])
                            initiated_layers[lyr_config['name']] = {'layer': layer_outputs, 'named_outs': named_outs, 'call_args': call_args, 'inputs': lyr_inputs}
                        elif lyr_name.upper() in ['TIMEDISTRIBUTED', 'BIDIRECTIONAL']:
                            wrp_layer = LAYERS[lyr_name.upper()]
                            initiated_layers[lyr_config['name']] = {'layer': wrp_layer}  # because wrapper layer name is property
                            continue
                        elif  "LAMBDA" in lyr_name.upper():
                            # lyr_config is serialized lambda layer, which needs to be deserialized
                            initialized_layer = tf.keras.layers.deserialize(lyr_config)
                            # layers_config['lambda']['config'] still contails lambda, so we need to replace the python
                            # object (lambda) with the serialized version (lyr_config) so that it can be saved as json file.
                            layers_config[lyr]['config'] = lyr_config
                            initiated_layers[lyr_config['name']] = {'layer': initialized_layer, 'named_outs': named_outs, 'call_args': call_args, 'inputs': lyr_inputs}
                        else:
                            if wrp_layer is not None:
                                initialized_layer = wrp_layer(LAYERS[lyr_name.upper()](*args, **lyr_config))
                                initiated_layers[lyr_config['name']] = {'layer': initialized_layer, 'named_outs': named_outs,
                                                                    'call_args': call_args, 'inputs': lyr_inputs}
                                wrp_layer = None
                            else:
                                initialized_layer = LAYERS[lyr_name.upper()](*args, **lyr_config)
                                initiated_layers[lyr_config['name']] = {'layer': initialized_layer, 'named_outs': named_outs,
                                                       'call_args': call_args, 'inputs': lyr_inputs}

                else:  # The inputs to this layer have been specified so they must exist in lyr_cache.
                    # it is an executable
                    if lyr_name.upper() in ACTIVATION_LAYERS:

                        layer_outputs = ACTIVATION_LAYERS[lyr_name.upper()](name=lyr_config['name'])
                        initiated_layers[lyr_config['name']] = {'layer': layer_outputs, 'named_outs': named_outs, 'call_args': call_args,
                                               'inputs': lyr_inputs}
                    elif lyr_name.upper() in ['TIMEDISTRIBUTED', 'BIDIRECTIONAL']:
                        wrp_layer = LAYERS[lyr_name.upper()]
                        initiated_layers[lyr_config['name']] = {'layer': wrp_layer}  # because wrapper layer name is property
                        continue
                    elif "LAMBDA" in lyr_name.upper():
                        initialized_layer = tf.keras.layers.deserialize(lyr_config)
                        layers_config[lyr]['config'] = lyr_config
                        initiated_layers[lyr_config['name']] = {'layer': initialized_layer, 'named_outs': named_outs, 'call_args': call_args,
                                               'inputs': lyr_inputs}
                    else:
                        if wrp_layer is not None:
                            initialized_layer = wrp_layer(LAYERS[lyr_name.upper()](*args, **lyr_config))
                            initiated_layers[lyr_config['name']] = {'layer': initialized_layer, 'named_outs': named_outs,
                                                                'call_args': call_args, 'inputs': lyr_inputs}
                            wrp_layer = None
                        else:
                            layer_initialized = LAYERS[lyr_name.upper()](*args, **lyr_config)
                            initiated_layers[lyr_config['name']] = {'layer': layer_initialized, 'named_outs': named_outs,
                                                   'call_args': call_args, 'inputs': lyr_inputs}

                if activation is not None:  # put the string back to dictionary to be saved in config file
                    lyr_config['activation'] = activation

                first_layer = False

        # inputs = [] todo, indentify input layers
        # for k,v in lyr_cache.items():
        #     # since the model is not build yet and we have access to only output tensors of each list, this is probably
        #     # the only way to know that how many `Input` layers were encountered during the run of this method. Each
        #     # tensor (except TimeDistributed) has .op.inputs attribute, which is empty if a tensor represents output of Input layer.
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
        #     if 'op' not in dir(layer_outputs):  # layer_outputs does not have `op`, which means it has no incoming node
        #         print("Warning: the output is of Input tensor class type")

        outs = None
        #if BACKEND == 'tensorflow':
            # outs = self.call(input_lyrs)
            # setattr(self, 'output_lyrs', outs)
            # if BACKEND == 'tensorflow':
            #     ## Reinitial
            #     super(Model, self).__init__(
            #         inputs=input_lyrs,
            #         outputs=outs)
                #MODEL.__init__(self, inputs=inputs, outputs=outs)

        return input_lyrs#, outs

    def call(self, inputs, training=None, mask=None):
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

        # hopefully this is just one tensor
        else:
            cache = {inputs.name: inputs}

        # todo keep tensor chache and layers cache separate

        for lyr, lyr_args in self.initiated_layers.items():

            tf_lyr_name = lyr.split('_')[0]

            if isinstance(lyr_args['layer'], tf.Tensor):
                # this must be an input layer
                assert is_input(lyr_args['layer'])
                outs = lyr_args['layer']

            elif tf_lyr_name.upper() in ['TIMEDISTRIBUTED', 'BIDIRECTIONAL']:
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

                if lyr_args['named_outs'] is not None:
                    if isinstance(outs, list):
                        assert len(lyr_args['named_outs']) == len(outs)
                        for name, out_tensor in zip(lyr_args['named_outs'], outs):
                            cache[name] = out_tensor
                    else:
                        cache[lyr_args['named_outs']] = outs

            cache[lyr] = outs
            prev_output_name = lyr

        outs = self.maybe_add_output_layer(outs, cache)

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

            if idx==0:
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

    def get_optimizer(self):
        opt_args = self.get_opt_args()
        optimizer = OPTIMIZERS[self.config['optimizer'].upper()](**opt_args)

        return optimizer

    def build(self, input_shape):
        if self.category == "DL" and BACKEND == 'tensorflow':
            # Initialize the graph
            self._is_graph_network = True
            self._init_graph_network(
                inputs=self.input_lyrs,
                outputs=self.output_lyrs
            )

            super().compile(
                loss=self.loss(), optimizer=self.get_optimizer(), metrics=self.get_metrics())

            self.info['model_parameters'] = self.trainable_parameters()

            if self.verbosity > 0:
                if 'tcn' in self.config['model']['layers']:
                    tcn.tcn_full_summary(self._model, expand_residual_blocks=True)
                else:
                    self.summary()

            self.plot_model(self)

        elif self.category == "ML":
            self.build_ml_model()

        if not getattr(self, 'from_check_point', False):
            # fit main fail so better to save config before as well. This will be overwritten once the fit is complete
            self.save_config()

        self.update_info()
        return

    def first_layer_shape(self):
        """ instead of tuple, returning a list so that it can be moified if needed"""
        if BACKEND == 'pytorch':
            if self.lookback == 1:
                return [-1, self.ins]
            else:
                return [-1, self.lookback, self.ins]

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
        # for pre_processing
        return self.call_fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.call_evaluate(*args, **kwargs)

    def predict(self,
                *args, **kwargs
                ):
        return self.call_predict(*args, **kwargs)

    def loss_name(self):
        return self.loss

    def fit_pytorch(self,  x, **kwargs):
        """Trains the pytorch model."""


        learner = self.torch_learner(model=self,
                          batch_size=self.config['batch_size'],
                          num_epochs=self.config['epochs'],
                          shuffle=self.config['shuffle'],
                          to_monitor=self.config['metrics'],
                          patience=self.config['patience']
                          )

        history = learner.fit(x, **kwargs)

        setattr(self, 'history', history)
        return history

    def predict_pytorch(self, x, **kwargs):

        if isinstance(x, torch.utils.data.Dataset):
            dataset = x
        elif isinstance(x, np.ndarray):
            dataset = to_torch_dataset(x=x)
        elif isinstance(x, list) and len(x)==1:
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
    if name.upper() != "TIMEDISTRIBUTED" and hasattr(tensor, 'op'):
        if hasattr(tensor.op, 'inputs'):
            _ins = tensor.op.inputs
            if len(_ins) == 0:
                _is_input = True

    return _is_input
