"""

This code is almost exact copy and past of keract library except few changes
in _get_nodes function where I added one if else statement The original code at
https://github.com/philipperemy/keract has following MIT licence as of 24 Nov 2020.

MIT License

Copyright (c) 2019 Philippe Rémy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import OrderedDict

from .backend import tf, keras

if keras is not None:
    K = keras.backend
    Sequential = tf.keras.Sequential
    Model = tf.keras.models.Model
else:
    K, Sequential, Model = None, None, None


def is_placeholder(n):
    return (hasattr(n, '_op') and n._op.type == 'Placeholder') or '_input' in str(n) or 'input' in str(n)

def n_(node, output_format, nested=False, module=None):
    if isinstance(node, list):
        node_name = '_'.join([str(n.name) for n in node])
    else:
        node_name = str(node.name)
    if module is not None and nested:
        node_name = module.name + '/' + node_name
    if output_format == 'simple' and ':' in node_name:
        return node_name.split(':')[0]
    elif output_format == 'full' and hasattr(node, 'output'):
        return node.output.name
    return node_name


def _evaluate(model: Model, nodes_to_evaluate, x, y=None, auto_compile=False):

    if not model._is_compiled:
        if auto_compile:
            model.compile(loss='mse', optimizer='adam')
        else:
            raise Exception('Compilation of the model required.')

    def eval_fn(k_inputs):
        try:
            return K.function(k_inputs, nodes_to_evaluate)(model._standardize_user_data(x, y))
        except AttributeError:  # one way to avoid forcing non eager mode.
            if y is None:  # tf 2.3.0 upgrade compatibility.
                return K.function(k_inputs, nodes_to_evaluate)(x)
            return K.function(k_inputs, nodes_to_evaluate)((x, y))  # although works.
        except ValueError as e:
            print('Run it without eager mode tf.compat.v1.disable_eager_execution()')
            raise e

    try:
        return eval_fn(model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    except Exception:
        return eval_fn(model._feed_inputs)


def get_gradients_of_trainable_weights(model, x, y):
    """
    Get the gradients of trainable_weights for the kernel and the bias nodes for all filters in each layer.
    Trainable_weights gradients are averaged over the minibatch.
    :param model: keras compiled model
    :param x: inputs for which gradients are sought (averaged over all inputs if batch_size > 1)
    :param y: outputs for which gradients are sought
    :return: dict mapping layers to corresponding gradients (filter_h, filter_w, in_channels, out_channels)
    """
    nodes = OrderedDict([(n.name, n) for n in model.trainable_weights])
    return _get_gradients(model, x, y, nodes)


def get_gradients_of_activations(model, x, y, layer_names=None, output_format='simple', nested=False):
    """
    Get gradients of the outputs of the activation functions, regarding the loss.
    Intuitively, it shows how your activation maps change over a tiny modification of the loss.
    :param model: keras compiled model or one of ['vgg16', 'vgg19', 'inception_v3', 'inception_resnet_v2',
    'mobilenet_v2', 'mobilenetv2'].
    :param x: Model input (Numpy array). In the case of multi-inputs, x should be of type List.
    :param y: Model target (Numpy array). In the case of multi-inputs, y should be of type List.
    :param layer_names: (optional) Single name of a layer or list of layer names for which activations should be
    returned. It is useful in very big networks when it is computationally expensive to evaluate all the layers/nodes.
    :param output_format: Change the output dictionary key of the function.
    - 'simple': output key will match the names of the Keras layers. For example Dense(1, name='d1') will
    return {'d1': ...}.
    - 'full': output key will match the full name of the output layer name. In the example above, it will
    return {'d1/BiasAdd:0': ...}.
    - 'numbered': output key will be an index range, based on the order of definition of each layer within the model.
    :param nested: (optional) If set, will move recursively through the model definition to retrieve nested layers.
                Recursion ends at leaf layers of the model tree or at layers with their name specified in layer_names.

                E.g., a model with the following structure

                -layer1
                    -conv1
                    ...
                    -fc1
                -layer2
                    -fc2

                ... yields a dictionary with keys 'layer1/conv1', ..., 'layer1/fc1', 'layer2/fc2'.
                If layer_names = ['layer2/fc2'] is specified, the dictionary will only hold one key 'layer2/fc2'.

                The layer names are generated by joining all layers from top level to leaf level with the separator '/'.
    :return: Dict {layer_names (specified by output_format) -> activation of the layer output/node (Numpy array)}.
    """
    nodes = OrderedDict()
    _get_nodes(model, nodes, output_format, nested=nested, layer_names=layer_names)
    return _get_gradients(model, x, y, nodes)


def _get_gradients(model, x, y, nodes):
    if model.optimizer is None:
        raise Exception('Please compile the model first. The loss function is required to compute the gradients.')

    nodes_names = nodes.keys()
    nodes_values = nodes.values()

    try:
        if not hasattr(model, 'total_loss'):
            raise Exception('Disable TF eager mode to use get_gradients.\n'
                            'Add this command at the beginning of your script:\n'
                            'tf.compat.v1.disable_eager_execution()')
        grads = model.optimizer.get_gradients(model.total_loss, nodes_values)
    except ValueError as e:
        if 'differentiable' in str(e):
            # Probably one of the gradients operations is not differentiable...
            grads = []
            differentiable_nodes = []
            for n in nodes_values:
                try:
                    grads.extend(model.optimizer.get_gradients(model.total_loss, n))
                    differentiable_nodes.append(n)
                except ValueError:
                    pass
            # nodes_values = differentiable_nodes
        else:
            raise e

    gradients_values = _evaluate(model, grads, x, y)

    return OrderedDict(zip(nodes_names, gradients_values))


def _get_nodes(module, nodes, output_format, nested=False, layer_names=None, depth=0):
    def update_node(n):
        is_node_a_model = isinstance(n, (Model, Sequential))
        if not is_placeholder(n):
            if is_node_a_model and nested:
                return
            try:
                mod = None if depth == 0 else module
                name = n_(n, output_format, nested, mod)
                if layer_names is None or name in layer_names:
                    if is_node_a_model:
                        if hasattr(n, '_layers'):
                            output = n._layers[-1].output
                        else:
                            output = n.layers[-1].output
                    else:
                        output = n.output
                    nodes.update({name: output})
            except AttributeError:
                pass

    try:
        layers = module._layers if hasattr(module, '_layers') else module.layers
    except AttributeError:
        return
    for layer in layers:
        update_node(layer)
        if nested:
            _get_nodes(layer, nodes, output_format, nested, layer_names, depth + 1)


def get_activations(model, x, layer_names=None, nodes_to_evaluate=None,
                    output_format='simple', nested=False, auto_compile=True):
    """
    Fetch activations (nodes/layers outputs as Numpy arrays) for a Keras model and an input X.
    By default, all the activations for all the layers are returned.
    :param model: Keras compiled model or one of ['vgg16', 'vgg19', 'inception_v3', 'inception_resnet_v2',
    'mobilenet_v2', 'mobilenetv2', ...].
    :param x: Model input (Numpy array). In the case of multi-inputs, x should be of type List.
    :param layer_names: (optional) Single name of a layer or list of layer names for which activations should be
    returned. It is useful in very big networks when it is computationally expensive to evaluate all the layers/nodes.
    :param nodes_to_evaluate: (optional) List of Keras nodes to be evaluated. Useful when the nodes are not
    in model.layers.
    :param output_format: Change the output dictionary key of the function.
    - 'simple': output key will match the names of the Keras layers. For example Dense(1, name='d1') will
    return {'d1': ...}.
    - 'full': output key will match the full name of the output layer name. In the example above, it will
    return {'d1/BiasAdd:0': ...}.
    - 'numbered': output key will be an index range, based on the order of definition of each layer within the model.
    :param nested: If specified, will move recursively through the model definition to retrieve nested layers.
                Recursion ends at leaf layers of the model tree or at layers with their name specified in layer_names.

                E.g., a model with the following structure

                -layer1
                    -conv1
                    ...
                    -fc1
                -layer2
                    -fc2

                ... yields a dictionary with keys 'layer1/conv1', ..., 'layer1/fc1', 'layer2/fc2'.
                If layer_names = ['layer2/fc2'] is specified, the dictionary will only hold one key 'layer2/fc2'.

                The layer names are generated by joining all layers from top level to leaf level with the separator '/'.
    :param auto_compile: If set to True, will auto-compile the model if needed.
    :return: Dict {layer_name (specified by output_format) -> activation of the layer output/node (Numpy array)}.
    """
    layer_names = [layer_names] if isinstance(layer_names, str) else layer_names
    # print('Layer names:', layer_names)
    nodes = OrderedDict()
    if nodes_to_evaluate is None:
        _get_nodes(model, nodes, output_format, nested, layer_names, auto_compile)
    else:
        if layer_names is not None:
            raise ValueError('Do not specify a [layer_name] with [nodes_to_evaluate]. It will not be used.')
        nodes = OrderedDict([(n_(node, 'full'), node) for node in nodes_to_evaluate])

    if len(nodes) == 0:
        if layer_names is not None:
            network_layers = ', '.join([layer.name for layer in model.layers])
            raise KeyError('Could not find a layer with name: [{}]. '
                           'Network layers are [{}]'.format(', '.join(layer_names), network_layers))
        else:
            raise ValueError('Nodes list is empty. Or maybe the model is empty.')

    # The placeholders are processed later (Inputs node in Keras). Due to a small bug in tensorflow.
    input_layer_outputs = []
    layer_outputs = OrderedDict()

    for key, node in nodes.items():
        if isinstance(node, list):
            for nod in node:
                if not is_placeholder(nod):
                    if key not in layer_outputs:
                        layer_outputs[key] = []
                    layer_outputs[key].append(nod)
        else:
            if not is_placeholder(node):
                layer_outputs.update({key: node})
    if nodes_to_evaluate is None or (layer_names is not None) and \
            any([n.name in layer_names for n in model.inputs]):
        input_layer_outputs = list(model.inputs)

    if len(layer_outputs) > 0:
        activations = _evaluate(model, layer_outputs.values(), x, y=None)
    else:
        activations = {}

    def craft_output(output_format_):
        inputs = [x] if not isinstance(x, list) else x
        activations_inputs_dict = OrderedDict(
            zip([n_(output, output_format_) for output in input_layer_outputs], inputs))
        activations_dict = OrderedDict(zip(layer_outputs.keys(), activations))
        result_ = activations_inputs_dict.copy()
        result_.update(activations_dict)

        if output_format_ == 'numbered':
            result_ = OrderedDict([(i, v) for i, (k, v) in enumerate(result_.items())])
        return result_

    result = craft_output(output_format)
    if layer_names is not None:  # extra check.
        result = {k: v for k, v in result.items() if k in layer_names}
    if nodes_to_evaluate is not None and len(result) != len(nodes_to_evaluate):
        result = craft_output(output_format_='full')  # collision detected in the keys.

    return result
