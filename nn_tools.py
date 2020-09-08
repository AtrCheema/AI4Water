from models.global_variables import keras, tcn, ACTIVATION_FNS, ACTIVATION_LAYERS

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
    scalers = {}
    cnn_counter = 0
    lstm_counter = 0
    act_counter = 0
    time_dist_counter = 0
    conv2d_lstm_counter = 0
    dense_counter = 0
    run_paras = AttributeNotSetYet("You must define the `run_paras` method first")


class NN(AttributeStore):

    def __init__(self, data_config:dict,
                 nn_config:dict
                 ):
        self.data_config = data_config
        self.nn_config=  nn_config

        super(NN, self).__init__()

    @property
    def lookback(self):
        return self.data_config['lookback']

    def conv_lstm_model(self, enc_config:dict, dec_config:dict, outs:int):
        """
         basic structure of a ConvLSTM based encoder decoder model.
        """
        sub_seq = self.nn_config['subsequences']
        sub_seq_lens = int(self.lookback / sub_seq)

        inputs = keras.layers.Input(shape=(sub_seq, 1, sub_seq_lens, self.ins))

        enc_outs = self.conv2d_lstm(enc_config, inputs)

        interm_outs = keras.layers.RepeatVector(self.outs)(enc_outs)

        lstm_outs = self.add_lstm(interm_outs, dec_config)

        predictions = keras.layers.Dense(outs)(lstm_outs)

        return inputs, predictions

    def add_lstm(self, inputs, config):

        self.lstm_counter += self.lstm_counter

        if 'name' not in config:
            config['name'] = 'lstm_lyr_' + str(self.lstm_counter)

        # check whether to apply any activation layer or not.
        act_layer = check_layer_existence(config, 'act_layer')

        config, activation = check_act_fn(config)

        lstm_activations = layers.LSTM(**config)(inputs)
        config['activation'] = activation

        # apply activation layer if applicable
        lstm_activations = self.add_act_layer(act_layer, lstm_activations)

        return lstm_activations

    def conv2d_lstm(self, config:dict, inputs):
        """
        Adds a CONVLSTM2D layer. It is possible to add activation layer after conv2d layer.
        """
        self.conv2d_lstm_counter += self.conv2d_lstm_counter
        if 'name' not in config:
            config['name'] = 'conv_lstm_' + str(self.conv2d_lstm_counter)

        # checks whether to apply activation layer at the end or not
        act_layer = check_layer_existence(config, 'act_layer')

        config, activation = check_act_fn(config)

        conv_lstm_outs = keras.layers.ConvLSTM2D(**config)(inputs)

        # apply activation layer if applicable
        conv_lstm_outs = self.add_act_layer(act_layer, conv_lstm_outs)

        return  keras.layers.Flatten()(conv_lstm_outs)


    def add_1d_cnn(self, inputs, config: dict):
        """
        adds a 1d CNN collowed by an optional activation layer, max pool layer and then flattens the output
        Fore more options in in `config` file see
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
        """
        self.cnn_counter += 1
        if 'name' in config:
            rand_name = config['name']
        else:
            rand = '1dcnn_' + str(self.cnn_counter)
            rand_name = rand
            config['name'] = rand

        # check whether to apply any activation layer or not.
        act_layer = check_layer_existence(config, 'act_layer')

        config, activation = check_act_fn(config)

        max_pool_size = config.pop('max_pool_size')

        cnn_activations = layers.Conv1D(**config)(inputs)
        config['activation'] = activation

        # apply activation layer if applicable
        cnn_activations = self.add_act_layer(act_layer, cnn_activations)

        max_pool_lyr = layers.MaxPooling1D(pool_size=max_pool_size,
                                           name='max_pool_lyr_'+rand_name)(cnn_activations)

        flat_lyr = layers.Flatten(name='flat_lyr_'+rand_name)(max_pool_lyr)

        return flat_lyr

    def cnn_model(self, cnn_config:dict, outs:int):
        """ basic structure of a simple CNN based model"""

        inputs = layers.Input(shape=(self.lookback, self.ins))

        # check whether to apply any activation layer or not.
        act_layer = check_layer_existence(cnn_config, 'act_layer')

        cnn_configs = check_cnn_config(cnn_config)

        cnn_outputs = inputs
        cnn_inputs = inputs
        for cnn in cnn_configs:
            cnn_outputs = self.add_1d_cnn(cnn_inputs, cnn)
            cnn_inputs = cnn_outputs

        # apply activation layer if applicable
        cnn_outputs = self.add_act_layer(act_layer, cnn_outputs)

        predictions = self.dense_layers(cnn_outputs, outs)

        return inputs, predictions

    def simple_lstm(self, lstm_config:dict, outs):
        """ basic structure of a simple LSTM based model"""

        inputs = layers.Input(shape=(self.lookback, self.ins))

        # check whether to apply any activation layer or not.
        act_layer = check_layer_existence(lstm_config, 'act_layer')

        lstm_configs = check_lstm_config(lstm_config)

        lstm_activations = inputs
        lstm_inputs = inputs
        for lstm in lstm_configs:
            lstm_activations = self.add_lstm(lstm_inputs, lstm)
            lstm_inputs = lstm_activations

        # apply activation layer if applicable
        lstm_activations = self.add_act_layer(act_layer, lstm_activations)

        predictions = self.dense_layers(lstm_activations, outs)

        return inputs, predictions

    def add_time_dist_cnn(self, cnn_config:dict, inputs):

        # check whether to apply any activation layer or not.
        act_layer = check_layer_existence(cnn_config, 'act_layer')

        cnn_config, activation = check_act_fn(cnn_config)

        if 'name' not in cnn_config:
            self.cnn_counter += 1
            cnn_config['name'] = 'td_cnn_' + str(self.cnn_counter)

        self.time_dist_counter += 1
        name = 'time_dist_' + str(self.time_dist_counter)

        cnn_lyr = layers.TimeDistributed(layers.Conv1D(**cnn_config),
                                         name=name)(inputs)
        cnn_config['activation'] = activation

        # apply activation layer if applicable
        cnn_lyr = self.add_act_layer(act_layer, cnn_lyr, True)

        return cnn_lyr

    def cnn_lstm(self, cnn_config:dict,  lstm_config:dict,  outs):
        """ blue print for developing a CNN-LSTM model.
        :param cnn_config: con contain input arguments for one or more than one cnns, all given as dictionaries
        :param lstm_config: cna contain input arguments for one or more than one LSTM layers, all given as dictionaries
        :param outs:, `int`, `dict` containing config for multiple dense layers
        It is possible to insert activation layer after each cnn/lstm and/or after whole lstm/cnn block
        """
        timesteps = self.lookback // self.nn_config['subsequences']

        inputs = layers.Input(shape=(None, timesteps, self.ins))

        # check whether to apply any activation layer or not.
        act_layer = check_layer_existence(cnn_config, 'act_layer')

        cnn_configs = check_cnn_config(cnn_config)

        cnn_outputs = inputs
        cnn_inputs = inputs
        for cnn in cnn_configs:
            cnn_outputs = self.add_time_dist_cnn(cnn, cnn_inputs)
            cnn_inputs = cnn_outputs

        # apply activation layer if applicable
        cnn_outputs = self.add_act_layer(act_layer, cnn_outputs, True)

        max_pool_lyr = layers.TimeDistributed(layers.MaxPooling1D(pool_size=cnn_config['max_pool_size']))(cnn_outputs)
        flat_lyr = layers.TimeDistributed(layers.Flatten())(max_pool_lyr)

        # check whether to apply any activation layer or not.
        act_layer = check_layer_existence(lstm_config, 'act_layer')

        lstm_configs = check_lstm_config(lstm_config)

        lstm_activations = flat_lyr
        lstm_inputs = flat_lyr
        for lstm in lstm_configs:
            lstm_activations = self.add_lstm(lstm_inputs, lstm)
            lstm_inputs = lstm_activations

        # apply activation layer if applicable
        lstm_activations = self.add_act_layer(act_layer, lstm_activations, True)

        predictions = self.dense_layers(lstm_activations, outs)

        return inputs, predictions

    def dense_layers(self, dense_inputs, out_config):
        """ adds one or more dense layers following their configurations in `out_config`.
        out_config, can be `int`, list or dictionary.
        out_config={1:{units=1}}  one dense layer which outputs 1
        out_config={8: {units=8}, 4: {units=2}, 1: {units=1, use_bias=False}}  adds three dense layers.
        For more options of a dense layer see https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        """
        out_config = self.check_dense_config(out_config)
        dense_outs = None
        for dense_units, dense_config in out_config.items():
            self.dense_counter += 1
            dense_config['name'] = 'dense_' + str(self.dense_counter)
            dropout_lyr = check_layer_existence(dense_config, 'dropout_layer')
            dense_config, activation = check_act_fn(dense_config)
            dense_outs = layers.Dense(**dense_config)(dense_inputs)
            dense_config['activation'] = activation
            dense_outs = self.add_dropout_layer(dropout_lyr, dense_outs)
            dense_inputs = dense_outs

        return dense_outs

    def lstm_cnn(self, lstm_config:dict, cnn_config:dict, outs:int):
        """ basic structure of a simple LSTM -> CNN based model"""

        inputs = layers.Input(shape=(self.lookback, self.ins))

        lstm_config['return_sequences'] = True   # forcing it to be True

        lstm_activations = self.add_lstm(inputs, lstm_config)

        cnn_outputs = self.add_1d_cnn(lstm_activations, cnn_config)

        predictions = layers.Dense(outs)(cnn_outputs)

        return inputs, predictions

    def lstm_autoencoder(self, enc_config:dict, dec_config:dict, composite:bool, outs:int):
        """
        basic structure of an LSTM autoencoder
        """
        inputs = layers.Input(shape=(self.lookback, self.ins))

        # build encoder
        encoder = self.add_lstm(inputs, enc_config)

        # define predict decoder
        decoder1 = layers.RepeatVector(self.lookback-1)(encoder)
        decoder1 = self.add_lstm(decoder1, dec_config)
        decoder1 = layers.Dense(outs)(decoder1)
        if composite:
            # define reconstruct decoder
            decoder2 = layers.RepeatVector(self.lookback)(encoder)
            decoder2 = layers.LSTM(100, activation='relu', return_sequences=True)(decoder2)
            decoder2 = layers.TimeDistributed(layers.Dense(self.ins))(decoder2)

            outputs = [decoder1, decoder2]
        else:
            outputs = decoder1

        return inputs, outputs

    def tcn_model(self, tcn_options:dict, outs:int):
        """
        basic tcn model
        """
        tcn_options['return_sequences'] = False

        inputs = layers.Input(batch_shape=(None, self.lookback, self.ins))

        tcn_outs = tcn.TCN(**tcn_options)(inputs)  # The TCN layers are here.
        predictions = layers.Dense(outs)(tcn_outs)

        return inputs,predictions

    def add_dropout_layer(self, dropout, inputs):
        outs = inputs
        if dropout is not None:
            assert isinstance(dropout, float)
            outs = layers.Dropout(dropout)(inputs)

        return outs

    def add_act_layer(self, act_layer, inputs, time_distributed:bool=False):
        # this is used if we want to apply activation as a separate layer

        outputs = inputs
        if act_layer is not None:
            assert isinstance(act_layer, str)
            self.act_counter += 1
            name = 'act_' + str(self.act_counter)
            if time_distributed:
                outputs = layers.TimeDistributed(ACTIVATION_LAYERS[act_layer.upper()](name=name))(inputs)
            else:
                outputs = ACTIVATION_LAYERS[act_layer.upper()](name=name)(inputs)

        return outputs

    def check_dense_config(self, config)->dict:
        """ possible values of config are
        1
        [1,2,3]
        {1: {activation='tanh'}, 2: {activatio: 'relu'}, 2: {activation: 'leakyrelu'}
        """
        if isinstance(config, int):
            out_config = {config: {'units': config}}
        elif isinstance(config, list):
            out_config = {}
            for i in config:
                if isinstance(i, int):
                    out_config[i] = {'units':i}
                else:
                    raise TypeError("config for each dense layer must be dictionary within one dictionary not list of dictionaries")
        elif isinstance(config, dict):
            out_config =  config
        else:
            raise TypeError("Unknown configuration for dense layer")

        return out_config



def check_lstm_config(lstm_config:dict)->list:

    if 'n_layers' in lstm_config:
        n_lstms = lstm_config['n_layers']
        assert len(lstm_config) > n_lstms, "{} lstm configs provided but cnofig says add {}".format(
            len(lstm_config), n_lstms)
        lstm_configs = []
        for k, v in lstm_config.items():
            if k not in ['n_layers']:
                assert isinstance(v, dict), "cnn config must be of type `dict` but {} provided".format(v)
                lstm_configs.append(v)
    else:
        lstm_configs = [lstm_config]

    return lstm_configs


def check_layer_existence(config:dict, layer_name:str):
    layer_existence = None
    if layer_name in config:
        layer_existence = config.pop(layer_name)
    return layer_existence


def check_act_fn(config:dict):
    """ it is possible that the config file does not have activation argument or activation is None"""
    activation = None
    if 'activation' in config:
        activation = config['activation']
    if activation is not None:
        assert isinstance(activation, str)
        config['activation'] = ACTIVATION_FNS[activation.upper()]

    return config, activation

def check_cnn_config(cnn_config:dict)->list:
    """ checks whether the `cnn_config` dictionary contains configuration for one cnn layer or multiple cnn layers
    """

    if 'n_layers' in cnn_config:
        n_cnns = cnn_config['n_layers']
        assert len(cnn_config) > n_cnns + 1, "{} cnn configs provided but cnofig says add {}".format(
            len(cnn_config) - 1, n_cnns)
        cnn_configs = []
        for k, v in cnn_config.items():
            if k not in ['n_layers', 'max_pool_size']:
                assert isinstance(v, dict), "cnn config must be of type `dict` but {} provided".format(v)
                cnn_configs.append(v)
    else:
        cnn_configs = [cnn_config]

    return cnn_configs