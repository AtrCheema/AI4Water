import numpy as np

from ai4water.backend import keras
K = keras.backend
Concatenate, Add, Reshape = keras.layers.Concatenate, keras.layers.Add, keras.layers.Reshape
Input, Dense, Lambda, Subtract = keras.layers.Input, keras.layers.Dense, keras.layers.Lambda, keras.layers.Subtract
Model = keras.models.Model


class NBeats(keras.layers.Layer):
    """
    This implementation is same as that of Philip peremy_ with few modifications.
    Here NBeats can be used as a layer. The output shape will be
    (batch_size, forecast_length, input_dim)
    Some other changes have also been done to make this layer compatable with ai4water.

    Example
    -------
        >>> x = np.random.random((100, 10, 3))
        >>> y = np.random.random((100, 1))
        ...
        >>> model = Model(model={"layers":
        >>>            {"Input": {"shape": (10, 3)},
        >>>             "NBeats": {"lookback": 10, "forecast_length": 1, "num_exo_inputs": 2},
        >>>             "Flatten": {},
        >>>             "Reshape": {"target_shape": (1,1)}}},
        >>>           ts_args={'lookback':10})
        ...
        >>> model.fit(x=x, y=y.reshape(-1,1,1))

    .. _peremy:
        https://github.com/philipperemy/n-beats/tree/master/nbeats_keras)
    """
    GENERIC_BLOCK = 'generic'
    TREND_BLOCK = 'trend'
    SEASONALITY_BLOCK = 'seasonality'

    def __init__(
            self,
            units: int = 256,
            lookback: int = 10,
            forecast_len: int = 2,
            stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
            nb_blocks_per_stack=3,
            thetas_dim=(4, 8),
            share_weights_in_stack=False,
            nb_harmonics=None,
            num_inputs=1,
            num_exo_inputs=0,
            **kwargs
    ):
        """
        Initiates the Nbeats layer

        Arguments:
            units :
                Number of units in NBeats layer. It determines the size of NBeats.
            lookback:
                Number of historical time-steps used to predict next value
            forecast_len :
            stack_types :
            nb_blocks_per_stack :
            theta_dim :
            share_weights_in_stack :
            nb_harmonics :
            num_inputs:
            num_exo_inputs:
            kwargs :
        """

        if num_inputs != 1:
            raise NotImplementedError

        self.stack_types = stack_types
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = thetas_dim
        self.units = units
        self.share_weights_in_stack = share_weights_in_stack
        self.lookback = lookback
        self.forecast_length = forecast_len
        self.input_dim = num_inputs
        self.exo_dim = num_exo_inputs
        self.exo_shape = (self.lookback, self.exo_dim)
        self._weights = {}
        self.nb_harmonics = nb_harmonics
        assert len(self.stack_types) == len(self.thetas_dim)
        super().__init__(**kwargs)

    def __call__(self, inputs,  *args, **kwargs):
        """The first num_inputs from inputs will be considered as the same feature
        which is being predicted. Since the ai4water, by default does nowcasting
        instead of forecasting, and NBeats using the target/ground truth as input,
        using NBeats for nowcasting does not make sense.

        For example in case, the inputs is of shape (100, 5, 3), then
        inputs consists of three inputs and we consider the first
        input as target feature and other two as exogenous features.
        """
        e_ = {}

        if self.has_exog():
            x = inputs[..., 0:self.input_dim]
            e = inputs[..., self.input_dim:]

            for k in range(self.exo_dim):
                e_[k] = Lambda(lambda z: z[..., k])(e)
        else:
            x = inputs

        x_ = {}
        for k in range(self.input_dim):
            x_[k] = Lambda(lambda z: z[..., k])(x)

        y_ = {}

        for stack_id in range(len(self.stack_types)):
            stack_type = self.stack_types[stack_id]
            nb_poly = self.thetas_dim[stack_id]
            for block_id in range(self.nb_blocks_per_stack):
                backcast, forecast = self.create_block(x_, e_, stack_id, block_id, stack_type, nb_poly)
                for k in range(self.input_dim):
                    x_[k] = Subtract()([x_[k], backcast[k]])
                    if stack_id == 0 and block_id == 0:
                        y_[k] = forecast[k]
                    else:
                        y_[k] = Add()([y_[k], forecast[k]])

        for k in range(self.input_dim):
            y_[k] = Reshape(target_shape=(self.forecast_length, 1))(y_[k])
        if self.input_dim > 1:
            y_ = Concatenate(axis=-1)([y_[ll] for ll in range(self.input_dim)])
        else:
            y_ = y_[0]

        return y_

    def has_exog(self):
        return self.exo_dim > 0

    def _r(self, layer_with_weights, stack_id):
        # mechanism to restore weights when block share the same weights.
        # only useful when share_weights_in_stack=True.
        if self.share_weights_in_stack:
            layer_name = layer_with_weights.name.split('/')[-1]
            try:
                reused_weights = self._weights[stack_id][layer_name]
                return reused_weights
            except KeyError:
                pass
            if stack_id not in self._weights:
                self._weights[stack_id] = {}
            self._weights[stack_id][layer_name] = layer_with_weights
        return layer_with_weights

    def create_block(self, x, e, stack_id, block_id, stack_type, nb_poly):

        # register weights (useful when share_weights_in_stack=True)
        def reg(layer):
            return self._r(layer, stack_id)

        # update name (useful when share_weights_in_stack=True)
        def n(layer_name):
            return '/'.join([str(stack_id), str(block_id), stack_type, layer_name])

        backcast_ = {}
        forecast_ = {}
        d1 = reg(Dense(self.units, activation='relu', name=n('d1')))
        d2 = reg(Dense(self.units, activation='relu', name=n('d2')))
        d3 = reg(Dense(self.units, activation='relu', name=n('d3')))
        d4 = reg(Dense(self.units, activation='relu', name=n('d4')))
        if stack_type == 'generic':
            theta_b = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_b')))
            theta_f = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_f')))
            backcast = reg(Dense(self.lookback, activation='linear', name=n('backcast')))
            forecast = reg(Dense(self.forecast_length, activation='linear', name=n('forecast')))
        elif stack_type == 'trend':
            theta_f = theta_b = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_f_b')))
            backcast = Lambda(trend_model, arguments={"is_forecast": False, "backcast_length": self.lookback,
                                                      "forecast_length": self.forecast_length})
            forecast = Lambda(trend_model, arguments={"is_forecast": True, "backcast_length": self.lookback,
                                                      "forecast_length": self.forecast_length})
        else:  # 'seasonality'
            if self.nb_harmonics:
                theta_b = reg(Dense(self.nb_harmonics, activation='linear', use_bias=False, name=n('theta_b')))
            else:
                theta_b = reg(Dense(self.forecast_length, activation='linear', use_bias=False, name=n('theta_b')))
            theta_f = reg(Dense(self.forecast_length, activation='linear', use_bias=False, name=n('theta_f')))
            backcast = Lambda(seasonality_model,
                              arguments={"is_forecast": False, "backcast_length": self.lookback,
                                         "forecast_length": self.forecast_length})
            forecast = Lambda(seasonality_model,
                              arguments={"is_forecast": True, "backcast_length": self.lookback,
                                         "forecast_length": self.forecast_length})
        for k in range(self.input_dim):
            if self.has_exog():
                d0 = Concatenate()([x[k]] + [e[ll] for ll in range(self.exo_dim)])
            else:
                d0 = x[k]
            d1_ = d1(d0)
            d2_ = d2(d1_)
            d3_ = d3(d2_)
            d4_ = d4(d3_)
            theta_f_ = theta_f(d4_)
            theta_b_ = theta_b(d4_)
            backcast_[k] = backcast(theta_b_)
            forecast_[k] = forecast(theta_f_)

        return backcast_, forecast_


def linear_space(backcast_length, forecast_length, fwd_looking=True):
    ls = K.arange(-float(backcast_length), float(forecast_length), 1) / backcast_length
    if fwd_looking:
        ls = ls[backcast_length:]
    else:
        ls = ls[:backcast_length]
    return ls


def seasonality_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.get_shape().as_list()[-1]
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    t = linear_space(backcast_length, forecast_length, fwd_looking=is_forecast)
    s1 = K.stack([K.cos(2 * np.pi * i * t) for i in range(p1)], axis=0)
    s2 = K.stack([K.sin(2 * np.pi * i * t) for i in range(p2)], axis=0)
    if p == 1:
        s = s2
    else:
        s = K.concatenate([s1, s2], axis=0)
    s = K.cast(s, np.float32)
    return K.dot(thetas, s)


def trend_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.shape[-1]
    t = linear_space(backcast_length, forecast_length, fwd_looking=is_forecast)
    t = K.transpose(K.stack([t ** i for i in range(p)], axis=0))
    t = K.cast(t, np.float32)
    return K.dot(thetas, K.transpose(t))
