
__all__ = ["DLRegressionExperiments"]

import os

from ._main import Experiments
from ai4water import Model
from ai4water.hyperopt import Integer, Real, Categorical
from ai4water.utils.utils import jsonize, dateandtime_now


class DLRegressionExperiments(Experiments):
    """
    a framework for comparing several basic DL architectures for a given data.

    Any method defining a model should have the following signature:
    ```
    python
    >>> def model_<name>(self, **kwargs)
    >>>         self.param_space = ...
    >>>         self.x0 = ...
    >>>     layers = update_layers(**kwargs) # update layers with kwargs
    >>>        return self._make_return(layers, **kwargs)
    ```
    To check the available models 
    >>> exp = DLRegressionExperiments(...)
    >>> exp.models

    This class can be extended to different output shapes by overwriting 
    `:py:meth:ai4water.experiments.DLExperiments._output_layer` method.

    If learning rate, batch size, and lookback are are to be optimzied,
    their space can be specified in the following way:
    >>> exp = DLRegressionExperiments(...)
    >>> exp.lookback_space = [Integer(1, 100, name='lookback')]

    Example
    -------
    >>> from ai4water.experiments import DLRegressionExperiments
    >>> from ai4water.datasets import busan_beach
    >>> data = busan_beach()
    >>> exp = DLRegressionExperiments(
    >>> input_features = data.columns.tolist()[0:-1],
    >>> output_features = data.columns.tolist()[-1:],
    >>> epochs=300,
    >>> train_fraction=1.0,
    >>> y_transformation="log",
    >>> x_transformation="minmax",
    >>> )

    >>> exp.fit(data=data)

    """
    def __init__(self,
        param_space=None,
        x0=None,
        cases: dict = None,
        exp_name: str = None,
        num_samples: int = 5,
        verbosity: int = 1,
        **model_kws):

        self.param_space = param_space
        self.x0 = x0
        self.model_kws = model_kws

        self.lookback_space = []
        self.batch_size_space = []
        self.lr_space = []

        exp_name = exp_name or 'DLExperiments' + f'_{dateandtime_now()}'

        super().__init__(cases=cases,
                         exp_name=exp_name,
                         num_samples=num_samples,
                         verbosity=verbosity)

    @property
    def lookback_space(self):
        return self._lookback_space
    
    @lookback_space.setter
    def lookback_space(self, space):
        self._lookback_space = space

    @property
    def batch_size_space(self):
        return self._batch_size_space
    
    @batch_size_space.setter
    def batch_size_space(self, bs_space):
        self._batch_size_space = bs_space

    @property
    def lr_space(self):
        return self._lr_space
    
    @lr_space.setter
    def lr_space(self, lr_space):
        self._lr_space = lr_space

    @property
    def static_space(self):
        _space = []
        if self.lookback_space:
            _space.append(self.lookback_space)
        if self.batch_size_space:
            _space.append(self.batch_size_space)
        if self.lr_space:
            _space.append(self.lr_space)
        return _space

    @property
    def static_x0(self):
        _x0 = []
        if self.lookback_space:
            _x0.append(self.model_kws.get('lookback', 8))
        if self.batch_size_space:
            _x0.append(self.model_kws.get('batch_size', 32))
        if self.lr_space:
            _x0.append(self.model_kws.get('lr', 0.001))
        return _x0

    @property
    def mode(self):
        return "regression"

    @property
    def tpot_estimator(self):
        return None

    def _build(self, title=None, **suggested_paras):

        suggested_paras = jsonize(suggested_paras)

        verbosity = max(self.verbosity-1, 0)
        if 'verbosity' in self.model_kws:
            verbosity = self.model_kws.pop('verbosity')

        model = Model(
            model = {"layers": suggested_paras.pop('layers')},
             prefix=title,
             verbosity=verbosity,
            **suggested_paras,
            **self.model_kws
            )
        
        setattr(self, 'model_', model)

        return model

    def build_from_config(self, data, config_path, weight_file, **kwargs):

        model = Model.from_config_file(config_path=config_path)
        weight_file = os.path.join(model.w_path, weight_file)
        model.update_weights(weight_file=weight_file)

        test_true, test_pred = model.predict(data=data, return_true=True)

        train_true, train_pred = model.predict(data='training', return_true=True)

        setattr(self, 'model_', model)

        return (train_true, train_pred), (test_true, test_pred)

    def model_Dense(self, **kwargs):
        # Dense 
        self.param_space = self._make_param_space([Integer(16, 100, name='dense_units')])

        self.x0 = self._make_x0([32])

        _layers = self._make_layers({
            "Dense": kwargs.get('dense_units', 32)
            })

        return self._make_return(_layers, **kwargs)

    def model_DenseAct(self, **kwargs):
        # Dense, Activation
        self.param_space = self._make_param_space([
            Integer(16, 100, name='dense_units'),
            Categorical(['relu', 'sigmoid', 'elu', 'tanh'], name="activation")
            ])

        self.x0 = self._make_x0([32, 'relu'])

        _layers = self._make_layers({
            "Dense": kwargs.get('dense_units', 32),
            "Activation": {"activation": kwargs.get('activation', 'relu')},
            })

        return self._make_return(_layers, **kwargs)

    def model_DenseActDense(self, **kwargs):
        # Dense, Activation, Dense
        self.param_space = self._make_param_space([
            Integer(16, 100, name='dense_units'),
            Categorical(['relu', 'sigmoid', 'elu', 'tanh'], name="activation"),
            Integer(10, 60, name='dense_units2')
            ])

        self.x0 = self._make_x0([32, 'relu', 16])

        _layers = self._make_layers({
            "Dense": kwargs.get('dense_units', 32),
            "Activation": {"activation": kwargs.get('activation', 'relu')},
            "Dense": kwargs.get('dense_units2', 16)
            })

        return self._make_return(_layers, **kwargs)

    def model_DenseActDenseAct(self, **kwargs):
        # Dense, Activation, Dense
        self.param_space = self._make_param_space([
            Integer(16, 100, name='dense_units'),
            Categorical(['relu', 'sigmoid', 'elu', 'tanh'], name="activation"),
            Integer(10, 60, name='dense_units2'),
            Categorical(['relu', 'sigmoid', 'elu', 'tanh'], name="activation2")
            ])

        self.x0 = self._make_x0([32, 'relu', 16, 'relu'])

        _layers = self._make_layers({
            "Dense": kwargs.get('dense_units', 32),
            "Activation": {"activation": kwargs.get('activation', 'relu')},
            "Dense": kwargs.get('dense_units2', 16),
            "Activation": {"activation": kwargs.get('activation2', 'relu')}
            })

        return self._make_return(_layers, **kwargs)

    def model_Lstm(self, **kwargs):
        # lstm, dense
        self.param_space = self._make_param_space([Integer(16, 100, name='lstm_units')])

        self.x0 = self._make_x0([32])

        _layers = self._make_layers({
            "LSTM": kwargs.get('lstm_units', 32)
            })

        return self._make_return(_layers, **kwargs)

    def model_LstmAct(self, **kwargs):
        # lstm, Activation
        self.param_space = self._make_param_space([
            Integer(16, 100, name='lstm_units'),
            Categorical(['relu', 'sigmoid', 'elu', 'tanh'], name="activation")
            ])

        self.x0 = self._make_x0([32, 'relu'])

        _layers = self._make_layers({
            "LSTM": {"units": kwargs.get('lstm_units', 32),},
            "Activation": {"activation": kwargs.get('activation', 'relu')},
                  })

        return self._make_return(_layers, **kwargs)

    def model_LstmActLstm(self, **kwargs):
        # lstm, Activation, lstm
        self.param_space = self._make_param_space([
            Integer(16, 100, name='lstm_units'),
            Categorical(['relu', 'sigmoid', 'elu', 'tanh'], name="activation"),
            Integer(10, 60, name='lstm_units2')
            ])

        self.x0 = self._make_x0([32, 'relu', 16])

        _layers = self._make_layers({
            "LSTM": {"units": kwargs.get('lstm_units', 32), "return_sequences": True},
            "Activation": {"activation": kwargs.get('activation', 'relu')},
            "LSTM": kwargs.get('lstm_units2', 16)
                  })

        return self._make_return(_layers, **kwargs)

    def model_Cnn1D(self, **kwargs):

        self.param_space = self._make_param_space([
            Integer(16, 100, name='filters'),
            Integer(3, 5, name='kernel_size')
        ])

        self.x0 = [32, 5]

        _layers = self._make_layers({
            "Conv1D": {"filters": kwargs.get('filters', 32), 
                       "kernel_size": kwargs.get('kernel_size', 5)},
            "MaxPool1D": 2,
            })

        return self._make_return(_layers, **kwargs)

    def model_Cnn1DAct(self, **kwargs):

        self.param_space = self._make_param_space([
            Integer(16, 100, name='filters'),
            Integer(3, 5, name='kernel_size'),
            Categorical(['relu', 'sigmoid', 'elu', 'tanh'], name="activation")
        ])

        self.x0 = [32, 5, 'relu']

        _layers = self._make_layers({
            "Conv1D": {"filters": kwargs.get('filters', 32), 
                       "kernel_size": kwargs.get('kernel_size', 5)},
            "MaxPool1D": 2,
            "Activation": {"activation": kwargs.get('activation', 'relu')},
            })

        return self._make_return(_layers, **kwargs)

    def _make_param_space(self, space:list)->list:
        """makes a parameter space"""
        return space + self.static_space

    def _make_x0(self, x0)->list:
        """makes initial/starting parameters"""  
        return x0 + self.static_x0

    def _make_layers(self, layers:dict)->dict:

        layers.update(self._output_layer())

        return layers
    
    def _make_return(self, layers, **kwargs)->dict:
        
        return {'layers': layers,    
            'ts_args': {'lookback': kwargs.get('lookback', self.model_kws.get('lookback', 8))},
            'batch_size': kwargs.get('batch_size', self.model_kws.get('batch_size', 8)),
            'lr': kwargs.get('lr', self.model_kws.get('batch_size', 0.001)),
            }
    
    def _output_layer(self)->dict:
        return {
            "Flatten_out": {},
            "Dense_out": 1
            }
