__all__ = ["DLRegressionExperiments", "DLClassificationExperiments"]

from ai4water.backend import tf
from ai4water.utils.utils import jsonize
from ai4water.hyperopt import Integer, Real, Categorical
from ai4water.utils.utils import dateandtime_now
from ai4water.models import MLP, CNN, LSTM, CNNLSTM, LSTMAutoEncoder, TFT, TCN

from ._main import Experiments
from .utils import dl_space


class DLRegressionExperiments(Experiments):
    """
    A framework for comparing several basic DL architectures for a given data.
    This class can also be used for hyperparameter optimization of more than
    one DL models/architectures. However, the parameters which determine
    the dimensions of input data such as ``lookback`` should are
    not allowed to optimize when using random or grid search.

    To check the available models
    >>> exp = DLRegressionExperiments(...)
    >>> exp.models

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
    ... # runt he experiments
    >>> exp.fit(data=data)

    """

    def __init__(
            self,
            input_features: list,
            param_space=None,
            x0=None,
            cases: dict = None,
            exp_name: str = None,
            num_samples: int = 5,
            verbosity: int = 1,
            **model_kws
    ):
        """initializes the experiment."""
        self.input_features = input_features
        self.param_space = param_space

        # batch_size and lr will come from x0 so should not be
        # in model_kws
        if 'batch_size' in model_kws:
            self.batch_size_x0 = model_kws.pop('batch_size')
        else:
            self.batch_size_x0 = 32

        if 'lr' in model_kws:
            self.lr_x0 = model_kws.pop('lr')
        else:
            self.lr_x0 = 0.001

        self.x0 = x0

        # during model initiation, we must provide input_features argument
        model_kws['input_features'] = input_features

        self.lookback_space = []
        self.batch_size_space = Categorical(categories=[4, 8, 12, 16, 32],
                                            name="batch_size")
        self.lr_space = Real(1e-5, 0.005, name="lr")

        exp_name = exp_name or 'DLExperiments' + f'_{dateandtime_now()}'

        super().__init__(
            cases=cases,
            exp_name=exp_name,
            num_samples=num_samples,
            verbosity=verbosity,
            **model_kws
        )

        self.spaces = dl_space(num_samples=num_samples)

    @property
    def category(self):
        return "DL"

    @property
    def input_shape(self) -> tuple:
        features = len(self.input_features)
        shape = features,
        if "ts_args" in self.model_kws:
            if "lookback" in self.model_kws['ts_args']:
                shape = self.model_kws['ts_args']['lookback'], features

        return shape

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
            _x0.append(self.model_kws.get('lookback', 5))
        if self.batch_size_space:
            _x0.append(self.batch_size_x0)
        if self.lr_space:
            _x0.append(self.lr_x0)
        return _x0

    @property
    def mode(self):
        return "regression"

    @property
    def tpot_estimator(self):
        return None

    def _pre_build_hook(self, **suggested_paras):
        """suggested_paras contain model configuration which
        may contain executable tf layers which should be
        serialized properly.
        """

        suggested_paras = jsonize(suggested_paras, {
            tf.keras.layers.Layer: tf.keras.layers.serialize})

        return suggested_paras

    def model_MLP(self, **kwargs):
        """multi-layer perceptron model"""

        self.param_space = self.spaces["MLP"]["param_space"] + self.static_space
        self.x0 = self.spaces["MLP"]["x0"] + self.static_x0

        _kwargs = {}
        for arg in ['batch_size', 'lr']:
            if arg in kwargs:
                _kwargs[arg] = kwargs.pop(arg)
        config = {'model': MLP(input_shape=self.input_shape,
                               mode=self.mode,
                               **kwargs)}
        config.update(_kwargs)
        return config

    def model_LSTM(self, **kwargs):
        """LSTM based model"""

        self.param_space = self.spaces["LSTM"]["param_space"] + self.static_space
        self.x0 = self.spaces["LSTM"]["x0"] + self.static_x0

        _kwargs = {}
        for arg in ['batch_size', 'lr']:
            if arg in kwargs:
                _kwargs[arg] = kwargs.pop(arg)
        config = {'model': LSTM(input_shape=self.input_shape,
                               mode=self.mode,
                                **kwargs)}
        config.update(_kwargs)
        return config

    def model_CNN(self, **kwargs):
        """1D CNN based model"""

        self.param_space = self.spaces["CNN"]["param_space"] + self.static_space
        self.x0 = self.spaces["CNN"]["x0"] + self.static_x0

        _kwargs = {}
        for arg in ['batch_size', 'lr']:
            if arg in kwargs:
                _kwargs[arg] = kwargs.pop(arg)
        config = {'model': CNN(input_shape=self.input_shape,
                               mode=self.mode,
                               **kwargs)}
        config.update(_kwargs)

        return config

    def model_CNNLSTM(self, **kwargs)->dict:
        """CNN-LSTM model"""

        self.param_space = self.spaces["CNNLSTM"]["param_space"] + self.static_space
        self.x0 = self.spaces["CNNLSTM"]["x0"] + self.static_x0

        _kwargs = {}
        for arg in ['batch_size', 'lr']:
            if arg in kwargs:
                _kwargs[arg] = kwargs.pop(arg)

        assert len(self.input_shape) == 2
        config = {'model': CNNLSTM(input_shape=self.input_shape,
                                   mode=self.mode,
                                   **kwargs)}
        config.update(_kwargs)
        return config

    def model_LSTMAutoEncoder(self, **kwargs):
        """LSTM based auto-encoder model."""

        self.param_space = self.spaces["LSTMAutoEncoder"]["param_space"] + self.static_space
        self.x0 = self.spaces["LSTMAutoEncoder"]["x0"] + self.static_x0

        _kwargs = {}
        for arg in ['batch_size', 'lr']:
            if arg in kwargs:
                _kwargs[arg] = kwargs.pop(arg)
        config = {'model': LSTMAutoEncoder(input_shape=self.input_shape,
                                           mode=self.mode,
                                           **kwargs)}
        config.update(_kwargs)
        return config

    def model_TCN(self, **kwargs):
        """Temporal Convolution network based model."""

        self.param_space = self.spaces["TCN"]["param_space"] + self.static_space
        self.x0 = self.spaces["TCN"]["x0"] + self.static_x0

        _kwargs = {}
        for arg in ['batch_size', 'lr']:
            if arg in kwargs:
                _kwargs[arg] = kwargs.pop(arg)
        config = {'model': TCN(input_shape=self.input_shape,
                               mode=self.mode,
                               **kwargs)}
        config.update(_kwargs)
        return config

    def model_TFT(self, **kwargs):
        """temporal fusion transformer model."""

        self.param_space = self.spaces["TFT"]["param_space"] + self.static_space
        self.x0 = self.spaces["TFT"]["x0"] + self.static_x0

        _kwargs = {}
        for arg in ['batch_size', 'lr']:
            if arg in kwargs:
                _kwargs[arg] = kwargs.pop(arg)
        config = {'model': TFT(input_shape=self.input_shape,
                               **kwargs)}
        config.update(_kwargs)
        return config


class DLClassificationExperiments(DLRegressionExperiments):
    """
    Compare multiple neural network architectures for a classification problem

    Examples
    ---------
    >>> from ai4water.datasets import MtropicsLaos
    >>> data = MtropicsLaos().make_classification(
    ...     input_features=['air_temp', 'rel_hum'],
    ...     lookback_steps=5)
    define inputs and outputs
    >>> inputs = data.columns.tolist()[0:-1]
    >>> outputs = data.columns.tolist()[-1:]
    create the experiments class
    >>> exp = DLClassificationExperiments(
    ...     input_features=inputs,
    ...     output_features=outputs,
    ...     epochs=5,
    ...     ts_args={"lookback": 5}
    )
    run the experiments
    >>> exp.fit(data=data, include=["TFT", "MLP"])

    """
    def __init__(
            self,
            exp_name=f"DLClassificationExperiments_{dateandtime_now()}",
            *args,
            **kwargs):

        super(DLClassificationExperiments, self).__init__(
            exp_name=exp_name,
            *args, **kwargs
        )

    @property
    def mode(self):
        return "classification"

    def metric_kws(self, metric_name:str=None):
        kws = {
            'precision': {'average': 'macro'},
            'recall': {'average': 'macro'},
            'f1_score': {'average': 'macro'},
        }
        return kws.get(metric_name, {})

