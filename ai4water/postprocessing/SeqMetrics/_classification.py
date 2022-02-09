
import numpy as np
from sklearn import preprocessing
# from sklearn.metrics import hinge_loss
from sklearn.metrics import balanced_accuracy_score
import sklearn

from .utils import list_subclass_methods
from ._main import Metrics

from ai4water.backend import get_attributes

CLS_METRICS = get_attributes(sklearn.metrics, '_classification', case_sensitive=True)


class ClassificationMetrics(Metrics):
    """Calculates classification metrics."""
    # todo add very major erro and major error

    def __init__(self, *args, multiclass=False, **kwargs):
        self.multiclass = multiclass
        super().__init__(*args, metric_type='classification', **kwargs)
        self.true_labels = self._true_labels()
        self.true_logits = self._true_logits()
        self.pred_labels = self._pred_labels()
        self.pred_logits = self._pred_logits()

        self.all_methods = list_subclass_methods(ClassificationMetrics, True)
        # self.all_methods = [m for m in all_methods if not m.startswith('_')]

    def __getattr__(self, item):

        def func(**kwargs):  # because we want .f1_score() and not .f1_score
            return CLS_METRICS[item](self.true_labels, self.pred_labels, **kwargs)
        return func


    @staticmethod
    def _minimal() -> list:
        """some minimal and basic metrics"""
        return list_subclass_methods(ClassificationMetrics, True)

    @staticmethod
    def _hydro_metrics() -> list:
        """some minimal and basic metrics"""
        return list_subclass_methods(ClassificationMetrics, True)

    def _num_classes(self):
        return len(self._classes())

    def _classes(self):
        array = self.true_labels
        return np.unique(array[~np.isnan(array)])

    def _true_labels(self):
        """retuned array is 1d"""
        if self.multiclass:
            if self.true.size == len(self.true):
                return self.true.reshape(-1,1)
            return np.argmax(self.true.reshape(-1,1), axis=1)
        # it should be 1 dimensional
        assert self.true.size == len(self.true)
        return self.true.reshape(-1,)

    def _true_logits(self):
        """returned array is 2d"""
        if self.multiclass:
            return self.true
        lb = preprocessing.LabelBinarizer()
        return lb.fit_transform(self.true)

    def _pred_labels(self):
        """returns 1d"""
        if self.multiclass:
            if self.predicted.size == len(self.predicted):
                return self.predicted.reshape(-1,1)
            return np.argmax(self.predicted.reshape(-1,1), axis=1)
        lb = preprocessing.LabelBinarizer()
        lb.fit(self.true_labels)
        return lb.inverse_transform(self.predicted)

    def _pred_logits(self):
        """returned array is 2d"""
        if self.multiclass:
            return self.true
        # we can't do it
        return None

    def cross_entropy(self, epsilon=1e-12):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions.
        Input: predictions (N, k) ndarray
               targets (N, k) ndarray
        Returns: scalar
        """
        predictions = np.clip(self.predicted, epsilon, 1. - epsilon)
        n = predictions.shape[0]
        ce = -np.sum(self.true * np.log(predictions + 1e-9)) / n
        return ce

    # def hinge_loss(self):
    #     """hinge loss using sklearn"""
    #     if self.pred_logits is not None:
    #         return hinge_loss(self.true_labels, self.pred_logits)
    #     return None

    def balanced_accuracy_score(self):
        return balanced_accuracy_score(self.true_labels, self.pred_labels)

    def accuracy(self, normalize=True):
        if normalize:
            return np.average(self.true_labels==self.pred_labels)
        return (self.true_labels==self.pred_labels).sum()
