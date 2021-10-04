
import pandas as pd


class ExplainerMixin(object):

    def __init__(
            self,
            path,
            data,
            features
    ):
        self.path = path
        self.data = data
        self.features = features

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, features):
        if type(self.data) == pd.DataFrame:

            features = self.data.columns.to_list()
        elif features is None:
            features = [f"Feature {i}" for i in range(self.data.shape[-1])]
        else:
            assert isinstance(features, list) and len(features) == self.data.shape[-1], f"""
                features must be given as list of length {self.data.shape[-1]} 
                but are of len {len(features)}
                """

            features = features
        self._features = features

