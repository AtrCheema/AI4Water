
from ai4water.backend import np, pd, os


class ExplainerMixin(object):

    def __init__(
            self,
            path,
            data,
            features,
            save=True,
            show=True,
    ):
        if not os.path.exists(path):
            os.makedirs(path)

        self.path = path
        self.data = data
        self.features = features
        self.save = save
        self.show = show

    @property
    def data_is_2d(self):
        if isinstance(self.data, np.ndarray) and self.data.ndim == 2:
            return True
        elif isinstance(self.data, pd.DataFrame):
            return True
        else:
            return False

    @property
    def data_is_3d(self):
        if isinstance(self.data, np.ndarray) and self.data.ndim == 3:
            return True
        return False

    @property
    def single_source(self):
        if isinstance(self.data, list) and len(self.data) > 1:
            return False
        else:
            return True

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, features):
        if self.data_is_2d:
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
        elif not self.single_source and features is None:
            features = []
            for data in self.data:
                if isinstance(data, pd.DataFrame):
                    _features = data.columns.to_list()
                else:
                    _features = [f"Feature {i}" for i in range(data.shape[-1])]

                features.append(_features)

        elif self.data_is_3d and features is None:
            features = [f"Feature {i}" for i in range(self.data.shape[-1])]

        self._features = features

    @property
    def unrolled_features(self):
        # returns the possible names of features if unrolled over time
        if not self.data_is_2d and self.single_source:
            features = self.features
            if features is None:
                features = [f"Feature_{i}" for i in range(self.data.shape[-1])]

            lookback = self.data.shape[1]
            features = [[f"{f}_{i}" for f in features] for i in range(lookback)]
            features = [item for sublist in features for item in sublist]
        else:
            features = None
        return features
