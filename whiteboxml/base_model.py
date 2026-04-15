from abc import ABC, abstractmethod
import json


class BaseModel(ABC):
    def __init__(self):
        self.params = {}

    def __repr__(self):
        return f"{self.__class__.__name__}(params={self.params})"

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def save_params(self, path):
        with open(path, 'w') as f:
            json.dump(self.params, f)

    def load_params(self, path):
        with open(path, 'r') as f:
            self.params = json.load(f)