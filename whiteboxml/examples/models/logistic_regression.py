# whiteboxml/linear_models/logistic_regression.py

import numpy as np
from whiteboxml.base_model import BaseModel


class LogisticRegression(BaseModel):
    def __init__(self, lr=0.01, n_iter=1000):
        super().__init__()
        self.lr = lr
        self.n_iter = n_iter

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        w = np.zeros(n_features)
        b = 0

        for _ in range(self.n_iter):
            z = np.dot(X, w) + b
            y_pred = self._sigmoid(z)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            w -= self.lr * dw
            b -= self.lr * db

        # 🔥 Guardamos en el formato estándar
        self.params["weights"] = w.tolist()
        self.params["bias"] = b

    def predict(self, X):
        if not self.params:
            raise Exception("El modelo debe entrenarse antes de predecir.")

        w = np.array(self.params["weights"])
        b = self.params["bias"]

        z = np.dot(X, w) + b
        y_pred = self._sigmoid(z)

        return (y_pred >= 0.5).astype(int)
