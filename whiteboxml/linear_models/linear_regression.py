import numpy as np
from whiteboxml.base_model import BaseModel


class LinearRegression(BaseModel):
    def __init__(self, lr=0.01, n_iter=1000):
        super().__init__()
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Inicialización
        w = np.zeros(n_features)
        b = 0

        # Descenso de gradiente
        for _ in range(self.n_iter):
            y_pred = np.dot(X, w) + b

            # Gradientes
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Actualización
            w -= self.lr * dw
            b -= self.lr * db

        # Guardamos parámetros en el formato estándar
        self.params["weights"] = w.tolist()
        self.params["bias"] = b

    def predict(self, X):
        if not self.params:
            raise Exception("El modelo debe entrenarse antes de predecir.")

        w = np.array(self.params["weights"])
        b = self.params["bias"]

        return np.dot(X, w) + b
