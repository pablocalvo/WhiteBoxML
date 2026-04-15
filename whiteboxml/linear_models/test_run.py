import numpy as np
from whiteboxml.linear_models.logistic_regression import LogisticRegression

# Datos simples
X = np.array([[0], [1], [2], [3]])
y = np.array([0, 0, 1, 1])

# Crear modelo
model = LogisticRegression(lr=0.1, n_iter=1000)

# Entrenar
model.fit(X, y)

# Predecir
preds = model.predict(X)

print("Predicciones:", preds)