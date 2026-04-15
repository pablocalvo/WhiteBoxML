# whiteboxml/linear_models/test_run.py

import numpy as np
from whiteboxml.linear_models import LogisticRegression


def main():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression()
    model.fit(X, y)

    preds = model.predict(X)

    print("Predicciones:", preds)


if __name__ == "__main__":
    main()