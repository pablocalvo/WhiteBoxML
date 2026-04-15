# whiteboxml/reporting/model_reporting.py

import numpy as np


class ModelReporting:
    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def get_metrics(self, model):
        """
        Calcula métricas básicas para cualquier modelo compatible con BaseModel.
        """
        predictions = model.predict(self.X_test)

        # Error Cuadratico Medio (MSE)
        mse = np.mean((self.y_test - predictions) ** 2)

        return {
            "model_type": type(model).__name__,
            "mse": round(mse, 4),
            "params_count": len(model.params.get("weights", []))
            if "weights" in model.params else 0
        }

    def compare_models(self, models_list):
        """
        Compara multiples modelos y muestra resultados en consola.
        """
        print(f"{'Modelo':<25} | {'MSE':<10}")
        print("-" * 40)

        for model in models_list:
            metrics = self.get_metrics(model)
            print(f"{metrics['model_type']:<25} | {metrics['mse']:<10}")