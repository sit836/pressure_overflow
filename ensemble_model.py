import numpy as np


class EnsembleModel:
    def __init__(self, models):
        self.models = models
        self.n = len(models)

    def predict(self, inputs):
        avg_pred = np.zeros(inputs.shape[0])
        for model in self.models:
            avg_pred += model.predict(inputs) / self.n
        return avg_pred
