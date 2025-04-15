from smt.surrogate_models import KRG
import numpy as np

class KRGModel:
    def __init__(self, task='regression', **config):
        super().__init__()
        self.task = task
        self.config = config
        self.model = None

    @classmethod
    def search_space(cls, data_size, task):
        return {
            'theta0': {'domain': [1e-2, 1e1], 'init_value': 1.0, 'low_cost_init_value': 1.0},
        }

    @classmethod
    def init(cls):
        return cls

    @classmethod
    def size(cls, config):
        return 1

    def fit(self, X_train, y_train, **kwargs):
        X_train = np.array(X_train)
        y_train = np.array(y_train).reshape(-1, 1)
        theta = self.config.get('theta0', 1.0)
        self.model = KRG(theta0=[theta])
        self.model.set_training_values(np.array(X_train), np.array(y_train).reshape(-1, 1))
        self.model.train()

    def predict(self, X_new):
        X_new = np.array(X_new)
        return self.model.predict_values(X_new).flatten()

    def cleanup(self):
        self.model = None

    @classmethod
    def cost_relative2lgbm(cls):
        return 10
