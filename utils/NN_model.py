from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np


class KerasNN:
    def __init__(self, task='regression', **config):
        super().__init__()
        self.task = task
        self.config = config
        self.scaler = StandardScaler()
        self.model = None

    @classmethod
    def init(cls):
        return cls

    @classmethod
    def search_space(cls, data_size=None, task=None):
        return {
            "epochs": {"domain": (10, 100), "init_value": 10, "low_cost_init_value": 10},
            "batch_size": {"domain": (16, 128), "init_value": 32},
            "hidden_units": {"domain": (16, 256), "init_value": 64},
        }

    @classmethod
    def cost_relative2lgbm(cls):
        return 10

    def _build_model(self, input_dim):
        hidden_units = self.config.get("hidden_units", 64)
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        print(int(hidden_units[1]))
        model.add(layers.Dense(int(hidden_units[1]), activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))
        return model

    def fit(self, X_train, y_train, **kwargs):
        epochs = self.config.get("epochs", 10)
        batch_size = self.config.get("batch_size", 32)
        X_scaled = self.scaler.fit_transform(X_train)
        self.model = self._build_model(input_dim=X_scaled.shape[1])
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        self.model.fit(
            X_scaled,
            y_train,
            epochs=int(epochs),
            batch_size=int(batch_size[1]),
            verbose=0
        )

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return (preds > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return np.hstack([1 - preds, preds])

    @classmethod
    def size(cls, config):
        return 64

    def cleanup(self):
        self.model = None
