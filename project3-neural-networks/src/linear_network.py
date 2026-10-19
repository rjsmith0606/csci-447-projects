"""
Linear networks (Figure 1 in the assignment): inputs feed DIRECTLY into the
output layer, no hidden layers.

- LogisticRegression : inputs -> softmax output (classification, cross-entropy)
- LinearRegression   : inputs -> single linear output node (regression, MSE),
                       trained by gradient descent (equivalent to OLS)
"""

import numpy as np


class LogisticRegression:
    def __init__(self, n_features: int, n_classes: int, learning_rate: float = 0.01):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y_onehot: np.ndarray, epochs: int = 100, batch_size: int | None = None):
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LinearRegression:
    def __init__(self, n_features: int, learning_rate: float = 0.01):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int | None = None):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
