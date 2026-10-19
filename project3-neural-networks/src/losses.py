"""
Loss functions.

- Cross-entropy for classification (softmax output)
- Mean squared error for regression (linear output) and for the autoencoder
  reconstruction loss  L(x, x_hat) = ||x - x_hat||^2
"""

import numpy as np


def cross_entropy(y_true_onehot: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    raise NotImplementedError


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError
