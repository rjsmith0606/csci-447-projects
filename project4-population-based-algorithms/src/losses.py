"""
Loss functions.

- Cross-entropy for classification (softmax output)
- Mean squared error for regression (linear output) and for the autoencoder
  reconstruction loss  L(x, x_hat) = ||x - x_hat||^2
"""

import numpy as np


def cross_entropy(y_true_onehot: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    """Mean cross-entropy between one-hot targets and predicted probabilities."""
    y_prob = np.clip(y_prob, eps, 1.0)  # avoid ln(0)
    return float(-np.mean(np.sum(y_true_onehot * np.log(y_prob), axis=1)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error between true and predicted values."""
    return float(np.mean((y_true - y_pred) ** 2))
