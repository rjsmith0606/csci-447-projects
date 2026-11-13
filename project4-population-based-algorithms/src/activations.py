"""
Activation functions and their derivatives.

Per the assignment, hidden layers may use ONLY logistic (sigmoid) or tanh.
No ReLU or other nonlinearities. Output is softmax (classification) or
linear / identity (regression).
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative expressed in terms of the activation a = sigmoid(z)."""
    return a * (1 - a)


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def tanh_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative expressed in terms of the activation a = tanh(z)."""
    return 1 - a ** 2


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise, numerically stable softmax."""
    z = z - z.max(axis=1, keepdims=True)  # subtract row max so exp() can't overflow
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def linear(z: np.ndarray) -> np.ndarray:
    return z
