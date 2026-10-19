"""
Activation functions and their derivatives.

Per the assignment, hidden layers may use ONLY logistic (sigmoid) or tanh.
No ReLU or other nonlinearities. Output is softmax (classification) or
linear / identity (regression).
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative expressed in terms of the activation a = sigmoid(z)."""
    raise NotImplementedError


def tanh(z: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def tanh_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative expressed in terms of the activation a = tanh(z)."""
    raise NotImplementedError


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise, numerically stable softmax."""
    raise NotImplementedError


def linear(z: np.ndarray) -> np.ndarray:
    raise NotImplementedError
