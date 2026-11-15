"""
Feedforward neural network trained with popullation based algorithms

"""

import numpy as np

from src.activations import sigmoid, sigmoid_derivative, softmax, linear
from src.losses import cross_entropy, mse


class NeuralNetwork:
    def __init__(self, layer_sizes: list[int], mode: str = "classification",
                 hidden_activation: str = "sigmoid", learning_rate: float = 0.01,
                 seed: int | None = None):

        # choose the hidden activation (the assignment allows only these two)
        if hidden_activation == "sigmoid":
            self.act, self.act_deriv = sigmoid, sigmoid_derivative
        else:
            raise ValueError(f"Unknown hidden_activation: {hidden_activation}")

        if mode == "classification":
            self.loss = cross_entropy
        elif mode == "regression":
            self.loss = mse
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.layer_sizes = layer_sizes      # e.g. [d, h1, h2, K]
        self.mode = mode
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)
        self.n_layers = len(layer_sizes) - 1  # number of weight matrices

        # one (W, b) pair per layer
        self.W, self.b = [], []

        self.activations = []     # filled by forward(), reused by backward()
        self.loss_history = []

    def forward(self, X: np.ndarray):
        a = X
        self.activations = [X]
        for i in range(self.n_layers):
            z = a @ self.W[i] + self.b[i]
            if i == self.n_layers - 1:
                a = softmax(z) if self.mode == "classification" else linear(z)
            else:
                a = self.act(z)
            self.activations.append(a)
        return a

    def fitness(self, weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        self.set_weights(weights)
        return self.loss(y, self.forward(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = self.forward(X)
        return np.argmax(out, axis=1) if self.mode == "classification" else out

    def predict_proba(self, X):
        return self.forward(X)

    def set_weights(self, weights: np.ndarray):
        """Unpack a flat weight vector into one (W, b) pair per layer."""
        self.W, self.b = [], []
        pos = 0
        for i in range(self.n_layers):
            n_in, n_out = self.layer_sizes[i], self.layer_sizes[i + 1]
            self.W.append(weights[pos:pos + n_in * n_out].reshape(n_in, n_out).copy())
            pos += n_in * n_out
            self.b.append(weights[pos:pos + n_out].copy())
            pos += n_out
