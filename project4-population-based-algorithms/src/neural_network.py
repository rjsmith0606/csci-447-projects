"""
Feedforward neural network trained with backpropagation (Figure 2).

Topology: Input -> Hidden 1 -> Hidden 2 -> Output
- Hidden activation: logistic OR tanh (choose one, document it in the report)
- Output: softmax + cross-entropy (classification) or linear + MSE (regression)
- Optional momentum

The class should also support:
- initializing the first-layer weights from a pre-trained encoder
- freezing chosen layers (for the autoencoder variant where encoder weights
  are locked) vs. fine-tuning the whole network
"""

import numpy as np

from src.activations import sigmoid, sigmoid_derivative, tanh, tanh_derivative, softmax, linear
from src.losses import cross_entropy, mse


class NeuralNetwork:
    def __init__(self, layer_sizes: list[int], mode: str = "classification",
                 hidden_activation: str = "sigmoid", learning_rate: float = 0.01,
                 seed: int | None = None):

        # choose the hidden activation (the assignment allows only these two)
        if hidden_activation == "sigmoid":
            self.act, self.act_deriv = sigmoid, sigmoid_derivative
        elif hidden_activation == "tanh":
            self.act, self.act_deriv = tanh, tanh_derivative
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
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.W.append(self.rng.normal(0.0, 0.01, size=(fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))

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

    def backward(self, X: np.ndarray, y: np.ndarray):
        out = self.forward(X)
        n = X.shape[0]
        dWs = [None] * self.n_layers
        dbs = [None] * self.n_layers

        delta = out - y
        for i in reversed(range(self.n_layers)):
            a_prev = self.activations[i]
            dWs[i] = a_prev.T @ delta / n
            dbs[i] = delta.mean(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * self.act_deriv(a_prev)
        return dWs, dbs

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100,
            batch_size: int | None = None, frozen_layers: list[int] | None = None):
        frozen = set(frozen_layers or [])   # layer indices to leave untouched
        n = X.shape[0]
        if batch_size is None or batch_size > n:
            batch_size = n

        for epoch in range(epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, batch_size): # type: ignore
                idx = order[start:start + batch_size] # type: ignore
                dWs, dbs = self.backward(X[idx], y[idx])

                for i in range(self.n_layers):
                    if i in frozen:
                        continue
                    self.W[i] -= self.learning_rate * dWs[i] # type: ignore
                    self.b[i] -= self.learning_rate * dbs[i] # type: ignore

            self.loss_history.append(self.loss(y, self.forward(X)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = self.forward(X)
        return np.argmax(out, axis=1) if self.mode == "classification" else out

    def predict_proba(self, X):
        return self.forward(X)

    def get_layer_weights(self, layer_index: int):
        return self.W[layer_index].copy(), self.b[layer_index].copy()

    def set_layer_weights(self, layer_index: int, W: np.ndarray, b: np.ndarray):
        self.W[layer_index] = W.copy()
        self.b[layer_index] = b.copy()
