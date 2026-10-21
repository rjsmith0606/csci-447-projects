"""
Linear networks (Figure 1 in the assignment): inputs feed DIRECTLY into the
output layer, no hidden layers.

- LogisticRegression : inputs -> softmax output (classification, cross-entropy)
- LinearRegression   : inputs -> single linear output node (regression, MSE),
                       trained by gradient descent (equivalent to OLS)
"""

import numpy as np
from activations import softmax, linear
from losses import cross_entropy, mse

class LogisticRegression:
    def __init__(self, n_features: int, n_classes: int, learning_rate: float = 0.01, seed: int | None = None):
        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(loc=0.0, scale=0.01, size=(n_features, n_classes))
        self.b = np.zeros(n_classes)

        self.loss_history = []

    def fit(self, X: np.ndarray, y_onehot: np.ndarray, epochs: int = 100, batch_size: int | None = None):
        """Fit the model to the data using mini-batch gradient descent."""
        n = X.shape[0]
        if batch_size is None or batch_size > n:
            batch_size = n # full-batch gradient descent

        for epoch in range(epochs):
            order = self.rng.permutation(n) # reshuffle every epoch

            for start in range(0, n, batch_size): # type: ignore
                idx = order[start:start + batch_size] # type: ignore
                Xb, yb = X[idx], y_onehot[idx]

                error = self.predict_proba(Xb) - yb     # p - y, shape (batch, K)
                dW = Xb.T @ error / len(idx)            # (d, K)
                db = error.mean(axis=0)                 # (K,)

                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db

            self.loss_history.append(cross_entropy(y_onehot, self.predict_proba(X)))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(X @ self.W + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the most likely class index for each row of X."""
        return np.argmax(self.predict_proba(X), axis=1)


class LinearRegression:
    def __init__(self, n_features: int, learning_rate: float = 0.01, seed: int | None = None):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(loc=0.0, scale=0.01, size=(n_features, 1))
        self.b = np.zeros(1)

        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int | None = None):
        """Fit the model to the data using mini-batch gradient descent."""
        n = X.shape[0]
        if batch_size is None or batch_size > n:
            batch_size = n # full-batch gradient descent

        for epoch in range(epochs):
            order = self.rng.permutation(n) # reshuffle every epoch

            for start in range(0, n, batch_size): # type: ignore
                idx = order[start:start + batch_size] # type: ignore
                Xb, yb = X[idx], y[idx]

                error = self.predict(Xb) - yb    
                dW = Xb.T @ error / len(idx)            
                db = error.mean(axis=0)                 

                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db

            self.loss_history.append(mse(y, self.predict(X)))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return linear(X @ self.W + self.b)
