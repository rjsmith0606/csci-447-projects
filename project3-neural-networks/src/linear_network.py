"""
Linear networks (Figure 1 in the assignment): inputs feed DIRECTLY into the
output layer, no hidden layers.

- LogisticRegression : inputs -> softmax output (classification, cross-entropy)
- LinearRegression   : inputs -> single linear output node (regression, MSE),
                       trained by gradient descent (equivalent to OLS)
"""

import numpy as np
from src.activations import softmax, linear
from src.losses import cross_entropy, mse

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
        """
        Fit the model to the data using mini-batch gradient descent.

        X:        (n, d) feature matrix
        y_onehot: (n, K) one-hot class matrix from prepare_xy
        epochs:   number of full passes over the training data
        batch_size: examples per weight update; None means use all n (full-batch)
        """

        n = X.shape[0]
        if batch_size is None or batch_size > n:
            batch_size = n # full-batch gradient descent

        # Full-batch if no size is given, or if the requested size is bigger than the data
        if batch_size is None or batch_size > n:
            batch_size = n  # full-batch gradient descent: one update per epoch

        for epoch in range(epochs):
            # Shuffle the row order each epoch so batches differ every pass and the
            # model doesn't see the data in the same sequence (or any file ordering)
            order = self.rng.permutation(n)

            # Walk through the shuffled rows one batch at a time
            for start in range(0, n, batch_size):  # type: ignore
                # Row indices for this batch (the last batch may be smaller than batch_size)
                idx = order[start:start + batch_size]  # type: ignore
                Xb, yb = X[idx], y_onehot[idx]  # features and one-hot targets for this batch

                # Forward pass + error. For softmax with cross-entropy, the gradient of
                # the loss with respect to the scores is simply (probabilities - targets).
                # A wrong, confident prediction gives a large error for that class.
                error = self.predict_proba(Xb) - yb     # p - y, shape (batch, K)

                # Average the gradient over the batch so the learning rate means the
                # same thing regardless of batch size
                dW = Xb.T @ error / len(idx)            # (d, K): how the loss changes with each weight
                db = error.mean(axis=0)                 # (K,): how the loss changes with each bias

                # Gradient descent step: move the parameters a small step (learning_rate)
                # in the direction that reduces the loss, i.e. against the gradient
                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db

            # After each epoch, record the loss on the WHOLE training set (not just the
            # last batch) so loss_history gives a smooth learning curve for tuning
            self.loss_history.append(cross_entropy(y_onehot, self.predict_proba(X)))

        # Return the model itself so calls can be chained: LogisticRegression(...).fit(X, y)
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
        """
        Fit the model to the data using mini-batch gradient descent.

        X:        (n, d) feature matrix
        y:        (n, 1) target values
        epochs:   number of full passes over the training data
        batch_size: examples per weight update; None means use all n (full-batch)
        """
        n = X.shape[0]  # number of training examples

        # Full-batch if no size is given, or if the requested size is bigger than the data
        if batch_size is None or batch_size > n:
            batch_size = n  # full-batch gradient descent: one update per epoch

        for epoch in range(epochs):
            # Shuffle the row order each epoch so batches differ every pass and the
            # model doesn't see the data in the same sequence (or any file ordering)
            order = self.rng.permutation(n)

            # Walk through the shuffled rows one batch at a time
            for start in range(0, n, batch_size):  # type: ignore
                # Row indices for this batch (the last batch may be smaller than batch_size)
                idx = order[start:start + batch_size]  # type: ignore
                Xb, yb = X[idx], y[idx]  # features and z-scored targets for this batch

                # Forward pass + error. With a linear output and MSE loss, the gradient of
                # the loss with respect to the prediction is proportional to
                # (prediction - target). The constant factor of 2 from differentiating
                # the square is dropped; it only rescales the learning rate.
                error = self.predict(Xb) - yb           # prediction - target, shape (batch, 1)

                # Average the gradient over the batch so the learning rate means the
                # same thing regardless of batch size
                dW = Xb.T @ error / len(idx)            # (d, 1): how the loss changes with each weight
                db = error.mean(axis=0)                 # (1,): how the loss changes with the bias

                # Gradient descent step: move the parameters a small step (learning_rate)
                # in the direction that reduces the loss, i.e. against the gradient
                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db

            # After each epoch, record the MSE on the WHOLE training set (not just the
            # last batch) so loss_history gives a smooth learning curve for tuning.
            # It should start near 1.0 because the target is z-scored, so predicting
            # the mean gives an MSE of about 1.
            self.loss_history.append(mse(y, self.predict(X)))

        # Return the model itself so calls can be chained: LinearRegression(...).fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return linear(X @ self.W + self.b)
