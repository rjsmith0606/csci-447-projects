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


class NeuralNetwork:
    def __init__(self, layer_sizes: list[int], mode: str = "classification",
                 hidden_activation: str = "sigmoid", learning_rate: float = 0.01,
                 momentum: float = 0.0, seed: int | None = None):
        """
        layer_sizes: e.g. [n_inputs, h1, h2, n_outputs]
        mode: "classification" (softmax + cross-entropy) or "regression" (linear + MSE)
        """
        raise NotImplementedError

    def forward(self, X: np.ndarray):
        raise NotImplementedError

    def backward(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100,
            batch_size: int | None = None, frozen_layers: list[int] | None = None):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_layer_weights(self, layer_index: int):
        raise NotImplementedError

    def set_layer_weights(self, layer_index: int, W: np.ndarray, b: np.ndarray):
        raise NotImplementedError
