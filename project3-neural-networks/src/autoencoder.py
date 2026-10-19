"""
Autoencoder and the autoencoder-based prediction network (Figure 3).

Stage 1 - Autoencoder: Input -> Encoding (e, strictly fewer nodes than inputs)
          -> Decoding (same number of nodes as inputs), trained with MSE so the
          output reconstructs the input.
Stage 2 - Prediction network: drop the decoder, attach a hidden layer and the
          prediction layer (softmax / linear) to the encoder output:
          Input -> Encoding -> Hidden 2 -> Prediction
          Train with backprop, either (a) encoder weights locked, or
          (b) whole network fine-tuned. Pick one and justify it in the report.
"""

import numpy as np

from src.neural_network import NeuralNetwork


class Autoencoder:
    def __init__(self, n_inputs: int, n_encoding: int, hidden_activation: str = "sigmoid",
                 learning_rate: float = 0.01, momentum: float = 0.0, seed: int | None = None):
        if n_encoding >= n_inputs:
            raise ValueError("Encoding layer must have strictly fewer nodes than the input layer.")
        raise NotImplementedError

    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int | None = None):
        raise NotImplementedError

    def encode(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reconstruction_error(self, X: np.ndarray) -> float:
        raise NotImplementedError


def build_autoencoder_network(autoencoder: Autoencoder, n_hidden2: int, n_outputs: int,
                              mode: str = "classification", freeze_encoder: bool = True,
                              **kwargs) -> NeuralNetwork:
    """Build Input -> Encoding -> Hidden 2 -> Prediction from a trained autoencoder."""
    raise NotImplementedError
