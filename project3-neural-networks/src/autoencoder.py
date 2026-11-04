"""
Autoencoder and the autoencoder-based prediction network.

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
from src.losses import mse
from src.neural_network import NeuralNetwork


class Autoencoder:
    def __init__(self, n_inputs: int, n_encoding: int, hidden_activation: str = "tanh", learning_rate: float = 0.01, seed: int | None = None):
            if n_encoding >= n_inputs:
                raise ValueError(
                    "Encoding layer must have fewer nodes than the input layer.")

            self.n_inputs = n_inputs
            self.n_encoding = n_encoding
            self.hidden_activation = hidden_activation
            self.learning_rate = learning_rate
            self.seed = seed

            # Regression feedforward network
            self.network = NeuralNetwork(
                        layer_sizes=[n_inputs, n_encoding, n_inputs],
                        mode="regression",
                        hidden_activation=hidden_activation,
                        learning_rate=learning_rate,
                        seed=seed,
                    )


    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int | None = None):
        self.network.fit(X, X, epochs=epochs, batch_size=batch_size)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.network.W[0] + self.network.b[0]
        return self.network.act(z)

    def reconstruction_error(self, X: np.ndarray) -> float:
        recon = self.network.forward(X)
        return mse(X, recon)

class AutoencoderPredictor:

    def __init__(self, n_inputs, n_encoding, n_hidden2,
                 n_outputs, mode="classification",
                 hidden_activation="tanh",
                 learning_rate=0.01, ae_epochs=500,
                 freeze_encoder=True, seed=None):
        self.n_inputs = n_inputs
        self.n_encoding = n_encoding
        self.n_hidden2 = n_hidden2
        self.n_outputs = n_outputs
        self.mode = mode
        self.hidden_activation = hidden_activation
        self.learning_rate = learning_rate
        self.ae_epochs = ae_epochs
        self.freeze_encoder = freeze_encoder
        self.seed = seed
        self._pretrained = False
        self.network = NeuralNetwork(
            layer_sizes=[n_inputs, n_encoding, n_hidden2, n_outputs],
            mode=mode,
            hidden_activation=hidden_activation,
            learning_rate=learning_rate,
            seed=seed,
        )

    def fit(self, X, y, epochs=100, batch_size=None):  # ← 4 spaces
        if not self._pretrained:
            ae = Autoencoder(
                n_inputs=self.n_inputs,
                n_encoding=self.n_encoding,
                hidden_activation=self.hidden_activation,
                learning_rate=self.learning_rate,
                seed=self.seed,
            )
            ae.fit(X, epochs=self.ae_epochs, batch_size=batch_size)
            W_enc, b_enc = ae.network.get_layer_weights(0)
            self.network.set_layer_weights(0, W_enc, b_enc)
            self._pretrained = True
        frozen = [0] if self.freeze_encoder else None
        self.network.fit(X, y, epochs=epochs,
                         batch_size=batch_size,
                         frozen_layers=frozen)
        return self


    def predict(self, X):
        return self.network.predict(X)

    def predict_proba(self, X):
        return self.network.predict_proba(X)




def build_autoencoder_network(autoencoder: Autoencoder,
                               n_hidden2: int,
                               n_outputs: int,
                               mode: str = "classification",
                               freeze_encoder: bool = True,
                               **kwargs) -> NeuralNetwork:
    """
    Build  Input → Encoding → Hidden 2 → Prediction from an already-trained Autoencoder.

    Copies the encoder's W[0], b[0] into a 4-layer NeuralNetwork.
    The caller is responsible for training (and optionally freezing
    layer 0 via frozen_layers=[0] in the .fit()).
    """

    net = NeuralNetwork(
        layer_sizes=[autoencoder.n_inputs,
                     autoencoder.n_encoding,
                     n_hidden2,
                     n_outputs],
        mode=mode,
        hidden_activation=autoencoder.hidden_activation,
        learning_rate=kwargs.get("learning_rate", 0.01),
        seed=kwargs.get("seed", None),
    )

    # Transfer trained encoder weights
    W_enc, b_enc = autoencoder.network.get_layer_weights(0)
    net.set_layer_weights(0, W_enc, b_enc)

    return net
