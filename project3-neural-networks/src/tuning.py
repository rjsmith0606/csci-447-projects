"""
Hyperparameter tuning.

Tune on a validation split carved out of the TRAINING data (never the test
fold). Things to tune and explain in the report:
- hidden nodes in layer 1 and layer 2 (feedforward network)
- encoding size (strictly < number of inputs), hidden 2 size (autoencoder net)
- learning rate, momentum, number of epochs / batch size

Save tuning grids and chosen values to results/tuning/ so the report can cite them.
"""

import numpy as np


def grid_search(model_factory, param_grid: dict, X: np.ndarray, y: np.ndarray,
                mode: str = "classification", seed: int = 0):
    """Return (best_params, results_table)."""
    raise NotImplementedError
