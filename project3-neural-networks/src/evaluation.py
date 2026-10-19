"""
5x2 cross-validation and metrics for all three network types.

5x2 CV: 5 repetitions of 2-fold CV (10 train/test runs per model per data set).
- Stratified folds for classification
- Same fold splits across all three models so results can be compared
  statistically (paired tests)
- Normalization fit on the training fold only
"""

import numpy as np


def classification_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError


def five_by_two_splits(y: np.ndarray, mode: str = "classification", seed: int = 0):
    """Yield (rep, fold, train_idx, test_idx) for the 10 runs."""
    raise NotImplementedError


def cross_validation_loop(X: np.ndarray, y: np.ndarray, model_factory,
                          mode: str = "classification", seed: int = 0):
    """
    Run 5x2 CV for one model on one data set.
    model_factory: callable returning a fresh, untrained model.
    Returns a list of per-fold score dicts.
    """
    raise NotImplementedError


def paired_t_test(scores_a: list[float], scores_b: list[float]):
    """Statistical comparison of two models' per-fold scores."""
    raise NotImplementedError
