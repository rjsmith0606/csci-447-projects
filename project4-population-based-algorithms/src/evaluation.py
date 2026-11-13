"""
5x2 cross-validation and metrics for all three network types.

5x2 CV: 5 repetitions of 2-fold CV (10 train/test runs per model per data set).
- Stratified folds for classification
- Same fold splits across all three models so results can be compared
  statistically
- Normalization fit on the training fold only
"""

import numpy as np
from src.losses import cross_entropy


def classification_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the fraction of misclassified examples."""
    return float(np.mean(np.asarray(y_true) != np.asarray(y_pred)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the mean squared error."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def five_by_two_splits(y_labels: np.ndarray, mode: str = "classification", seed: int = 0):
    """Yield (rep, fold, train_idx, test_idx) for the 10 runs."""

    rng = np.random.default_rng(seed)  # its own generator: model randomness can't change the splits
    y_labels = np.asarray(y_labels)
    n = len(y_labels)
 
    for rep in range(5):
        if mode == "classification":
            half_a, half_b = [], []
            for c in np.unique(y_labels):
                idx = rng.permutation(np.where(y_labels == c)[0])  # shuffle within the class
                cut = len(idx) // 2
                half_a.append(idx[:cut])
                half_b.append(idx[cut:])
            a, b = np.concatenate(half_a), np.concatenate(half_b)
        elif mode == "regression":
            idx = rng.permutation(n)
            a, b = idx[: n // 2], idx[n // 2:]
        else:
            raise ValueError("Unsupported mode. Use 'classification' or 'regression'.")
 
        yield rep, 0, a, b   # train on A, test on B
        yield rep, 1, b, a   # train on B, test on A

def cross_validation_loop(X, y, y_labels, model_factory, mode="classification",
                          epochs=100, batch_size=None, seed=None):
    """
    Run 5x2 CV (5 repetitions of 2-fold CV = 10 runs) for ONE model type on ONE data set"""
    labels_for_split = y_labels if mode == "classification" else np.zeros(len(X))
 
    if mode == "classification":
        results = {"error": [], "cross_entropy": []}
    else:
        results = {"mse": []}
 
    for rep, fold, train_idx, test_idx in five_by_two_splits(labels_for_split, mode, seed=seed): # type: ignore
        model = model_factory()
        model.fit(X[train_idx], y[train_idx], epochs=epochs, batch_size=batch_size)
 
        if mode == "classification":
            results["error"].append(classification_error(y_labels[test_idx], model.predict(X[test_idx])))
            results["cross_entropy"].append(cross_entropy(y[test_idx], model.predict_proba(X[test_idx])))
        else:
            results["mse"].append(mean_squared_error(y[test_idx], model.predict(X[test_idx])))
 
    return results