"""
Hyperparameter tuning with random search.

Workflow (per model, per data set):
1. holdout_split() once with a fixed seed: ~20% becomes the tuning (validation) set,
   the other ~80% is what the 5x2 CV later runs on. Share this split across all three models.
2. random_search() trains each sampled config on the 80% and scores it on the 20%.
3. Fix the winning config and run cross_validation_loop on the 80%.
"""

import numpy as np
import copy

from src.evaluation import classification_error
from src.losses import mse


# ---------------------------------------------------------------------------
# Holdout split
# ---------------------------------------------------------------------------

def holdout_split(n, y_labels=None, holdout_size=0.2, seed=0):
    """
    Split row indices into (cv_idx, val_idx).

    n:            number of rows.
    y_labels:     (n,) integer class indices for a stratified split (classification),
                  or None for a plain random split (regression).
    holdout_size: fraction held out for tuning (0.2).
    Returns two sorted index arrays. Use X[cv_idx] for the 5x2 CV data and X[val_idx]
    for the tuning set.
    """
    rng = np.random.default_rng(seed)

    if y_labels is not None:
        y_labels = np.asarray(y_labels)
        held = []
        for c in np.unique(y_labels):
            idx = rng.permutation(np.where(y_labels == c)[0])  # shuffle within the class
            held.append(idx[: int(round(len(idx) * holdout_size))])
        val_idx = np.concatenate(held)
    else:
        val_idx = rng.permutation(n)[: int(round(n * holdout_size))]

    val_idx = np.sort(val_idx)
    train_idx = np.setdiff1d(np.arange(n), val_idx)
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Search spaces (candidate values for each hyperparameter)
# ---------------------------------------------------------------------------

# TODO - Finish hyperparameter search spaces for all three algorithms. The values below are just placeholders.
SEARCH_SPACES = {
    "genetic_algorithm": {
        "pop_size": [10, 20, 50],
        "n": [200, 500, 1000],
        "mutation_rate": [0.01, 0.05, 0.1, 0.2],
        "crossover_rate": [0.6, 0.8, 0.9],
        "tournament_size": [2, 3, 5],
    },
    "differential_evolution": {
        "pop_size": [10, 20, 50],
        "n": [50, 100, 200],
        "scaling_factor": [0.3, 0.5, 0.7, 0.9],
        "crossover_prob": [0.1, 0.3, 0.5, 0.9],
    },
}


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------

def random_search(n_trials, X_train, y_train, X_val, y_val, y_labels_val, network, algorithm,
                  mode="classification", seed=0):
    """
    Randomly sample hyperparameter configurations, train each on the training set,
    score it on the validation set, and return the best one.

    n_trials:         number of configs to sample.
    X_train, y_train: training data (the ~80% from holdout_split).
    X_val, y_val:     validation / tuning data (the ~20% from holdout_split).
    y_labels_val:     (n_val,) integer class labels for X_val (classification),
                      or None for regression.
    network:          untrained model instance (deep-copied for every run).
    algorithm:        training function, e.g. genetic_algorithm or differential_evolution.
    mode:             "classification" (scored by classification error) or "regression" (MSE).
    seed:             seeds the config sampling, so the search is reproducible.

    Returns (best_params, best_score, trials), where trials is a list of
    (params, score) for every sampled config, in the order they were tried.
    """
    name = algorithm.__name__
    if name not in SEARCH_SPACES:
        raise ValueError(f"No search space defined for algorithm: {name}")
    space = SEARCH_SPACES[name]

    rng = np.random.default_rng(seed)

    best_params = None
    best_score = float("inf")
    trials = []
    for _ in range(n_trials):
        # Pick one value per hyperparameter (indexing keeps plain Python ints/floats)
        sampled_params = {k: v[rng.integers(len(v))] for k, v in space.items()}

        # Train a fresh copy on the training set and score it on the validation set
        model = copy.deepcopy(network)  # fresh, untrained copy
        algorithm(model, X_train, y_train, **sampled_params)

        if mode == "classification":
            score = classification_error(y_labels_val, model.predict(X_val))
        else:
            score = mse(y_val, model.predict(X_val))
        trials.append((sampled_params, score))

        # Keep the best config seen so far
        if score < best_score:
            best_score = score
            best_params = sampled_params

    return best_params, best_score, trials
