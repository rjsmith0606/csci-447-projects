"""
5x2 cross-validation with a held-out pruning set, plus error/MSE metrics
"""

import copy
import numpy as np

from src.decision_tree import DecisionTree
from src.pruning import reduced_error_prune


def train_test_split(feature_data, label_data, test_size=0.5, rng=None):
    """
    Split the dataset into training and testing sets.

    Parameters:
    feature_data (np.ndarray): Feature data.
    label_data (np.ndarray): Label data.
    test_size (float): Proportion of the dataset to include in the test split (default is 0.5).
    rng (np.random.Generator): Optional random generator so runs can be reproduced.

    Returns:
    tuple: Training and testing sets for features and labels.
    """
    rng = rng if rng is not None else np.random.default_rng()

    # Shuffle the data
    indices = np.arange(len(feature_data))
    rng.shuffle(indices)

    # Calculate the split index
    split_index = int(len(feature_data) * (1 - test_size))

    feature_train = np.asarray(feature_data[indices[:split_index]], dtype=float)
    feature_test = np.asarray(feature_data[indices[split_index:]], dtype=float)
    label_train = np.asarray(label_data[indices[:split_index]])
    label_test = np.asarray(label_data[indices[split_index:]])

    return feature_train, feature_test, label_train, label_test


def classification_error(label_true, label_pred):
    """Classification error rate."""
    return np.mean(label_true != label_pred)


def mean_squared_error(label_true, label_pred):
    """Mean squared error."""
    return np.mean((label_true - label_pred) ** 2)


def predict_all(tree, feature_data):
    """Run every row of feature_data through the tree."""
    return np.array([tree.predict_single(tree.root, x) for x in feature_data])


def cross_validation_loop(feature_data, label_data, feature_types, mode="classification",
                          n_repeats=5, test_size=0.5, prune_size=0.2, seed=None):
    """
    5x2 cross-validation that compares an unpruned tree to a reduced-error-pruned tree.

    1. Pull out `prune_size` (20%) of the data ONCE. It is only used to choose prunes.
    2. On the remaining 80%, repeat n_repeats times: split into two folds (A, B),
       train on A / test on B, then train on B / test on A.
    3. For every fold: grow the tree fully, score it on the test fold (unpruned),
       prune a COPY of it using the pruning set, then score the copy on the same test fold.

    Parameters:
    feature_data (np.ndarray): Feature data (categoricals label-encoded, columns ordered like feature_types).
    label_data (np.ndarray): Label data.
    feature_types (dict): {feature_name: 'numeric' | 'categorical'}
    mode (str): 'classification' or 'regression'.
    n_repeats (int): Number of 2-fold repetitions (5 -> 10 scores per method).
    test_size (float): Fraction of the 80% used as the test fold in each 2-fold split (0.5).
    prune_size (float): Fraction of the full dataset held out for pruning (0.2).
    seed (int): Optional seed for reproducibility.

    Returns:
    dict: {'unpruned': [10 errors], 'pruned': [10 errors]}, paired by fold (index i is the same fold in both).
    """
    if mode == "classification":
        metric = classification_error
    elif mode == "regression":
        metric = mean_squared_error
    else:
        raise ValueError("Unsupported mode. Use 'classification' or 'regression'.")

    rng = np.random.default_rng(seed)

    # Fixed 20% pruning set, never used for training or testing
    cv_feature, prune_features, cv_label, prune_label = train_test_split(
        feature_data, label_data, test_size=prune_size, rng=rng)

    results = {"unpruned": [], "pruned": []}

    for _ in range(n_repeats):
        fold_A_feature, fold_B_feature, fold_A_label, fold_B_label = train_test_split(
            cv_feature, cv_label, test_size=test_size, rng=rng)

        # Train on A / test on B, then reverse the roles
        for train_X, train_y, test_X, test_y in [
            (fold_A_feature, fold_A_label, fold_B_feature, fold_B_label),
            (fold_B_feature, fold_B_label, fold_A_feature, fold_A_label),
        ]:
            tree = DecisionTree(mode=mode)
            tree.build_tree(train_X, train_y, feature_types)
            results["unpruned"].append(metric(test_y, predict_all(tree, test_X)))

            # # Prune a copy so the unpruned tree is left untouched
            pruned_tree = copy.deepcopy(tree)
            reduced_error_prune(pruned_tree, prune_features, prune_label)
            results["pruned"].append(metric(test_y, predict_all(pruned_tree, test_X)))

    return results