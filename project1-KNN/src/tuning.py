"""
Hyperparameter tuning for the KNN algorithm.
"""
import numpy as np
import random as random

from src.evaluation import train_test_split, classification_error, mean_squared_error
from src.knn import k_nearest_neighbors_classification, k_nearest_neighbors_regression
from src.reduction import edited_knn, condensed_knn

def sample_uniform(low, high):
    return random.uniform(low, high)
 
 
def sample_loguniform(low, high):
    """Sample on a log scale -- use this for gamma, which can span
    several orders of magnitude (e.g. 0.001 to 10). Provides a more uniform sampling across the range of values.
    A plain uniform draw would spend most of its picks on values between, say, 1 and 10, and rarely land anywhere
    near 0.001 to 0.01"""
    log_low, log_high = np.log10(low), np.log10(high)
    return 10 ** random.uniform(log_low, log_high)
 
 
def sample_int_odd(low, high):
    """Sample an odd integer k, to reduce classification tie frequency."""
    choices = [i for i in range(low, high + 1) if i % 2 == 1]
    return random.choice(choices)

def tune_knn_classification_params(features, labels, n=30):
    """
    Tune hyperparameters for KNN classification using random search.
    Returns the best k value.
    """
    k_range=(1, max(1, int((np.sqrt(len(labels)))))) # Set upper limit to half the square root of the number of samples
    iterations = n
    best_score = float('inf')
    best_k = None

    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.2)

    for _ in range(iterations):
        k = sample_int_odd(*k_range)
        score = classification_error(label_test, k_nearest_neighbors_classification(feature_train, label_train, feature_test, k=k))

        if score < best_score:
            best_score = score
            best_k = k

    return (best_k, best_score)


def tune_knn_regression_params(features, labels, n=30):
    """
    Tune hyperparameters for KNN regression using random search.
    Returns the best k value.
    """
    k_range=(1, max(1, int((np.sqrt(len(labels)))))) # Set upper limit to half the square root of the number of samples
    gamma_range=(0.0001, 100) 
    iterations = n
    best_score = float('inf')
    best_k = None
    best_gamma = None

    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.2)

    for _ in range(iterations):
        k = sample_int_odd(*k_range)
        gamma = sample_loguniform(*gamma_range)
        score = mean_squared_error(label_test, k_nearest_neighbors_regression(feature_train, label_train, feature_test, k=k, gamma=gamma))

        if score < best_score:
            best_score = score
            best_k = k
            best_gamma = gamma

    return (best_k, best_gamma, best_score)


def tune_epsilon_param(features, labels, n=30, method='edited'):
    """
    Tune the epsilon hyperparameter for KNN regression using random search.
    Returns the best epsilon value.
    """
    epsilon_range= np.std(labels) * np.array([0.001, 0.15]) # 1% to 20% of the standard deviation
    iterations = n
    best_score = float('inf')
    best_epsilon = None
    best_edited_features_len = None

    for _ in range(iterations):

        epsilon = sample_uniform(*epsilon_range)

        if method == 'edited':
            edited_features, edited_labels = edited_knn(features, labels, epsilon=epsilon, task='regression')
        elif method == 'condensed':
            edited_features, edited_labels = condensed_knn(features, labels, epsilon=epsilon, task='regression')
        else:
            raise ValueError("Invalid method specified. Choose 'edited' or 'condensed'.")

        feature_train, feature_test, label_train, label_test = train_test_split(edited_features, edited_labels, test_size=0.2)
        score = mean_squared_error(label_test, k_nearest_neighbors_regression(feature_train, label_train, feature_test, k=1))

        if score < best_score:
            best_score = score
            best_epsilon = epsilon
            best_edited_features_len = len(edited_features)

    return best_epsilon, best_score, best_edited_features_len