"""
Implementation of the K-Nearest Neighbors (KNN) algorithm.
"""
import numpy as np
from collections import Counter
from src.distance import euclidean_distance, manhattan_distance

def k_nearest_neighbors_classification(features_train, labels_train, features_test, k, distance_metric='euclidean'):
    """
    Predict the class labels for the test data using the KNN algorithm.

    Parameters:
    features_train (np.ndarray): Training data features.
    labels_train (np.ndarray): Training data labels.
    features_test (np.ndarray): Test data features.
    k (int): Number of nearest neighbors to consider.
    distance_metric (str): Distance metric to use ('Euclidean' or 'Manhattan').

    Returns:
    np.ndarray: Predicted class labels for the test data.
    """

    predictions = []
    
    for test_point in features_test:
        # Calculate distances from the test point to all training points
        if distance_metric == 'euclidean':
            distances = [euclidean_distance(test_point, train_point) for train_point in features_train]
        elif distance_metric == 'manhattan':
            distances = [manhattan_distance(test_point, train_point) for train_point in features_train]
        else:
            raise ValueError("Unsupported distance metric. Use 'euclidean' or 'manhattan'.")

        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:k]

        # Get the labels of the k nearest neighbors
        k_labels = labels_train[k_indices]

        # Determine the most common label among the neighbors
        most_common_label = Counter(k_labels).most_common(1)[0][0]

        predictions.append(most_common_label)

    return np.array(predictions)


def k_nearest_neighbors_regression(features_train, labels_train, features_test, k, gamma=1.0,
                                   distance_metric='euclidean'):
    """
    Predict values for test data using a Gaussian Kernel for regression.
    Kernel: K(x, xq) = exp(-gamma * ||x - xq||^2) [1]
    """
    if distance_metric != 'euclidean':
        raise ValueError("Regression requires 'euclidean' distance for the Gaussian kernel [1].")

    predictions = []

    for test_point in features_test:
        # 1. Calculate Euclidean distances from the test point to all training points
        distances = np.array([euclidean_distance(test_point, train_point) for train_point in features_train])

        # 2. Get indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:k]

        # 3. Extract the labels and distances for those k neighbors
        k_labels = labels_train[k_indices]
        k_dists = distances[k_indices]

        # 4. Apply the Gaussian Kernel to calculate weights [1]
        # Weight = exp(-gamma * distance^2)
        weights = np.exp(-gamma * (k_dists ** 2))

        # 5. Calculate the weighted average: sum(weight * label) / sum(weights) [1]
        prediction = np.sum(weights * k_labels) / np.sum(weights)
        predictions.append(prediction)

    return np.array(predictions)