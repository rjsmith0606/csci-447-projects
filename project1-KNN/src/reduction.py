"""
Implementation of edited KNN and condensed KNN.
"""

import numpy as np
from knn import k_nearest_neighbors_classification, k_nearest_neighbors_regression


def edited_knn(features, labels, epsilon=0.1, task='classification'):
    """
    Implements Edited KNN.
    Removes examples that are misclassified by their 1-nearest neighbor.
    """
    # Requirement: Use k=1 during the editing process [1]
    k_edit = 1
    reduced_features = []
    reduced_labels = []

    for i in range(len(features)):
        # Create training set excluding the current point to avoid self-matching
        train_feat = np.delete(features, i, axis = 0)
        train_lab = np.delete(labels, i, axis = 0)
        test_feat = features[i].reshape(1, -1)

        if task == 'classification':
            pred = k_nearest_neighbors_classification(train_feat, train_lab, test_feat, k = k_edit)
            # Keep if the 1-NN prediction matches the actual label [1]
            if pred[0] == labels[i]:
                reduced_features.append(features[i])
                reduced_labels.append(labels[i])

        elif task == 'regression':
            pred = k_nearest_neighbors_regression(train_feat, train_lab, test_feat, k = k_edit)
            # Keep if the prediction error is within the tuned threshold epsilon [1]
            if abs(pred[0] - labels[i]) <= epsilon:
                reduced_features.append(features[i])
                reduced_labels.append(labels[i])

    return np.array(reduced_features), np.array(reduced_labels)