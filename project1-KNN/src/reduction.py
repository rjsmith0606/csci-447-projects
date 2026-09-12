"""
Implementation of edited KNN and condensed KNN.
"""

import numpy as np
from src.knn import k_nearest_neighbors_classification, k_nearest_neighbors_regression


def edited_knn(features, labels, epsilon=0.1, task='classification'):
    """
    Implements Edited KNN.
    Removes examples that are misclassified by their 1-nearest neighbor.
    """
    # Requirement: Use k=1 during the editing process [1]
    k_edit = 1
    # Normalize pandas inputs so indexing below is always positional.
    features = np.asarray(features)
    labels = np.asarray(labels)
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



def condensed_knn(features, labels, epsilon=0.1, task='classification'):
    """
    Implements Condensed Nearest Neighbor (CNN).
    Starts with a seed and adds examples that are misclassified by the current reduced set.
    """
    # Requirement: Use k=1 during the condensing process [1]
    k_condense = 1
    # Normalize pandas inputs so indexing below is always positional.
    features = np.asarray(features)
    labels = np.asarray(labels)

    # Start with the first example as the seed
    reduced_features = [features[0]]
    reduced_labels = [labels[0]]

    # The remaining examples to be checked
    remaining_features = list(features[1:])
    remaining_labels = list(labels[1:])

    changed = True
    while changed and remaining_features:
        changed = False
        idx = 0
        while idx < len(remaining_features):
            test_feat = remaining_features[idx].reshape(1, -1)
            actual_label = remaining_labels[idx]

            if task == 'classification':
                pred = k_nearest_neighbors_classification(
                    np.array(reduced_features), np.array(reduced_labels), test_feat, k=k_condense
                )
                misclassified = pred[0] != actual_label


            elif task == 'regression':
                # Note: Condensed KNN for regression typically requires an epsilon threshold
                # similar to Edited KNN to define what "misclassified" means.
                # This is a simplified version; you may need to tune epsilon here.
                pred = k_nearest_neighbors_regression(
                    np.array(reduced_features), np.array(reduced_labels), test_feat, k=k_condense
                )
                misclassified = abs(pred[0] - actual_label) > epsilon

            if misclassified: # type: ignore
                reduced_features.append(remaining_features.pop(idx))
                reduced_labels.append(remaining_labels.pop(idx))
                changed = True
                # do NOT increment idx — list shrank, next element shifted into this index
            else:
                idx += 1

    return np.array(reduced_features), np.array(reduced_labels)

