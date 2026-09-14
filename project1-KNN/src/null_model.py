"""
A null model for comparison purposes.
"""

import numpy as np

def null_model_classification(label_column):
    """
    Predicts the most frequent class label for all instances.

    Parameters:
    label_column (np.ndarray): The column of labels from the training data.

    Returns:
    np.ndarray: Predicted class labels for all instances.
    """
    # Find the most common label in the training data
    most_common_label = max(set(label_column), key=list(label_column).count)
    
    # Return an array filled with the most common label
    label_pred = np.full(shape=(len(label_column),), fill_value=most_common_label)
    return np.mean(label_column != label_pred)


def null_model_regression(label_column):
    """
    Predicts the mean value of the label column for all instances.

    Parameters:
    label_column (np.ndarray): The column of labels from the training data.

    Returns:
    np.ndarray: Predicted values for all instances.
    """
    # Calculate the mean of the label column
    mean_value = np.mean(label_column)
    
    # Return an array filled with the mean value
    label_pred = np.full(shape=(len(label_column),), fill_value=mean_value)
    return np.mean((label_column - label_pred) ** 2)