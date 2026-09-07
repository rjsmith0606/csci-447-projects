"""
5x2 cross-validation loop and error/MSE metrics
"""

import pandas as pd
import numpy as np

from src.knn import k_nearest_neighbors_classification, k_nearest_neighbors_regression

def train_test_split(feature_data, label_data, test_size=0.5):
    """
    Split the dataset into training and testing sets.

    Parameters:
    feature_data (np.ndarray): Feature data.
    label_data (np.ndarray): Label data.
    test_size (float): Proportion of the dataset to include in the test split (default is 0.5).

    Returns:
    tuple: Training and testing sets for features and labels.
    """
    # Shuffle the data
    indices = np.arange(len(feature_data))
    np.random.shuffle(indices)

    # Calculate the split index
    split_index = int(len(feature_data) * (1 - test_size))

    # Split the data into training and testing sets
    feature_train = np.asarray(feature_data.iloc[indices[:split_index]], dtype=float)
    feature_test = np.asarray(feature_data.iloc[indices[split_index:]], dtype=float)
    label_train = np.asarray(label_data.iloc[indices[:split_index]])
    label_test = np.asarray(label_data.iloc[indices[split_index:]])


    return feature_train, feature_test, label_train, label_test


def classification_error(label_true, label_pred):
    """
    Calculate the classification error between true and predicted labels.

    Parameters:
    label_true (np.ndarray): True labels.
    label_pred (np.ndarray): Predicted labels.

    Returns:
    float: Classification error rate.
    """
    return np.mean(label_true != label_pred)


def mean_squared_error(label_true, label_pred):
    """
    Calculate the mean squared error between true and predicted labels.

    Parameters:
    label_true (np.ndarray): True labels.
    label_pred (np.ndarray): Predicted labels.

    Returns:
    float: Mean squared error.
    """
    return np.mean((label_true - label_pred) ** 2)


def cross_validation_loop(feature_data, label_data, k=3, model="classification", n_splits=5, test_size=0.5):
    """
    Perform a 5x2 cross-validation loop for model evaluation.

    Parameters:
    feature_data (np.ndarray): Feature data.
    label_data (np.ndarray): Label data.
    model: The type of model to evaluate ('classification' or 'regression').
    n_splits (int): Number of splits for cross-validation (default is 5).
    test_size (float): Proportion of the dataset to include in the test split (default is 0.5).

    Returns:
    list: A list of mean squared errors or classification errors for each fold.
    """

    error_list = []

    for _ in range(n_splits):
        # Split the data into training and testing sets
        fold_A_feature, fold_B_feature, fold_A_label, fold_B_label = train_test_split(feature_data, label_data, test_size=test_size)
        

        # Fit the model on the training data
        if model == "classification":
            label_pred = k_nearest_neighbors_classification(fold_A_feature, fold_A_label, fold_B_feature, k)
            error_list.append(classification_error(fold_B_label, label_pred))

            # Reverse the roles of fold A and fold B for the second evaluation
            label_pred = k_nearest_neighbors_classification(fold_B_feature, fold_B_label, fold_A_feature, k)
            error_list.append(classification_error(fold_A_label, label_pred))

        elif model == "regression":
            # NOTE: KNN regression is not implemented yet, so this part will raise NotImplementedError
            label_pred = k_nearest_neighbors_regression(fold_A_feature, fold_A_label, fold_B_feature, k)
            error_list.append(mean_squared_error(fold_B_label, label_pred))
            
            # Reverse the roles of fold A and fold B for the second evaluation
            label_pred = k_nearest_neighbors_regression(fold_B_feature, fold_B_label, fold_A_feature, k)
            error_list.append(mean_squared_error(fold_A_label, label_pred))
        else:
            raise ValueError("Unsupported model type. Use 'classification' or 'regression'.")

    return error_list