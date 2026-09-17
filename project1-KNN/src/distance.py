"""
This module contains functions for calculating distances between data points, which are used in the KNN algorithm.
"""
import numpy as np

def euclidean_distance(A, B, batch_size=500):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    point1 (list or tuple): The first point as a list or tuple of coordinates.
    point2 (list or tuple): The second point as a list or tuple of coordinates.

    Returns:
    float: The Euclidean distance between the two points.
    """
    # if len(point1) != len(point2):
    #     raise ValueError("Points must have the same number of dimensions.")
    
    # distance = sum((a - b) ** 2 for a, b in zip(point1, point2)) ** 0.5
    # return distance

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.shape[1] != B.shape[1]:
        raise ValueError("Points must have the same number of dimensions.")

    n_a = A.shape[0]
    out = np.empty((n_a, B.shape[0]), dtype=float)

    for start in range(0, n_a, batch_size):
        end = min(start + batch_size, n_a)
        # (batch, 1, d) - (1, n_b, d) -> (batch, n_b, d), then sum of squares over d
        diff = A[start:end, np.newaxis, :] - B[np.newaxis, :, :]
        out[start:end] = np.sqrt(np.sum(diff ** 2, axis=2))

    return out



def manhattan_distance(A, B, batch_size=500):
    """
    Calculate the Manhattan distance between two points.

    Parameters:
    point1 (list or tuple): The first point as a list or tuple of coordinates.
    point2 (list or tuple): The second point as a list or tuple of coordinates.

    Returns:
    float: The Manhattan distance between the two points.
    """
    # if len(point1) != len(point2):
    #     raise ValueError("Points must have the same number of dimensions.")
    
    # distance = sum(abs(a - b) for a, b in zip(point1, point2))
    # return distance

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.shape[1] != B.shape[1]:
        raise ValueError("Points must have the same number of dimensions.")

    n_a = A.shape[0]
    out = np.empty((n_a, B.shape[0]), dtype=float)

    for start in range(0, n_a, batch_size):
        end = min(start + batch_size, n_a)
        diff = A[start:end, np.newaxis, :] - B[np.newaxis, :, :]
        out[start:end] = np.sum(np.abs(diff), axis=2)

    return out