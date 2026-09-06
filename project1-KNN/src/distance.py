"""
This module contains functions for calculating distances between data points, which are used in the KNN algorithm.
"""

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    point1 (list or tuple): The first point as a list or tuple of coordinates.
    point2 (list or tuple): The second point as a list or tuple of coordinates.

    Returns:
    float: The Euclidean distance between the two points.
    """
    if len(point1) != len(point2):
        raise ValueError("Points must have the same number of dimensions.")
    
    distance = sum((a - b) ** 2 for a, b in zip(point1, point2)) ** 0.5
    return distance


def manhattan_distance(point1, point2):
    """
    Calculate the Manhattan distance between two points.

    Parameters:
    point1 (list or tuple): The first point as a list or tuple of coordinates.
    point2 (list or tuple): The second point as a list or tuple of coordinates.

    Returns:
    float: The Manhattan distance between the two points.
    """
    if len(point1) != len(point2):
        raise ValueError("Points must have the same number of dimensions.")
    
    distance = sum(abs(a - b) for a, b in zip(point1, point2))
    return distance