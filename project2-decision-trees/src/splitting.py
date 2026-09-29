# Splitting
import numpy as np

def entropy(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    ps = counts / len(y)
    return float(-np.sum(ps * np.log2(ps)))

def gain_ratio(y, split_y):
    # split_y is a list of arrays, one for each child
    parent_entropy = entropy(y)
    n = len(y)

    weighted_entropy = 0.0
    split_info = 0.0
    for child in split_y:
        if len(child) == 0: continue
        weight = len(child) / n
        weighted_entropy += weight * entropy(child)
        split_info -= weight * np.log2(weight)

    gain = parent_entropy - weighted_entropy
    return gain / split_info if split_info > 0 else 0.0

def calculate_mse(y):
    if len(y) == 0: return 0.0
    return np.mean((y - np.mean(y))**2)

def regression_split_score(y, split_y):
    n = len(y)
    weighted_mse = sum([len(child)/n * calculate_mse(child) for child in split_y])
    return weighted_mse