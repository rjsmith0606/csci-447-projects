# Splitting
import numpy as np

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def gain_ratio(y, split_y):
    # split_y is a list of arrays, one for each child
    parent_entropy = entropy(y)
    n = len(y)

    weighted_entropy = 0
    split_info = 0
    for child in split_y:
        weight = len(child) / n
        weighted_entropy += weight * entropy(child)
        split_info += weight * np.log2(weight) if weight > 0 else 0

    gain = parent_entropy - weighted_entropy
    return gain / split_info if split_info != 0 else 0

def calculate_mse(y):
    if len(y) == 0: return 0
    return np.mean((y - np.mean(y))**2)

def regression_split_score(y, split_y):
    n = len(y)
    weighted_mse = sum([len(child)/n * calculate_mse(child) for child in split_y])
    return weighted_mse