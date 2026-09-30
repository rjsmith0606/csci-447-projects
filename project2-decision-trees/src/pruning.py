import numpy as np


def reduced_error_prune(tree, validation_features, validation_labels):
    """
    Perform reduced error pruning (post-pruning) on a decision tree.
    - Tag verticies for possible pruning
    - Test possible pruned trees by removing tagged vertices one at a time
    - Test the best against the original on validation set
    - Accept pruned tree if no worse
    - Repeat until performance on validation set starts to degrade

    Parameters:
    - tree: The decision tree to be pruned.
    - validation_features: The validation set features
    - validation_labels: The validation set labels

    Returns:
    - Nothing, but edits the tree in place.
    """
    pass