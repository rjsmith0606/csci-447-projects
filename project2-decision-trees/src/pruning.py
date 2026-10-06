# Pruning

import numpy as np
from src.tree_node import TreeNode

def evaluate_accuracy(tree, X, y):
    """
    Calculates the accuracy (classification) or MSE (regression)
    of the tree on a given dataset.
    """
    preds = [tree.predict_single(tree.root, x) for x in X]
    if tree.mode == 'classification':
        return np.mean(np.array(preds) == y)
    else:
        # For regression, we return negative MSE because pruning logic
        # usually looks for 'higher' scores to keep.
        return -np.mean((np.array(preds) - y)**2)

def evaluate_accuracy_subtree(tree, node, X, y):
    """
    Helper to predict outcomes using a specific subtree node as the root.
    """
    if len(X) == 0:
        return 0.0

    preds = []
    for x in X:
        curr = node
        while not curr.is_leaf:
            val = x[curr.feature_index]
            # Determine branch based on split type (numeric vs categorical)
            if isinstance(curr.threshold, (int, float)):
                branch = 'left' if val <= curr.threshold else 'right'
            else:
                branch = val

            # Fallback to node value if the category was not seen during training
            if branch not in curr.children:
                preds.append(curr.value)
                break
            curr = curr.children[branch]
        else:
            preds.append(curr.value)

    preds = np.array(preds)
    if tree.mode == 'classification':
        return np.mean(preds == y)
    else:
        return -np.mean((preds - y)**2)

def reduced_error_pruning(tree, X_val, y_val):
    """
    Performs Reduced Error Pruning (post-pruning) on a decision tree.
    Edits the tree in place.
    """
    def prune_recursive(node, X, y):
        if node.is_leaf:
            return node

        # 1. Prune children first (Bottom-Up approach)
        for key in list(node.children.keys()):
            # Filter the validation data that actually reaches this child
            if isinstance(node.threshold, (int, float)):
                mask = (X[:, node.feature_index] <= node.threshold) if key == 'left' else (X[:, node.feature_index] > node.threshold)
            else:
                mask = X[:, node.feature_index] == key

            # Recursively prune the child
            node.children[key] = prune_recursive(node.children[key], X[mask], y[mask])

        if len(y) == 0:
            return TreeNode(is_leaf=True, value=node.value)

        # 2. Evaluate current subtree performance
        current_score = evaluate_accuracy_subtree(tree, node, X, y)

        # 3. Evaluate performance if this node were converted to a leaf
        # Use the stored node.value (majority class or mean)
        if tree.mode == 'classification':
            leaf_score = np.mean([node.value == val for val in y])
        else:
            leaf_score = -np.mean((node.value - y)**2)

        # 4. Prune if the leaf performs better or equal to the subtree
        if leaf_score >= current_score:
            return TreeNode(is_leaf=True, value=node.value)

        return node

    # Start pruning from the root using the validation set
    tree.root = prune_recursive(tree.root, X_val, y_val)