# Regression and Classifcation Tree

import numpy as np
from src.tree_node import TreeNode
from src.splitting import gain_ratio, regression_split_score, calculate_mse

class DecisionTree:
    def __init__(self, mode='classification'):
        self.mode = mode
        self.root = None

    # Builds tree by looking at every possible split point. Uses a max the Gain Ratio (for classification) or min MSE (for regression)
    def build_tree(self, X, y, feature_types, is_root=True):
        """
        Builds a decision tree recursively.
 
        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        y (np.ndarray): Target vector of shape (n_samples,)
        feature_types (dict): Dictionary mapping feature names to their types ('numeric' or 'categorical')
        is_root (bool): True for the top-level call (sets self.root); the recursive calls pass False
 
        Returns:
        TreeNode: Root node of the built decision tree.
        """
        # Value this node would predict if it were a leaf (also stored on internal nodes for pruning / unseen categories)
        if self.mode == 'classification':
            classes, counts = np.unique(y, return_counts=True)
            node_value = classes[counts.argmax()]
        else:
            node_value = np.mean(y)
 
        # Base case: all same class or no features left creates leaf
        if len(np.unique(y)) == 1:
            node = TreeNode(is_leaf=True, value=y[0])
            if is_root: self.root = node
            return node
 
        # Set up for search of best split. A split has to beat "no split" (gain ratio > 0, or MSE lower than the parent's)
        best_gain = 0 if self.mode == 'classification' else calculate_mse(y) - 1e-12
        best_split = None
 
        for feat_idx in range(X.shape[1]):
            feat_name = list(feature_types.keys())[feat_idx]
            col = X[:, feat_idx]
 
            if len(np.unique(col)) < 2: # constant feature can't split anything
                continue
 
            # Checking all possible splits for numeric features
            if feature_types[feat_name] == 'numeric':
                vals = np.unique(col)
                thresholds = (vals[:-1] + vals[1:]) / 2   # empty if only one unique value
                for t in thresholds:
                    left_mask = col <= t
                    right_mask = ~left_mask
                    split_y = [y[left_mask], y[right_mask]]
                    score = gain_ratio(y, split_y) if self.mode == 'classification' else regression_split_score(y, split_y)
 
                    if (self.mode == 'classification' and score > best_gain) or \
                       (self.mode == 'regression' and score < best_gain):
                        best_gain, best_split = score, (feat_idx, t, 'numeric', split_y)
            else: # Checking all possible splits for categorical features
                values = np.unique(col)
                split_y = [y[col == v] for v in values]
                score = gain_ratio(y, split_y) if self.mode == 'classification' else regression_split_score(y, split_y)
 
                if (self.mode == 'classification' and score > best_gain) or \
                   (self.mode == 'regression' and score < best_gain):
                    best_gain, best_split = score, (feat_idx, values, 'categorical', split_y)
 
        if best_split is None: # if it has no valid split node becomes leaf
            node = TreeNode(is_leaf=True, value=node_value)
            if is_root: self.root = node
            return node
 
        feat_idx, thresh, split_type, _ = best_split
        node = TreeNode(feature_index=feat_idx, threshold=thresh, value=node_value)
 
        if split_type == 'numeric':
            node.children = {
                'left': self.build_tree(X[X[:, feat_idx] <= thresh], y[X[:, feat_idx] <= thresh], feature_types, is_root=False),
                'right': self.build_tree(X[X[:, feat_idx] > thresh], y[X[:, feat_idx] > thresh], feature_types, is_root=False)
            }
        else:
            for val in thresh:
                mask = X[:, feat_idx] == val
                node.children[val] = self.build_tree(X[mask], y[mask], feature_types, is_root=False)
 
        if is_root: self.root = node
        return node


    def predict_single(self, node, x):
        """
        Predict the class or value for a single sample x using the decision tree rooted at node.
 
        Parameters:
        node (TreeNode): The root of the decision tree to use for prediction.
        x (np.ndarray): The input sample for which to predict the class or value.
 
        Returns:
        The predicted class or value for the sample x.
        """
        if node.is_leaf: return node.value
        if node.feature_index is None: return node.value
 
        val = x[node.feature_index]
        if isinstance(node.threshold, (int, float)):
            branch = 'left' if val <= node.threshold else 'right'
        else:
            branch = val
 
        if branch not in node.children: return node.value # category never seen at this node in training
 
        return self.predict_single(node.children[branch], x)