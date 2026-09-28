# Regression and Classifcation Tree

import numpy as np
from tree_node import TreeNode
from splitting import gain_ratio, regression_split_score

class DecisionTree:
    def __init__(self, mode='classification'):
        self.mode = mode
        self.root = None

    # Builds tree by looking at every possible split point. Uses a max the Gain Ratio (for classification) or min MSE (for regression)
    def build_tree(self, X, y, feature_types):
        # Base case: all same class or no features left creates leaf
        if len(np.unique(y)) == 1:
            return TreeNode(is_leaf=True, value=y[0])

        # Set up for search of best split
        best_gain = -1 if self.mode == 'classification' else float('inf')
        best_split = None

        for feat_idx in range(X.shape[1]):
            feat_name = list(feature_types.keys())[feat_idx]
            col = X[:, feat_idx]

            # Checking all possible splits for numeric features
            if feature_types[feat_name] == 'numeric':
                thresholds = np.unique(col)
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
            return TreeNode(is_leaf=True, value=np.bincount(y).argmax() if self.mode == 'classification' else np.mean(y))

        feat_idx, thresh, split_type, _ = best_split
        node = TreeNode(feature_index=feat_idx, threshold=thresh)

        if split_type == 'numeric':
            node.children = {
                'left': self.build_tree(X[X[:, feat_idx] <= thresh], y[X[:, feat_idx] <= thresh], feature_types),
                'right': self.build_tree(X[X[:, feat_idx] > thresh], y[X[:, feat_idx] > thresh], feature_types)
            }
        else:
            for val in thresh:
                mask = X[:, feat_idx] == val
                node.children[val] = self.build_tree(X[mask], y[mask], feature_types)

        return node



    def predict_single(self, node, x):
        if node.is_leaf: return node.value
        if node.feature_index is None: return node.value

        val = x[node.feature_index]
        if isinstance(node.threshold, (int, float)):
            branch = 'left' if val <= node.threshold else 'right'
        else:
            branch = val

        return self.predict_single(node.children[branch], x)