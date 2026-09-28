# Tree Node
class TreeNode:
    def __init__(self, is_leaf=False, value=None, feature_index=None, threshold=None, children=None):
        self.is_leaf = is_leaf # if true it stores a value
        self.value = value  # Class label for classification, mean for regression
        self.feature_index = feature_index  # Index of feature used for splitting
        self.threshold = threshold  # Used for numeric binary splits
        self.children = children if children is not None else {} # Map of value/branch -> TreeNode