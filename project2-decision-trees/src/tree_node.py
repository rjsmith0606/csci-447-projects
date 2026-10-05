# Tree Node
class TreeNode:
    def __init__(self, is_leaf=False, value=None, feature_index=None, threshold=None,
                 children=None, split_type=None, n_samples=0):
        self.is_leaf = is_leaf              # if True, node just returns `value`
        self.value = value                  # majority class / mean of training rows reaching this node
                                            # (stored on INTERNAL nodes too -- pruning and unseen-category fallback need it)
        self.feature_index = feature_index  # index of feature used for splitting
        self.threshold = threshold          # numeric splits only (None for categorical)
        self.split_type = split_type        # 'numeric' | 'categorical' | None for leaves
        self.children = children if children is not None else {}  # 'left'/'right' or category value -> TreeNode
        self.n_samples = n_samples          # number of training rows that reached this nodes