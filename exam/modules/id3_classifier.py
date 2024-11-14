import numpy as np

class ID3Classifier:
    def __init__(self, max_depth=None):
        self.tree = None
        self.max_depth = max_depth

    def fit(self, X, y, features):
        self.tree = self.id3(X, y, features)

    def id3(self, X, y, features, depth=0):
        if len(np.unique(y)) == 1:
            return np.unique(y)[0]

        if len(features) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return np.bincount(y).argmax()

        gains = [self.information_gain(X, y, feature) for feature in features]
        print("Information gains for features:", gains)  # Debugging print
        
        # If all gains are 0, return the majority class
        if np.all(np.array(gains) == 0):
            return np.bincount(y).argmax()
        
        best_feature_idx = np.argmax(gains)
        best_feature = features[best_feature_idx]

        tree = {best_feature: {}}
        unique_values = np.unique(X[:, best_feature_idx])

        for value in unique_values:
            subset_X = X[X[:, best_feature_idx] == value]
            subset_y = y[X[:, best_feature_idx] == value]
            new_features = [feat for feat in features if feat != best_feature]
            subtree = self.id3(subset_X, subset_y, new_features, depth + 1)
            tree[best_feature][value] = subtree

        return tree

    def predict(self, X):
        y_pred = []
        for instance in X:
            y_pred.append(self._traverse_tree(instance, self.tree))
        return np.array(y_pred)

    def _traverse_tree(self, instance, tree):
        if isinstance(tree, dict):
            feature = list(tree.keys())[0]
            value = instance[feature]
            subtree = tree[feature].get(value, np.bincount(tree).argmax())
            return self._traverse_tree(instance, subtree)
        return tree

    def information_gain(self, X, y, feature_idx):
        original_entropy = self.entropy(y)
        unique_values, counts = np.unique(X[:, feature_idx], return_counts=True)

        if len(unique_values) == 1:
            return 0  # No information gain if the feature doesn't split the data

        weighted_entropy = 0
        for value, count in zip(unique_values, counts):
            subset_y = y[X[:, feature_idx] == value]
            weighted_entropy += (count / len(y)) * self.entropy(subset_y)

        info_gain = original_entropy - weighted_entropy
        if np.isnan(info_gain) or info_gain < 0:
            info_gain = 0
        return info_gain

    def entropy(self, y):
        unique_classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))
