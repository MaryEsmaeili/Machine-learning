import numpy as np
from id3_classifier import ID3Classifier

class ID3ClassifierWithThresholds(ID3Classifier):
    def _id3(self, data, features, depth=0):
        """Recursive ID3 algorithm with threshold splits for continuous features."""
        X, y = data[:, :-1], data[:, -1]
        
        if np.all(y == y[0]):
            return y[0]
        
        if not features or (self.max_depth is not None and depth == self.max_depth):
            return np.bincount(y.astype(int)).argmax()

        best_feature, best_threshold = self._get_best_split(data, features)
        if best_threshold is not None:
            tree = {f"{best_feature} <= {best_threshold}": {}}
            left_subset = data[data[:, best_feature] <= best_threshold]
            right_subset = data[data[:, best_feature] > best_threshold]
            tree[f"{best_feature} <= {best_threshold}"]["left"] = self._id3(left_subset, features, depth + 1)
            tree[f"{best_feature} <= {best_threshold}"]["right"] = self._id3(right_subset, features, depth + 1)
        else:
            tree = {best_feature: {}}
            for value in np.unique(X[:, best_feature]):
                subset = data[X[:, best_feature] == value]
                tree[best_feature][value] = self._id3(subset, features, depth + 1)
                
        return tree

    def _get_best_split(self, data, features):
        """Find the best feature and threshold to split on for continuous features."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(len(features)):
            thresholds = self._get_possible_thresholds(data[:, feature_idx])
            for threshold in thresholds:
                gain = self._information_gain_threshold(data, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        return best_feature, best_threshold

    def _predict_single(self, x):
        """Predict a single instance using the decision tree."""
        tree = self.tree
        while isinstance(tree, dict):
            # Extract the feature and threshold if it is a threshold-based split
            feature = next(iter(tree))  # Get the first key of the tree (e.g., 'Feature 0 <= 1.5')

            # Print the feature string for debugging
            print(f"Current feature string: {feature}")

            if "<=" in feature:
                # Handle threshold-based splits (e.g., 'Feature 0 <= 1.5')
                feature_idx_str, threshold_str = feature.split(" <= ")
                feature_idx = int(feature_idx_str.split()[1])  # Extract the feature index (e.g., '0')
                threshold = float(threshold_str)  # Extract the threshold (e.g., '1.5')

                # Compare the feature value in the test instance with the threshold
                if x[feature_idx] <= threshold:
                    tree = tree[feature]['left']
                else:
                    tree = tree[feature]['right']
            else:
                # Handle categorical splits (if any)
                feature_idx = int(feature.split()[1])  # Extract feature index (e.g., '0')
                feature_value = x[feature_idx]
                if feature_value in tree[feature]:
                    tree = tree[feature][feature_value]
                else:
                    return None  # Handle unseen values if necessary
        return tree



    def _get_possible_thresholds(self, feature_column):
        """Calculate possible thresholds as the midpoints between sorted unique values."""
        sorted_values = np.unique(feature_column)
        thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2
        return thresholds

    def _information_gain_threshold(self, data, feature_idx, threshold):
        """Calculate the information gain for a given feature and threshold split."""
        X, y = data[:, :-1], data[:, -1]
        total_entropy = self._entropy(y)
        left_split = y[X[:, feature_idx] <= threshold]
        right_split = y[X[:, feature_idx] > threshold]
        weighted_entropy = (len(left_split) / len(y)) * self._entropy(left_split) + (len(right_split) / len(y)) * self._entropy(right_split)
        return total_entropy - weighted_entropy
