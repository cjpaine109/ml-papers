# -------------------------------------------------------------------------
# ID3 Decision Tree Algorithm Implementation
# Reference: "Induction of Decision Trees" by J.R. Quinlan
# Author: Conner Paine
# Date: 06/12/2024
# -------------------------------------------------------------------------

# IMPORTS
import numpy as np
import pandas as pd
from collections import Counter


class ID3DecisionTree:
    def __init__(self):
        """Initialize an empty decision tree."""
        self.tree = None

    def entropy(self, y):
        """
        Calculate the entropy of a dataset.

        Parameters:
        y (array-like): Target values.

        Returns:
        float: Entropy of the target values.
        """
        class_counts = np.bincount(y)
        probs = class_counts / len(y)
        return -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
    
    def information_gain(self, X, y, feature):
        """
        Calculate the information gain of a feature.

        Parameters:
        X (array-like): Feature dataset.
        y (array-like): Target values.
        feature (int): Index of the feature.

        Returns:
        float: Information gain of the feature.
        """
        total_entropy = self.entropy(y)
        feature_values = X[:, feature]
        unique_values = np.unique(feature_values)
        attribute_gain = 0

        for value in unique_values:
            subset_y = y[feature_values == value]
            subset_proportions = len(subset_y) / len(y)
            attribute_gain += subset_proportions * self.entropy(subset_y)

        return total_entropy - attribute_gain
    
    def best_feature_(self, X, y):
        """
        Determine the best feature to split on.

        Parameters:
        X (array-like): Feature dataset.
        y (array-like): Target values.

        Returns:
        int: Index of the best feature.
        """
        features = list(range(X.shape[1]))
        best_feature = None
        max_gain = -np.inf

        for feature in features:
            gain = self.information_gain(X, y, feature)
            if gain > max_gain:
                max_gain = gain
                best_feature = feature

        return best_feature
    
    def fit(self, X, y, features=None):
        """
        Build the decision tree using the ID3 algorithm.

        Parameters:
        X (array-like): Feature dataset.
        y (array-like): Target values.
        features (list, optional): List of feature indices to consider. Defaults to None.

        Returns:
        dict: The trained decision tree.
        """
        if features is None:
            features = list(range(X.shape[1]))
        
        if len(np.unique(y)) == 1:
            return y[0]
        
        if len(features) == 0:
            return Counter(y).most_common(1)[0][0]
        
        best_feature = self.best_feature_(X, y)
        unique_values = np.unique(X[:, best_feature])
        tree = {best_feature: {}}

        for value in unique_values:
            subset_X = X[X[:, best_feature] == value]
            subset_y = y[X[:, best_feature] == value]
            new_features = [f for f in features if f != best_feature]
            tree[best_feature][value] = self.fit(subset_X, subset_y, new_features)

        self.tree = tree
        return tree
    
    def predict_one(self, x, tree):
        """
        Predict the class label for a single instance.

        Parameters:
        x (dict): A single instance with feature values.
        tree (dict): The decision tree.

        Returns:
        int: The predicted class label.
        """
        if not isinstance(tree, dict):
            return tree
        feature = list(tree.keys())[0]
        if x[feature] in tree[feature]:
            return self.predict_one(x, tree[feature][x[feature]])
        else:
            subtree = tree[feature]
            return Counter([self.predict_one(x, subtree[value]) for value in subtree if value in subtree]).most_common(1)[0][0]
        
    def predict(self, X):
        """
        Predict the class labels for a dataset.

        Parameters:
        X (array-like): Dataset with feature values.

        Returns:
        array: Array of predicted class labels.
        """
        return np.array([self.predict_one(x, self.tree) for x in X])


if __name__ == "__main__":
    # Sample data from paper
    data = {
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
        'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    }

    df = pd.DataFrame(data)
    X = df.drop('Play', axis=1).values
    y = df['Play'].apply(lambda x: 1 if x == 'Yes' else 0).values

    # Train ID3 Decision Tree
    id3 = ID3DecisionTree()
    id3.fit(X, y)

    # Predict on the training data
    predictions = id3.predict(X)

    # Calculate accuracy
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y, predictions)
    print("Accuracy:", acc)
    print(predictions)
    print(y)