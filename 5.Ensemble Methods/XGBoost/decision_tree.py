import heapq
from abc import ABC, abstractmethod

import numpy as np

from XGBoost.utils import split_by_threshold


class DecisionNode:

    def __init__(self, feature_i=None,
                 threshold=None,
                 value=None,
                 left_branch=None,
                 right_branch=None):
        self.feature_i = feature_i
        self.threshold = threshold
        self.value = value
        self.left_branch = left_branch  # greater than or equal
        self.right_branch = right_branch


class DecisionTree(ABC):

    def __init__(self, min_samples_split=2,
                 min_impurity=1e-7,
                 reg_lambda=0,
                 max_depth=float("inf"),
                 min_child_weight=1,
                 loss=None):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.loss = loss
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight

    def fit(self, X, y):
        """ Build decision tree """
        self.root = self.__build_tree(X, y, 0)

    def __build_tree(self, X, y, current_depth=0):
        """ Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data"""

        n_samples, n_features = X.shape
        quantile_data = np.array([[np.percentile(feature, level * 10, axis=0)
                                   for level in range(1, 10)]
                                  for feature in X.T])
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Calculate the impurity for each feature

            results = []
            for index in range(n_features):
                result = self.get_best_split(X, y, index, quantile_data)
                results.append(result)
            this_spilt = heapq.nlargest(1, results, key=lambda x: x[1])[0]
            spilt_index, largest_impurity, spilt_threshold, left, right = this_spilt

            if largest_impurity > self.min_impurity:
                # Build subtrees for the right and left branches
                X_left, y_left = X[left, :], y[left]
                X_right, y_right = X[right, :], y[right]
                l_branch = self.__build_tree(X_left, y_left, current_depth + 1)
                r_branch = self.__build_tree(X_right, y_right, current_depth + 1)
                return DecisionNode(feature_i=spilt_index, threshold=spilt_threshold,
                                    left_branch=l_branch, right_branch=r_branch)

        leaf_value = self._leaf_value_calculation(y)

        return DecisionNode(value=leaf_value)

    def get_best_split(self, X, y, feature_index, spilt_matrix):
        best_impurity = -np.inf
        groups = (None, None)
        spilt_threshold = None

        quantile_values = spilt_matrix[feature_index]
        for threshold in quantile_values:
            left_index, right_index = split_by_threshold(X, feature_index, threshold)

            if len(left_index) > 0 and len(right_index) > 0:
                # Select the y-values of the two sets
                y1 = y[left_index]
                y2 = y[right_index]
                impurity, *weights = self._impurity_calculation(y, y1, y2)

                if impurity > best_impurity and self.min_child_weight <= min(weights):
                    spilt_threshold = threshold
                    best_impurity = impurity
                    groups = left_index, right_index
        spilt_info = feature_index, best_impurity, spilt_threshold, *groups
        return spilt_info

    def predict_value(self, x, tree=None):
        """ Do a recursive search down the tree and make a prediction of the data sample by the
            value of the leaf that we end up at """
        if tree is None:
            tree = self.root
        # If we have a value (i.e we're at a leaf) => return value as the prediction
        if tree.value is not None:
            return tree.value
        # Choose the feature that we will test
        feature_value = x[tree.feature_i]

        # Determine if we will follow left or right branch
        branch = tree.right_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.left_branch
        elif feature_value == tree.threshold:
            branch = tree.left_branch
        # Test subtree
        return self.predict_value(x, branch)

    def predict(self, X):
        """ Classify samples one by one and return the set of labels """
        y_pred = np.array([self.predict_value(sample) for sample in X])
        return y_pred

    @abstractmethod
    def _impurity_calculation(self, y, y1, y2):
        pass

    @abstractmethod
    def _leaf_value_calculation(self, y):
        pass


class XGBoostRegressionTree(DecisionTree):

    def _split(self, y):
        """ y contains y_true in left half of the middle column and
        y_pred in the right half. Split and return the two matrices """
        col = int(np.shape(y)[1] / 2)
        y, y_pred = y[:, :col], y[:, col:]
        return y, y_pred

    def _gain(self, y, y_pred):
        nominator = (y * self.loss.gradient(y, y_pred)).sum() ** 2
        child_weight = self.loss.hess(y, y_pred).sum()
        denominator = child_weight ** 2 + self.reg_lambda
        gain_value = 0.5 * (nominator / denominator)

        return gain_value, child_weight

    def _impurity_calculation(self, y, y1, y2):
        # Split
        y, y_pred = self._split(y)
        y1, y1_pred = self._split(y1)
        y2, y2_pred = self._split(y2)

        left_gain, left_weight = self._gain(y1, y1_pred)
        right_gain, right_weight = self._gain(y2, y2_pred)
        gain, _ = self._gain(y, y_pred)

        return left_gain + right_gain - gain, left_weight, right_weight

    def _leaf_value_calculation(self, y):
        y, y_pred = self._split(y)

        gradient = np.sum(self.loss.gradient(y, y_pred), axis=0)
        hessian = np.sum(self.loss.hess(y, y_pred), axis=0)
        update_approximation = -gradient / (hessian + self.reg_lambda)

        return update_approximation
