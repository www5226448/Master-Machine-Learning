import os
import random

import numpy as np
from tqdm import tqdm

from XGBoost.decision_tree import XGBoostRegressionTree
from XGBoost.utils import to_categorical, LogisticLoss, SquareLoss


class XGBoost:

    def __init__(self,
                 n_estimators=30,
                 max_depth=10,
                 learning_rate=0.1,
                 min_samples_split=2,
                 reg_lambda=1,
                 min_impurity=1e-7,
                 min_child_weight=1,
                 sub_sample=0.8,
                 sub_feature=0.8,
                 n_jobs=None,
                 random_state=None
                 ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.sub_sample = sub_sample
        self.sub_feature = sub_feature
        if n_jobs is None:
            self.n_jobs = os.cpu_count()
        self.random_state = random_state

    def __initalize(self):
        self.estimators = [
            XGBoostRegressionTree(
                min_samples_split=self.min_samples_split,
                min_impurity=self.min_impurity,
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                loss=self.loss)
            for _ in range(self.n_estimators)
        ]

    def __sample(self, X):
        random.seed(a=self.random_state)  # fixed a random seed
        n, m = X.shape
        sub_n = int(n * self.sub_sample)
        sub_m = int(m * self.sub_feature)
        self.shuffle_feture = [random.sample([i for i in range(m)], sub_m)
                               for _ in range(self.n_estimators)]
        self.shuffle_sample = [random.sample([i for i in range(n)], sub_n)
                               for _ in range(self.n_estimators)]

    def fit(self, X, y):
        self.__initalize()
        self.__sample(X)
        y_pred = np.ones_like(y)
        for i, tree in tqdm(enumerate(self.estimators)):
            sample_index = self.shuffle_sample[i]
            feture_index = self.shuffle_feture[i]

            sub_X = X[sample_index, :][:, feture_index]

            sub_y = y[sample_index]
            sub_y_pred = y_pred[sample_index]

            y_and_pred = np.concatenate((sub_y, sub_y_pred), axis=1)

            tree.fit(sub_X, y_and_pred)
            update_pred = tree.predict(X[:, feture_index])

            y_pred += np.multiply(self.learning_rate, update_pred)

        return self

    def predict(self, X):
        y_pred = None
        for i, tree in enumerate(self.estimators):

            feture_index = self.shuffle_feture[i]

            sub_X = X[:, feture_index]
            update_pred = tree.predict(sub_X)
            if y_pred is None:
                y_pred = np.ones_like(update_pred)
            y_pred += np.multiply(self.learning_rate, update_pred)

        return y_pred


class XGBRegressor(XGBoost):
    def __init__(self, n_estimators=30,
                 max_depth=8,
                 learning_rate=1,
                 min_samples_split=2,
                 min_impurity=1e-7,
                 min_child_weight=1,
                 sub_sample=0.8,
                 sub_feature=0.8,
                 reg_lambda=1,
                 n_jobs=1,
                 random_state=None):
        super().__init__(n_estimators=n_estimators,
                         max_depth=max_depth,
                         learning_rate=learning_rate,
                         min_samples_split=min_samples_split,
                         min_impurity=min_impurity,
                         min_child_weight=min_child_weight,
                         sub_sample=sub_sample,
                         sub_feature=sub_feature,
                         reg_lambda=reg_lambda,
                         n_jobs=n_jobs,
                         random_state=random_state)
        self.loss = SquareLoss()

    def fit(self, X, y):
        y = y[:, np.newaxis]
        super().fit(X, y)
        return self

    def predict(self, X):
        y_pred = super().predict(X)
        return y_pred.flatten()


class XGBClassifier(XGBoost):
    def __init__(self, n_estimators=30,
                 max_depth=8,
                 learning_rate=0.3,
                 min_samples_split=2,
                 min_impurity=1e-7,
                 min_child_weight=1,
                 sub_sample=0.8,
                 sub_feature=0.8,
                 reg_lambda=1,
                 n_jobs=1,
                 random_state=None):
        super().__init__(n_estimators=n_estimators,
                         max_depth=max_depth,
                         learning_rate=learning_rate,
                         min_samples_split=min_samples_split,
                         min_impurity=min_impurity,
                         min_child_weight=min_child_weight,
                         sub_sample=sub_sample,
                         sub_feature=sub_feature,
                         reg_lambda=reg_lambda,
                         n_jobs=n_jobs,
                         random_state=random_state
                         )
        self.loss = LogisticLoss()

    def fit(self, X, y):
        y = to_categorical(y)
        super().fit(X, y)
        return self

    def predict_prob(self, X):
        y_pred = super().predict(X)
        # Turn into probability distribution (Softmax)
        y_prob = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        return y_prob

    def predict(self, X):
        y_prob = self.predict_prob(X)
        # Set label to the value that maximizes probability
        y_pred = np.argmax(y_prob, axis=1)
        return y_pred


if __name__ == '__main__':
    from sklearn.datasets import make_classification, make_regression
    from sklearn.metrics import mean_squared_error, accuracy_score

    for i in range(20, 200, 10):
        X, y = make_classification(n_samples=300, n_classes=3, n_informative=4, n_features=8)
        clf = XGBClassifier(n_estimators=i).fit(X, y)
        print(accuracy_score(y, clf.predict(X)))

        X, y = make_regression(n_samples=300)
        clf = XGBRegressor(n_estimators=i).fit(X, y)
        print(mean_squared_error(y, clf.predict(X)))
