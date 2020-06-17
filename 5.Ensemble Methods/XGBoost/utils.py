from abc import abstractmethod, ABC

import numpy as np
from scipy.special import expit


def to_categorical(x, n_col=None):
    """ One-hot encoding of nominal values """
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


def split_by_threshold(X, feature_i, threshold):
    left_index = np.where(X[:, feature_i] >= threshold)[0]
    right_index = np.where(X[:, feature_i] < threshold)[0]
    return left_index, right_index


class Loss(ABC):

    @abstractmethod
    def gradient(self, y, y_perd):
        pass

    @abstractmethod
    def hess(self, y, y_pred):
        pass


class LogisticLoss(Loss):

    def gradient(self, y, y_pred):
        p = expit(y_pred)
        return p - y

    def hess(self, y, y_pred):
        p = expit(y_pred)
        return p * (1 - p)


class SquareLoss(Loss):

    def gradient(self, y, y_pred):
        return y_pred - y

    def hess(self, y, y_pred):
        return np.ones_like(y)
