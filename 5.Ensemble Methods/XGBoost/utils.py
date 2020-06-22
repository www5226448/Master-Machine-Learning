from abc import abstractmethod, ABC

import numpy as np
from scipy.special import expit
from sklearn.metrics import mean_squared_error, log_loss,roc_auc_score,recall_score,precision_score



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

    @abstractmethod
    def cost(self, y, y_pred):
        pass



class CrossEntropy(Loss):

    def gradient(self, y, y_pred):
        p = expit(y_pred)
        return p - y

    def hess(self, y, y_pred):
        p = expit(y_pred)
        return p * (1 - p)

    def cost(self, y, y_pred):
        loss = log_loss(y, y_pred)
        return loss

    def auc(self,y,y_pred):
        return roc_auc_score(y,y_pred,average='micro')

    def recall(self,y,y_pred):
        y_pred=np.argmax(y_pred, axis=1) #transform to class label
        return recall_score(y,y_pred,average='micro')

    def precise(self,y,y_pred):
        y_pred = np.argmax(y_pred, axis=1)  # transform to class label
        return precision_score(y,y_pred,average='micro')





class MeanSquare(Loss):

    def gradient(self, y, y_pred):
        return y_pred - y

    def hess(self, y, y_pred):
        return np.ones_like(y)

    def cost(self, y, y_pred):
        loss = mean_squared_error(y, y_pred)
        return loss


def train_valiate_test(X,y,ratio=(5,3,2)):
    ratio=np.array(ratio)/np.sum(ratio)
    if len(X)!=len(y):
        raise ValueError('The X and y must be has same size')
    l,r,*_=ratio

    n=X.shape[0]
    L,R=int(l*n),int(r*n+l*n)

    train=X[:L],y[:L]
    valiate=X[L:R],y[L:R]
    test=X[R:],y[R:]

    return train,valiate,test



