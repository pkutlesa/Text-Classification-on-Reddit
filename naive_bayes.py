import numpy as np


class BernoulliNaiveBayes(object):
    # constructor, alpha: laplace smoothing parameter (default=1.0)
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.theta_k = None
        self.theta_j_k = None

    def fit(self, X, y):
        count_sample = X.shape[0]
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.theta_k = [np.log(len(i) / count_sample) for i in separated]
        count_j_k = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
        smoothing = 2 * self.alpha
        count_k = np.array([len(i) + smoothing for i in separated])
        self.theta_j_k = count_j_k / count_k[np.newaxis].T
        return self

    def predict_log_probabilities(self, X):
        return [(np.log(self.theta_j_k) * x + np.log(1 - self.theta_j_k) * np.abs(x - 1)
                 ).sum(axis=1) + self.theta_k for x in X]

    def predict(self, X):
        return np.argmax(self.predict_log_probabilities(X), axis=1)
