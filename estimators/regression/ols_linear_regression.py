import numpy as np

from metrics.regression_metrics import RegressionMetrics

seed = 42
np.random.seed(seed)


class OLSLinearRegression:
    def __init__(self, n_features=1):
        self.n_features = n_features
        self.theta = np.random.randn(n_features)

    def fit(self, X_train, y_train):
        self.theta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
        return self.theta

    def predict(self, X_test):
        if X_test.shape[1] != self.theta.shape[0]:
            X_pred = np.c_[np.ones(len(X_test)), X_test]
        else:
            X_pred = X_test

        return np.dot(X_pred, self.theta)

    def compute_performance(self, X_test, y_test):
        pred = self.predict(X_test)
        metrics = RegressionMetrics(y_test, pred)
        return metrics.compute_errors()
