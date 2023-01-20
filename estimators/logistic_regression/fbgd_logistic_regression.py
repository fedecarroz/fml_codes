import numpy as np

from helpers.sigmoid import sigmoid
from metrics.classification_metrics import ClassificationMetrics

seed = 42
np.random.seed(seed)


class FBGDLogisticRegression:
    def __init__(self, n_features=1, epochs=2000, alpha=1e-2, penalty="l2", lmd=0):
        self.n_features = n_features
        self.epochs = epochs
        self.alpha = alpha
        self.penalty = penalty
        self.lmd = lmd
        self.lmd_ = None
        self.theta = np.random.randn(n_features)

    def __cost_func(self, m, pred, y):
        if self.penalty == "l2":
            reg_cost = np.sum(self.lmd_ / 2 * np.dot(self.theta.T, self.theta))
        else:
            reg_cost = np.sum(self.lmd_ / 2 * np.abs(self.theta))

        error = np.dot(y.T, np.log(pred)) + np.dot((1 - y).T, np.log(1 - pred))
        return (-1 / m) * (error - reg_cost)

    def fit(self, X_train, y_train, X_val, y_val):
        m = X_train.shape[0]
        m_val = X_val.shape[0]

        cost_history = np.zeros(self.epochs)
        cost_history_val = np.zeros(self.epochs)
        theta_history = np.zeros((self.epochs, self.n_features))

        self.lmd_ = np.full(self.n_features, self.lmd)
        self.lmd_[0] = 0

        for step in range(self.epochs):
            exp = np.dot(X_train, self.theta)
            pred = sigmoid(exp)
            error = pred - y_train

            exp_val = np.dot(X_val, self.theta)
            pred_val = sigmoid(exp_val)

            if self.penalty == "l2":
                reg_gd = self.lmd_ * self.theta
            else:
                reg_gd = (self.lmd_ * np.abs(self.theta) / self.theta) / 2

            self.theta -= (1 / m) * self.alpha * (np.dot(error, X_train) + reg_gd)

            cost_history[step] = self.__cost_func(m, pred, y_train)
            cost_history_val[step] = self.__cost_func(m_val, pred_val, y_val)
            theta_history[step] = self.theta.T

        return cost_history, cost_history_val, theta_history

    def predict(self, X_test):
        if X_test.shape[1] != self.theta.shape[0]:
            X_pred = np.c_[np.ones(len(X_test)), X_test]
        else:
            X_pred = X_test

        h = np.dot(X_pred, self.theta)
        return sigmoid(h)

    def compute_performance(self, X_test, y_test):
        pred = self.predict(X_test)
        metrics = ClassificationMetrics(y_test, pred)
        return metrics.compute_errors()

    def get_params(self):
        return {
            'alpha': self.alpha,
            'penalty': self.penalty,
            'lmd': self.lmd,
        }

    def set_params(self, params: dict):
        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params.keys():
                raise Exception("No such parameter")

            setattr(self, key, value)

    def learning_curves(self, X_train, y_train, X_val, y_val):
        print("Calculating learning curves. Please wait...")
        m = X_train.shape[0]
        cost_history = np.zeros(m)
        cost_history_val = np.zeros(m)

        for i in range(m):
            c_h, c_h_v, _ = self.fit(X_train[:i + 1], y_train[:i + 1], X_val, y_val)
            cost_history[i] = c_h[-1]
            cost_history_val[i] = c_h_v[-1]

        return cost_history, cost_history_val

    def cost_grid(self, X, Y, A, B, first_dim, second_dim):
        result = np.zeros((100, 100))

        for r in range(result.shape[0]):
            for c in range(result.shape[1]):
                temp_theta = self.theta[:]
                temp_theta[first_dim] = A[r, c]
                temp_theta[second_dim] = B[r, c]
                result[r, c] = np.average((X @ temp_theta - Y) ** 2) * 0.5

        return result
