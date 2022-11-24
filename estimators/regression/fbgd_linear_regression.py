import numpy as np

from estimators.base_estimator import BaseEstimator
from metrics.regression_metrics import RegressionMetrics

np.random.seed(42)


class FBGDLinearRegression(BaseEstimator):
    def __init__(self, n_features=1, n_steps=2000, alpha=1e-2, penalty="l2", lmd=0):
        self.n_features = n_features
        self.n_steps = n_steps
        self.alpha = alpha
        self.penalty = penalty
        self.lmd = lmd
        self.lmd_ = None
        self.theta = np.random.randn(n_features)

    def cost_func(self, m, pred, y):
        if self.penalty == "l2":
            reg_cost = np.sum(self.lmd_ * np.dot(self.theta.T, self.theta))
        else:
            reg_cost = np.sum(self.lmd_ * np.abs(self.theta))

        error = pred - y
        return 1 / (2 * m) * (np.dot(error.T, error) + reg_cost)

    def fit(self, x_train, y_train, x_val, y_val):
        m = x_train.shape[0]
        m_val = x_val.shape[0]

        cost_history = np.zeros(self.n_steps)
        cost_history_val = np.zeros(self.n_steps)
        theta_history = np.zeros((self.n_steps, self.n_features))

        self.lmd_ = np.full(self.n_features, self.lmd)
        self.lmd_[0] = 0

        for step in range(self.n_steps):
            pred = np.dot(x_train, self.theta)
            pred_val = np.dot(x_val, self.theta)

            error = pred - y_train

            if self.penalty == "l2":
                reg_gd = np.dot(self.lmd_, self.theta)
            else:
                reg_gd = (np.dot(self.lmd_, np.abs(self.theta) / self.theta)) / 2

            self.theta -= (1 / m) * ((self.alpha * np.dot(x_train.T, error)) + reg_gd)

            cost_history[step] = self.cost_func(m, pred, y_train)
            cost_history_val[step] = self.cost_func(m_val, pred_val, y_val)
            theta_history[step] = self.theta.T

        return cost_history, cost_history_val, theta_history

    def predict(self, x):
        x_pred = np.c_[np.ones(x.shape[0]), x]
        return np.dot(x_pred, self.theta)

    def compute_performance(self, x, y):
        pred = self.predict(x)
        metrics = RegressionMetrics(pred, y)
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

    def cost_grid(self, X, Y, A, B, first_dim, second_dim):
        result = np.zeros((100, 100))

        for r in range(result.shape[0]):
            for c in range(result.shape[1]):
                temp_theta = self.theta[:]
                temp_theta[first_dim] = A[r, c]
                temp_theta[second_dim] = B[r, c]
                result[r, c] = np.average((X @ temp_theta - Y) ** 2) * 0.5

        return result