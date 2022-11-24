import numpy as np

from estimators.base_estimator import BaseEstimator


def learning_curves(estimator: BaseEstimator, x_train, y_train, x_val, y_val):
    m = x_train.shape[0]
    cost_history = np.zeros(m)
    cost_history_val = np.zeros(m)

    for i in range(m):
        c_h, c_h_v, _ = estimator.fit(x_train[:i + 1], y_train[:i + 1], x_val, y_val)
        cost_history[i] = c_h[-1]
        cost_history_val[i] = c_h_v[-1]

    return cost_history, cost_history_val
