import matplotlib.pyplot as plt
import numpy as np

from helpers.sigmoid import sigmoid, sigmoid_derivative
from metrics.regression_metrics import RegressionMetrics

np.random.seed(42)


class NeuralNetwork:
    def __init__(self, layers, epochs=700, alpha=1e-2, lmd=1):
        self.layers = layers
        self.epochs = epochs
        self.alpha = alpha
        self.lmd = lmd

        self.w = {}
        self.b = {}
        self.loss = []
        self.loss_val = []

    def init_parameters(self):
        L = len(self.layers)

        for l in range(1, L):
            self.w[l] = np.random.randn(self.layers[l], self.layers[l - 1])
            self.b[l] = np.ones((self.layers[l], 1))

    def forward_propagation(self, X):
        layers = len(self.w)
        values = {}

        for i in range(1, layers + 1):
            if i == 1:
                values['Z' + str(i)] = np.dot(self.w[i], X.T) + self.b[i]
                values['A' + str(i)] = sigmoid(values['Z' + str(i)])
            elif i == layers:
                values['Z' + str(i)] = np.dot(self.w[i], values['A' + str(i - 1)]) + self.b[i]
                values['A' + str(i)] = values['Z' + str(i)]
            else:
                values['Z' + str(i)] = np.dot(self.w[i], values['A' + str(i - 1)]) + self.b[i]
                values['A' + str(i)] = sigmoid(values['Z' + str(i)])

        return values

    def compute_cost(self, AL, y):
        m = y.shape[0]
        layers = len(AL) // 2
        pred = AL['A' + str(layers)]

        cost = (np.average((pred - y) ** 2)) / 2
        reg_sum = 0
        for l in range(1, layers + 1):
            reg_sum += (np.sum(np.square(self.w[l])))
        L2_reg = reg_sum * (self.lmd / (2 * m))
        return cost + L2_reg

    def compute_cost_derivative(self, a, y):
        return a - y

    def backpropagation_step(self, values, X, y):
        m = X.shape[0]
        layers = len(self.w)
        params_upd = {}
        dZ = None
        for i in range(layers, 0, -1):
            if i == layers:
                dA = self.compute_cost_derivative(values['A' + str(i)], y)
                dZ = dA
            else:
                dA = np.dot(self.w[i + 1].T, dZ)
                dZ = np.multiply(dA, sigmoid_derivative(values['A' + str(i)]))

            if i == 1:
                params_upd['W' + str(i)] = (1 / m) * (np.dot(dZ, X) + self.lmd * self.w[i])
            else:
                params_upd['W' + str(i)] = (1 / m) * (np.dot(dZ, values['A' + str(i - 1)].T) + self.lmd * self.w[i])

            params_upd['B' + str(i)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        return params_upd

    def update(self, upd):
        layers = len(self.w)

        for i in range(1, layers + 1):
            self.w[i] -= self.alpha * upd['W' + str(i)]
            self.b[i] -= self.alpha * upd['B' + str(i)]

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.loss = []
        self.loss_val = []
        self.init_parameters()

        for i in range(self.epochs):
            values = self.forward_propagation(X_train)
            grads = self.backpropagation_step(values, X_train, y_train)
            self.update(grads)

            cost = self.compute_cost(values, y_train)
            self.loss.append(cost)

            if X_val is not None and y_val is not None:
                values_val = self.forward_propagation(X_val)
                cost_val = self.compute_cost(values_val, y_val)
                self.loss_val.append(cost_val)

    def predict(self, X_test):
        values = self.forward_propagation(X_test)
        layers = len(values) // 2
        pred = values['A' + str(layers)]
        return pred

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Loss curve')
        plt.show()

    def compute_performance(self, x, y):
        pred = self.predict(x)
        metrics = RegressionMetrics(y, pred[-1])
        return metrics.compute_errors()

    def get_params(self):
        return {
            "layers": self.layers,
            "alpha": self.alpha,
            "lmd": self.lmd,
        }

    def set_params(self, params):
        self.loss = []
        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params.keys():
                raise Exception('No such parameter')

            setattr(self, key, value)

    def learning_curves(self, X_train, y_train, X_val, y_val):
        print("Calculating learning curves. Please wait...")
        m = X_train.shape[0]
        cost_history = np.zeros(m)
        cost_history_val = np.zeros(m)

        for i in range(m):
            self.fit(X_train[:i + 1], y_train[:i + 1], X_val, y_val)
            cost_history[i] = self.loss[-1]
            cost_history_val[i] = self.loss_val[-1]

        return cost_history, cost_history_val

    def plot_learning_curves(self, c_h, c_h_v):
        plt.plot(c_h)
        plt.plot(c_h_v)
        plt.title("Learning curves")
        plt.show()
