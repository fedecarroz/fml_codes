import matplotlib.pyplot as plt
import numpy as np

from metrics.classification_metrics import ClassificationMetrics

np.random.seed(42)


def sigmoid(n):
    return 1 / (1 + np.exp(-n))


def sigmoid_derivative(n):
    return n * (1 - n)


class NeuralNetwork:
    def __init__(self, layers, epochs=700, alpha=1e-2, lmd=1):
        self.layers = layers
        self.n_layers = len(layers)
        self.epochs = epochs
        self.alpha = alpha
        self.lmd = lmd

        self.w = {}
        self.b = {}
        self.loss = []
        self.loss_val = []

    def init_parameters(self):
        for i in range(1, self.n_layers):
            self.w[i] = np.random.randn(self.layers[i], self.layers[i - 1])
            self.b[i] = np.ones((self.layers[i], 1))

    def forward_propagation(self, X):
        values = {}

        for i in range(1, self.n_layers):
            if i == 1:
                values["Z" + str(i)] = np.dot(self.w[i], X.T) + self.b[i]
            else:
                values["Z" + str(i)] = (
                    np.dot(self.w[i], values["A" + str(i - 1)]) + self.b[i]
                )

            values["A" + str(i)] = sigmoid(values["Z" + str(i)])

        return values

    def compute_cost(self, values, y):
        m = y.shape[0]
        pred = values["A" + str(self.n_layers - 1)]

        cost = -np.average(y.T * np.log(pred) + (1 - y.T) * np.log(1 - pred))
        reg_sum = 0
        for i in range(1, self.n_layers):
            reg_sum += np.sum(np.square(self.w[i]))
        L2_reg = reg_sum * (self.lmd / (2 * m))

        return cost + L2_reg

    def compute_cost_derivative(self, values, y):
        return -(np.divide(y.T, values) - np.divide(1 - y.T, 1 - values))

    def backpropagation_step(self, values, X, y):
        m = y.shape[0]
        params_upd = {}
        dZ = None
        for i in range(self.n_layers - 1, 0, -1):
            if i == (self.n_layers - 1):
                dA = self.compute_cost_derivative(values["A" + str(i), y])
            else:
                dA = np.dot(self.w[i + 1].T, dZ)

            dZ = np.multiply(dA, sigmoid_derivative(values["A" + str(i)]))

            if i == 1:
                params_upd["W" + str(i)] = (1 / m) * (
                    np.dot(dZ, X) + self.lmd * self.w[i]
                )
            else:
                params_upd["W" + str(i)] = (1 / m) * (
                    np.dot(dZ, values["A" + str(i - 1)].T) + self.lmd * self.w[i]
                )

            params_upd["B" + str(i)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        return params_upd

    def update(self, upd):
        for i in range(1, self.n_layers):
            self.w[i] -= self.alpha * upd["W" + str(i)]
            self.b[i] -= self.alpha * upd["B" + str(i)]

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
        pred = values["A" + str(self.n_layers - 1)]
        return np.round(pred)

    def compute_performance(self, X, y):
        pred = self.predict(X)
        metrics = ClassificationMetrics(y, pred[-1])
        return metrics.compute_errors()

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Loss curve")
        plt.show()

    def plot_loss_val(self):
        plt.plot(self.loss_val)
        plt.xlabel("epochs")
        plt.ylabel("loss val")
        plt.title("Loss val curve")
        plt.show()

    def get_params(self):
        return {
            "layers": self.layers,
            "epochs": self.epochs,
            "alpha": self.alpha,
            "lmd": self.lmd,
        }

    def set_params(self, params):
        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params.keys():
                raise Exception("No such parameter")

            setattr(self, key, value)

        return self

    def learning_curves(self, X_train, y_train, X_val, y_val):
        print("Calculating learning curves. Please wait...")
        m = len(X_train)
        cost_history = np.zeros(m)
        cost_history_val = np.zeros(m)
        for i in range(m):
            self.fit(X_train[: i + 1], y_train[: i + 1], X_val, y_val)
            cost_history[i] = self.loss[-1]
            cost_history_val[i] = self.loss_val[-1]

        return cost_history, cost_history_val

    def plot_learning_curves(self, c_h, c_h_v):
        plt.plot(c_h)
        plt.plot(c_h_v)
        plt.title("Learning curves")
        plt.show()
