import numpy as np
from matplotlib import pyplot as plt

from metrics.classification_metrics import ClassificationMetrics

np.random.seed(42)


class NeuralNetwork:
    def __init__(self, layers, epochs=700, alpha=1e-2, lmd=1):
        self.layers = layers
        self.epochs = epochs
        self.alpha = alpha
        self.lmd = lmd

        self.w = {}
        self.b = {}
        # self.A = {}
        # self.Z = {}
        # self.dA = {}
        # self.dZ = {}
        self.X = None
        self.y = None
        self.loss = []

    def init_parameters(self):
        L = len(self.layers)

        for l in range(1, L):
            self.w[l] = np.random.randn(self.layers[l], self.layers[l - 1])
            self.b[l] = np.zeros((self.layers[l], 1))

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, A):
        g_prime = A * (1 - A)
        return g_prime

    def forward_propagation(self):
        layers = len(self.w)
        values = {}

        for i in range(1, layers + 1):
            if i == 1:
                values['Z' + str(i)] = np.dot(self.w[i], self.X.T) + self.b[i]
            else:
                values['Z' + str(i)] = np.dot(self.w[i], values['A' + str(i - 1)]) + self.b[i]

            values['A' + str(i)] = self.sigmoid(values['Z' + str(i)])

        return values

    def compute_cost(self, AL):
        m = self.y.shape[0]
        layers = len(AL) // 2
        y_pred = AL['A' + str(layers)]

        cost = -np.average(self.y.T * np.log(y_pred) + (1 - self.y.T) * np.log(1 - y_pred))
        reg_sum = 0
        for l in range(1, layers):
            reg_sum += (np.sum(np.square(self.w[l])))
        L2_reg = reg_sum * (self.lmd / (2 * m))
        return cost + L2_reg

    def compute_cost_derivative(self, AL):
        return -(np.divide(self.y.T, AL) - np.divide(1 - self.y.T, 1 - AL))

    def backpropagation_step(self, values):
        m = self.X.shape[0]
        layers = len(self.w)
        params_upd = {}

        for i in range(layers, 0, -1):
            if i == layers:
                dA = self.compute_cost_derivative((values['A' + str(i)]))
            else:
                dA = np.dot(self.w[i + 1].T, dZ)

            dZ = np.multiply(dA, self.sigmoid_derivative(values['A' + str(i)]))

            if i == 1:
                params_upd['W' + str(i)] = (1 / m) * (np.dot(dZ, self.X) + self.lmd * self.w[i])
            else:
                params_upd['W' + str(i)] = (1 / m) * (np.dot(dZ, values['A' + str(i - 1)].T) + self.lmd * self.w[i])

            params_upd['B' + str(i)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        return params_upd

    def update(self, upd):
        layers = len(self.w)

        for i in range(1, layers + 1):
            self.w[i] -= self.alpha * upd['W' + str(i)]
            self.b[i] -= self.alpha * upd['B' + str(i)]

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.init_parameters()

        for i in range(self.epochs):
            A_list = self.forward_propagation()
            cost = self.compute_cost(A_list)
            grads = self.backpropagation_step(A_list)
            self.update(grads)
            self.loss.append(cost)

    def predict(self, X_test):
        # self.X = np.c_[np.ones(X_test.shape[0]), X_test]
        self.X = X_test
        AL = self.forward_propagation()
        layers = len(AL) // 2
        y_pred = AL['A' + str(layers)]
        return np.round(y_pred)

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Loss curve')
        plt.show()

    def compute_performance(self, x, y):
        pred = self.predict(x)
        metrics = ClassificationMetrics(y, pred[-1])
        return metrics.compute_errors()

    def get_params(self) -> dict:
        return {
            "alpha": self.alpha,
            "lmd": self.lmd
        }

    def set_params(self, params):
        self.loss = []
        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params.keys():
                raise Exception("No such parameter")

            setattr(self, key, value)
