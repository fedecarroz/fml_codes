import numpy as np


def sigmoid(num):
    return 1 / (1 + np.exp(-num))


def sigmoid_derivative(num):
    return num * (1 - num)
