import numpy as np


def sigmoid(num):
    return 1 / (1 + np.exp(-num))
