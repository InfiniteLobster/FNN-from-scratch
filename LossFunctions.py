import numpy as np


def MeanSquaredError(ground_truth, prediction):
    error = 0.5 * np.sum((prediction - ground_truth) ** 2)
    return error

def mse_derivative(y_pred, y_true):
    return y_pred - y_true
