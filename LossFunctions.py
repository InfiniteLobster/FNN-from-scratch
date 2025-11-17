import numpy as np

def MeanSquaredError(targets, predictions):
    """
    Computes Mean Squared Error for column-vector inputs.
    targets:      (n, 1)
    predictions:  (n, 1)
    """
    diff = predictions - targets
    return 0.5 * np.sum(diff ** 2)


def MeanSquaredErrorDerivative(targets, predictions):
    """
    Derivative of MSE with respect to predictions.
    dL/da = (pred - target)
    targets:      (n, 1)
    predictions:  (n, 1)
    """
    return (predictions - targets)
