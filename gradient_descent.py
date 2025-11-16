import numpy as np

#Basic gradient descent

def gradient_descent(parameters, gradients, learning_rate):
    """
    parameters: dict of parameters (W1, b1, W2, b2, ...)
    gradient: dict of corresponding gradients (dW1, db1, ...)
    """
    for name in params:
        parameters[name] -= learning_rate * gradients[name]


# Stochastic gradient descent

def iterate_minibatches(X, y, batch_size, shuffle=True):
    """
    Yield mini-batches of (X_batch, y_batch).

    X: array of shape (N, D)
    y: array of shape (N,) or (N, C)
    """
    N = X.shape[0]
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, N, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]

def sgd_update(params, grads, learning_rate):
    """
    Perform one SGD parameter update.

    params: dict of parameter arrays (e.g. {"W1": W1, "b1": b1, ...})
    grads:  dict of gradient arrays with same keys ({"W1": dW1, "b1": db1, ...})
    """
    for name in params:
        params[name] -= learning_rate * grads[name]
