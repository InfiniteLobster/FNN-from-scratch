import numpy as np
#identity
def identity(x):
    return x

def der_identity(x):
    return 1
#Sigmoid
def sigmoid(x):
    # numerically stable
    return 1.0 / (1.0 + np.exp(-x))

def der_sigmoid(x):
    s = sigmoid(x)
    return s * (1.0 - s)
#Tanh
def tanh(x):
    return np.tanh(x)

def der_tanh(x):
    t = np.tanh(x)
    return 1.0 - t**2
#ReLU
def relu(x):
    return np.maximum(0.0, x)

def der_relu(x):
    return 1.0 * (x > 0.0)
# Leaky ReLU

_LEAKY_SLOPE = 0.01

def leaky_relu(x):
    return np.where(x > 0.0, x, _LEAKY_SLOPE * x)

def der_leaky_relu(x):
    return np.where(x > 0.0, 1.0, _LEAKY_SLOPE)
#Softmax (vector input)
def softmax(x):
    """
    Softmax supports vector or column-vector input.
    Uses numerical stabilization.
    """
    x_shifted = x - np.max(x, axis=0, keepdims=True)
    exp_values = np.exp(x_shifted)
    return exp_values / np.sum(exp_values, axis=0, keepdims=True)

def der_softmax(x):
    """
    Softmax derivative returns the *diagonal* part only.
    Full Jacobian is layer-dependent; usually combined with cross-entropy.
    Here we return the element-wise derivative: s*(1-s).
    """
    s = softmax(x)
    return s * (1 - s)
