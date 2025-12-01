import numpy as np
#this is .py file with activation functions and their derivatives


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

def leaky_relu(x, slope=0.01):
    return np.where(x > 0, x, slope * x)

def der_leaky_relu(x, slope=0.01):
    return np.where(x > 0, 1.0, slope)
#Softmax (scalar input) - if softmax is used in scalar context, then it is dividing by itself, so it would yield 1 and derivative would be always 0. In practice it should never happen, but it is up to user choice
def softmax(x):
    return 1

def der_softmax(x):
    return 0

#Softmax (vector/array input)
def softmax_vec(x):#
    #
    x_shifted = x - np.max(x, axis=0, keepdims=True)#keepdims=True keeps shape for broadcasting(which should be impossible due to failsafes in previous steps)
    exp_values = np.exp(x_shifted)
    return exp_values / np.sum(exp_values, axis=0, keepdims=True)

def der_softmax_vec(x):
    """
    Softmax derivative returns the *diagonal* part only.
    Full Jacobian is layer-dependent; usually combined with cross-entropy.
    Here we return the element-wise derivative: s*(1-s).
    """
    s = softmax_vec(x)
    return s * (1 - s)
