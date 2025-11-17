import numpy as np

def identity(x):
    return x

def identity_derivative(x):
    # derivative of f(x) = x is 1 for every element
    return np.ones_like(x)
