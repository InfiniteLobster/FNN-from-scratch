import numpy as np
#this is .py file with loss functions and their derivatives


# ================================================
# Mean Squared Error (MSE)
# ================================================
def MeanSquaredError(targets, predictions):
    diff = predictions - targets
    return 0.5 * np.sum(diff ** 2)

def MeanSquaredErrorDerivative(targets, predictions):
    return (predictions - targets)


# ================================================
# Binary Cross-Entropy (for sigmoid output)
# ================================================
def BinaryCrossEntropy(targets, predictions, eps=1e-12):
    """
    targets: (1, batch)
    predictions: (1, batch)
    """
    predictions = np.clip(predictions, eps, 1 - eps)
    return -np.mean(targets * np.log(predictions) +
                    (1 - targets) * np.log(1 - predictions))

def BinaryCrossEntropyDerivative(targets, predictions, eps=1e-12):
    predictions = np.clip(predictions, eps, 1 - eps)
    return (predictions - targets) / (predictions * (1 - predictions))


# ================================================
# Softmax + Cross-Entropy (for multi-class)
# ================================================
def SoftmaxCrossEntropy(target_one_hot, softmax_vals):#in this implementation softmax is calculated in layer, but cross entropy should only be used with softmax in output layer
    # Cross-entropy loss calculation
    loss_sum = -np.sum(target_one_hot * np.log(softmax_vals + 1e-12), axis = 0)# 1e-12 is added for numerical stability
    loss_mean = np.mean(loss_sum)
    #returning loss
    return loss_mean


def SoftmaxCrossEntropyDerivative(target_one_hot, softmax_vals):
    return (softmax_vals - target_one_hot)
