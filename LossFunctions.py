import numpy as np

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
def SoftmaxCrossEntropy(target_one_hot, logits):
    """
    logits: raw scores from last layer (before softmax)
    target_one_hot: one-hot encoded targets
    """
    # Numerical stability: subtract max for each column
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    exp_vals = np.exp(shifted)
    softmax_vals = exp_vals / np.sum(exp_vals, axis=0, keepdims=True)

    # Cross-entropy loss
    loss = -np.sum(target_one_hot * np.log(softmax_vals + 1e-12))
    return loss / logits.shape[1]


def SoftmaxCrossEntropyDerivative(target_one_hot, logits):
    """
    Efficient gradient:
    grad = softmax(logits) - target_one_hot
    """
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    exp_vals = np.exp(shifted)
    softmax_vals = exp_vals / np.sum(exp_vals, axis=0, keepdims=True)

    return (softmax_vals - target_one_hot)
