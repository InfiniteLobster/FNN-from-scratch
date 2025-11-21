from TrainingFunctions import backwards
from LossFunctions import MeanSquaredError, MeanSquaredErrorDerivative
import numpy as np

def train_sgd(network, inputs, targets, epochs, learning_rate):
    
    
    
    # inputs:  shape (n_features, n_samples)
    # targets: shape (n_outputs, n_samples)
    
    n_samples = inputs.shape[1]

    for epoch in range(epochs):
        # shuffle sample order
        indices = np.random.permutation(n_samples)

        for idx in indices:
            # extracting just one sample
            input_sample  = inputs[:,  idx:idx+1]   # (n_features, 1)
            target_sample = targets[:, idx:idx+1]   # (n_outputs, 1)

            # computing gradients
            grad_W = backwards(network, input_sample, target_sample,
                                MeanSquaredErrorDerivative)

            # gradient descent step
            for i in range(len(network.weights_list)):
                network.weights_list[i] -= learning_rate * grad_W[i]
