

import numpy as np
from TrainingFunctions import backwards
from LossFunctions import MeanSquaredErrorDerivative
import SuppFunctions  # for optional gradient clipping


# 1) Mini-batch Stochastic Gradient Descent
def train_minibatch_sgd(network,
                        inputs,
                        targets,
                        epochs,
                        learning_rate,
                        batch_size,
                        loss_derivative=MeanSquaredErrorDerivative):

    n_samples = inputs.shape[1]

    for epoch in range(epochs):

        # Random permutation of indices to create random mini-batches
        indices = np.random.permutation(n_samples)

        # Iterate over the dataset in chunks of size 'batch_size'
        for start in range(0, n_samples, batch_size):
            # Compute the slice of indices for this mini-batch
            batch_idx = indices[start:start + batch_size]

            # Prepare a list of accumulated gradients, one array per layer.
            # Each is initialized as zeros with the same shape as that layer's weights.
            grad_acc = [np.zeros_like(W) for W in network.weights_list]

            # Loop over all samples in this mini-batch
            for idx in batch_idx:
                # Select one sample (column) and target
                input_sample = inputs[:, idx:idx + 1]
                target_sample = targets[:, idx:idx + 1]

                # Compute gradients for this sample
                grad_W = backwards(network,
                                   input_sample,
                                   target_sample,
                                   loss_derivative)

                # Accumulate gradients for each layer.
                for i in range(len(grad_acc)):
                    grad_acc[i] += grad_W[i]

            # The last mini-batch may contain fewer than 'batch_size' samples,
            # so we compute the effective size explicitly.
            batch_size_effective = len(batch_idx)

            # Update each layer's weight matrix 
            for i in range(len(network.weights_list)):
                # Average gradient over the mini-batch
                grad_avg = grad_acc[i] / batch_size_effective

                # clip gradient to prevent exploding gradients (optional)
                grad_avg = SuppFunctions.clip_gradient(grad_avg)

                #Gradient descent update
                network.weights_list[i] -= learning_rate * grad_avg



# 3) Mini-batch Adam Optimizer
def train_minibatch_adam(network,
                         inputs,
                         targets,
                         epochs,
                         learning_rate,
                         batch_size,
                         loss_derivative=MeanSquaredErrorDerivative,
                         beta1=0.9,
                         beta2=0.999,
                         epsilon=1e-8):

    n_samples = inputs.shape[1]

    # Initialize first and second moment estimates for each layer's weights.
    m = [np.zeros_like(W) for W in network.weights_list]  # first moment
    v = [np.zeros_like(W) for W in network.weights_list]  # second moment

    # Global time step (counts how many parameter updates we have done so far).
    t = 0

    for epoch in range(epochs):

        # Shuffle indices for this epoch
        indices = np.random.permutation(n_samples)

        # Loop over mini-batches
        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]

            # Accumulate gradients over the mini-batch
            grad_acc = [np.zeros_like(W) for W in network.weights_list]

            for idx in batch_idx:
                input_sample = inputs[:, idx:idx + 1]
                target_sample = targets[:, idx:idx + 1]

                grad_W = backwards(network,
                                   input_sample,
                                   target_sample,
                                   loss_derivative)

                for i in range(len(grad_acc)):
                    grad_acc[i] += grad_W[i]

            batch_size_effective = len(batch_idx)

            # Each mini-batch update increments the time step.
            t += 1

            # Update each layer with Adam rules
            for i in range(len(network.weights_list)):
                # Average gradient over the mini-batch
                g = grad_acc[i] / batch_size_effective

                # clip to control exploding gradients (optional)
                g = SuppFunctions.clip_gradient(g)

                
                # Update biased first moment estimate (moving average of gradients)
                m[i] = beta1 * m[i] + (1.0 - beta1) * g

                # Update biased second moment estimate (moving average of squared gradients)
                v[i] = beta2 * v[i] + (1.0 - beta2) * (g * g)

                
                # Because m and v are initialized at zero, they are biased towards 0
                # especially during the first steps. Bias correction compensates this.
                m_hat = m[i] / (1.0 - beta1 ** t)
                v_hat = v[i] / (1.0 - beta2 ** t)

                
                
                update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

                # Apply update
                network.weights_list[i] -= update
