import numpy as np
from LossFunctions import *
from SuppFunctions import *




# 1. MINI-BATCH STOCHASTIC GRADIENT DESCENT 


def train_minibatch_sgd(network,
                        inputs,
                        targets,
                        epochs,
                        learning_rate,
                        batch_size,
                        loss_derivative,
                        grad_clip = 0):
    """
    Train the FNN using mini-batch SGD.

    inputs:  np.ndarray of shape (n_features, n_samples)
    targets: np.ndarray of shape (n_outputs, n_samples)
    """

    # Number of training samples 
    n_samples = inputs.shape[1] 
    #sometimes Np due to its mechanics transposes the array. So far it happend only for targets. This is fix that resolves this issue
    if(targets.shape[0] != network.weights_list[-1].shape[0]):
        targets = targets.T
    # Loop through the dataset
    for epoch in range(epochs):
        # Shuffle sample indices so that batches are random each epoch
        indices = np.random.permutation(n_samples)
        #preparing range and getting its last value for bacth loop
        range_batches = range(0, n_samples, batch_size)
        last_value = (((n_samples-1)//batch_size)*batch_size)# n_samples/batch_size gives amount of steps that fits into range, -1 is used to not accidentely "go over" the range as it is < n_samples. Multiplying gives precise number
        # Process the dataset in chunks of size batch_size
        for start in range_batches:
            #getting indices for this batch
            batch_idx = indices[start:(start + batch_size)]
            #slicing current batch data
            input_sample = inputs[:,batch_idx]
            target_sample = targets[:,batch_idx]
            #propagation of input(-s in case of batches) through network
            out = network.forward(input_sample)
            #propagating error backwards through network
            grad_W = network.backward(out[0],out[1],target_sample,loss_derivative)
            #As last mini-batch may contain fewer than 'batch_size' samples, the size of batch has to be computed as it is not the same every time (checking index of)
            if(start == last_value):
                batch_size_effective = len(batch_idx)
            else:
                batch_size_effective = batch_size
            # iterating through weights arrays (of layers) to update each layer's weight matrix 
            for i in range(len(network.weights_list)):
                #getting current layer
                weights_array = network.weights_list[i]
                # Average gradient over the mini-batch for current layer
                grad_avg = grad_W[i] / batch_size_effective
                #gradient clipping if selected
                if(grad_clip != 0):
                    grad_avg = clip_gradient(grad_avg,grad_clip)
                #updating weights
                weights_array -= learning_rate * grad_avg
                #assignign updates weights to network property
                network.weights_list[i] = weights_array
    #returning trained network
    return network


# 2. MINI-BATCH ADAM OPTIMIZER


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

    
    # Initialize Adam moment vectors for each layer.
    #
    # Shapes are identical to corresponding weight matrices.
    # m[i]: first moment (mean of gradients)
    # v[i]: second moment (mean of squared gradients)
    
    m = [np.zeros_like(W) for W in network.weights_list]
    v = [np.zeros_like(W) for W in network.weights_list]

    t = 0  # Adam time step counter (increments per batch)

    # Loop over dataset
    for epoch in range(epochs):

        # Shuffle data order this epoch
        indices = np.random.permutation(n_samples)

        # Mini-batch loop
        for start in range(0, n_samples, batch_size):

            batch_idx = indices[start:start + batch_size]

            # Extract batch: shapes (features, B) and (outputs, B)
            x_batch = inputs[:, batch_idx]
            y_batch = targets[:, batch_idx]

            # Forward and Backward on this batch 
            z_values, a_values = network.forward(x_batch)
            grad_W = network.backward(z_values, a_values, y_batch, loss_derivative)

            B = x_batch.shape[1]  # batch size (maybe smaller on last batch)

            # Increment Adam time step
            t += 1

            
            # Adam update for each layer
            # 
            for i in range(len(network.weights_list)):

                # Convert summed gradient to averaged gradient
                g = grad_W[i] / B

              

            
                # 1. Update biased first moment estimate
              
                m[i] = beta1 * m[i] + (1.0 - beta1) * g

                
                # 2. Update biased second moment estimate
                
                v[i] = beta2 * v[i] + (1.0 - beta2) * (g * g)

            
                # 3. Bias corrections:
             
                
                m_hat = m[i] / (1.0 - beta1 ** t)
                v_hat = v[i] / (1.0 - beta2 ** t)

                
                # 4. Compute parameter update:
                
                update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

                
                # 5. Apply update (gradient descent direction)
                
                network.weights_list[i] -= update
    #
    return network