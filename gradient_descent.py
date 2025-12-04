import numpy as np
from LossFunctions import *
from SuppFunctions import * # for clip_gradient
from ErrorClasses import *


# Helper: makes sure that targets have correct shape (for some reason NumPy sometimes(not always, was caught only in one dataset) transposes the arrays before passing them to function, so they arrive in wrong shape. This function should fix it)
def _prepare_data(shape_output_layer,  targets):
    #it is first checked if problem with matching sizes occurs
    if (targets.shape[0] != shape_output_layer):
        #transposing the array to hopefully resolve the issue
        targets = targets.T
        #it is checked if operation resolved the issue
        if(targets.shape[0] != shape_output_layer):
            #if shapes doesn't match after transpose, then there is problem with input to the function, not NumPy error
            raise NotSupportedInputGiven("Network training","Targets does not match network outputs")
    #returning values (both unchanged and changed, because if code reaches this step everything should be allright. If it is not error is thrown in nested if statments)
    return targets


#SGD training loop
def train_minibatch_sgd(network,
                        inputs,
                        targets,
                        epochs,
                        learning_rate,
                        batch_size,
                        loss_derivative,
                        l1_coeff=0.0,
                        l2_coeff=0.0,
                        grad_clip=0.0):
    """
    Train the FNN using mini-batch SGD.

    inputs:  np.ndarray of shape (n_features, n_samples)
    targets: np.ndarray of shape (n_outputs, n_samples)
    """
    #checking for NumPy transpose mistake and solving it(throwing error if there is other problem with shapes of output layer and given targets)
    targets = _prepare_data(network.weights_list[-1].shape[0], targets)
    #getting number of examples to train network on(information needed for training)
    n_samples = inputs.shape[1]
    #training loop
    for epoch in range(epochs):
        #Shuffle sample indices so that batches are random each epoch
        indices = np.random.permutation(n_samples)
        #Process the dataset in chunks of size batch_size (batching)
        for start in range(0, n_samples, batch_size):
            #getting indices of examples to be used for this batch training
            batch_idx = indices[start:start + batch_size]
            #getting data for current batch into variables
            inputs_batch = inputs[:, batch_idx]
            targets_batch = targets[:, batch_idx]
            #getting batch size for gradient division (as all batches are summed, so they need to be divided to get average across batches). It needs to be checked as last batch might not have the same amount of examples in it as the rest
            batch_size_effective = inputs_batch.shape[1]  # effective batch size
            #forward pass to get network output
            z_values, a_values = network.forward(inputs_batch)
            #Backward pass (gradients summed over batch) -> getting gradients
            grad_W = network.backward(z_values, a_values, targets_batch, loss_derivative)
            #weights update(iterating through weight arrays of layers)
            for iLayer in range(len(network.weights_list)):
                #getting weights array of current layer
                weights_array = network.weights_list[iLayer]
                # Convert summed gradient to average gradient
                grad = grad_W[iLayer] / batch_size_effective
                #L2 regularization(if declared)
                if l2_coeff != 0.0:
                    reg = l2_coeff * weights_array.copy()#calculating regularization value
                    reg[:, 0] = 0.0  ##we do not regularize bias column
                    grad += reg#applying regularization
                #L1 regularization(if declared)
                    reg_l1 = l1_coeff * np.sign(weights_array.copy())#calculating regularization value
                    reg_l1[:, 0] = 0.0##we do not regularize bias column
                    grad += reg_l1#applying regularization
                #gradient clipping(if declared)
                if grad_clip != 0.0:
                    grad = clip_gradient(grad, grad_clip)
                #gradient descent step
                network.weights_list[iLayer] = weights_array - learning_rate * grad
    #trained network and last gradient (weights) are returned. Gradient is returned for sweep
    return network,grad_W


#SGD with momentum training loop
def train_minibatch_sgd_momentum(network,
                                 inputs,
                                 targets,
                                 epochs,
                                 learning_rate,
                                 batch_size,
                                 loss_derivative,
                                 momentum=0.9,
                                 l1_coeff=0.0,
                                 l2_coeff=0.0,
                                 grad_clip=0.0):
    """
    Train the FNN using mini-batch SGD with classical momentum.
    """
    #checking for NumPy transpose mistake and solving it(throwing error if there is other problem with shapes of output layer and given targets)
    targets = _prepare_data(network.weights_list[-1].shape[0], targets)
    #getting number of examples to train network on(information needed for training)
    n_samples = inputs.shape[1]
    #declaring Velocity (momentum) variable for each layer (for pre-allocation)
    velocity = [np.zeros_like(W) for W in network.weights_list]
    #training loop
    for epoch in range(epochs):
        #Shuffle sample indices so that batches are random each epoch
        indices = np.random.permutation(n_samples)
        #Process the dataset in chunks of size batch_size (batching)
        for start in range(0, n_samples, batch_size):
            #getting indices of examples to be used for this batch training
            batch_idx = indices[start:start + batch_size]
            #getting data for current batch into variables
            inputs_batch = inputs[:, batch_idx]
            targets_batch = targets[:, batch_idx]
            #getting batch size for gradient division (as all batches are summed, so they need to be divided to get average across batches). It needs to be checked as last batch might not have the same amount of examples in it as the rest
            batch_size_effective = inputs_batch.shape[1]
            #forward pass to get network output
            z_values, a_values = network.forward(inputs_batch)
            #Backward pass (gradients summed over batch) -> getting gradients
            grad_W = network.backward(z_values, a_values, targets_batch, loss_derivative)
            #weights update(iterating through weight arrays of layers)
            for iLayer in range(len(network.weights_list)):
                #getting weights array of current layer
                weights_array = network.weights_list[iLayer]
                #converting summed gradient to average gradient
                grad = grad_W[iLayer] / batch_size_effective
                #L2 regularization(if declared)
                if l2_coeff != 0.0:
                    reg = l2_coeff * weights_array.copy()#calculating regularization value
                    reg[:, 0] = 0.0#we do not regularize bias column
                    grad += reg#applying regularization
                #L1 regularization(if declared)
                if l1_coeff != 0.0:
                    reg_l1 = l1_coeff * np.sign(weights_array.copy())#calculating regularization value
                    reg_l1[:, 0] = 0.0#we do not regularize bias column
                    grad += reg_l1#applying regularization
                #gradient clipping(if declared)
                if grad_clip != 0.0:
                    grad = clip_gradient(grad, grad_clip)
                #Momentum update: v = mu * v - lr * g
                velocity[iLayer] = momentum * velocity[iLayer] - learning_rate * grad
                #Gradient descent step
                network.weights_list[iLayer] = weights_array + velocity[iLayer]
    #trained network and last gradient (weights) are returned. Gradient is returned for sweep
    return network,grad_W
#RMSPROP training loop
def train_minibatch_rmsprop(network,
                            inputs,
                            targets,
                            epochs,
                            learning_rate,
                            batch_size,
                            loss_derivative,
                            beta=0.9,
                            epsilon=1e-8,
                            l1_coeff=0.0,
                            l2_coeff=0.0,
                            grad_clip=0.0):
    """
    Train the FNN using mini-batch RMSprop.

    beta: decay rate for the running average of squared gradients
    """
    #checking for NumPy transpose mistake and solving it(throwing error if there is other problem with shapes of output layer and given targets)
    targets = _prepare_data(network.weights_list[-1].shape[0], targets)
    #getting number of examples to train network on(information needed for training)
    n_samples = inputs.shape[1]
    #declaring average of squared gradients variable for each layer (for pre-allocation) 
    rms = [np.zeros_like(W) for W in network.weights_list]
    #training loop
    for epoch in range(epochs):
        #Shuffle sample indices so that batches are random each epoch
        indices = np.random.permutation(n_samples)
        #Process the dataset in chunks of size batch_size (batching)
        for start in range(0, n_samples, batch_size):
            #getting indices of examples to be used for this batch training
            batch_idx = indices[start:start + batch_size]
            #getting data for current batch into variables
            inputs_batch = inputs[:, batch_idx]
            targets_batch = targets[:, batch_idx]
            #getting batch size for gradient division (as all batches are summed, so they need to be divided to get average across batches). It needs to be checked as last batch might not have the same amount of examples in it as the rest
            batch_size_effective = inputs_batch.shape[1]
            #forward pass to get network output
            z_values, a_values = network.forward(inputs_batch)
            #Backward pass (gradients summed over batch) -> getting gradients
            grad_W = network.backward(z_values, a_values, targets_batch, loss_derivative)
            #weights update(iterating through weight arrays of layers)
            for iLayer in range(len(network.weights_list)):
                #getting weights array of current layer
                weights_array = network.weights_list[iLayer]
                # Convert summed gradient to average gradient
                grad = grad_W[iLayer] / batch_size_effective
                #L2 regularization(if declared)
                if l2_coeff != 0.0:
                    reg = l2_coeff * weights_array.copy()#calculating regularization value
                    reg[:, 0] = 0.0#we do not regularize bias column
                    grad += reg#applying regularization
                #L1 regularization(if declared)
                if l1_coeff != 0.0:
                    reg_l1 = l1_coeff * np.sign(weights_array.copy())#calculating regularization value
                    reg_l1[:, 0] = 0.0#we do not regularize bias column
                    grad += reg_l1#applying regularization
                #gradient clipping(if declared)
                if grad_clip != 0.0:
                    grad = clip_gradient(grad, grad_clip)
                #Updating running average of squared gradients
                rms[iLayer] = beta * rms[iLayer] + (1.0 - beta) * (grad * grad)
                #RMSprop parameter update
                update = learning_rate * grad / (np.sqrt(rms[iLayer]) + epsilon)
                #gradient descent step
                network.weights_list[iLayer] = weights_array - update
    #trained network and last gradient (weights) are returned. Gradient is returned for sweep
    return network,grad_W


#Nestorov accelerated gradient (NAG) training loop
def train_minibatch_nag(network,
                        inputs,
                        targets,
                        epochs,
                        learning_rate,
                        batch_size,
                        loss_derivative,
                        momentum=0.9,
                        l1_coeff=0.0,
                        l2_coeff=0.0,
                        grad_clip=0.0):
    """
    Train the FNN using mini-batch Nesterov Accelerated Gradient (NAG).
    """
    #checking for NumPy transpose mistake and solving it(throwing error if there is other problem with shapes of output layer and given targets)
    targets = _prepare_data(network.weights_list[-1].shape[0], targets)
    #getting number of examples to train network on(information needed for training)
    n_samples = inputs.shape[1]
    #declaring Velocity (momentum) variable for each layer (for pre-allocation)
    velocity = [np.zeros_like(W) for W in network.weights_list]
    #training loop
    for epoch in range(epochs):
        #Shuffle sample indices so that batches are random each epoch
        indices = np.random.permutation(n_samples)
        #Process the dataset in chunks of size batch_size (batching)
        for start in range(0, n_samples, batch_size):
            #getting indices of examples to be used for this batch training
            batch_idx = indices[start:start + batch_size]
            #getting data for current batch into variables
            inputs_batch = inputs[:, batch_idx]
            targets_batch = targets[:, batch_idx]
            #getting batch size for gradient division (as all batches are summed, so they need to be divided to get average across batches). It needs to be checked as last batch might not have the same amount of examples in it as the rest
            batch_size_effective = inputs_batch.shape[1]  # effective batch size
            #Look-ahead step: w_lookahead = w + momentum * v
            for iLayer in range(len(network.weights_list)):
                network.weights_list[iLayer] = network.weights_list[iLayer] + momentum * velocity[iLayer]
            #forward pass to get network output at look-ahead weights
            z_values, a_values = network.forward(inputs_batch)
            #Backward pass at look-ahead weights (gradients summed over batch) -> getting gradients
            grad_W = network.backward(z_values, a_values, targets_batch, loss_derivative)
            #Update velocity and weights (Nesterov rule) - > iterating through weight arrays of layers
            for iLayer in range(len(network.weights_list)):
                #getting weights array of current layer
                weights_array = network.weights_list[iLayer]
                # Convert summed gradient to average gradient
                grad = grad_W[iLayer] / batch_size_effective
                #L2 regularization(if declared)
                if l2_coeff != 0.0:
                    reg = l2_coeff * weights_array.copy()#calculating regularization value
                    reg[:, 0] = 0.0#we do not regularize bias column
                    grad += reg#applying regularization
                #L1 regularization(if declared)
                if l1_coeff != 0.0:
                    reg_l1 = l1_coeff * np.sign(weights_array.copy())#calculating regularization value
                    reg_l1[:, 0] = 0.0#we do not regularize bias column
                    grad += reg_l1#applying regularization
                #gradient clipping(if declared)
                if grad_clip != 0.0:
                    grad = clip_gradient(grad, grad_clip)
                # Save previous velocity
                v_prev = velocity[iLayer].copy()
                # NAG velocity update:
                velocity[iLayer] = momentum * velocity[iLayer] - learning_rate * grad
                # We are currently at w_look = w_t + momentum * v_prev.
                # We want final w_{t+1} = w_t + v_new.
                # => w_{t+1} = w_look + (v_new - momentum * v_prev)
                network.weights_list[iLayer] = weights_array + (velocity[iLayer] - momentum * v_prev)
    #trained network and last gradient (weights) are returned. Gradient is returned for sweep
    return network,grad_W


#ADAM OPTIMIZER training loop
def train_minibatch_adam(network,
                         inputs,
                         targets,
                         epochs,
                         learning_rate,
                         batch_size,
                         loss_derivative,
                         beta1=0.9,
                         beta2=0.999,
                         epsilon=1e-8,
                         l1_coeff=0.0,
                         l2_coeff=0.0,
                         grad_clip=0.0):

    #checking for NumPy transpose mistake and solving it(throwing error if there is other problem with shapes of output layer and given targets)
    targets = _prepare_data(network.weights_list[-1].shape[0], targets)
    #getting number of examples to train network on(information needed for training)
    n_samples = inputs.shape[1]
    # Initialize Adam moment vectors for each layer.
    mom1 = [np.zeros_like(W) for W in network.weights_list]  # first moment
    mom2 = [np.zeros_like(W) for W in network.weights_list]  # second moment
    step_count = 0  # Adam time step counter (increments per batch)
    # Loop over dataset
    for epoch in range(epochs):
        # Shuffle data order this epoch
        indices = np.random.permutation(n_samples)
        #Process the dataset in chunks of size batch_size (batching)
        for start in range(0, n_samples, batch_size):
            #getting indices of examples to be used for this batch training
            batch_idx = indices[start:start + batch_size]
            #getting data for current batch into variables
            inputs_batch = inputs[:, batch_idx]
            targets_batch = targets[:, batch_idx]
            #getting batch size for gradient division (as all batches are summed, so they need to be divided to get average across batches). It needs to be checked as last batch might not have the same amount of examples in it as the rest
            batch_size_effective = inputs_batch.shape[1]  # effective batch size
            #forward pass to get network output
            z_values, a_values = network.forward(inputs_batch)
            #Backward pass (gradients summed over batch) -> getting gradients
            grad_W = network.backward(z_values, a_values, targets_batch, loss_derivative)
            #increment of Adam time step
            step_count += 1
            #weights update(iterating through weight arrays of layers)
            for iLayer in range(len(network.weights_list)):
                #getting weights array of current layer
                weights_array = network.weights_list[iLayer]
                # Convert summed gradient to average gradient
                grad = grad_W[iLayer] / batch_size_effective
                #L2 regularization(if declared)
                if l2_coeff != 0.0:
                    reg = l2_coeff * weights_array.copy()#calculating regularization value
                    reg[:, 0] = 0.0#we do not regularize bias column
                    grad += reg#applying regularization
                #L1 regularization(if declared)
                if l1_coeff != 0.0:
                    reg_l1 = l1_coeff * np.sign(weights_array.copy())#calculating regularization value
                    reg_l1[:, 0] = 0.0#we do not regularize bias column
                    grad += reg_l1#applying regularization
                #gradient clipping(if declared)
                if grad_clip != 0.0:
                    grad = clip_gradient(grad, grad_clip)
                #Updating biased first moment estimate
                mom1[iLayer] = beta1 * mom1[iLayer] + (1.0 - beta1) * grad
                #Updating biased second moment estimate
                mom2[iLayer] = beta2 * mom2[iLayer] + (1.0 - beta2) * (grad * grad)
                #making bias corrections
                m_hat = mom1[iLayer] / (1.0 - beta1 ** step_count)
                v_hat = mom2[iLayer] / (1.0 - beta2 ** step_count)
                #computing parameter update
                update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                #apply update (gradient descent step)
                network.weights_list[iLayer] = weights_array - update
    #trained network and last gradient (weights) are returned. Gradient is returned for sweep
    return network,grad_W
