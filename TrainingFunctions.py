import SuppFunctions
import numpy as np 

def backwards(network, inputs, targets, loss_derivative):
    """
    This function computes the gradients of all weight matrices based on provided
    input and desired target output using standard backpropagation.
    """

    # ---- Forward pass ----
    # Forward propagation must be performed first because backpropagation requires
    # all intermediate z (pre-activation) and a (post-activation) values.
    z_values, a_values = network.forward(inputs)

    # Number of layers in the network (i.e., weight matrices)
    num_layers = len(network.weights_list)

    # List to store computed gradients (same shape as weight matrices)
    grad_W = [None] * num_layers

    # ---- Output layer gradient initialisation ----
    # Gradient of the loss with respect to activations of the final layer.
    a_L = a_values[-1]
    z_L = z_values[-1]
    grad_a_L = loss_derivative(targets, a_L)

    # Retrieve activation derivatives for each neuron separately
    act_funcs_last = network.activ_functions_list[-1]
    activation_deriv_last = np.zeros_like(z_L)

    for i, act in enumerate(act_funcs_last):
        der_func = SuppFunctions.getDer(act)
        activation_deriv_last[i, 0] = der_func(z_L[i, 0])

    # Combine chain rule: dL/dz = dL/da * da/dz
    grad_z = grad_a_L * activation_deriv_last

    # ---- Backpropagation through all layers ----
    for layer_index in reversed(range(num_layers)):

        # Retrieve activation of previous layer (or input)
        a_prev = a_values[layer_index]
        a_prev_with_bias = SuppFunctions.addBiasInput(a_prev)

        # Weight gradient: matrix multiplication between gradients and previous activations
        grad_W[layer_index] = grad_z @ a_prev_with_bias.T

        grad_W[layer_index] = SuppFunctions.clip_gradient(grad_W[layer_index])

        # For hidden layers: propagate gradients backward
        if layer_index > 0:

            W = network.weights_list[layer_index]

            # Backpropagate error through weights
            back_signal = W.T @ grad_z

            # Remove bias component
            back_signal_no_bias = back_signal[1:, :]

            # Retrieve activation derivatives for each neuron in this layer
            act_funcs_prev = network.activ_functions_list[layer_index - 1]
            activation_deriv_prev = np.zeros_like(z_values[layer_index - 1])

            for i, act in enumerate(act_funcs_prev):
                der_func = SuppFunctions.getDer(act)
                activation_deriv_prev[i, 0] = der_func(z_values[layer_index - 1][i, 0])

            # Update gradient to pass down to next layer
            grad_z = back_signal_no_bias * activation_deriv_prev

    # Return all layer gradients
    return grad_W