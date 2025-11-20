import SuppFunctions
import numpy as np 

def backwards(network, inputs, targets, loss_derivative):
    """
    Standard backpropagation to compute gradients for all layers.
    Works for arbitrary-depth networks with per-neuron activation functions.
    """

    # ---- Forward pass ----
    z_values, a_values = network.forward(inputs)
    num_layers = len(network.weights_list)

    # z_values structure:
    #   z_values[0] = input
    #   z_values[1] = pre-activation layer 1
    #   z_values[2] = pre-activation layer 2
    #   ...
    #   z_values[num_layers] = final layer
    #
    # weights_list[k] corresponds to z_values[k+1]

    grad_W = [None] * num_layers

    # ---- Output layer ----
    a_L = a_values[-1]            # activation output
    z_L = z_values[-1]            # pre-activation output
    grad_a_L = loss_derivative(targets, a_L)

    # Compute derivative of activation for output layer
    act_funcs_last = network.activ_functions_list[-1]
    activation_deriv_last = np.zeros_like(z_L)

    for i, act in enumerate(act_funcs_last):
        der_func = SuppFunctions.getDer(act)
        activation_deriv_last[i, 0] = der_func(z_L[i, 0])

    # dL/dz for output:
    grad_z = grad_a_L * activation_deriv_last

    # ---- Backpropagate through all layers ----
    for layer_index in reversed(range(num_layers)):

        # Correct z-indexing:
        # weights_list[layer_index] ↔ z_values[layer_index+1]
        z_current = z_values[layer_index + 1]
        a_prev    = a_values[layer_index]

        # Add bias to previous activations
        a_prev_with_bias = SuppFunctions.addBiasInput(a_prev)

        # Weight gradient
        grad_W[layer_index] = grad_z @ a_prev_with_bias.T
        grad_W[layer_index] = SuppFunctions.clip_gradient(grad_W[layer_index])

        # Skip backprop for input layer
        if layer_index == 0:
            break

        # ---- Compute gradient for next (previous) layer ----

        W = network.weights_list[layer_index]

        # back_signal = W^T * grad_z
        back_signal = W.T @ grad_z

        # Remove bias row → derives only wrt non-bias activations
        back_signal_no_bias = back_signal[1:, :]

        # Retrieve pre-activation for previous layer
        # z_values[layer_index] is correct for previous layer
        prev_z = z_values[layer_index]

        # Activation derivatives for previous layer
        act_funcs_prev = network.activ_functions_list[layer_index - 1]
        activation_deriv_prev = np.zeros_like(prev_z)

        for i, act in enumerate(act_funcs_prev):
            der_func = SuppFunctions.getDer(act)
            activation_deriv_prev[i, 0] = der_func(prev_z[i, 0])

        # dL/dz for the previous layer
        grad_z = back_signal_no_bias * activation_deriv_prev

    return grad_W
