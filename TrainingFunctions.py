import SuppFunctions


def backwards(network, inputs, targets, loss_derivative):
    """
    This function computes the gradients of all weight matrices based on provided input
    and desired target output.
    """

    # Forward pass is performed first, as backward propagation requires
    # stored z (matrix multiplication results) and a (activation results)
    # values for every layer in the network. These lists contain one entry
    # for each layer, with index 0 representing the input layer.
    z_values, a_values = network.forward(inputs)

    # Number of layers in the network equals number of weight matrices.
    num_layers = len(network.weights_list)

    # List for storing calculated gradients. Each entry corresponds to the
    # gradient matrix for one layer and matches the shape of that layer’s weights.
    grad_W = [None] * num_layers

    # For the output layer, the gradient of the loss with respect to activation
    # must first be computed. This is done using the loss derivative function
    # supplied by the user (e.g., MSE derivative).
    a_L = a_values[-1]
    z_L = z_values[-1]
    grad_a_L = loss_derivative(targets, a_L)

    # Activation derivatives must be computed to continue the chain rule.
    # Instead of hard-coding identity, derivative is retrieved dynamically
    # using getDer() based on the activation function assigned to this layer.
    activ_functions_last = network.activ_functions_list[-1]
    act_last = activ_functions_last[0]                   # activation function of neurons in this layer
    act_der_last = SuppFunctions.getDer(act_last)        # derivative function
    activation_deriv_last = act_der_last(z_L)

    # Gradient of loss with respect to z of the final layer is obtained by
    # element-wise multiplication of previous gradient and activation derivative.
    grad_z = grad_a_L * activation_deriv_last

    # Gradients must now be propagated through every layer. Backward propagation
    # proceeds from the last layer toward the first, therefore reverse iteration.
    for layer_index in reversed(range(num_layers)):

        # a_prev represents the activation of the previous layer.
        # Since bias is used in this implementation, the bias input must be added
        # before forming the delta-weight product.
        a_prev = a_values[layer_index]
        a_prev_with_bias = SuppFunctions.addBiasInput(a_prev)

        # Gradient of loss with respect to weights is computed as outer product
        # between current layer’s grad_z and previous layer’s activations.
        grad_W[layer_index] = grad_z @ a_prev_with_bias.T

        # For all layers except the first, the gradient signal must be propagated
        # backwards. This requires multiplication by the transpose of weight matrix,
        # and removing the first row that corresponds to bias influence.
        if layer_index > 0:

            W = network.weights_list[layer_index]

            # Propagate error backwards through weights.
            back_signal = W.T @ grad_z

            # First entry corresponds to bias and must be excluded.
            back_signal_no_bias = back_signal[1:, :]

            # Derivative of activation for the previous layer is retrieved dynamically
            # using getDer(), ensuring support for any activation function.
            activ_functions_prev = network.activ_functions_list[layer_index - 1]
            act_prev = activ_functions_prev[0]
            act_der_prev = SuppFunctions.getDer(act_prev)
            activation_deriv_prev = act_der_prev(z_values[layer_index - 1])

            # Update grad_z for the next iteration (layer below).
            grad_z = back_signal_no_bias * activation_deriv_prev

    # The list of gradients, one for each layer, is returned for use in gradient descent.
    return grad_W
