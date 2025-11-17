import SuppFunctions
from ActivFunctions import identity_derivative, identity


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
    # Since identity activation is used in current implementation, its derivative
    # is simply a matrix of ones matching z’s shape.
    activation_deriv_last = identity_derivative(z_L)

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

            # Derivative of activation for the previous layer (identity again).
            activation_deriv_prev = identity_derivative(z_values[layer_index - 1])

            # Update grad_z for the next iteration (layer below).
            grad_z = back_signal_no_bias * activation_deriv_prev

    # The list of gradients, one for each layer, is returned for use in gradient descent.
    return grad_W
