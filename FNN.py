from LossFunctions import MeanSquaredError,mse_derivative
from ActivFunctions import identity_derivative
import numpy as np

class FNN:


#constructor
    def __init__(self,weights,activ_functions):
        #
        self.weights = weights
#methods
    def forward(self,input):
        output = input

        return output
    
    def backward(self, input_data, target_output, predicted_output):

        # ---- STEP 1: LOSS AND LOSS DERIVATIVE ----
        L = MeanSquaredError(target_output, predicted_output)
        dL = mse_derivative(predicted_output, target_output)

        # ---- PREPARE LIST TO STORE GRADIENTS ----
        grads = []

        # ---- STEP 2: INITIAL DELTA FOR LAST LAYER ----
        delta = dL * identity_derivative(self.z_values[-1])

        # ---- LOOP BACKWARD THROUGH ALL LAYERS ----
        # (start at last layer index, go backwards to layer 0)
        for layer_index in reversed(range(len(self.weights))):

            # 1. INPUT TO THIS LAYER
            if layer_index == 0:
                input_to_layer = input_data
            else:
                input_to_layer = self.a_values[layer_index - 1]

            # 2. COMPUTE WEIGHT GRADIENT FOR THIS LAYER
            dW = np.outer(delta, input_to_layer)
            grads.insert(0, dW)   # insert at front to preserve layer order

            # 3. IF THIS IS NOT THE FIRST LAYER, PROPAGATE ERROR BACK
            if layer_index > 0:

                # Remove the bias column
                W_no_bias = self.weights[layer_index][:, 1:]

                # Backpropagate the error
                error_prev = W_no_bias.T @ delta

                # Compute delta for the previous layer
                z_prev = self.z_values[layer_index - 1]
                delta = error_prev * identity_derivative(z_prev)

            # If layer_index == 0: stop (no previous layer)

        # ---- RETURN ALL WEIGHT GRADIENTS ----
        return grads
