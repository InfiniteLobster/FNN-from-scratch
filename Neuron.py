#
import numpy as np
from InitFunctions import  *
from SuppFunctions import  *
#
class Neuron:
    #instance attributes
    #self.weight_vector - this variable holds vector representing the weight values of neuron. It is represented by 2d array with one row (to represent neuron as part of 1 neuron layer), so by row vector. The 0th index is assumed to represent bias.
    #self.activ_function - this variable holds activation function of neuron.
    #constructor
    def __init__(self,weights,activ_function, method_ini = "Zero", datatype_weights = "float64", random_lower_bound = 0.0, random_upper_bound = 1.0):
        #activation function assignment (activ function is first to assign due to length of processing. It is shorter then for weight. So, if error occurs during it the weight processign would not be done unnecessarly)
        if (callable(activ_function)):#if given variable is callable then there is high chance that's the wanted activation function. It is not ideal solution, but it is optimal
            self.activ_function = activ_function
        else:#if given variable is not an activation function, then class object can not be initialized due to lack (no activ function) or too much (e.g. vector of activ functions) of information. TO DO: implementing proper error
            raise NotImplementedError
        #based on the input (weights) the weight vector of the object would be assigned (or created, initialized). The choice of method is based on the data type of input
        type_weights = type(weights)
        #weights assignment
        if (type_weights == int):#this is case when weights values are not given by the user, only dimension(as it is for single neuron, e.g. one row of weights). In this case weights needs to be initialized. it is assumed that given value is without bias.
            #basic version (zero initialization) is always done as a base to ommit need to pass datatype information down to initialization methods
            weights_conversion = np.zeros([1,(weights+1)],dtype=datatype_weights)#to represent neuron weight vector as one row of layer weight array, the weight vector is a 2D array with only one row(row vector). Given number of weights is assumed to be without bias, so it is added for creation.
            #weights are initialized based on selected method
            match method_ini:#(TO DO: implementing methods)
                case "Random":# in the random initialization the weights values are generated randomly as real numbers between given bounds
                    weights_conversion = randomIni(weights_conversion,random_lower_bound,random_upper_bound)
            #after initialization weight are assigned to object property
            self.weight_vector = weights_conversion
        elif (type_weights == np.ndarray):#this is the case when already initialized weights are passed for neuron object creation
            #not all data types can represent or can be converted to represent weights. They need to be rational number (at least in this implementation). At this point it is verified if passed data meets that criterium.
            if (isRationalNumber(weights)):
                #object weights must be of default dtype or as specified by user. It is checked if that is the case, if not then weights dtype is converted
                weights_dtype = ensureDtypeNpArray(weights, datatype_weights)
                #given variable dimensions are checked to identify if it can be passed as object weight vector and by which method.
                number_dimensions = weights_dtype.ndim
                if(number_dimensions == 1):
                    #to represent neuron weight vector as one row of layer weight array, the weight vector is a 2D array with only one row(row vector).
                    weights_vector_row = weights_dtype[np.newaxis,:]
                    #transformation are finished and results can be assigned as object property
                    self.weight_vector = weights_vector_row
                elif(number_dimensions == 2):
                    #for proper representation the weight_vector of neuron needs to be row vector. It is checked if that's the case and changed if not.
                    if(isRowVector(weights_dtype)):
                        self.weight_vector = weights_dtype
                    elif(isColVector(weights_dtype)):
                        self.weight_vector = weights_dtype.T
                    else:#if none of above true, then given variable is array and not vector. Appropriate error must be passed to user. TO DO: error implementation
                        raise NotImplementedError
                else:#if given NumPy array is not vector than it is unappropriate for use and appropriate error should be thrown. TO DO: implementing proper error (can be probably the same as above)
                    raise NotImplementedError
            else:#given variable is of unappropriate datatype. Error should be thrown. TO DO: error implementation
                raise NotImplementedError
        else:#given variable is of type for which object initialization is not implemented. Appropriate error sould be passed. TO DO: implementing proper error
            raise NotImplementedError
    #simple input processing by network
    def forward(self,input):
        #gettin instance attributes to separate variables for readability
        weight_vector = self.weight_vector
        activ_function = self.activ_function
        #input needs to be in form of np array. 
        if(type(input) == np.ndarray):
            #input to Neuron needs to be a rational number for the operations to be completed. Thus it needs to be verified.
            if(isRationalNumber(input)):
                #depending on the array dimensionality the input can be interpreted differently. Due to this it needs to be checked and passed to proper method of handling.
                number_dimensions = input.ndim
                if(number_dimensions == 1):#given input vector needs to be represented as a column vector in 2D array format for matrix operations. Proper transformations are done below to do so.
                    #the input needs to be represent as a 2D array with only one column(column vector) for matrix operations usage.
                    input_format = input[:,np.newaxis]
                elif(number_dimensions == 2):
                    input_format = getProperInputArray(input)
                else:#if data is of dimensions that handling is not implement proper error should be thrown. TO DO: implement proper error
                    raise NotImplementedError 
                #bias input is added to input vector
                input_ready = addBiasInput(input_format)
            else:#if given data is not proper, the error should be thrown. TO DO: proper error.
                raise NotImplementedError
        else:#if given data is not in proper format, the error should be thrown. TO DO: proper error.
            raise NotImplementedError
        #before proceeding with input propagation through neuron it needs to be verified if input is compatible.
        if(weight_vector.shape[1] == input_ready.shape[0]):#if they are compatible for matrix multiplication than operation can proceed
            #input is multiplied by weights for forward pass (matrix multiplication)
            matrix_multi = weight_vector @ input_ready
            #the results of input "passing through" weights needs to be put through activation function
            activation_out = activ_function(matrix_multi)
            #both matrix multiplication results (z) and activation results (a) are send out
            output = [matrix_multi,activation_out]
        else:#if inproper input was given and operation cannot proceed proper error should be raised. TO DO: implement proper error
            raise NotImplementedError
        #results are returned
        return output
  