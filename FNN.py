import numpy as np
from InitFunctions import  *
from SuppFunctions import  *

class FNN:


#constructor
    def __init__(self,weights,activ_functions,method_ini = "Zero", datatype_weights = "float64", random_lower_bound = 0.0, random_upper_bound = 1.0):
        #weights assignment
        weights_dtype = type(weights)
        if(weights_dtype == np.ndarray):#based on input types the weight assignment would process differently, so it is split by ifelse construct
            weights_dim = weights.ndim#depending on dimension of given data, the interpretation of weights assignment is different (or can be impossible), so it is evaluated and logic proceedes through ifelse construct
            if(weights_dim == 1):#if vector is given then it is assumed that it is for weight assignment, they should be initialized
                if(np.issubdtype(weights.dtype, np.integer)):#numbers in vector needs to be integers to represnt dimensions of layer and neurons in it
                    #TO DO: this part needs to change for real usability.
                    if(weights.size == 3):#as weights of single layer are represented by 2D array, then if you need to represent multiple of them you only need three numbers for dimensions. If there is less or more than ambiguity is created. To omit it, the initialization can proceed only with 3 numbers in vector.
                        nNeurons = weights[0]#first number is assumed to be for layer size (number of neurons in layer)
                        nWeights = weights[1]#second number is assumed to be for number of weights in first layer (bias is not counted).For rest of the layers, the number of inputs depends on number of neurons in previous layer
                        nLayers = weights[2]#thrid number is assumed to be for number of layers (input layer is not imputed as in design it is not represent in weight matrix list)
                        #list to hold 2D weight arrays of layers is declared for proper assignment in loop.
                        weights_layers = []
                        #to initialize
                        for iLayer in range(nLayers):
                            #weights are initializaed by basic method
                            weights_conversion = np.zeros([nNeurons,(nWeights+1)],dtype=datatype_weights)
                            #for each subsequent layer, the number of weights depends on number of neurons in previous layer, so its assignment needs to be adjusted
                            nWeights = nNeurons
                            #if method other than basic method (zero) was selected, than weights are initialized according to it
                            match method_ini:#(TO DO: implementing methods)
                                case "Random":# in the random initialization the weights values are generated randomly as real numbers between given bounds
                                    weights_conversion = randomIni(weights_conversion,random_lower_bound,random_upper_bound)
                            weights_layers.append(weights_conversion)
                        #after initialization weights are assigned to object property
                        self.weights_list= weights_layers
                    else:
                        raise NotImplementedError
                else:#if in given vector the numbers are not integers, then they cannot be interpreted as dimensions for initialization, so weight creation is impossible. TO DO: Proper error should be thrown
                    raise NotImplementedError
            elif(weights_dim == 2):
                raise NotImplementedError
            else:#array of incompatible size was given. Processing is not possible, so proper error should be thrown. TO DO: Implement proper error
                raise NotImplementedError
        elif(weights_dtype == list):#additional option of initialization through list can be considered
            raise NotImplementedError
        else:#input that is not compatible was given. Operation cannot proceed, so proper error should be thrown. TO DO: Implement proper error
            raise NotImplementedError
        #activation function assignment 
        if (callable(activ_functions)):#if given variable is callable then there is high chance that's the wanted activation function. It is not ideal solution, but it is optimal. In this case only one activ function is given and it is interpreted as all neurons should have same activ function
            #list of length equal to number of neurons is created. Based on assumption all activation functions would be the same if only one activation function is given .
            activ_functions_list = []
            for iLayer in range(len(self.weights_list)):
                activ_functions_base = [activ_functions] * self.weights_list[iLayer].shape[0]
                activ_functions_list.append(activ_functions_base)
            #ready activ function list is passed to instance atrribute
            self.activ_functions_list = activ_functions_list
        elif(type(activ_functions) == np.ndarray):#different activation function for each neuron in layer would posssible if gven list of them (tool for customization). TO DO: implementation
            raise NotImplementedError
        else:#if given variable is not an activation function, then class object can not be initialized due to lack (no activ function) of data. TO DO: implementing proper error
            raise NotImplementedError
#methods
    def forward(self,input):
        #getting instance attributes to separate variables for readability
        weights_list = self.weights_list
        activ_functions_list = self.activ_functions_list
        #input needs to be in form of np array and that needs to be verified. 
        if(type(input) == np.ndarray):
            #input to Layer needs to be a rational number for the operations to be completed. Thus it needs to be verified.
            if(isRationalNumber(input)):
                #depending on the array dimensionality the input can be interpreted differently. Due to this it needs to be checked and passed to proper method of handling.
                number_dimensions = input.ndim
                if(number_dimensions == 1):#given input vector needs to be represented as a column vector in 2D array format for matrix operations. Proper transformations are done below to do so.
                    #the input needs to be represent as a 2D array with only one column(column vector) for matrix operations usage (in case of vector).
                    input_format = input[:,np.newaxis]
                elif(number_dimensions == 2):#given array might be not only array, but also row and column vector. If only vector is given, than it has to be in column vector form. To ensure proper form some verification and if necessery processing is done
                    input_format = getProperInputArray(input)
                else:#if data is of dimensions that handling is not implement proper error should be thrown. TO DO: implement proper error
                    raise NotImplementedError 
            else:#if given data is not proper, the error should be thrown. TO DO: proper error.
                raise NotImplementedError
        else:#if given data is not in proper format, the error should be thrown. TO DO: proper error.
            raise NotImplementedError
        #matix multiplications results and activation results needs to be saved for output. For this purpose variables with lists are created
        z = [input_format]#in this implementation a first value of z(matrix multi results) would be input values (can be interpreted as representation of input layer, which is not represented by any network attribute in this implementation) as for input layer it is input.
        a = [input_format]#in this implementation a first value of a(activation results) would be input values (can be interpreted as representation of input layer, which is not represented by any network attribute in this implementation) as for input layer it is identity value.
        #input needs to propagate through all layers in network and it is achieved by looping through them.
        for iLayer in range(len(weights_list)):
            #variables necessery for current layer propagations are assigned to local variables for readability
            weights_array = weights_list[iLayer]
            activ_functions = activ_functions_list[iLayer]
            input = a[-1]
            #bias input is added to input vector/array to account for bias, which is 0 weight
            input_ready = addBiasInput(input)
            #it needs to be verified if matrix multiplication can be performed. Theoretically it should be strictly necessery only for first input as rest should be accounted by architecture, but it is present for all if some undetected mistake was made in architecture creation.
            if(weights_array.shape[1] == input_ready.shape[0]):
                #input is multiplied by weights for forward pass (matrix multiplication)
                matrix_multi = weights_array @ input_ready
                #the results of input "passing through" weights needs to be put through activation functions
                activation_out = activationLayer(matrix_multi,activ_functions)
                #calculated values are passed to lists storing them
                z.append(matrix_multi)
                a.append(activation_out)
            else:#if inproper input was given and operation cannot proceed proper error should be raised. TO DO: implement proper error
                raise NotImplementedError
        #results are joined together into list for proper output
        output = [matrix_multi,activation_out]
        #results are returned
        return output