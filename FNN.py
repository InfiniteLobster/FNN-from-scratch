import numpy as np
from InitFunctions import  *
from SuppFunctions import  *
from ErrorClasses import *

class FNN:
#instance attributes
#self.weights_list - this variable holds list of arrays representing the weight values of layer neurons for each layer in the network. Weights are represented by 2d array where rows represent individual neurons and columns represent wieghts of neurons. The 0th column index is assumed to represent bias.
#self.activ_functions_list - this variable holds list of "list of activation function of neurons in layer" for all layers.

#constructor
    def __init__(self,weights,activ_functions,method_ini = "Zero", datatype_weights = "float64", random_lower_bound = 0.0, random_upper_bound = 1.0):
        #if given weight information is list of neuron numbers in each dimension than it needs to be converted into np array (for code uniformity).
        if(type(weights) == list):#it first need to be checked if given input is list
            #
            if(len(weights) == 0):
                raise NotImplementedError 
            #
            if(all(isinstance(weight, int) for weight in weights)):#now it is checked if it is list of integer values (not np arrays like with other initialization method)
                try:
                    weights = np.asanyarray(weights)#compared to Neuron class, the list can only represent dimensions, so it has to be in format for dimensions not weights (dtype = datatype_weights is missing here)
                except Exception as error_caught:
                    if(isinstance(error_caught,ValueError)):
                        raise NotSupportedInputGiven("weights initialization","Values given in list are not numbers and thus can not be used as network dimensions.")
                else:
                    raise error_caught
        #weights assignment
        if(type(weights) == np.ndarray):#based on input types the weight assignment would process differently, so it is split by ifelse construct
            if(weights.ndim == 1):#if vector is given then it is assumed that it is for weight initialization by number of neurons in each layer information
                if(np.issubdtype(weights.dtype, np.integer)):#numbers in vector needs to be integers to represnt dimensions of layer and neurons in it
                    if(weights.size > 1):#you need at least 2 layers counting input to create smallest network
                        nWeights = weights[0]#first number is assumed to be for number of neurons in input layers
                        nLayers = (len(weights) - 1)#the number of layers to be initialized is smaller by 1, because input layer is not represented by weights in this implementation
                        #list to hold 2D weight arrays of layers is declared for proper assignment in loop.
                        weights_layers = []
                        #to initialize network, all layers (except input) needs to be initialize separately to ensure that is done properly.
                        for iLayer in range(nLayers):
                            #getting number of neurons in current layer
                            nNeurons = weights[iLayer+1]
                            #weights are initializaed by basic method
                            weights_conversion = np.zeros([nNeurons,(nWeights+1)],dtype=datatype_weights)#bias is accounted for by adding 1
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
                        raise NotSupportedInputGiven("weights initialization","At least two numbers are needed for FNN creation: FNN needs at least input and output layer.")
                else:#if in given vector the numbers are not integers, then they cannot be interpreted as number of neurons, so weight creation is impossible. TO DO: Proper error should be thrown
                    raise NotSupportedInputGiven("weights initialization","Given values in vector must be integers to properly represent number of neurons in layers")
            else:#array of incompatible size was given. Processing is not possible, so proper error should be thrown.
                raise NotSupportedArrayDimGiven("1")
        elif(type(weights) == list):#in this case network is initialized by giving all weights
            #first it is verified if in all elements of list there are arrays representing layers
            if(all(isinstance(weight, np.ndarray) for weight in weights)):
                #list to hold 2D weight arrays of layers is declared for proper assignment in loop.
                weights_layers = []
                #to initialize network weight layers, all layers needs to be initialize separately to ensure that is done properly (verification if dimensions match so that input propagation is possible).
                for iLayer in range(nLayers):
                    #weight array is assigned to variable
                    weights_array = weights[iLayer]
                    if(iLayer>0):#for all, but first layer it needs to be checked if given weight arrays allow the input propagation through network. If not error should be thrown
                        #
                        if(weights_array.shape[1]== weights_layers[-1].shape[0]):
                            weights_layers.append(weights_array)
                        else:
                            raise NotImplementedError 
                    else:
                        #
                        if((weights_array.shape[0] > 0) | (weights_array.shape[1] > 0)):
                            weights_layers.append(weights_array)
                        else:
                            raise NotImplementedError 
            else:
                raise NotImplementedError
        else:#input that is not compatible was given. Operation cannot proceed, so proper error should be thrown.
            raise NotSupportedInputGiven("weights initialization")
        #activation function assignment 
        if (callable(activ_functions)):#if given variable is callable then there is high chance that's the wanted activation function. It is not ideal solution, but it is optimal. In this case only one activ function is given and it is interpreted as all neurons should have same activ function
            #list of length equal to number of neurons is created. Based on assumption all activation functions would be the same if only one activation function is given .
            activ_functions_list = []
            for iLayer in range(len(self.weights_list)):
                activ_functions_base = [activ_functions] * self.weights_list[iLayer].shape[0]
                activ_functions_list.append(activ_functions_base)
            #ready activ function list is passed to instance atrribute
            self.activ_functions_list = activ_functions_list
        elif(type(activ_functions) == list):#different activation function for each neuron in layer would posssible if given list of them (tool for customization). TO DO: implementation
            raise NotImplementedError
        else:#if given variable is not an activation function, then class object can not be initialized due to lack (no activ function) of data. TO DO: implementing proper error
            raise NotImplementedError
#methods
    #simple input processing by network
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
        output = [z,a]
        #results are returned
        return output