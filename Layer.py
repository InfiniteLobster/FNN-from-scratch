import numpy as np
from InitFunctions import  *
from SuppFunctions import  *
from ErrorClasses import *

class Layer:
#instance attributes
#self.weights_array - this variable holds array representing the weight values of layer neurons. It is represented by 2d array where rows represent individual neurons and columns represent wieghts of neurons. The 0th column index is assumed to represent bias.
#self.activ_functions - this variable holds list of activation function of neurons in layer.

#constructor
    def __init__(self,weights,activ_functions,method_ini = "Zero", datatype_weights = "float64", random_lower_bound = 0.0, random_upper_bound = 1.0, random_mean = 0.0, random_std = 1.0):
        #conversion of given weight input if it is list instead of np array (not always possible as numbers are needed)
        if(type(weights) == list):
            try:
                weights = np.array(weights)#compared to Neuron class, the list can only represent dimensions, so it has to be in format for dimensions not weights (dtype = datatype_weights is missing here)
            except Exception as error_caught:
                if(isinstance(error_caught,ValueError)):
                    raise NotSupportedInputGiven("weights initialization","Values given in list are not numbers and thus can not be used as weight array dimensions.")#not being integers is caught in other section below. There is no need to repeat.
                else:
                    raise error_caught
        #weights assignment
        if(type(weights) == np.ndarray):#based on input types the weight assignment would process differently, so it is split by ifelse construct
            weights_dim = weights.ndim#depending on dimension of given data, the interpretation of weights assignment is different (or can be impossible), so it is evaluated and logic proceedes through ifelse construct
            if(weights_dim == 1):#if vector is given then it is assumed that it is for weight assignment, they should be initialized. It would also serve as initialization of instance by list (converted earlier to np.array so this segment of code can be used in both cases).
                if(np.issubdtype(weights.dtype, np.integer)):#numbers in vector needs to be integers to represnt dimensions of layer and neurons in it
                    if(weights.size == 2):#as weights of single layer are represented by 2D array, then only two numbers for dimensions are needed. If there is less or more than ambiguity is created. To omit it, the initialization can proceed only with 2 numbers.
                        nNeurons = weights[0]#first number is assumed to be for layer size (number of neurons in layer)
                        nWeights = weights[1]#second number is assumed to be for number of weights in neurons of layer (bias is not counted)
                        #weights are initializaed by basic method
                        shape_ini = [nNeurons,(nWeights+1)]#1 is added to weights, because by convention given number does not include bias, which needs to be accounted during matrix initialization as it is part of weight matrix
                        #weight array is initialized according to choosen method
                        match method_ini:
                            case "Zero":
                                weights_ini = zeroIni(shape_ini,datatype_weights)
                            case "RandomUni":
                                weights_ini = randomIniUniform(shape_ini,datatype_weights, lower_bound= random_lower_bound, upper_bound= random_upper_bound)
                            case "RandomNor":
                                weights_ini = randomIniNormal(shape_ini,datatype_weights, mean = random_mean, std = random_std)
                            case "XavUni":
                                weights_ini = xavIniUniform(shape_ini,datatype_weights)
                            case "XavNor":
                                weights_ini = xavIniNormal(shape_ini,datatype_weights)
                            case "HeUni":
                                weights_ini = heIniUniform(shape_ini,datatype_weights)
                            case "HeNor":
                                weights_ini = heIniNormal(shape_ini,datatype_weights)
                        #after initialization weights are assigned to object property
                        self.weights_array= weights_ini
                    else:
                        raise NotSupportedInputGiven("weights initialization","In this implementation initialization by vector (1 dim) is only supported for dimensions to initialize. As there are only 2 dimensions to initialize layer any other are unsuitable and raise error due to ambiguity")
                else:#if in given vector the numbers are not integers, then they cannot be interpreted as dimensions for initialization, so weight creation is impossible. TO DO: Proper error should be thrown
                    raise NotSupportedInputGiven("weights initialization","Given values are not integers and thus can not represent of layer weights matrix.")
            elif(weights_dim == 2):#if given array has 2 dimensions, then it is assumed that it is weight array of Layer instance.
                #to create layer, there needs to be information to represent at least 1 neuron with at least one weight (bias if one), i.e. weight array can not be empty.
                if((weights.shape[0] >= 1) & (weights.shape[1] >= 1 )):
                    #only rational numbers can represent weight values in this implementation, so it needs to be verified if that;s the case and raise error if not.
                    if(isRationalNumber(weights)):#if numbers are rational than they can be weights
                        #assigning given weights to object instance property
                        self.weights_array= weights
                    else:#if number are not rational, than error needs to be raised
                        raise NotSupportedInputGiven("weights initialization","Given values are not a rational numbers and thus can not be used as weight values.")
                else:
                    raise NotSupportedInputGiven("weights initialization","Given array have insuffiecient information to represent weights of layer.")
            else:#array of incompatible size was given. Processing is not possible, so proper error should be thrown. TO DO: Implement proper error
                raise NotSupportedArrayDimGiven("1,2")
        else:#input that is not compatible was given. Operation cannot proceed, so proper error should be thrown. TO DO: Implement proper error
            raise NotSupportedInputGiven("weights initialization","Not supported data type given.")
        #activation function assignment 
        if (callable(activ_functions)):#if given variable is callable then there is high chance that's the wanted activation function. It is not ideal solution, but it is optimal. In this case only one activ function is given and it is interpreted as all neurons should have same activ function
            #list of length equal to number of neurons is created. Based on assumption all activation functions would be the same if only one activation function is given .
            activ_functions_base = [activ_functions] * self.weights_array.shape[0]
            #ready activ function list is passed to instance atrribute
            self.activ_functions = activ_functions_base
        elif(type(activ_functions) == list):#different activation function for each neuron in layer would posssible if given list of them (tool for customization). TO DO: implementation
            #number of given activation functions should be equal to neurons in layer or be only one to be put (copy) as the same for all neurons
            num_activ_functions = len(activ_functions)
            #based on number of elements in the list different initialization method is used (or error raised if uncompatible information was given)
            if(num_activ_functions == 1):#if there is only one element, then it is assumed that it is activation function for all of neurons
                #list of length equal to number of neurons is created. Based on assumption all activation functions would be the same if only one activation function is given .
                activ_functions_base = [activ_functions[0]] * self.weights_array.shape[0]
                #ready activ function list is passed to instance atrribute
                self.activ_functions = activ_functions_base
            elif(num_activ_functions == weights.shape[0]):#number of activation function should match the number of neurons (or be single one to be the same for the whole layer) for proper initialization
                #each activation function would be appended to this list
                activ_functions_base = []
                #each element of list needs to be tested if it is functiuon (same mechanism as before).
                for iNeuron in range(num_activ_functions):
                    #current element of the list is assigned to variable for better readability.
                    element = activ_functions[iNeuron]
                    #verification if element of list is a activ function is done.
                    if(callable(element)):
                        activ_functions_base.append(element)
                    else:
                        raise NotSupportedInputGiven("activation functions initialization","Given variable is not function")
                self.activ_functions = activ_functions_base
            else:#if size of given list is uncompatible, then initialization is impossible and proper error should be thrown
                raise NotSupportedInputGiven("activation functions initialization","Given activation functions number does not match number of neurons in layer")
        else:#if given variable is not an activation function, then class object can not be initialized due to lack (no activ function) of data. TO DO: implementing proper error
            raise NotSupportedInputGiven("activation functions initialization","Not supported data type given.")
#methods
    #simple input processing by network
    def forward(self,input):
        #getting instance attributes to separate variables for readability
        weights_array = self.weights_array
        activ_functions = self.activ_functions
        #conversion of given input if it is list instead of np array (not always possible as numbers are needed)
        if(type(input) == list):
            try:
                input = np.asanyarray(input, dtype = weights_array.dtype)#input should be in same format as weights to stay consistent with data types in calculations.
            except Exception as error_caught:
                if(isinstance(error_caught,ValueError)):
                    raise NotSupportedInputGiven("input propagation","Values given in list are not numbers and thus can not be used as input to neuron.")
                else:
                    raise error_caught
        #input needs to be in form of np array. 
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
                else:#if data is of dimensions that handling is not implement proper error should be thrown. 
                    raise NotSupportedArrayDimGiven("1,2")
            else:#if given data is not proper, the error should be thrown. 
                raise NotSupportedInputGiven("input propagation","Given values are not a rational numbers and thus can not be used to get output from  layer neurons.")
        else:#if given data is not in proper format, the error should be thrown. 
            raise NotSupportedInputGiven("input propagation","Not supported data type given.")
        #bias input is added to input vector/array to account for bias, which is 0 weight
        input_ready = addBiasInput(input_format)
        #before proceeding with input propagation through layer it needs to be verified if input is compatible.
        if(weights_array.shape[1] == input_ready.shape[0]):#if they are compatible for matrix multiplication than operation can proceed
            #input is multiplied by weights for forward pass (matrix multiplication)
            z = weights_array @ input_ready
            #the results of input "passing through" weights needs to be put through activation functions
            a = activationLayer(z,activ_functions)
            #both matrix multiplication results (z) and activation results (a) are send out
            output = [z,a]
        else:#if inproper input was given and operation cannot proceed and proper error should be raised. 
            raise NotSupportedInputGiven("input propagation","Input does not match network, i.e. matrix multiplication cannot be done due to mismatch of dimensions.")
        #results are returned
        return output