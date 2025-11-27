import numpy as np
from InitFunctions import  *
from SuppFunctions import  *
from ErrorClasses import *

class FNN:
#instance attributes
#self.weights_list - this variable holds list of arrays representing the weight values of layer neurons for each layer in the network. Weights are represented by 2d array where rows represent individual neurons and columns represent wieghts of neurons. The 0th column index is assumed to represent bias.
#self.activ_functions_list_list - this variable holds list of "list of activation function of neurons in layer" for all layers.

#constructor
    def __init__(self,weights,activ_functions,method_ini = "Zero", datatype_weights = "float64", random_lower_bound = - 1.0, random_upper_bound = 1.0):
        #if given weight information is list of neuron numbers in each dimension than it needs to be converted into np array (for code uniformity).
        if(type(weights) == list):#it first need to be checked if given input is list
            #to initialize weights some information needs to be given. If given list is empty (so it has no information) proper error should be thrown.
            if(len(weights) == 0):
                raise NotSupportedInputGiven("weights initialization","Given list is empty")
            #only integers can represent number of nurons in each layer. For this reason it needs to be verified if that's the case. If not proper error should be thrown
            if(all(isinstance(weight, int) for weight in weights)):#now it is checked if it is list of integer values (not np arrays like with other initialization method)
                try:
                    weights = np.array(weights)#compared to Neuron class, the list can only represent number of neurons (integers), so it has to be in format for dimensions not weights (floats). (dtype = datatype_weights is missing here)
                except Exception as error_caught:#if value error is caught, then no-standard error message should be given for clarity.
                    if(isinstance(error_caught,ValueError)):
                        raise NotSupportedInputGiven("weights initialization","Values given in list are not integer numbers and thus can not be used as neuron number.")
                    #if other error than value error is caight, then it should thrown with its message.
                    raise error_caught
        #weights assignment
        if(type(weights) == np.ndarray):#based on input types the weight assignment would process differently, so it is split by ifelse construct
            if(weights.ndim == 1):#if vector is given then it is assumed that it is for weight initialization by number of neurons in each layer information
                if(np.issubdtype(weights.dtype, np.integer)):#numbers in vector needs to be integers to represnt dimensions of layer and neurons in it
                    if(weights.size > 1):#you need at least 2 layers counting input to create smallest network by initialization. The reason for that is how each layer connects, i.e. the number of neurons in previous layer needs to match number of weights in next, so to create weight array for first hidden layer, the input layer infromation is needed (despite input layer not being represented in weight list in this implementation)
                        nWeights = weights[0]#first number is assumed to be for number of neurons in input layers
                        nLayers = (len(weights) - 1)#the number of layers to be initialized is smaller by 1. It is so, because input layer is not represented by weights in this implementation and thus it doesn't have to be initialized
                        #list to hold 2D weight arrays of layers is declared for proper assignment in loop.
                        weights_layers = []
                        #to initialize network, all layers (except input) needs to be initialized separately to ensure that it is done properly.
                        for iLayer in range(nLayers):
                            #getting number of neurons in current layer
                            nNeurons = weights[iLayer+1]#as [0] layer corresponds to inpout layer, the information of hidden layers starts from [1], so the information retrieval is shifted by 1
                            #weights are initializaed by basic method
                            weights_conversion = np.zeros([nNeurons,(nWeights+1)],dtype=datatype_weights)#bias is accounted for by adding 1 to number of weights.
                            #for each subsequent layer, the number of weights depends on number of neurons in previous layer, so its assignment needs to be adjusted
                            nWeights = nNeurons#assigned value is used in next iteration. Of course it is unused for output layer as there is no next layer. So, this assignment is redundant for last iteration, but it is cheaper to leave as it is, than to create if statment.
                            #if method other than basic method (zero) was selected, than weights are initialized according to it
                            match method_ini:#(TO DO: implementing other methods. Possible TO DO: different ini method for each layer)
                                case "Random":# in the random initialization the weights values are generated randomly as real numbers between given bounds
                                    weights_conversion = randomIni(weights_conversion,random_lower_bound,random_upper_bound)
                            #after initialization by chosen method, the layer is added to the list.
                            weights_layers.append(weights_conversion)
                        #after initialization weights are assigned to object property
                        self.weights_list= weights_layers
                    else:
                        raise NotSupportedInputGiven("weights initialization","At least two numbers are needed for FNN creation: FNN needs at least input and output layer (single layer network).")
                else:#if in given vector the numbers are not integers, then they cannot be interpreted as number of neurons, so weight creation is impossible.
                    raise NotSupportedInputGiven("weights initialization","Given values in vector must be integers to properly represent number of neurons in layers")
            else:#array of incompatible size was given. Processing is not possible, so proper error should be thrown. 
                raise NotSupportedArrayDimGiven("1")
        elif(type(weights) == list):#in this case network is initialized by taking all weight values from given information.
            #first it is verified if in all elements of list there are arrays representing layers
            if(all(isinstance(weight, np.ndarray) for weight in weights)):#all elements of the list must be np arrays to represent weight arrays of each layer
                #list to hold 2D weight arrays of layers is declared for proper assignment in loop.
                weights_layers = []
                #to initialize network weight layers, all layers needs to be initialize separately to ensure that it is done properly (verification if dimensions match so that input propagation is possible).
                for iLayer in range(len(weights)):
                    #weight array is assigned to variable for code readability
                    weights_array = weights[iLayer]#contrary to previous initialization method, here there is no need to include information about input layer as number of weights is already known (although it doesn't mean thath it is correct, so it is verified in next steps). 
                    if(iLayer>0):#for all, but first layer, it needs to be checked if given weight arrays allow the input propagation through network. If not, then proper error should be thrown
                        #it is verified if number of neurons for previous layer matches number of weights for current layer
                        if(weights_array.shape[1] == (weights_layers[-1].shape[0] + 1)):#previous layer would always be last element in list. Number of rows shape[0] corresponds to number of neurons (+1 is added as accounting for bias term) and columns shape[1] corresponds to number of weights. There is no need to check empty scenario specifically here, as with first layer case, because there is no possibility at this point of zero weight shape in previous layer (first layer would throw error and possibility of proceeding (becoming previous layer) with zero weight shape is discarded here with shape comparision(can not be true if there is no possibility of 0 in previous layer)).
                            #if everything is correct, then array is added as next element of the list
                            weights_layers.append(weights_array)
                        else:#if there is missmatch which makes input propagation impossible proper error should be thrown
                            raise NotSupportedInputGiven("weights initialization",f"There is missmatch between number of neurons in layer {iLayer -1} and number of weights in layer {iLayer}")
                    elif(iLayer == 0):#the only plausible leftover case is when there is only one layer, so there is no need for verification
                        #to discard empty weight array scenario (zero weigth shape) for whole weight list, it needs to be checked if such situation does not occur in the first layer (more explanation in case above).
                        if((weights_array.shape[0] > 0) | (weights_array.shape[1] > 0)):
                            #if everything is correct, then array is added as next element of the list (in this case it is always first element)
                            weights_layers.append(weights_array)
                        else:
                            raise NotSupportedInputGiven("weights initialization","Weight array of first layer is empty.")
                    else:#there should be no option of this error occuring if program runs correctly, but if some problem connected with running (not code itself) occurs, then some unexpected outcome might occur. For this reason error with proper information and placing should be thrown.
                        raise NotSupportedInputGiven("weights initialization","Unexpected error. Some corruption occured.")
                #after all verification (if there was something wrong with the code the error would be raised) given list of weight array can be assigned as object property.
                self.weights_list = weights_layers
            else:
                raise NotSupportedInputGiven("weights initialization","Not all elements in list are np arrays, so they can not represent weight arrays of layer.")
        else:#input that is not compatible was given. Operation cannot proceed, so proper error should be thrown.
            raise NotSupportedInputGiven("weights initialization","Not supported data type given.")
        #activation function assignment 
        if (callable(activ_functions)):#if only single activation function is given, then it needs to be converted into single element list for code unification.
            #given value creates one element list with itself
            activ_functions = [activ_functions]
        if(type(activ_functions) == list):#activation function list initialization always starts from list. 
            #if there are only 2 elements some special case initialization can occur. So, such case must be verified and proceed properly.
            if(len(activ_functions) == 2):
                #if all elements of the list are lists (assumed to be lists of activation layers for 2 layer network), then standard initialization should be done.
                if(all(isinstance(activ_function_list, list) for activ_function_list in activ_functions)):
                    #setting proper value for flag variable
                    standardImplement = True
                elif(all(callable(activ_function) for activ_function in activ_functions)):#if two elements are functions (assumed to be activation functions), then special initialization occurs, where first activation function is used for all hidden layers (and neurons in them), but output layer, which uses second activation function
                    #setting proper value for flag variable (it is only done for principle as false values for this var are never used/compared)
                    standardImplement = False
                    #list to hold all lists of layer activation function is declared to hold initialization results for each layer and then be passed as whole.
                    activ_functions_list_list = []
                    #initialization goes through all layers, but last(output layer)
                    for iLayer in range(len(self.weights_list) - 1):
                        activ_functions_base = [activ_functions[0]] * self.weights_list[iLayer].shape[0]
                        activ_functions_list_list.append(activ_functions_base)
                    #activation function is assigned for last layer.
                    activ_functions_base = [activ_functions[1]] * self.weights_list[iLayer].shape[0]
                    activ_functions_list_list.append(activ_functions_base)
                    #ready activ function list is passed to instance atrribute
                    self.activ_functions_list_list = activ_functions_list_list
                else:#if none of the above occurs, then proper error should be thrown as incompatible input was given
                    NotSupportedInputGiven("activation functions initialization","Not supported data type given in list.")
            else:
                #setting proper value for flag variable
                standardImplement = True
            #if special case initialization did not occur than standard (assumed to be the most commonly used) would proceed
            if(standardImplement == True):
                #based on input varaible properties initialization would proceed differently
                if((len(activ_functions) == 1 )&(callable(activ_functions[0]))):#if given list has only one element and it is function, than it is assumed that the same activation function should be used by all neruons in the network
                    #list to hold all lists of layer activation function is declared to hold initialization results for each layer and then be passed as whole.
                    activ_functions_list_list = []
                    #initialization goes through all layers
                    for iLayer in range(len(self.weights_list)):
                        activ_functions_base = activ_functions * self.weights_list[iLayer].shape[0]
                        activ_functions_list_list.append(activ_functions_base)
                    #ready activ function list is passed to instance atrribute
                    self.activ_functions_list_list = activ_functions_list_list
                elif((len(activ_functions)) == (len(self.weights_list))):#it first needs to be verified if there is enough information for each layer
                    #based on informationt ype in given list, initialization proceeds differently, so it is verified how exactly it should proceed.
                    if(all(isinstance(activ_function_list, list) for activ_function_list in activ_functions)):#if all elements of the given list are lists (to be verified if lists of functions at later step) than initialization by assignment can proceed
                        #list to hold all lists of layer activation function is declared to hold initialization results for each layer and then be passed as whole.
                        activ_functions_list_list = []
                        #assignment goes through all layers
                        for iLayer in range(len(activ_functions)):
                            #gettign current activation function list into variable for code readability
                            activ_functions_list = activ_functions[iLayer]
                            #it needs to be verified if all elements of activation function list are activation functions and if number of activation function matches number of neurons in layer 
                            if((self.weights_list[iLayer].shape[0] == len(activ_functions_list)) & (all(callable(activ_function) for activ_function in activ_functions[iLayer]))):
                                #if everything is as it should be, then assignment proceeds
                                activ_functions_list_list.append(activ_functions_list)
                            else:#if given input does not fit criteria, then proper error should be thrown.
                                raise NotSupportedInputGiven("activation functions initialization","There is missmatch between number of neurons and activation functions in layer {iLayer} or one(or more) given activation function is not a function")
                        self.activ_functions_list_list = activ_functions_list_list
                    elif(all(callable(activ_function) for activ_function in activ_functions)):#if all elements of list are functions (assumed to be activation functions) than assigment by initialization (same activ function for all neurons in layer) proceeds
                        #list to hold all lists of layer activation function is declared to hold initialization results for each layer and then be passed as whole.
                        activ_functions_list_list = []
                        #initialization goes through all layers
                        for iLayer in range(len(self.weights_list)):
                            activ_functions_base = [activ_functions[iLayer]] * self.weights_list[iLayer].shape[0]
                            activ_functions_list_list.append(activ_functions_base)
                        #ready activ function list is passed to instance atrribute
                        self.activ_functions_list_list = activ_functions_list_list
                    else:
                        raise NotSupportedInputGiven("activation functions initialization","Not supported data type given in list.")
                else:
                    raise NotSupportedInputGiven("activation functions initialization","Not enough or too much information was given to proceed with initialization")
        else:
            raise NotSupportedInputGiven("activation functions initialization","Not supported data type given.")
#methods
    #simple input processing by network
    def forward(self,input):
        #getting instance attributes to separate variables for readability
        weights_list = self.weights_list
        activ_functions_list_list = self.activ_functions_list_list
        #conversion of given input if it is list instead of np array (not always possible as numbers are needed)
        if(type(input) == list):
            try:
                input = np.asanyarray(input, dtype = weights_list[0].dtype)#input should be in same format as weights to stay consistent with data types in calculations.
            except Exception as error_caught:
                if(isinstance(error_caught,ValueError)):
                    raise NotSupportedInputGiven("input propagation","Values given in list are not numbers and thus can not be used as input to neuron.")
                else:
                    raise error_caught
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
                else:#if data is of dimensions that handling is not implement proper error should be thrown. 
                    raise NotSupportedArrayDimGiven("1,2")
            else:#if given data is not proper, the error should be thrown.
                raise NotSupportedInputGiven("input propagation","Given values are not a rational numbers and thus can not be used to get output from the network.")
        else:#if given data is not in proper format, the error should be thrown. TO DO: proper error.
            raise NotSupportedInputGiven("input propagation","Not supported data type given.")
        #matix multiplications results and activation results needs to be saved for output. For this purpose variables with lists are created
        z = [input_format]#in this implementation a first value of z(matrix multi results) would be input values (can be interpreted as representation of input layer, which is not represented by any network attribute in this implementation) as for input layer it is input.
        a = [input_format]#in this implementation a first value of a(activation results) would be input values (can be interpreted as representation of input layer, which is not represented by any network attribute in this implementation) as for input layer it is identity value.
        #input needs to propagate through all layers in network and it is achieved by looping through them.
        for iLayer in range(len(weights_list)):
            #variables necessery for current layer propagations are assigned to local variables for readability
            weights_array = weights_list[iLayer]
            activ_functions_list = activ_functions_list_list[iLayer]
            input = a[-1]
            #bias input is added to input vector/array to account for bias, which is 0 weight
            input_ready = addBiasInput(input)
            #it needs to be verified if matrix multiplication can be performed. Theoretically it should be strictly necessery only for first input as rest should be accounted by architecture, but it is present for all if some undetected mistake was made in architecture creation.
            if(weights_array.shape[1] == input_ready.shape[0]):
                #input is multiplied by weights for forward pass (matrix multiplication)
                matrix_multi = weights_array @ input_ready
                #the results of input "passing through" weights needs to be put through activation functions
                activation_out = activationLayer(matrix_multi,activ_functions_list)
                #calculated values are passed to lists storing them
                z.append(matrix_multi)
                a.append(activation_out)
            else:#if inproper input was given and operation cannot proceed proper error should be raised.
                raise NotSupportedInputGiven("input propagation","Input does not match network, i.e. matrix multiplication cannot be done due to mismatch of dimensions.")
        #results are joined together into list for proper output
        output = [z,a]
        #results are returned
        return output