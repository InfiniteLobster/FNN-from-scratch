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
                    if(weights.size == 3):#as weights of single layer are represented by 2D array, then if you need to represent multiple of them you only need three numbers for dimensions. If there is less or more than ambiguity is created. To omit it, the initialization can proceed only with 3 numbers in vector.
                        nNeurons = weights[0]#first number is assumed to be for layer size (number of neurons in layer)
                        nWeights = weights[1]#second number is assumed to be for number of weights in neurons of layer (bias is not counted)
                        nLayers = weights[2]#thrid number is assumed to be for number of layers (input layer is not imputed as in design it is not represent in weight matrix list)
                        #
                        weights_layers = []
                        #
                        for iLayer in range(nLayers):
                            #weights are initializaed by basic method
                            weights_conversion = np.zeros([nNeurons,(nWeights+1)],dtype=datatype_weights)
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
        output = input

        return output