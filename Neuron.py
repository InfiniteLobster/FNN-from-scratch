#
import numpy as np
from InitFunctions import  *
from SuppFunctions import  *
#
class Neuron:
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
        if (type_weights == int):#this is case when weights values are not given by the user, only dimensions. In this case weights needs to be initialized
            #basic version (zero initialization) is always done as a base to ommit need to pass datatype information down to initialization methods
            weights_conversion = np.zeros([1,weights],dtype=datatype_weights)#to represent neuron weight vector as one row of layer weight array, the weight vector is a 2D array with only one row(row vector).
            #weights are initialized based on selected method
            match method_ini:#(TO DO: implementing methods)
                case "Random":# 
                    weights_conversion = randomIniVector(weights_conversion,random_lower_bound,random_upper_bound)
            #after initialization weight are assigned to object property
            self.weight_vector = weights_conversion
        elif (type_weights == np.ndarray):
            #not all data types can represent or can be converted to represent weights. They need to be rational number (at least in this implementation).
            if (IsRationalNumber(weights.dtype)):
                #object weights must be of default dtype or as specified by user. It is checked if that is the case, if not then weights dtype is converted
                weights_dtype = ensureDtypeNpArray(weights, datatype_weights)
                #given variable dimensions are checked to identify if it can be passed as object weight vector and by whihc method.
                number_dimensions = weights_dtype.ndim
                if(number_dimensions == 1):
                    #to represent neuron weight vector as one row of layer weight array, the weight vector is a 2D array with only one row(row vector).
                    weights_vector_row = weights_dtype[np.newaxis,:]
                    #transformation are finished and results can be assigned as object property
                    self.weight_vector = weights_vector_row
                elif(number_dimensions == 2):
                    #for proper representation the weight_vector of neron needs to be row vector. It is checked if that's the case and changed if not.
                    if(isRowVector(weights_dtype)):
                        self.weight_vector = weights_vector_row
                    elif(isColVector(weights_dtype)):
                        self.weight_vector = weights_vector_row.T
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
        #
        weight_vector = self.weight_vector
        #
        if(type(input) == np.ndarray):
            #
            #
            if(IsRationalNumber(input)):
                number_dimensions = input.ndim
                if(number_dimensions == 1):
                    raise NotImplementedError 
                elif(number_dimensions == 2):
                    raise NotImplementedError 
                else:
                    raise NotImplementedError 
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return output
  