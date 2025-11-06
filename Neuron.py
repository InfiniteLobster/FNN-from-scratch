import numpy as np
from InitFunctions import  *

class Neuron:
    

    def __init__(self,weights,activ_function, method_ini = "Zero", datatype_weights = "float64"):
        #
        type_weights = type(weights)
        #weights assignment
        if (type_weights == int):#this is case when only weights values are not given by the user, only dimensions. In this case weights needs to be initialized
            #
            weights_conversion = np.zeros([1,weights],dtype=datatype_weights)
            #weights are initialized based on selected method
            match method_ini:#(TO DO: implementing methods)
                case "Random":# 
                    weights_conversion = randomIniVector(weights_conversion)
            #after initialization
            self.weight_vector = weights_conversion
        elif (type_weights == np.ndarray):
            #given variable dimensions are checked to identify if it can be passed as object weight vector
            number_dimensions = weights.ndim
            if(number_dimensions == 1):
                #all weight values needs to be numbers and cannot be complex (at least in our implementation)
                if (np.issubdtype(weights.dtype, np.integer) & np.issubdtype(weights.dtype, np.floating)):# the check of dtypes is shortened by "cutting" the data type tree (from https://numpy.org/doc/stable/reference/arrays.scalars.html), so that only it is compare to highest appropariate (leaving only desired data types) node.
                    #object weights must be of default dtype or as specified by user. It is checked if that is the case, if not then weights dtype is converted
                    if(weights.dtype == "float64"):
                       self.weight_vector = weights
                    else:
                        self.weight_vector = weights.astype(datatype_weights)
                else:#if given weight vector has undesired dtype, then appropriate error should be given. TO DO: implementing proper error
                    raise NotImplementedError
            else:#if given NumPy array is not vector than it is unappropriate for use and appropriate error should be thrown. TO DO: implementing proper error
                raise NotImplementedError
        #
        if (callable(activ_function)):#types.FunctionType is used as function is not default data type in python
            self.activ_function = activ_function
        else:#if given variable is not an activation function, then class object can not be initialized due to lack (no activ function) or too much (e.g. vector of activ functions) of information. TO DO: implementing proper error
            raise NotImplementedError

  