import numpy as np
from InitFunctions import  *
from SuppFunctions import  *

class Neuron:
    

    def __init__(self,weights,activ_function, method_ini = "Zero", datatype_weights = "float64", random_lower_bound = 0.0, random_upper_bound = 1.0):
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
            #given variable dimensions are checked to identify if it can be passed as object weight vector
            number_dimensions = weights.ndim
            if(number_dimensions == 1):
                #all weight values needs to be numbers and cannot be complex (at least in our implementation)
                if (np.issubdtype(weights.dtype, np.integer) & np.issubdtype(weights.dtype, np.floating)):# the check of dtypes is shortened by "cutting" the data type tree (from https://numpy.org/doc/stable/reference/arrays.scalars.html), so that only it is compare to highest appropariate (leaving only desired data types) node.
                    #object weights must be of default dtype or as specified by user. It is checked if that is the case, if not then weights dtype is converted
                    weights_dtype = insureDtypeNpArray(weights, datatype_weights)
                    #to represent neuron weight vector as one row of layer weight array, the weight vector is a 2D array with only one row(row vector).
                    weights_vector_row = weights_dtype[np.newaxis,:]
                    #transformation are finished and results can be assigned as object property
                    self.weight_vector = weights_vector_row
                else:#if given weight vector has undesired dtype, then appropriate error should be given. TO DO: implementing proper error
                    raise NotImplementedError
            elif(number_dimensions == 2):
                if (np.issubdtype(weights.dtype, np.integer) & np.issubdtype(weights.dtype, np.floating)):# the check of dtypes is shortened by "cutting" the data type tree (from https://numpy.org/doc/stable/reference/arrays.scalars.html), so that only it is compare to highest appropariate (leaving only desired data types) node.
                    #object weights must be of default dtype or as specified by user. It is checked if that is the case, if not then weights dtype is converted
                    weights_dtype = insureDtypeNpArray(weights, datatype_weights)
                    #TO DO: implement checkign if given array is row vector (if not then convert)
                    #transformation are finished and results can be assigned as object property
                    self.weight_vector = weights_vector_row
                else:#if given weight vector has undesired dtype, then appropriate error should be given. TO DO: implementing proper error
                    raise NotImplementedError
            else:#if given NumPy array is not vector than it is unappropriate for use and appropriate error should be thrown. TO DO: implementing proper error
                raise NotImplementedError
        #activ function assignment
        if (callable(activ_function)):#types.FunctionType is used as function is not default data type in python
            self.activ_function = activ_function
        else:#if given variable is not an activation function, then class object can not be initialized due to lack (no activ function) or too much (e.g. vector of activ functions) of information. TO DO: implementing proper error
            raise NotImplementedError

  