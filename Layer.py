import numpy as np


class Layer:
    


    def __init__(self,weights,activ_functions,method_ini = "Zero", datatype_weights = "float64"):
        #
        type_weights = type(weights)
        type_activ = type(activ_functions)
        #
        if (type_weights == int):#this is case when only weights values are not given by the user, only dimensions. In this case weights needs to be initialized
            #weights are initialized based on selected method
            match method_ini:#this is default version, not very usefull (TO DO: implementing other methods)
                case "Zero":
                    self.weights = np.zeros([weights,1],dtype=datatype_weights)
                case "Random":# TO DO
                    k = 1
        elif (type_weights == np.ndarray):
            self.k = 1


        self.WeightArray = 1