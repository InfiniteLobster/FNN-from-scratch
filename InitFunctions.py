import numpy as np
import random


def randomIniVector(vector,lower_bound = 0.0,upper_bound = 1.0, datatype_weights = "float64"):
    #
    vector_init = vector.astype(dtype = datatype_weights)
    #
    len_vector = vector.size
    #
    for index in range(len_vector):
        #
        value_ran = random.uniform(lower_bound,upper_bound)
        #
        vector_init[0,index] = value_ran
    return vector_init