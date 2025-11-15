import numpy as np
import random

#
def randomIni(array,lower_bound = 0.0,upper_bound = 1.0):
    #
    array_init = array
    #
    shape_array = array_init.shape
    #
    for iRow in range(shape_array[0]):
        for iCol in range(shape_array[1]):
            #
            value_ran = random.uniform(lower_bound,upper_bound)
            #
            array_init[iRow,iCol] = value_ran
    #
    return array_init