import numpy as np

#
def ensureDtypeNpArray(array_in,data_type):
    #
    if(array_in.dtype == data_type):
        array_out = array_in
    else:
        array_out = array_in.astype(data_type)
    #
    return array_out
#
def isRationalNumber(array):
    answer = (np.issubdtype(array.dtype, np.integer) | np.issubdtype(array.dtype, np.floating))
    return answer
#
def isRowVector(array):
    shape = array.shape
    answer = shape[0] == 1 & shape[1] >= 1
    return answer
#
def isColVector(array):
    shape = array.shape
    answer = shape[0] >= 1 & shape[1] == 1
    return answer
#
def addBiasInput(input):
    #
    shapeInput = input.shape
    #
    newInput = np.empty([(shapeInput[0] + 1),shapeInput[1]],dtype = input.dtype)
    #
    newInput[0,:] = 1.0
    newInput[1:,:] = input
    #
    return newInput