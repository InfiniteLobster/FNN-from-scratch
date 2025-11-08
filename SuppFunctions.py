import numpy as np

#
def ensureDtypeNpArray(array_in,data_type):
    #
    if(array_out.dtype == data_type):
        array_out = array_in
    else:
        array_out = array_in.astype(data_type)
    #
    return array_out
#
def IsRationalNumber(array):
    answer = (np.issubdtype(array.dtype, np.integer) & np.issubdtype(array.dtype, np.floating))
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