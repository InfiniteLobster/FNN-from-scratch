import numpy as np


def insureDtypeNpArray(array_in,data_type):
    #
    if(array_out.dtype == data_type):
        array_out = array_in
    else:
        array_out = array_in.astype(data_type)
    #
    return array_out