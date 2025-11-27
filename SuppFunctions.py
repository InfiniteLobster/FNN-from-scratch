import numpy as np
from ActivFunctions import (
    identity, der_identity,
    sigmoid, der_sigmoid,
    tanh, der_tanh,
    relu, der_relu,
    leaky_relu, der_leaky_relu,
    softmax, der_softmax
)

der_map = {
    identity: der_identity,
    sigmoid: der_sigmoid,
    tanh: der_tanh,
    relu: der_relu,
    leaky_relu: der_leaky_relu,
    softmax: der_softmax
}

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
    answer = ((np.issubdtype(array.dtype, np.integer)) | (np.issubdtype(array.dtype, np.floating)))
    return answer
#
def isRowVector(array):
    shape = array.shape
    answer = (shape[0] == 1 )& (shape[1] >= 1)
    return answer
#
def isColVector(array):
    shape = array.shape
    answer = (shape[0] >= 1) & (shape[1] == 1)
    return answer
#
def isArray(array):
    shape = array.shape
    answer = (shape[0] >= 1) & (shape[1] >= 1)
    return answer
#
def getProperInputArray(array):
    #for proper representation the input, the input vector needs to be a column vector or array. It is checked if that's the case and changed if not.
    if(isRowVector(array)):
        input_format = array.T
    elif(isColVector(array)):
        input_format = array
    elif(isArray(array)):
        input_format = array
    else:#if none of above true, then given variable is in non implemented format. Appropriate error must be passed to user. TO DO: error implementation
        raise NotImplementedError
    #input is returned in proper format
    return input_format
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
#
def activationLayer(input,activ_functions):
    #
    shapeInput = input.shape
    #
    output = np.empty(shapeInput)
    #iterating through inputs
    for iInput in range(shapeInput[1]):
        #getting current input
        input_current = input[:,iInput]
        #for all neurons in input
        for iNeuron in range(input_current.shape[0]):
            #
            activ_function_current = activ_functions[iNeuron]
            #
            output_current = activ_function_current(input_current[iNeuron])
            #
            output[iNeuron,iInput] = output_current
    return output
#
der_map = {
    identity: der_identity,
    sigmoid: der_sigmoid,
    tanh: der_tanh,
    relu: der_relu,
    leaky_relu: der_leaky_relu,
    softmax: der_softmax
}
def getDer(func):
    if func in der_map:
        return der_map[func]
    raise NotImplementedError(f"Derivative not implemented for {func.__name__}")
#
def clip_gradient(grad, threshold=1.0):
    norm = np.linalg.norm(grad)
    if norm > threshold:
        grad = grad * (threshold / norm)
    return grad
#
def testInputFormat(base_list,network):
    #inputs
    inputTestList = base_list
    inputTestVect1D = np.array(base_list)
    inputTestVect2Drow = inputTestVect1D[np.newaxis,:] 
    inputTestVect2Dcol = inputTestVect2Drow.T
    inputTestArray = np.concatenate((inputTestVect2Dcol,inputTestVect2Dcol), axis = 1) 
    #forward pass
    o1 = network.forward(inputTestList)
    o2 = network.forward(inputTestVect1D)
    o3 = network.forward(inputTestVect2Drow)
    o4 = network.forward(inputTestVect2Dcol)
    o5 = network.forward(inputTestArray)
    if(type(o1[0]) == list):#test network
        #testlayer/neuron
        if((o1[-1][1].all() == o2[-1][1].all()) & (o1[-1][1].all() == o3[-1][1].all()) & (o1[-1][1].all() == o4[-1][1].all()) & (o1[-1][1].all() == o5[-1][1].all())):
            return True
        else:
            return False
    else:
        #testlayer/neuron
        if((o1[1].all() == o2[1].all()) & (o1[1].all() == o3[1].all()) & (o1[1].all() == o4[1].all()) & (o1[1].all() == o5[1].all())):
            return True
        else:
            return False