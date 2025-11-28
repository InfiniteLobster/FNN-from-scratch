import numpy as np
from ActivFunctions import *
from ErrorClasses import *
#this is .py file with supporting functions 


#this function checks if np array is of desired data type and if not changes it into desired data type
def ensureDtypeNpArray(array_in,data_type):
    #verification of dtype
    if(array_in.dtype == data_type):#if data type is as desired, then input array does not change
        array_out = array_in
    else:#if not the same, then it is different and needs to be changed
        array_out = np.asarray(array_in,data_type)
    #output out
    return array_out
#this function evaluates if data type of given array is composed of rational numbers
def isRationalNumber(array):
    #verification of data type is rational
    answer = ((np.issubdtype(array.dtype, np.integer)) | (np.issubdtype(array.dtype, np.floating)))
    #output out
    return answer
#this function verifies if given 2D array is row vector
def isRowVector(array):
    #getting shape of array
    shape = array.shape
    #checking if requirments for row vector are met, i.e only one row and at least one column (second argument is more of verification if given array was 2D)
    answer = (shape[0] == 1 )& (shape[1] >= 1)
    #returning results
    return answer
#this function verifies if given 2D array is column vector
def isColVector(array):
    #getting shape of array
    shape = array.shape
    #checking if requirments for column vector are met, i.e only one column and at least one row (first argument is more of verification if given array was 2D)
    answer = (shape[0] >= 1) & (shape[1] == 1)
    #returning results
    return answer
#this function verifies if given 2D array is array (basically it is checke dif it is empty in any of dimensions)
def isArray(array):
    #getting shape of array
    shape = array.shape
    #checking if requirments for array are met, i.e. having informations in 2 dimensions
    answer = (shape[0] >= 1) & (shape[1] >= 1)
    return answer
#this function ensures that given input would be suitable for input propagation through ANN
def getProperInputArray(array):
    #for proper representation the input, the input vector needs to be a column vector or array. It is checked if that's the case and changed if not.
    if(isRowVector(array)):#if it is vector input, then it has to be in column format. So, if it is in row format than it has to be transposed.
        input_format = array.T
    elif(isColVector(array)):
        input_format = array
    elif(isArray(array)):
        input_format = array
    else:#if none of above true, then given variable is in non implemented format. Appropriate error must be passed to user. 
        raise NotSupportedInputGiven("input propagation","Given input is in dimensions for which input propagation is not implemented")
    #input is returned in proper format
    return input_format
#this function adds input values to represent bias (1s) into given input
def addBiasInput(input):
    #getting shape of given input to know how big holding variable should be initialized
    shapeInput = input.shape
    #initializing holding variable for assignment of inputs
    newInput = np.empty([(shapeInput[0] + 1),shapeInput[1]],dtype = input.dtype)#as single input is assumed to be in column format, than additional row has to be added for bias input
    #assigning values to proper places, i.e. bias inputs as first elements as biases are [0] elements in neuron weights
    newInput[0,:] = 1.0
    newInput[1:,:] = input
    #returning input with added bias input
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