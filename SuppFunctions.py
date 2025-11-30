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
#this function puts matrix multiplication results (input propagation) through activation functions for a layer. As in this implementation each neuron in layer can have different activation function each neurons activation needs to be done separately
def activationLayer(input,activ_functions_list):
    #to create properly sized output variable (pre-allocation) information of input(output of matrix multiplication) shape is needed
    shapeInput = input.shape
    #output variable is declared for pre-allocation
    output = np.empty(shapeInput)
    for iNeuron in range(input.shape[0]):
        #selecting input for current neuron (row vector of current neuron)
        input_current = input[iNeuron,:]
        #getting activation function of current neuron
        activ_function_current = activ_functions_list[iNeuron]
        #puttign input through activation function
        output_current = activ_function_current(input_current)
        #assignign neuron output to output variable
        output[iNeuron,:] = output_current
    #returning the output
    return output
#
def derLoss(targets,a_output,loss_derivative):
    #
    shapeInput = a_output.shape
    #
    dL_dy_all = np.empty(shapeInput)
    #
    for iInput in range(shapeInput[1]):
        #
        y_col = targets[:,iInput]
        a_col = a_output[:,iInput]
        #
        for iNeuron in range(y_col.shape[0]):
            #
            y = y_col[iNeuron]
            a = a_col[iNeuron]
            #
            dL_dy = loss_derivative(y,a)
            # 
            dL_dy_all[iNeuron,iInput] = dL_dy
    #returning the output
    return dL_dy_all
#
def derLoss_vector(targets,a_output,loss_derivative):
    #calculating loss
    dL_dy = loss_derivative(targets,a_output)
    #returning the output
    return dL_dy
#
def getDelta(der_prev,z,activ_functions_list):

    if(all((activ_function.__code__.co_code == activ_functions_list[0].__code__.co_code) for activ_function in activ_functions_list)):
            #
            if(activ_functions_list[0].__code__.co_code == softmax.__code__.co_code):
                activ_function = softmax_vec
            else:
                activ_function = activ_functions_list[0]
            #
            activ_function_der = getDer(activ_function)
            #
            a_der = activ_function_der(z)
            #
            delta = der_prev * a_der
    else:#
        #
        delta = gradLoss(der_prev,z,activ_functions_list) 
    return delta
#
def gradLoss(derPrev,z_output,activ_functions_list):
    #
    shapeInput = derPrev.shape
    #
    delta = np.empty(shapeInput)
    #
    for iInput in range(shapeInput[1]):
        #
        der_col = derPrev[:,iInput]
        z_col = z_output[:,iInput]
        #
        for iNeuron in range(der_col.shape[0]):
            #
            der = der_col[iNeuron]
            z = z_col[iNeuron]
            #
            activ_function = activ_functions_list[iNeuron]
            activ_function_der = getDer(activ_function)
            #
            a_der = activ_function_der(z)
            #
            delta_neur = der * a_der
            #
            delta[iNeuron,iInput] = delta_neur
    #
    return delta
#
der_map = {
    identity: der_identity,
    sigmoid: der_sigmoid,
    tanh: der_tanh,
    relu: der_relu,
    leaky_relu: der_leaky_relu,
    softmax: der_softmax,
    softmax_vec: der_softmax_vec
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
#this function test given network with given input transformed for different possible input data formats. Essentially output for all input data formats should be the same and this is verified.
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
        if((o1[1].all() == o2[1].all()) & (o1[1].all() == o3[1].all()) & (o1[1].all() == o4[1].all()) & (o1[1].all() == o5[1].all())):#in case of ANN only output of last layer is compared
            return True
        else:
            return False