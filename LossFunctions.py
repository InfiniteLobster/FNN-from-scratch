import numpy as np
#this is .py file with loss functions and their derivatives


# ================================================
# Mean Squared Error (MSE)
# ================================================
def MeanSquaredError(targets, predictions):
    #getting the difference between ground truth and predictions
    diff = predictions - targets
    #squaring the difference
    sqr = diff**2
    #getting mean of squared values
    mean = np.mean(sqr)
    #putting results into output variable
    output = mean
    #retunring results
    return output

def MeanSquaredErrorDerivative(targets, predictions):
    #getting number of examples for proper division
    num_examples = predictions.shape[1]
    #getting the difference between ground truth and predictions
    diff = predictions - targets
    #multiplying by 2 
    mult = 2 * diff
    #dividing results with number of examples (averaging)
    div = mult/num_examples
    #putting results into output variable
    output = div
    #retunring results
    return output
# ================================================
# Binary Cross-Entropy (for sigmoid output)
# ================================================
def BinaryCrossEntropy(targets, predictions, eps=1e-12):
    #predictions are clipped for numerical stability
    predictions = np.clip(predictions, eps, 1 - eps)
    #calculating components separatly for code clarity
    first_component = targets * np.log(predictions)
    second_component = (1 - targets) * np.log(1 - predictions)
    #adding components and averaging them 
    avg = np.mean(first_component + second_component)
    #assigning avg results to variable (plus negating it)
    output = -avg            
    #returning results
    return output

def BinaryCrossEntropyDerivative(targets, predictions, eps=1e-12):
    #predictions are clipped for numerical stability
    predictions = np.clip(predictions, eps, 1 - eps) 
    #calculating components separatly for code clarity
    first_component = (predictions - targets)
    second_component = (predictions * (1 - predictions))
    #dividing components to reach results
    output = first_component / second_component
    #returning results
    return output
#Cross-Entropy
def CrossEntropy(targets, predictions):
    #
    #prob = predictions[np.arrange(len(targets)),targets]
    #
    loss = -np.sum(targets * np.log(predictions + 1e-15))#1e-15 is added for numerical stability ->predictions can become zero, which lead to divide by zero error
    #
    return loss
def CrossEntropyDerivative(targets, predictions):
    #
    #grad = np.zeros_like(predictions)
    #grad[targets,np.arrange(predictions.shape[1])] = -1 / predictions[targets,np.arrange(predictions.shape[1])]
    #
    output = -targets / (predictions +1e-15)
    #
    return output
# ================================================
# Softmax + Cross-Entropy (for multi-class)
# ================================================
#this is only used in backpropagation and for this purpose only derivative is needed
def SoftmaxCrossEntropyDerivative(target_one_hot, softmax_vals):
    return (softmax_vals - target_one_hot)
