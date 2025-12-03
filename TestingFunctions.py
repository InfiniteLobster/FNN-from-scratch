import numpy as np

#this is .py file with model testing functions 


#this function calculates confusion matrix elements (true positive, true negative, false positive, false negative) based on predictions and ground truths
def getConfMatCompBin(ground_truth,predictions):
    #calculating elements
    true_positive = np.sum((predictions == 1)&(ground_truth == 1))
    true_negative = np.sum((predictions == 0)&(ground_truth == 0))
    false_positive = np.sum((predictions == 1)&(ground_truth == 0))
    false_negative = np.sum((predictions == 0)&(ground_truth == 1))
    #joinding all components into list so they can be outputted together
    confComp = [true_positive,true_negative,false_positive,false_negative]
    #returning results
    return confComp
#this function calculates accuracy of model results (based on confusion matrix results)
def getAccuracyBin(true_positive,true_negative,false_positive,false_negative):
    if((true_positive.size > 0) | (true_negative.size > 0) | (false_positive.size > 0)| (false_negative.size > 0)):
        #accuracy is ratio of true results out of all results
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative) 
    else:
        #
        accuracy = 0
    #returning results
    return accuracy
#this function calculates precision of model results (based on confusion matrix results)
def getPrecisionBin(true_positive,false_positive):
    if((true_positive.size > 0) | (false_positive.size > 0) ):
        #accuracy is ratio of true results out of all results
        precision = (true_positive) / (true_positive + false_positive) 
    else:
        #
        precision = 0
    #returning results
    return precision
##this function calculates recall of model results (based on confusion matrix results)
def getRecallBin(true_positive,false_negative):
    if((true_positive.size > 0) | (false_negative.size > 0) ):
        #accuracy is ratio of true results out of all results
        recall = (true_positive) / (true_positive + false_negative) 
    else:
        #
        recall = 0
    #returning results
    return recall
#
def getAccuracy(ground_truth,predictions):
    #
    total = predictions.size
    #
    accurate = 0
    #
    for iExample in range(total):
        #
        predictions_this = predictions[iExample]
        ground_truth_this = ground_truth[iExample]
        #
        if((predictions_this == ground_truth_this)):
            accurate = accurate + 1
    #
    accuracy = accurate/total
    #
    return accuracy