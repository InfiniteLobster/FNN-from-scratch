#importing libraries needed for this file operation
import numpy as np
import matplotlib.pyplot as plt
#this is .py file with model testing functions 


#this function calculates confusion matrix elements (true positive, true negative, false positive, false negative) based on predictions and ground truths for binary classification 
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
#this function plots confusion matrix for binary classification
def plotConfMatBin(confComp, class_names=None, title="Confusion Matrix"):
    #changing confusion matrix components format into array, so plot can be colored to indicate intensity
    confComp_array = np.array([[confComp[0],confComp[3]],[confComp[2],confComp[1]]])
    #creating figure to 'hold' plot
    plt.figure(figsize=(7, 6))
    #adding colors based on intensity
    plt.imshow(confComp_array, interpolation="nearest")
    #creating 'ticks' in plot for each case in confusion matrix, i.e. true_positive, true_negative, false_positive, false_negative
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, [class_names[1],class_names[0]])#predicted labels
    plt.yticks(tick_marks, [class_names[1],class_names[0]])#true labels
    #setting values at proper places in plot
    plt.text(0,0,str(confComp[0]),fontsize=48,ha="center", va="center")#true_positive
    plt.text(0,1,str(confComp[2]),fontsize=48,ha="center", va="center")#false_positive
    plt.text(1,0,str(confComp[3]),fontsize=48,ha="center", va="center")#false_negative
    plt.text(1,1,str(confComp[1]),fontsize=48,ha="center", va="center")#true_negative
    #adding descriptive features
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.colorbar()
    #showing the results of plotting (returning confusion matrix)
    plt.show()
#this function calculates accuracy of model results (based on confusion matrix results) for binary classification
def getAccuracyBin(true_positive,true_negative,false_positive,false_negative):
    #calculating accuracy (accuracy is ratio of true results out of all results)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative) 
    #returning results
    return accuracy
#this function calculates precision of model results (based on confusion matrix results) for binary classification
def getPrecisionBin(true_positive,false_positive):
    #calculating precision (precision is ratio of true positive predictions out of all positive predicitions)
    precision = (true_positive) / (true_positive + false_positive) 
    #returning results
    return precision
##this function calculates recall of model results (based on confusion matrix results) for binary classification
def getRecallBin(true_positive,false_negative):
    #calculating recal (precision is ratio of true positives out of prediciton that are positve in ground truth)
    recall = (true_positive) / (true_positive + false_negative) 
    #returning results
    return recall
#this function gets accuracy of multi class model
def getAccuracy(ground_truth,predictions):
    #number of examples to process (and divide by) is acertained
    total = predictions.size
    #setting variable for counting
    accurate = 0
    #iterating through examples
    for iExample in range(total):
        #getting pair of values (prediction and ground truth) for current examples
        predictions_this = predictions[iExample]
        ground_truth_this = ground_truth[iExample]
        #evaluating if correct class was predicted
        if((predictions_this == ground_truth_this)):
            #if correct class was predicted, than counter for accurate predictions is increased.
            accurate = accurate + 1
    #getting final results (ratio of correct predictions to all examples)
    accuracy = accurate/total
    #returning results
    return accuracy
#this function calculates confusion matrix elements for multi-class classification 
def getConfMatCompMulti(ground_truth,predictions):
    #getting number of classes to know dimesnions of confusion matrix to create
    num_classes = predictions.shape[0]
    #creating confusion matrix variable for pre-allocation
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    #iterating through results to add numbers at proper places (to compare prediction and true value)
    for true_axis, predi_axis in zip(ground_truth, predictions):
        #adding value at place corresponding to situation (on y axis(rows) true labels are located, on x axis(columns) predicted labels are located)
        confusion_matrix[true_axis, predi_axis] += 1
    #returning confusion matrix (np array)
    return confusion_matrix
#this function plots confusion matrix for multi-class classification
def plot_confusion_matrix(confusion_matrix, class_names=None, title="Confusion Matrix"):
    #getting shape of confusion matrix(based on numbers of classes) to know how plot should be created
    num_classes = confusion_matrix.shape[0]
    #in case when class names are not given, number labels are used (based on index)
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    #creating figure to 'hold' plot
    plt.figure(figsize=(7, 6))
    #adding colors based on intensity
    plt.imshow(confusion_matrix, interpolation="nearest")
    #creating 'ticks' in plot for each case in confusion matrix
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    #getting theshold for visualization purposes
    thresh = confusion_matrix.max() / 2.0
    #adding values of confusion matrix to plots by iterating through them
    for iTrueAx in range(num_classes):
        for jPrediAx in range(num_classes):
            plt.text(
                jPrediAx, iTrueAx, #seting coordinates on the plot
                str(confusion_matrix[iTrueAx, jPrediAx]),#giving value of the tick
                ha="center", va="center",#centering the value
                color="white" if confusion_matrix[iTrueAx, jPrediAx] > thresh else "black"# due to coloring of plot to show intensity, in some cases text may by invisible. By changing its color depneding on threshold improve visibility of values
            )
    #adding descriptive features
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.colorbar()
    #showing the results of plotting (returning confusion matrix)
    plt.show()