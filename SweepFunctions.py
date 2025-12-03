#general libraries
import pandas as pd
import numpy as np
import copy
import wandb
import keras
#project code
from FNN import FNN
from gradient_descent import *
from ActivFunctions import  *
from LossFunctions import *
from SuppFunctions import *
from TestingFunctions import *


#
def getLossDer(name):
    #
    match name:
        case "MeanSquaredError":
            loss_derivative = MeanSquaredErrorDerivative
        case "CrossEntropy":
            loss_derivative = CrossEntropyDerivative
        case "SoftmaxCrossEntropy":
            loss_derivative = SoftmaxCrossEntropyDerivative
    #
    return loss_derivative
#
def getDataset(name):
    #checking which dataset is used
    match name:
        case "MNIST":        
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        case "CIFAR":
            (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    #preparing dataset for training and testing
    x_train_flattened = x_train.reshape(x_train.shape[0],-1)
    x_test_flattened = x_test.reshape(x_test.shape[0],-1)
    y_train_flattened = y_train.reshape(-1)
    y_test_flattened = y_test.reshape(-1)
    #one_hot_encoding targets(y)
    dataTarget_train = one_hot_encode(y_train_flattened.T,10)
    dataTarget_test = one_hot_encode(y_test_flattened.T,10)
    #normalization(plus transpose to allign dataset to network architecture)
    dataInput_train = (x_train_flattened.T/255.0)
    dataInput_test =  (x_test_flattened.T/255.0)
    #returning dataset to train
    return dataTarget_train,dataTarget_test,dataInput_train,dataInput_test
#
def getActivFunct(name): 
    #
    match name:
        case "identity":
            activ_function = identity
        case "sigmoid":
            activ_function = sigmoid
        case "tanh":
            activ_function = tanh
        case "relu":
            activ_function = relu
        case "leaky_relu":
            activ_function = leaky_relu
        case "softmax":
            activ_function = softmax
    #
    return activ_function
#
def getLossFunct(name): 
    #
    match name:
        case "MeanSquaredError":
            loss_function = MeanSquaredError
        case "CrossEntropy":
            loss_function = CrossEntropy
        case "SoftmaxCrossEntropy":
            loss_function = CrossEntropy
    #
    return loss_function





#
def train_one_epoch(net, dataInput_train, dataTarget_train, cfg):
    #getting training parameters from the config variable
    cfg_optimizer = cfg.optimizer
    cfg_lr = cfg.learning_rate
    cfg_batch_size = cfg.batch_size
    cfg_loss_function = cfg.loss_function
    cfg_l_method = cfg.l_method
    cfg_l_coeff = cfg.l_coeff
    cfg_grad_clip = cfg.grad_clip
    cfg_momentum = cfg.momentum
    cfg_beta = cfg.beta
    cfg_beta1 = cfg.beta1
    cfg_beta2 = cfg.beta2
    #getting derivative of the loss function
    loss_derivative = getLossDer(cfg_loss_function)
    #selecting regularization method and regularization coefficient value
    match cfg_l_method:
        case "none":
            l1_coeff_in = 0.0
            l2_coeff_in = 0.0
        case "l1":
            l1_coeff_in = cfg_l_coeff
            l2_coeff_in = 0.0
        case "l2":
            l1_coeff_in = 0.0
            l2_coeff_in = cfg_l_coeff
    #training model for one epoch based on optimizer selected
    match cfg_optimizer:
        case "sgd":
            net_out = train_minibatch_sgd(net, dataInput_train, dataTarget_train, 1, cfg_lr, cfg_batch_size, loss_derivative,l1_coeff=l1_coeff_in,l2_coeff=l2_coeff_in, grad_clip = cfg_grad_clip)
        case "sgd_momentum":
            net_out = train_minibatch_sgd_momentum(net, dataInput_train, dataTarget_train, 1, cfg_lr, cfg_batch_size, loss_derivative, momentum = cfg_momentum,l1_coeff=l1_coeff_in,l2_coeff=l2_coeff_in, grad_clip = cfg_grad_clip)
        case "rmsprop":
            net_out = train_minibatch_rmsprop(net, dataInput_train, dataTarget_train, 1, cfg_lr, cfg_batch_size, loss_derivative, beta=cfg_beta, l1_coeff=l1_coeff_in, l2_coeff=l2_coeff_in, grad_clip = cfg_grad_clip)
        case "nag":
            net_out = train_minibatch_nag(net, dataInput_train, dataTarget_train, 1, cfg_lr, cfg_batch_size, loss_derivative, momentum = cfg_momentum, l1_coeff=l1_coeff_in, l2_coeff=l2_coeff_in, grad_clip = cfg_grad_clip)
        case "adam":
            net_out = train_minibatch_adam(net, dataInput_train, dataTarget_train, 1, cfg_lr, cfg_batch_size, loss_derivative, beta1 = cfg_beta1, beta2 = cfg_beta2, l1_coeff=l1_coeff_in, l2_coeff=l2_coeff_in, grad_clip = cfg_grad_clip)
    #returning trained network
    return net_out
#
#
def main(args=None):
    #
    project = args.project if args else None
    #
    with wandb.init(project=project) as run:
        #getting configuration
        cfg = run.config
        #getting run parameters from configuration
        cfg_dataset = cfg.dataset
        cfg_epochs = cfg.epochs
        cfg_num_hidden_layers = cfg.num_hidden_layers
        cfg_num_hidden_units = cfg.num_hidden_units
        cfg_activation_hidden = cfg.activation_hidden
        cfg_activation_output = cfg.activation_output
        cfg_loss_function = cfg.loss_function
        cfg_weights_init = cfg.weights_init
        #
        dataTarget_train,dataTarget_test,dataInput_train,dataInput_test = getDataset(cfg_dataset)
        #
        num_input = dataInput_train.shape[0]
        num_output = dataTarget_train.shape[0]
        #
        hidden_layers = [cfg_num_hidden_units] * cfg_num_hidden_layers
        #
        arch_net = hidden_layers
        arch_net.insert(0,num_input)
        arch_net.append(num_output)
        #
        activation_hidden = getActivFunct(cfg_activation_hidden)
        activation_output = getActivFunct(cfg_activation_output)
        #declaring network
        net = FNN(arch_net,
                            [activation_hidden,activation_output],
                            method_ini = cfg_weights_init
                            )
        #
        loss_function = getLossFunct(cfg_loss_function)
        #decoding test set(needed for network accuracy)
        dataTarget_test_decode = one_hot_decode(dataTarget_test)
        dataTarget_train_decode = one_hot_decode(dataTarget_train)
        # Execute the training loop and log the performance values to W&B
        for epoch in np.arange(1, (cfg_epochs+1)):
            #
            net = train_one_epoch(net, dataInput_train, dataTarget_train, cfg)
            #
            prediction_train = net.predictClassMulti(dataInput_train)
            prediction_val = net.predictClassMulti(dataInput_test)
            #
            loss_train = loss_function(dataTarget_train,prediction_train)
            accuracy_train = getAccuracy(dataTarget_train_decode,prediction_train)
            #
            loss_val = loss_function(dataTarget_test,prediction_val)
            accuracy_val = getAccuracy(dataTarget_test_decode,prediction_val)
            #
            run.log(
                {
                    "epoch": epoch,
                    "train_acc": accuracy_train,
                    "train_loss": loss_train,
                    "val_acc": accuracy_val, 
                    "val_loss": loss_val,
                }
            )