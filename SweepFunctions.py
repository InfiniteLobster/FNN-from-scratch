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
        case "None":
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
        # 
        cfg = run.config
        #
        cfg_dataset = cfg.dataset
        cfg_epochs = cfg.epochs
        cfg_hidden_layers = cfg.hidden_layers
        #
        dataTarget_train,dataTarget_test,dataInput_train,dataInput_test = getDataset(name)



        # Execute the training loop and log the performance values to W&B
        for epoch in np.arange(1, cfg_epochs):
            train_acc, train_loss = train_one_epoch(epoch, lr, batch_size)
            val_acc, val_loss = evaluate_one_epoch(epoch)
            run.log(
                {
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "val_acc": val_acc, # Metric optimized
                    "val_loss": val_loss,
                }
            )