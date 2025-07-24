import numpy as np
import math
import struct
import h5py
from data_process import *
from activation_backward import *
from activation_funcs import *
from initialization import *

def forward_propagation_acti(X, W, b, activation):
    linear_op = np.dot(W, X) + b
    activated_op = activation_f(linear_op, activation)
    return activated_op, linear_op

def forward_propagation(X, Y, lambd, parameters, activations):
    In = X
    l = len(activations) + 1
    temp = {}
    temp['activated_op0'] = X
    for i in range(1, l):
        previous_In = In
        W = parameters['W' + str(i)]
        b = parameters['b' + str(i)]
        activated_op, linear_op = forward_propagation_acti(previous_In, W, b, activations[i - 1])
        temp['activated_op' + str(i)] = activated_op
        temp['linear_op' + str(i)] = linear_op
    y_hat = activated_op
    cost = compute_cost_L2(y_hat, Y, parameters, lambd)
    return y_hat, cost, temp

def backward_propagation_acti(dAct, liner_op, Act_prev, W, activations):
    N = Act_prev.shape[1]
    dliner = activation_func_backward(dAct, liner_op, activations)
    dW = np.dot(dliner, Act_prev.T) / N
    db = np.sum(dliner, axis = 1, keepdims = True) / N
    dAct_prev = np.dot(W.T, dliner)
    return dAct_prev, dW, db

def backward_propagation(temp, label, parameters, activation, lambd):
    gradients = {}
    l = len(activation)
    dAct_prev, dW, db = softmax_backward(temp['activated_op' + str(l)], label, 
                                         temp['activated_op' + str(l - 1)], parameters['W' + str[l], lambd])
    for i in range(l - 1, 0, -1):
        dAct_prev, dW, db = backward_propagation_acti(dAct_prev, temp['liner_op' + str[i]], temp['activated_op' + str[i - 1]], 
                                                      parameters['W' + str[i], activation[i - 1]])
        gradients['dW' + str[i]] = dW
        gradients['db' + str[i]] = db
    return gradients
    

def softmax_backward(y_hat, label, Act_prev, W, lambd):
    N = Act_prev.shape[1]
    dliner = y_hat - label
    dW = np.dot(dliner, Act_prev.T) / N + (lambd / N) * W
    db = np.sum(dliner, axis = 1, keepdims = True) / N
    dAct_prev = np.dot(W.T, dliner)
    return dAct_prev, dW, db

def SGD_optimization_parameters(parameters, grads, learning_rate, t = 1, decay_rate = 0):
    # time-based learning rate
    adjusted_lr = learning_rate / (1 + decay_rate * t)
    L = len(parameters) // 2
    for i in range(1, L + 1):
        parameters['W' + str[i]] -= adjusted_lr * grads['dW' + str(i)]
        parameters['b' + str(i)] -= adjusted_lr * grads['db' + str(i)]
    return parameters

def compute_cost_L2(y_hat, Y, parameters, lambd):
    """Searched online, I did not write this"""
    """
    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (10, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), shape (10, number of examples)
    parameters -- python dictionary containing parameters of the model
    
    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]
    L = len(parameters) // 2
    y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10) 
    cross_entropy_cost = -np.sum(Y * np.log(y_hat)) / m
    w_sum = 0
    for i in range(1, L):
        w_sum += np.sum(np.square(parameters['W'+str(i)]))
    L2_regularization_cost = (lambd/(2*m)) * w_sum
    cost = cross_entropy_cost + L2_regularization_cost
    return cost