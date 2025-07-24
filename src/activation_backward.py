import numpy as np
from activation_funcs import *

def relu_backward(dAct, liner_op):
    dliner = np.array(dAct, copy = True)
    dliner[liner_op <= 0] = 0
    return dliner

def leaky_relu_backward(dAct, liner_op):
    dliner = np.array(dAct, copy = True)
    dliner[liner_op <= 0] = dliner[liner_op <= 0] * 0.01
    return dliner

def sigmoid_backward(dAct, liner_op):
    temp = sigmoid(liner_op)
    dliner = dAct * temp * (1 - temp)
    return dliner

def tanh_backward(dAct, liner_op):
    temp = tanh(liner_op)
    dliner = dAct * (1 - temp ** 2)
    return dliner

def activation_func_backward(dAct, liner_op, activation):
    if activation == 'relu_backward':
        return relu_backward(dAct, liner_op)
    elif activation == 'leaky_relu_backward':
        return leaky_relu_backward(dAct, liner_op)
    elif activation == 'sigmoid_backward':
        return sigmoid_backward(dAct, liner_op)
    elif activation == 'tanh_backward':
        return tanh_backward(dAct, liner_op)
    else:
        raise ValueError(f"unsupported activation func {activation}")