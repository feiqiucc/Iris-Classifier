from data_process import *
import math
import numpy as np

def init_data():
    # reshape to (1, m) so that easier to broadcast
    (train_features, train_labels),(dev_features, dev_labels), (test_features, test_labels) = load_data()
    train_labels = train_labels.reshape(1, train_features.shape[1])
    dev_labels = dev_labels.reshape(1, dev_features.shape[1])
    test_labels = test_labels.reshape(1, test_features.shape[1])
    return train_features, train_labels, dev_features, dev_labels, test_features, test_labels

def he_init_normal(x_in, x_out):
    # recommend relu
    W = np.random.randn(x_out, x_in) * np.sqrt(2.0 / x_in)
    b = np.zeros((x_out, 1))
    return W, b

def xavier_init_uniform(x_in, x_out):
    #recommend tanh sigmoid
    limit = np.sqrt(6 / (x_in + x_out))
    W = np.random.rand(x_out, x_in) * np.sqrt(1.0 / x_in)
    b = np.zeros((x_out, 1))
    return W, b

def init_parameters(layer_dim, init_method):
    parameters = {}
    l = len(layer_dim)
    for l in range(1, l):
        if init_method[l-1] == 'he':
            parameters['W' + str[l]], parameters['b' + str(l)] = he_init_normal(layer_dim[l-1], layer_dim[l])
        elif init_method[l-1] == 'xavier':
            parameters['W' + str[l]], parameters['b' + str(l)] = xavier_init_uniform(layer_dim[l-1], layer_dim[l])
        else:
            raise ValueError(f"unsupported initialization method: {init_method}")
