import numpy as np

def relu(X):
    return np.maximum(0, X)

def leaky_relu(X):
    return np.maximum(0, X) + (0.01 * np.minimum(0, X))

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def tanh(X):
    return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))

def elu(X, alpha):
    return np.where(X > 0, X, alpha * (np.exp(X) - 1))

def softmax(X):
    expX = np.exp(X - np.max(X))
    return expX / np.sum(expX, axis=0, keepdims=True)

def activation_f(X: np.ndarray, func: str, alpha:float = 1.0):
    X = np.asarray(X)
    if func == 'relu':
        return relu(X)
    elif func == 'sigmoid':
        return sigmoid(X)
    elif func == 'leaky_relu':
        return leaky_relu(X)
    elif func == 'tanh':
        return tanh(X)
    elif func == 'elu':
        return elu(X, alpha)
    elif func == 'softmax':
        return softmax(X)
    else:
        raise ValueError(f"This activation: {func} function is not supported.")
    


