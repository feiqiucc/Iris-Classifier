import numpy as np
from data_process import *
from initialization import init_data
from model import forward_propagation

def evaluate(parameters, feature, label):
    y_hat, _, _ = forward_propagation(feature, label, 0, parameters, ['leaky_relu', 'softmax'])
    predict = np.argmax(y_hat, axis = 0)
    accuracy = np.mean(predict == label) * 100
    return accuracy

print("This file is running")

if __name__ == "__main__":
    parameters = load_model('models/model0.h5')
    train_features, train_labels, dev_features, dev_labels, test_features, test_labels = init_data()
    train_accuracy = evaluate(parameters, train_features, train_labels)
    dev_accuracy = evaluate(parameters, dev_features, dev_labels)
    test_accuracy = evaluate(parameters, test_features, test_labels)

    print(f"Train accuracy: {train_accuracy:.2f}%")
    print(f"Dev accuracy: {dev_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")