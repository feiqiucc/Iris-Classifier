import numpy as np
import pandas as pd
import h5py

def my_split_np(features, labels, test_ratio=0.3, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    test_size = int(len(features) * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return features[train_idx], features[test_idx], labels[train_idx], labels[test_idx]

def save_model(parameters, filename="model.h5"):
    with h5py.File(filename, 'w') as f:
        for key, value in parameters.items():
            f.create_dataset(key, data=value)
    print(f"Model saved to {filename}")

def load_model(filename="model.h5"):
    parameters = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            parameters[key] = np.array(f[key])
    print(f"Model loaded from {filename}")
    return parameters   

def load_data():
    data = pd.read_csv("dataset/iris.csv")
    features = data[['sepal.length'], ['sepal.width'], ['petal.length'], ['petal.width']].values
    labels = data['variety'].map({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}).values
    train_features, temp_features, train_labels, temp_labels = my_split_np(features, labels, 0.2, 42)
    dev_features, test_features, dev_labels, test_labels = my_split_np(temp_features, temp_labels, 0.5, 18)
    return ((train_features.T, train_labels),
            (dev_features.T, dev_labels),
            (test_features.T, test_labels))

