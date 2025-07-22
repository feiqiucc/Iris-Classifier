import numpy as np
import pandas as pd
import pickle as p

def my_split_np(features, labels, test_ratio=0.3, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    test_size = int(len(features) * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return features[train_idx], features[test_idx], labels[train_idx], labels[test_idx]

def save_m(parameters, filename="model.pkl"):
    with open(filename, 'wb') as f:
        p.dump(parameters, f)
    print(f"this model is save to {filename}")

def load_m(parameters, filename="model.pkl"):
    with open(filename, 'rb') as f:
        parameters = p.load(f)
    print(f"model is loaded from {filename}")
    return parameters

def load_data():
    data = pd.read_csv("dataset/iris.csv")
    features = data[['sepal.length'], ['sepal.width'], ['petal.length'], ['petal.width']].values
    labels = data['variety'].map({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}).values
    train_features, temp_features, train_labels, temp_labels = my_split_np(features, labels, 0.3, 42)
    dev_features, test_features, dev_labels, test_labels = my_split_np(temp_features, temp_labels, 0.5, 18)
    return ((train_features.T, train_labels),
            (dev_features.T, dev_labels),
            (test_features.T, test_labels))

