import torch
import torch.nn as nn
import torch.optim as optim

import pandas

df = pandas.read_csv('iris/iris.data')
df = df.dropna()
df = df.sample(120)

X = torch.tensor(df.iloc[:, :4].values).float()
Y = torch.tensor(df.iloc[:, 4].values)

in_features = 4
hidden_dim = 32
out_features = 3

model = nn.Sequential(
    nn.Linear(in_features, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, out_features),
    nn.Sigmoid()
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 301):
    outputs = model(X)
    loss = criterion(outputs, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 30 == 0:
        print(f'{epoch}\t: {loss.item():.3f}')
