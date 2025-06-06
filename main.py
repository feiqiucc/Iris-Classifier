import torch
import torch.nn as nn
import torch.optim as optim


in_features = 16
hidden_dim = 64
out_features = 2

model = nn.Sequential(
    nn.Linear(in_features, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, out_features)
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    outputs = model(...)
    loss = criterion(outputs, ...)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
