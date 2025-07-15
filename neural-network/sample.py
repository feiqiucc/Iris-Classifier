import numpy as np
import torch
import torch.nn as nn

loss_func = nn.MSELoss()

xb = torch.rand(100)
yb = xb * 2 + 1
max_iter = 1000

model = nn.Sequential(
    nn.Linear(1, 1)
)

optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
for iter in range(max_iter):
    xb = torch.tensor(x_train, dtype=torch.float).view(-1,1)
    yb = torch.tensor(y_train, dtype=torch.float).view(-1,1)
    target = model(xb)
    loss = loss_func(target, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
