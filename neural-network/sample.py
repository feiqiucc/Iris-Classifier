import torch
import torch.nn as nn

loss_func = nn.MSELoss()

x_train = torch.rand(100) * 10
y_train = x_train * 2 + 1
max_iter = 1500

model = nn.Sequential(
    nn.Linear(1, 100),
    nn.ReLU(),
    nn.Linear(100,1)
)

optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-3)
for iter in range(max_iter):
    xb = torch.tensor(x_train, dtype=torch.float).view(-1,1)
    yb = torch.tensor(y_train, dtype=torch.float).view(-1,1)
    target = model(xb)
    loss = loss_func(target, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if iter % 100 == 0:
        print(f"Iter {iter}: Loss = {loss.item():.6f}")

print(model(torch.tensor([1.0])))