import torch
import torch.nn as nn
import math

from dl.modules.mlp import MLP

# prepare data
x = torch.linspace(-math.pi, math.pi, 1000).unsqueeze(1)
y = torch.sin(x)

# train code
model = MLP(1, 128, 1, depth=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(2000):
    pred = model(x)
    loss = loss_fn(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.6f}")