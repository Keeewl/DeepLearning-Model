import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_dim, eps=1e-5):
        super().__init__()

        self.normalized_dim = normalized_dim
        self.eps = eps

        # learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_dim))
        self.beta = nn.Parameter(torch.zeros(normalized_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.gamma * x_hat + self.beta

        return y