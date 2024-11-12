import torch
import torch.nn as nn


class PositionalEmbedding1D(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        return x + self.pos_embedding
