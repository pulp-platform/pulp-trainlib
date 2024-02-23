import numpy as np
from torch import float32
from torch import nn
from torch import Tensor 
from torch import cuda
import torch
from torch.nn import functional as F

class TestModel(nn.Module):
    """Just testing the GELU activation"""
    def __init__(self):
        super().__init__()
        self.act = nn.GELU(approximate='tanh')
        self.scores = None # for visualization

    def forward(self, x):
        x = self.act(x)
        return x