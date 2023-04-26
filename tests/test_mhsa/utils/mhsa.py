import numpy as np
from torch import float32
from torch import nn
from torch import Tensor 
from torch import cuda
import torch
from torch.nn import functional as F

def own_softmax(x):
    #print("Softmax input: ")
    #print(x)
    
    maxes = torch.amax(x, (1,2), keepdim=True)
    #print("Maxes: ")
    #print(maxes)

    #print("x - maxes: ")
    #print(x - maxes)

    x_exp = torch.exp((x-maxes))
    #print("X_exp: ")
    #print(x_exp*100000)
    x_exp_sum = torch.sum(x_exp, (1,2), keepdim=True)
    #print("X_exp_sum: ")
    #print(x_exp_sum)

    return x_exp/x_exp_sum

class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.proj_in = nn.Linear(dim, 3*dim, bias=False)
        self.proj_out = nn.Linear(dim, dim, bias=False)
        self.dim = dim
        self.n_heads = num_heads
        self.head_dim = dim // num_heads
        self.scaling = (self.head_dim) ** -0.5
        self.scores = None # for visualization
        #self.softmax = nn.Softmax(dim=-1)
        self.softmax = own_softmax

    def forward(self, x, tgt_len):
        q, k, v = self.proj_in(x).chunk(3, dim=-1)
        #q = q * self.scaling
        q = q.contiguous().view(tgt_len, self.n_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, self.n_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, self.n_heads, self.head_dim).transpose(0, 1)

        #print("q: ")
        #print(q.shape)
        #print(q)
        #print("k: ")
        #print(k)
        #print("v: ")
        #print(v)

        scores = torch.bmm(q, k.transpose(1, 2))

        #print("scores: ")
        #print(scores.shape)
        #print(scores)
        assert list(scores.size()) == [self.n_heads, tgt_len, tgt_len]

        #print("Scaling factor: ")
        #print(self.scaling)
        scores = scores * self.scaling
        scores = self.softmax(scores)
        #print("scores post softmax: ")
        #print(scores)

        scores = torch.bmm(scores, v)
        assert list(scores.size()) == [self.n_heads, tgt_len, self.head_dim]

        scores = scores.transpose(0, 1).contiguous().view(tgt_len, self.dim)
        self.scores = scores
        h = self.proj_out(scores)
        return h