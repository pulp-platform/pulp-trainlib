import numpy as np
from torch import float32
from torch import nn
from torch import Tensor 
from torch import cuda
import torch
from torch.nn import functional as F

def own_softmax(x):
    maxes = torch.amax(x, (1,2), keepdim=True)

    x_exp = torch.exp((x-maxes))
    x_exp_sum = torch.sum(x_exp, (1,2), keepdim=True)

    return x_exp/x_exp_sum

def own_partial_softmax(x):
    n_heads = x.shape[-3]
    seq_length = x.shape[-1]
    B = 8
    eps_max = B / (2**B)
    x_copy = x.detach().numpy().astype(np.float32)

    x_copy = x_copy / eps_max

    exp_partial_sum = np.zeros((n_heads, seq_length), dtype = np.float32)
    global_max = np.full((n_heads, seq_length), -np.Infinity, dtype = np.float32)
    shift_sum = np.zeros((n_heads, seq_length), dtype = np.float32)

    current_max = np.max(x_copy, axis = -1)
    
    max_shift = (current_max - global_max) * eps_max

    shift_sum[current_max > global_max] = max_shift[current_max > global_max]
    global_max[current_max > global_max] = current_max[current_max > global_max]

    diff = np.repeat(global_max, seq_length).reshape(n_heads, seq_length, seq_length) - x_copy
    shift = diff * eps_max

    exp_sum = np.sum(1 / 2**shift, axis = -1)
    exp_partial_sum = (exp_partial_sum / 2**(shift_sum.astype(np.float32))) + exp_sum
    exp_partial_sum_inverse = 1 / exp_partial_sum

    diff = np.repeat(global_max, seq_length).reshape(n_heads, seq_length, seq_length) - x_copy
    shift = diff * eps_max

    return torch.from_numpy(np.repeat(exp_partial_sum_inverse, seq_length).reshape(n_heads, seq_length, seq_length) / 2**shift)





class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, att_dim):
        super().__init__()
        self.proj_in = nn.Linear(dim, 3*att_dim, bias=False)
        self.proj_out = nn.Linear(att_dim, dim, bias=False)
        self.dim = dim
        self.att_dim = att_dim
        self.n_heads = num_heads
        self.head_dim = att_dim // num_heads
        self.scaling = (self.head_dim) ** -0.5
        self.scores = None # for visualization
        self.softmax = own_partial_softmax

    def forward(self, x, tgt_len):
        q, k, v = self.proj_in(x).chunk(3, dim=-1)
        q = q.contiguous().view(tgt_len, self.n_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, self.n_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, self.n_heads, self.head_dim).transpose(0, 1)

        scores = torch.bmm(q, k.transpose(1, 2))

        assert list(scores.size()) == [self.n_heads, tgt_len, tgt_len]

        scores = scores * self.scaling
        scores = self.softmax(scores)

        scores = torch.bmm(scores, v)
        assert list(scores.size()) == [self.n_heads, tgt_len, self.head_dim]

        scores = scores.transpose(0, 1).contiguous().view(tgt_len, self.att_dim)
        self.scores = scores
        h = self.proj_out(scores)
        return h