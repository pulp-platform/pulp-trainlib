import numpy as np
import torch
from torch import nn

from SoftmaxFastExp import SoftmaxFastExp


def q_rsqrt(x):
    with torch.no_grad():
        y = np.asarray((x,), dtype=np.float32)
        x2 = y * 0.5
        i = y.view(np.int32)
        i = np.right_shift(i, 1)
        i = 0x5f3759df - i
        y = i.view(np.float32)
        y = y * (1.5 - (x2 * y * y))

        result = torch.from_numpy(y)

    return result


def own_softmax(x):
    maxes = torch.max(x, -1, keepdim=True)[0]
    # maxes = torch.swapaxes(maxes, -2, -1)
    x_exp = torch.exp((x-maxes))
    x_exp_sum = torch.sum(x_exp, -1, keepdim=True)

    return x_exp/x_exp_sum


def threshold(x):
    log2 = 0.6931471805599453
    log2_2 = 0.4804530139182014
    log2_3 = 0.3330246519889294
    log2_4 = 0.2308350985830834
    log2_5 = 0.1600026977571413
    x[x < 3.14] = (1 - log2 * x + 0.5 * np.power(x, 2) * log2_2 - 0.16 * np.power(x, 3) * log2_3 + 0.0416 *
                   np.power(x, 4) * log2_4 - 0.008 * np.power(x, 5) * log2_5)[x < 3.14]
    x[x >= 3.14] = 0
    return x


def own_partial_softmax_simple(x):
    n_heads = x.shape[-3]
    seq_length = x.shape[-1]
    x_copy = x.detach().numpy().astype(np.float32)

    print("Softmax input:")
    print(x)
    
    lines_max = np.max(x_copy, axis=-1)
    diff = np.repeat(lines_max, seq_length).reshape(n_heads, seq_length, seq_length) - x_copy
    
    exp_sum = np.sum(1 / 2**diff, axis=-1)
    exp_sum_inverse = 1 / exp_sum
    
    return torch.from_numpy(np.repeat(exp_sum_inverse, seq_length).reshape(n_heads, seq_length, seq_length) / 2**diff)


def own_partial_softmax_simple_approximate(x):
    n_heads = x.shape[-3]
    seq_length = x.shape[-1]
    x_copy = x.detach().numpy().astype(np.float32)

    print("Softmax input:")
    print(x)
    
    lines_max = np.max(x_copy, axis=-1)
    diff = np.repeat(lines_max, seq_length).reshape(n_heads, seq_length, seq_length) - x_copy
    
    exp_sum = np.sum(threshold(diff.copy()), axis=-1)
    exp_sum_inverse = 1 / exp_sum
    
    return torch.from_numpy(np.repeat(exp_sum_inverse, seq_length).reshape(n_heads, seq_length, seq_length) *
                            threshold(diff.copy()))


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
        # self.scaling = (self.head_dim) ** -0.5
        self.scaling = q_rsqrt(self.head_dim)
        self.scores = None  # for visualization
        # self.softmax = own_softmax
        # self.softmax = own_partial_softmax_simple
        self.softmax = SoftmaxFastExp

    def forward(self, x, tgt_len):
        # OP 1
        qkv = self.proj_in(x)

        # OP 2
        q = qkv[..., :int(qkv.shape[-1] / 3)]
        k = qkv[..., int(qkv.shape[-1] / 3):2 * int(qkv.shape[-1] / 3)]
        v = qkv[..., 2 * int(qkv.shape[-1] / 3):]

        q = q.contiguous().view(tgt_len, self.n_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(tgt_len, self.n_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(tgt_len, self.n_heads, self.head_dim).transpose(0, 1)

        # OP 3
        scores = torch.bmm(q, k.transpose(1, 2))

        assert list(scores.size()) == [self.n_heads, tgt_len, tgt_len]

        # OP 4
        scores = scores * self.scaling

        # OP 5
        # scores = self.softmax(scores)
        scores = SoftmaxFastExp.apply(scores)

        # OP 6
        scores = torch.bmm(scores, v)
        assert list(scores.size()) == [self.n_heads, tgt_len, self.head_dim]

        scores = scores.transpose(0, 1).contiguous().view(tgt_len, self.att_dim)
        self.scores = scores

        # OP 7
        h = self.proj_out(scores)
        return h
