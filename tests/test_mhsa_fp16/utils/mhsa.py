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
    maxes = maxes.float()
    x_copy = x.float()

    x_exp = torch.exp((x_copy-maxes))
    x_exp = x_exp.bfloat16()
    x_exp_sum = torch.sum(x_exp, -1, keepdim=True)

    return x_exp/x_exp_sum


class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, att_dim, bf16_format):
        super().__init__()
        self.proj_in = nn.Linear(dim, 3*att_dim, bias=False)
        self.proj_out = nn.Linear(att_dim, dim, bias=False)
        self.dim = dim
        self.att_dim = att_dim
        self.n_heads = num_heads
        self.head_dim = att_dim // num_heads
        # self.scaling = q_rsqrt(self.head_dim).bfloat16()
        # self.scaling = (1 / math.sqrt(self.head_dim))
        self.scaling = q_rsqrt(self.head_dim)
        self.scores = None  # for visualization
        self.softmax = SoftmaxFastExp
        self.bf16_format = bf16_format

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

        if self.bf16_format == 0:
            scores = scores.half()
        else:
            scores = scores.bfloat16()

        # OP 6
        scores = torch.bmm(scores, v)
        assert list(scores.size()) == [self.n_heads, tgt_len, self.head_dim]

        scores = scores.transpose(0, 1).contiguous().view(tgt_len, self.att_dim)
        self.scores = scores

        # OP 7
        h = self.proj_out(scores)
        return h
