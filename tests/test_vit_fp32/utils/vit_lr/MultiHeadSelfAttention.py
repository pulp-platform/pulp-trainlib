import torch.nn as nn

from utils.vit_lr.SoftmaxFastExp import SoftmaxFastExp
from utils.vit_lr.vit_lr_utils import q_rsqrt


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, att_dim, device):
        super().__init__()

        assert (
            att_dim % n_heads == 0
        ), "Attention dimension not compatible with the given number of heads."

        self.n_heads = n_heads
        self.att_dim = att_dim
        self.head_dim = att_dim // n_heads

        self.proj_q = nn.Linear(dim, att_dim, bias=True)
        self.proj_k = nn.Linear(dim, att_dim, bias=True)
        self.proj_v = nn.Linear(dim, att_dim, bias=True)

        self.scaling = q_rsqrt(self.head_dim).to(device)

        self.softmax = SoftmaxFastExp.apply
        # self.softmax = torch.nn.Softmax(dim=-1)

        self.proj_out = nn.Linear(att_dim, dim, bias=False)

    def forward(self, x, tgt_len):
        # OP 1
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        q = q.contiguous().view(tgt_len, self.n_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(tgt_len, self.n_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(tgt_len, self.n_heads, self.head_dim).transpose(0, 1)

        # OP 3
        scores = q @ k.transpose(1, 2)
        assert list(scores.size()) == [
            self.n_heads,
            tgt_len,
            tgt_len,
        ]

        # OP 4
        scores = scores * self.scaling

        # OP 5
        scores = self.softmax(scores)

        # OP 6
        scores = scores @ v
        assert list(scores.size()) == [
            self.n_heads,
            tgt_len,
            self.head_dim,
        ]

        scores = scores.transpose(0, 1).contiguous().view(tgt_len, self.att_dim)

        # OP 7
        h = self.proj_out(scores)
        return h
