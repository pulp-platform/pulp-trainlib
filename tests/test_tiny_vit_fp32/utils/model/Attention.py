import itertools

import torch
from torch import nn


class Attention(torch.nn.Module):
    def __init__(
        self,
        dim,
        key_dim,
        num_heads=8,
        attn_ratio=4,
        resolution=(14, 14),
    ):
        super().__init__()

        # (h, w)
        assert isinstance(resolution, tuple) and len(resolution) == 2

        self.ab = None

        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        n = len(points)
        attention_offsets = {}
        idxs = []

        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets))
        )

        self.register_buffer(
            "attention_bias_idxs", torch.LongTensor(idxs).view(n, n), persistent=False
        )

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)

        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        b, n, _ = x.shape

        # Normalization
        x = self.norm(x)

        qkv = self.qkv(x)

        # (B, N, num_heads, d)
        q, k, v = qkv.view(b, n, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.d], dim=3
        )

        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + (
            self.attention_biases[:, self.attention_bias_idxs]
            if self.training
            else self.ab
        )

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(b, n, self.dh)
        x = self.proj(x)

        return x
