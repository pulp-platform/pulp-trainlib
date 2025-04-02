import torch.nn.functional as F
from model.Attention import Attention
from model.Conv2dBN import Conv2dBN
from model.DropPath import DropPath
from model.Mlp import Mlp
from model.SparseAttention import SparseAttention
from torch import nn


class TinyViTBlock(nn.Module):
    r"""TinyViT Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int, int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        local_conv_size (int): the kernel size of the convolution between
                               Attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        local_conv_size=3,
        activation=nn.GELU,
        use_nsa=False,
    ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads

        assert window_size > 0, "window_size must be greater than 0"

        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)

        if use_nsa:
            self.attn = SparseAttention(
                dim=dim,
                dim_head=head_dim,
                heads=num_heads,
                sliding_window_size=512,
                compress_block_size=32,
                selection_block_size=64,
                num_selected_blocks=16,
            )
        else:
            self.attn = Attention(
                dim=dim,
                key_dim=head_dim,
                num_heads=num_heads,
                attn_ratio=1,
                resolution=window_resolution,
            )

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=mlp_activation,
            drop=drop,
        )

        pad = local_conv_size // 2

        self.local_conv = Conv2dBN(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim
        )

    def forward(self, x):
        h, w = self.input_resolution
        b, l, c = x.shape

        assert l == h * w, "input feature has wrong size"

        res_x = x

        if h == self.window_size and w == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(b, h, w, c)
            pad_b = (self.window_size - h % self.window_size) % self.window_size
            pad_r = (self.window_size - w % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            ph, pw = h + pad_b, w + pad_r
            nh = ph // self.window_size
            nw = pw // self.window_size

            # window partition
            x = (
                x.view(b, nh, self.window_size, nw, self.window_size, c)
                .transpose(2, 3)
                .reshape(b * nh * nw, self.window_size * self.window_size, c)
            )

            x = self.attn(x)

            # window reverse
            x = (
                x.view(b, nh, nw, self.window_size, self.window_size, c)
                .transpose(2, 3)
                .reshape(b, ph, pw, c)
            )

            if padding:
                x = x[:, :h, :w].contiguous()

            x = x.view(b, l, c)

        x = res_x + self.drop_path(x)

        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.local_conv(x)
        x = x.view(b, c, l).transpose(1, 2)

        x = x + self.drop_path(self.mlp(x))

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"
        )
