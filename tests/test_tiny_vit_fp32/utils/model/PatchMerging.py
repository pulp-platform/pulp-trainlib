from model.Conv2dBN import Conv2dBN
from torch import nn


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim, activation):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim

        self.act = activation()

        self.conv1 = Conv2dBN(dim, out_dim, 1, 1, 0)
        self.conv2 = Conv2dBN(out_dim, out_dim, 3, 2, 1, groups=out_dim)
        self.conv3 = Conv2dBN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        if x.ndim == 3:
            h, w = self.input_resolution
            b = len(x)

            # (B, C, H, W)
            x = x.view(b, h, w, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)

        x = self.conv3(x)

        x = x.flatten(2).transpose(1, 2)

        return x
