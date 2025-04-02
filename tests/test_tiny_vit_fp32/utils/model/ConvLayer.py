from model.MBConv import MBConv
from torch import nn


class ConvLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        activation,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        out_dim=None,
        conv_expand_ratio=4.0,
    ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                MBConv(
                    dim,
                    dim,
                    conv_expand_ratio,
                    activation,
                    drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation
            )
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x
