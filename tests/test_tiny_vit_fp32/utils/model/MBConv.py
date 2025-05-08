from model.Conv2dBN import Conv2dBN
from model.DropPath import DropPath
from torch import nn


class MBConv(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio, activation, drop_path):
        super().__init__()

        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2dBN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()

        self.conv2 = Conv2dBN(
            self.hidden_chans,
            self.hidden_chans,
            ks=3,
            stride=1,
            pad=1,
            groups=self.hidden_chans,
        )

        self.act2 = activation()

        self.conv3 = Conv2dBN(self.hidden_chans, self.out_chans, ks=1, bn_weight_init=0)
        self.act3 = activation()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        x = self.drop_path(x)

        x += shortcut
        x = self.act3(x)

        return x
