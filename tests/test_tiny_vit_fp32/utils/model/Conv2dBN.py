import torch


class Conv2dBN(torch.nn.Sequential):
    def __init__(
        self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1.0
    ):
        super().__init__()

        self.add_module(
            "c", torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False)
        )

        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)

        self.add_module("bn", bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()

        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]

        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5

        m = torch.nn.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
        )

        m.weight.data.copy_(w)
        m.bias.data.copy_(b)

        return m
