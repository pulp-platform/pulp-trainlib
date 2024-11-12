import torch
from torch.autograd import Function

from utils.vit_lr.vit_lr_utils import fastexp_gist


class SoftmaxFastExp(Function):
    @staticmethod
    def forward(ctx, input):
        maxes = torch.max(input, -1, keepdim=True)[0]
        # maxes = torch.swapaxes(maxes, -2, -1)
        x_exp = fastexp_gist((input - maxes))
        x_exp_sum = torch.sum(x_exp, -1, keepdim=True)
        output = x_exp / x_exp_sum
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        out_data = ctx.saved_tensors[0]
        sums = torch.sum(grad_output * out_data, -1, keepdim=True).repeat(
            1, 1, 1, grad_output.shape[-1]
        )
        grad_input = (grad_output - sums) * out_data

        return grad_input
