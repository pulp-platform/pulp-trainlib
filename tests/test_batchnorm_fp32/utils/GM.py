import argparse
import math

import torch
from torch import nn

import dump_utils as dump


class DNN(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.l1 = nn.BatchNorm2d(
            num_features=in_c,
            eps=0.00001,
            momentum=0,
            affine=True,
        )

    def forward(self, x):
        x1 = self.l1(x)

        return x1


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-CI", type=int, default=2)
    parser.add_argument("-HI", type=int, default=3)
    parser.add_argument("-WI", type=int, default=4)
    parser.add_argument("-BATCH_SIZE", type=int, default=8)
    parser.add_argument("-STEP", type=str, default="FORWARD")
    parser.add_argument("-NUM_CORES", type=int, default=1)
    parser.add_argument("-HWC", type=int, default=0)

    return parser.parse_args()


def write_init_defines(
    in_c,
    in_h,
    in_w,
    l1_hk,
    l1_wk,
    l1_hpad,
    l1_wpad,
    l1_hstr,
    l1_wstr,
    eps,
    batch_size,
    step,
):
    f = open("init-defines.h", "w")

    f.write("// Layer1\n")
    f.write("#define Tin_C_l1 " + str(in_c) + "\n")
    f.write("#define Tin_H_l1 " + str(in_h) + "\n")
    f.write("#define Tin_W_l1 " + str(in_w) + "\n\n")

    f.write("#define Tout_C_l1 " + str(in_c) + "\n")
    f.write(
        "#define Tout_H_l1 "
        + str(math.floor((in_h - l1_hk + 2 * l1_hpad + l1_hstr) / l1_hstr))
        + "\n"
    )
    f.write(
        "#define Tout_W_l1 "
        + str(math.floor((in_w - l1_wk + 2 * l1_wpad + l1_wstr) / l1_wstr))
        + "\n\n"
    )

    f.write("#define Tker_H_l1 " + str(l1_hk) + "\n")
    f.write("#define Tker_W_l1 " + str(l1_wk) + "\n\n")

    f.write("#define Tstr_H_l1 " + str(l1_hstr) + "\n")
    f.write("#define Tstr_W_l1 " + str(l1_wstr) + "\n\n")

    f.write("#define Tpad_H_l1 " + str(l1_hpad) + "\n")
    f.write("#define Tpad_W_l1 " + str(l1_wpad) + "\n\n")

    # Define epsilon as float
    f.write("#define EPS " + str(eps) + "f\n\n")

    f.write(f"#define {step}\n\n")
    f.write(f"// Batch size\n#define BATCH_SIZE {batch_size}\n")

    f.close()


def write_io_data(net, in_c, in_h, in_w, input_test, output_test, label, batch_size):
    f = open("io_data.h", "w")

    f.write("// Init weights\n")
    f.write(f"#define WGT_SIZE_L1 2 * {in_c}\n")
    f.write(
        "PI_L2 float init_WGT_l1[WGT_SIZE_L1] = {"
        + dump.tensor_to_string(net.l1.weight.data)
        + dump.tensor_to_string(net.l1.bias.data)
        + "};\n\n"
    )

    f.write("#define BN_WGT_G_SIZE 2 * " + str(net.l1.weight.data.numel()) + "\n")
    f.write(
        "PI_L2 float BN_WGT_GRAD[BN_WGT_G_SIZE] = {"
        + dump.tensor_to_string(net.l1.weight.grad)
        + dump.tensor_to_string(net.l1.bias.grad)
        + "};\n\n"
    )

    f.write("// Input and Output data\n")
    f.write(f"#define IN_SIZE {batch_size} * {in_c*in_h*in_w}\n")
    f.write(
        "PI_L1 float INPUT[IN_SIZE] = {" + dump.tensor_to_string(input_test) + "};\n\n"
    )

    f.write("#define BN_IN_G_SIZE " + str(input_test.grad.numel()) + "\n")
    f.write(
        "PI_L2 float BN_IN_GRAD[BN_IN_G_SIZE] = {"
        + dump.tensor_to_string(input_test.grad)
        + "};\n\n"
    )

    f.write(f"#define OUT_SIZE {batch_size} * {in_c*in_h*in_w}\n")
    f.write(
        "PI_L2 float REFERENCE_OUTPUT[OUT_SIZE] = {"
        + dump.tensor_to_string(output_test)
        + "};\n\n"
    )

    f.write("PI_L1 float LABEL[OUT_SIZE] = {" + dump.tensor_to_string(label) + "};\n")

    f.close()


def main():
    # Set seed
    torch.manual_seed(42)

    # Get arguments
    args = get_args()

    # Get layer parameters
    in_c = args.CI
    in_h = args.HI
    in_w = args.WI

    batch_size = args.BATCH_SIZE
    step = args.STEP

    l1_hk = l1_wk = l1_hstr = l1_wstr = 1
    l1_hpad = l1_wpad = 0

    # Initialize network
    net = DNN(in_c=in_c)
    for p in net.parameters():
        nn.init.normal_(p, mean=0.0, std=1.0)
    net.zero_grad()

    # Write to "init-defines.h"
    write_init_defines(
        in_c=in_c,
        in_h=in_h,
        in_w=in_w,
        l1_hk=l1_hk,
        l1_wk=l1_wk,
        l1_hpad=l1_hpad,
        l1_wpad=l1_wpad,
        l1_hstr=l1_hstr,
        l1_wstr=l1_wstr,
        eps=net.l1.eps,
        batch_size=batch_size,
        step=step,
    )

    # Sample input data
    input_test = torch.torch.div(
        torch.randint(1000, [batch_size, in_c, in_h, in_w]), 1000
    )
    input_test.requires_grad = True

    # Get test output
    output_test = net(input_test)
    output_test.retain_grad()

    # Generate sample label
    label = torch.ones_like(output_test)

    # Compute loss and perform backward pass
    loss_fn = nn.MSELoss()
    loss = loss_fn(output_test, label)
    loss.backward()

    # Write to "io_data.h"
    write_io_data(net, in_c, in_h, in_w, input_test, output_test, label, batch_size)


if __name__ == "__main__":
    main()
