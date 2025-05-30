"""
Copyright (C) 2021-2022 ETH Zurich and University of Bologna
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Authors: Francesco Conoscenti (francesco.conoscenti@studio.unibo.it), Alberto Dequino (alberto.dequino@unibo.it),
         Calin Diaconu (calin.diaconu@studio.unibo.it)
"""

import argparse
from copy import deepcopy

import numpy as np  # Matrix and vector computation package
import torch
import torch.nn as nn

import dump_utils as dump
import mhsa


class MyNet(nn.Module):
    # Define a simple network with a mhsa layer for testing
    def __init__(self, in_h, in_w, n_heads, att_dim):
        super().__init__()
        self.mhsa = mhsa.MultiHeadedSelfAttention(
            dim=in_w, num_heads=n_heads, att_dim=att_dim
        )

    def forward(self, x, tgt_len):
        return self.mhsa(x=x, tgt_len=tgt_len)

def hook_fn1(_, __, o):
    # Hook to write output gradients
    f = open("mhsa-grads.h", "w")

    print("------------Output Grad------------")
    for grad in o:
        try:
            output_grad = torch.transpose(grad, 0, 1).bfloat16()
            f.write("#define G_OUTPUT_SIZE " + str(output_grad.numel()) + "\n")
            print(output_grad)

            if current_step == "BACKWARD":
                f.write(
                    "PI_L2 fp16 OUTPUT_GRAD[G_OUTPUT_SIZE] = {"
                    + dump.tensor_to_string(output_grad)
                    + "};\n"
                )
            else:
                f.write(
                    "PI_L2 fp16 OUTPUT_GRAD[G_OUTPUT_SIZE] = {"
                    + dump.tensor_to_string(output_grad)
                    + "};\n"
                )
        except AttributeError:
            print("None found for Gradient (output)")

    f.close()

if __name__ == "__main__":
    # ~~~~~~~~~~ INTRO ~~~~~~~~~~
    # Set the seed for reproducibility
    np.random.seed(seed=1)  # <----- Sneed
    torch.manual_seed(0)

    # Visualize data with more precision
    torch.set_printoptions(precision=10, sci_mode=False)

    # Set up parser
    parser = argparse.ArgumentParser("MHSA Layer Test")
    parser.add_argument("--in_width", type=int, default=8)  # Token size
    parser.add_argument("--in_height", type=int, default=4)  # Sequence length
    parser.add_argument("--ch_in", type=int, default=1)
    parser.add_argument("--ch_out", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--weight", type=float, default=0.1)
    parser.add_argument("--att_dim", type=int, default=8)
    parser.add_argument(
        "--step", type=str, default="FORWARD"
    )  # Possible steps: FORWARD, BACKWARD

    args = parser.parse_args()

    # Read arguments
    in_h = args.in_height
    in_w = args.in_width
    ch_in = args.ch_in
    ch_out = args.ch_out
    n_heads = args.n_heads
    current_step = args.step
    weight_init = args.weight
    att_dim = args.att_dim
    head_dim = int(att_dim / n_heads)

    # Write net step to file
    f_step = open("step-check.h", "w")
    f_step.write("#define " + str(current_step) + "\n")
    f_step.close()

    # Write input/output weights to file
    f = open("net-args.h", "w")

    f.write("#define EMBED_SIZE " + str(att_dim) + "\n")
    f.write("#define HIDDEN_SIZE " + str(in_w) + "\n")
    f.write("#define SEQ_LEN " + str(in_h) + "\n")
    f.write("#define N_HEADS " + str(n_heads) + "\n")

    f.close()

    # Define network and add hook
    net = MyNet(in_h=in_h, in_w=in_w, n_heads=n_heads, att_dim=att_dim).bfloat16()
    net.zero_grad()

    # ~~~~~~~~~~ MANAGE INPUT ~~~~~~~~~~
    # Generate random input data
    inp = torch.randn(ch_in, in_h, in_w).bfloat16()
    inp.requires_grad = True

    # Print input data to terminal
    print("------------Input sequence------------")
    print(inp)

    # Write transpose of input data to file
    # inp_copy = torch.transpose(inp, -1, -2)

    f = open("input-sequence.h", "w")
    f.write("#define INPUT_SIZE " + str(inp.numel()) + "\n")
    f.write(
        "PI_L2 fp16 INPUT[INPUT_SIZE] = {" + dump.tensor_to_string(inp) + "};\n"
    )
    f.close()

    # ~~~~~~~~~~ MANAGE INPUT WEIGHTS ~~~~~~~~~~
    # Generate random input weights
    in_wgt_init_tensor_q = torch.randn(att_dim, in_w).bfloat16()
    in_wgt_init_tensor_k = torch.randn(att_dim, in_w).bfloat16()
    in_wgt_init_tensor_v = torch.randn(att_dim, in_w).bfloat16()

    in_bias_init_tensor_q = torch.zeros(att_dim).bfloat16()
    in_bias_init_tensor_k = torch.zeros(att_dim).bfloat16()
    in_bias_init_tensor_v = torch.zeros(att_dim).bfloat16()

    # Copy input weights to network
    with torch.no_grad():
        net.mhsa.proj_q.weight.data = deepcopy(in_wgt_init_tensor_q)
        net.mhsa.proj_k.weight.data = deepcopy(in_wgt_init_tensor_k)
        net.mhsa.proj_v.weight.data = deepcopy(in_wgt_init_tensor_v)

        net.mhsa.proj_q.bias.data = deepcopy(in_bias_init_tensor_q)
        net.mhsa.proj_k.bias.data = deepcopy(in_bias_init_tensor_k)
        net.mhsa.proj_v.bias.data = deepcopy(in_bias_init_tensor_v)

    # Print input weights to terminal
    print("Shape input weights:")
    print(net.mhsa.proj_q.weight.shape)
    #print("Shape input biases:")
    #print(net.mhsa.proj_q.bias.shape)
    print("q:")
    print(net.mhsa.proj_q.weight.data)
    print("k:")
    print(net.mhsa.proj_k.weight.data)
    print("v:")
    print(net.mhsa.proj_v.weight.data)
    print("\n")

    # Write input weights to init file
    f = open("attention-defines.h", "w")
    f.write("\n\n// Input Projections Weight Initialization\n")
    f.write("#define INPUT_WGT_SIZE (" + str(in_wgt_init_tensor_q.numel()) + ")\n")
    f.write(
        "PI_L2 fp16 INPUT_WEIGHTS_Q[INPUT_WGT_SIZE] = {"
        + dump.tensor_to_string(in_wgt_init_tensor_q)
        + "};\n"
    )
    f.write(
        "PI_L2 fp16 INPUT_WEIGHTS_K[INPUT_WGT_SIZE] = {"
        + dump.tensor_to_string(in_wgt_init_tensor_k)
        + "};\n"
    )
    f.write(
        "PI_L2 fp16 INPUT_WEIGHTS_V[INPUT_WGT_SIZE] = {"
        + dump.tensor_to_string(in_wgt_init_tensor_v)
        + "};\n"
    )

    f.write("\n\n// Input Projections Bias Initialization\n")
    f.write("#define INPUT_BIAS_SIZE (" + str(in_bias_init_tensor_q.numel()) + ")\n")
    f.write(
        "PI_L2 fp16 INPUT_BIASES_Q[INPUT_BIAS_SIZE] = {"
        + dump.tensor_to_string(in_bias_init_tensor_q)
        + "};\n"
    )
    f.write(
        "PI_L2 fp16 INPUT_BIASES_K[INPUT_BIAS_SIZE] = {"
        + dump.tensor_to_string(in_bias_init_tensor_k)
        + "};\n"
    )
    f.write(
        "PI_L2 fp16 INPUT_BIASES_V[INPUT_BIAS_SIZE] = {"
        + dump.tensor_to_string(in_bias_init_tensor_v)
        + "};\n"
    )
    f.close()

    # ~~~~~~~~~~ MANAGE OUTPUT WEIGHTS ~~~~~~~~~~
    # Generate random output weights
    output_proj_wgt_init_tensor = torch.randn(in_w, att_dim).bfloat16()
    output_proj_bias_init_tensor = torch.zeros(in_w).bfloat16()

    # Copy output weights to network
    with torch.no_grad():
        net.mhsa.proj_out.weight.data = deepcopy(output_proj_wgt_init_tensor)
        net.mhsa.proj_out.bias.data = deepcopy(output_proj_bias_init_tensor)

    # Print output weights to terminal
    print("Shape output projection weights:")
    print(net.mhsa.proj_out.weight.data.shape)
    print(net.mhsa.proj_out.weight.data)
    print("\n")

    # Write output weights to init file
    f = open("output-defines.h", "w")
    f.write("\n\n")
    f.write(
        "#define OUTPUT_WGT_SIZE (" + str(output_proj_wgt_init_tensor.numel()) + ")\n"
    )
    f.write(
        "PI_L2 fp16 ATTENTION_OUTPUT_WEIGHTS[OUTPUT_WGT_SIZE] = {"
        + dump.tensor_to_string(output_proj_wgt_init_tensor)
        + "};\n"
    )
    f.write(
        "PI_L2 fp16 ATTENTION_OUTPUT_BIASES["
        + str(in_w)
        + "] = {"
        + dump.tensor_to_string(output_proj_bias_init_tensor)
        + "};\n\n"
    )
    f.close()

    # ~~~~~~~~~~ COMPUTE OUTPUT ~~~~~~~~~~
    # Compute output
    label = torch.ones(in_h, in_w).bfloat16()
    criterion = nn.MSELoss()
    out = net(x=inp, tgt_len=in_h).bfloat16()

    # Print output to terminal
    print("out: ")
    print(out.size())
    print(label.size())
    print(out)

    # Compute loss
    loss = criterion(out.cuda(), label.cuda())

    # Write output to file
    out_copy = torch.transpose(out, -1, -2)

    f = open("output-sequence.h", "w")
    f.write("#define OUTPUT_SIZE " + str(out.numel()) + "\n")
    f.write(
        "PI_L2 fp16 OUTPUT[OUTPUT_SIZE] = {" + dump.tensor_to_string(out) + "};\n"
    )
    f.close()

    # Compute gradients
    net.zero_grad()
    loss.backward()

    input_wgt_grad_q = net.mhsa.proj_q.weight.grad
    input_wgt_grad_k = net.mhsa.proj_k.weight.grad
    input_wgt_grad_v = net.mhsa.proj_v.weight.grad
    output_wgt_grad = net.mhsa.proj_out.weight.grad
    input_grad = inp.grad.transpose(1, 2)

    # Write gradients to file
    f = open("mhsa-grads.h", "a")

    f.write("#define G_INPUT_WGT_SIZE " + str(input_wgt_grad_q.numel()) + "\n")
    f.write(
        "PI_L2 fp16 INPUT_WGT_GRAD_Q[G_INPUT_WGT_SIZE] = {"
        + dump.tensor_to_string(input_wgt_grad_q)
        + "};\n"
    )
    f.write(
        "PI_L2 fp16 INPUT_WGT_GRAD_K[G_INPUT_WGT_SIZE] = {"
        + dump.tensor_to_string(input_wgt_grad_k)
        + "};\n"
    )
    f.write(
        "PI_L2 fp16 INPUT_WGT_GRAD_V[G_INPUT_WGT_SIZE] = {"
        + dump.tensor_to_string(input_wgt_grad_v)
        + "};\n"
    )

    f.write("#define G_OUTPUT_WGT_SIZE " + str(output_wgt_grad.numel()) + "\n")
    f.write(
        "PI_L2 fp16 OUTPUT_WGT_GRAD[G_OUTPUT_WGT_SIZE] = {"
        + dump.tensor_to_string(output_wgt_grad)
        + "};\n"
    )

    f.write("#define G_IN_SIZE " + str(input_grad.numel()) + "\n")
    f.write(
        "PI_L2 fp16 INPUT_GRAD[G_IN_SIZE] = {"
        + dump.tensor_to_string(input_grad)
        + "};\n"
    )

    f.close()

    # Write attention scores to file
    f = open("attention_scores.h", "w")
    f.write("#define ATTENTION_S_LENGTH " + str(net.mhsa.scores.numel()) + "\n")
    f.write(
        "PI_L2 fp16 ATTENTION_SCORES[ATTENTION_S_LENGTH] = {"
        + dump.tensor_to_string(torch.transpose(net.mhsa.scores, 0, 1))
        + "};\n"
    )
    f.close()
