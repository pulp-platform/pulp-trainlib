'''
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
'''

'''
Authors: Alberto Dequino (alberto.dequino@unibo.it)
'''

from copy import deepcopy
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import dump_utils as dump
import numpy as np  # Matrix and vector computation package
import random
import test_model

# Set the seed for reproducability
np.random.seed(seed=1) # <----- Sneed
torch.manual_seed(0)


##################################################################################################################################

#Visualize data with more precision
torch.set_printoptions(precision=10, sci_mode=False) 

parser = argparse.ArgumentParser("GELU Layer Test")
parser.add_argument( '--in_width', type=int, default=8)
parser.add_argument( '--in_height', type=int, default=8)
parser.add_argument( '--ch_in', type=int, default=1)
parser.add_argument( '--bf16_format', type=int, default=1) # if == 1, data format if bfloat16, if 0 is float16
parser.add_argument( '--step', type=str, default='FORWARD')     # Possible steps: FORWARD, BACKWARD_GRAD, BACKWARD_ERROR

args = parser.parse_args()

# Network parameters in_size
in_h = args.in_height
in_w = args.in_width
ch_in = args.ch_in
ch_out = ch_in
current_step = args.step
bf16_format = args.bf16_format

# Net step
f_step = open('step-check.h', 'w')
f_step.write('#define ' + str(current_step) + '\n')
f_step.close()

# Data file
f = open("init-defines.h", "w") 

f.write('#define Tin_C_l1 '+str(ch_in)+'\n')
f.write('#define Tin_H_l1 '+str(in_h)+'\n')
f.write('#define Tin_W_l1 '+str(in_w)+'\n')
f.write('#define Tout_C_l1 '+str(ch_out)+'\n')


f.close()

if bf16_format == 1:
  inp_1 = torch.randn(ch_in, in_h, in_w).bfloat16()
  inp_2 = torch.randn(ch_in, in_h, in_w).bfloat16()
else:
  inp_1 = torch.randn(ch_in, in_h, in_w).half()
  inp_2 = torch.randn(ch_in, in_h, in_w).half()

# Write input sequence
print("------------Input sequence------------")
f = open("input-sequence.h", "w")
f.write("#define INPUT_SIZE "+str(inp_1.numel())+'\n')
#print(inp_1)

if bf16_format == 1:
  inp_1_copy = inp_1.bfloat16()
  inp_2_copy = inp_2.bfloat16()
else:
  inp_1_copy = inp_1.half()
  inp_2_copy = inp_2.half()

if current_step=='FORWARD':
  f.write('PI_L2 fp16 INPUT_1[INPUT_SIZE] = {'+dump.tensor_to_string(inp_1)+'};\n')
  f.write('PI_L2 fp16 INPUT_2[INPUT_SIZE] = {'+dump.tensor_to_string(inp_2)+'};\n')
else:
  f.write('PI_L2 fp16 INPUT_1[INPUT_SIZE] = {'+dump.tensor_to_string(inp_1)+'};\n')
  f.write('PI_L2 fp16 INPUT_2[INPUT_SIZE] = {'+dump.tensor_to_string(inp_2)+'};\n')
f.close()


out = torch.add(inp_1, inp_2)

print("out: ")
print(out.size())
print(out)

if bf16_format == 1:
  out_copy = out.bfloat16()
else:
  out_copy = out.half()

f = open("skip-output.h", "w")
f.write('#define OUTPUT_SIZE '+str(out.numel())+'\n')
f.write('PI_L2 fp16 OUTPUT[OUTPUT_SIZE] = {'+dump.tensor_to_string(out_copy)+'};\n')
f.close()