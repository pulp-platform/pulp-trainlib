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
parser.add_argument( '--ch_out', type=int, default=1)  
parser.add_argument( '--bf16_format', type=int, default=1) # if == 1, data format if bfloat16, if 0 is float16
parser.add_argument( '--step', type=str, default='FORWARD')     # Possible steps: FORWARD, BACKWARD_GRAD, BACKWARD_ERROR

args = parser.parse_args()

# Network parameters in_size
in_h = args.in_height
in_w = args.in_width
ch_in = args.ch_in
ch_out = args.ch_out
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
  net = test_model.TestModel().bfloat16()
elif bf16_format == 0: 
  net = test_model.TestModel().half()
net.zero_grad()

def hook_fn1(m, i, o):

  cont = 0
  input_grad = []
  weight_grad = []
  output_grad = []
  f = open("gelu-grads.h", "w")

  print("------------Output Grad------------")
  for grad in o:
    try:
      output_grad = grad
      f.write('#define G_OUTPUT_SIZE '+str(output_grad.numel())+'\n')
      print(output_grad)
      if current_step=='BACKWARD_GRAD' or current_step=='BACKWARD_ERROR':
          f.write('PI_L2 fp16 OUTPUT_GRAD[G_OUTPUT_SIZE] = {'+dump.tensor_to_string(output_grad)+'};\n')
      else:
          f.write('PI_L2 fp16 OUTPUT_GRAD[G_OUTPUT_SIZE] = {'+dump.tensor_to_string(output_grad)+'};\n')

    except AttributeError:
      print ("None found for Gradient (output)")

  f.close()


def hook_fn2(m, i, o):

     cont = 0
     input_grad = []
     weight_grad = []
     output_grad = []
     f = open("gelu-output.h", "w")

     print("------------Output------------")
     for grad in o:
       try:
         if cont==0:
          output_grad = grad
          f.write('#define OUTPUT_SIZE '+str(output_grad.numel())+'\n')
          print(output_grad)
          f.write('PI_L2 fp16 OUTPUT[OUTPUT_SIZE] = {'+dump.tensor_to_string(output_grad)+'};\n')
         cont+=1
       except AttributeError:
         print ("None found for Output")
     f.close()


gradsRnn = net.act.register_full_backward_hook(hook_fn1)

if bf16_format == 1:
  inp = torch.randn(ch_in, in_h, in_w).bfloat16()
else:
  inp = torch.randn(ch_in, in_h, in_w).half()

inp.requires_grad = True

# Write input sequence
print("------------Input sequence------------")
f = open("input-sequence.h", "w")
f.write("#define INPUT_SIZE "+str(inp.numel())+'\n')
print(inp)

if bf16_format == 1:
  inp_copy = inp.bfloat16()
else:
  inp_copy = inp.half()

if current_step=='FORWARD':
  f.write('PI_L2 fp16 INPUT[INPUT_SIZE] = {'+dump.tensor_to_string(inp)+'};\n')
else:
  f.write('PI_L2 fp16 INPUT[INPUT_SIZE] = {'+dump.tensor_to_string(inp)+'};\n')
f.close()


out = net(x=inp)

print("out: ")
print(out.size())
print(out)

if bf16_format == 1:
  out_copy = out.bfloat16()
else:
  out_copy = out.half()

f = open("gelu-output.h", "w")
f.write('#define OUTPUT_SIZE '+str(out.numel())+'\n')
f.write('PI_L2 fp16 OUTPUT[OUTPUT_SIZE] = {'+dump.tensor_to_string(out_copy)+'};\n')
f.close()