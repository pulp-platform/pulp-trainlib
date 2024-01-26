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
Authors: Francesco Conoscenti (francesco.conoscenti@studio.unibo.it), Alberto Dequino (alberto.dequino@unibo.it)
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
import mhsa_partial_softmax as mhsa

# Set the seed for reproducability
np.random.seed(seed=1) # <----- Sneed


##################################################################################################################################

#Visualize data with more precision
torch.set_printoptions(precision=10, sci_mode=False) 

parser = argparse.ArgumentParser("MHSA Layer Test")
parser.add_argument( '--in_width', type=int, default=8) # Token size
parser.add_argument( '--in_height', type=int, default=4) # Sequence length
parser.add_argument( '--ch_in', type=int, default=1)
parser.add_argument( '--ch_out', type=int, default=1)  
parser.add_argument( '--n_heads', type=int, default=8)
parser.add_argument( '--weight', type=float, default=0.1)
parser.add_argument( '--att_dim', type=int, default=8)
parser.add_argument( '--step', type=str, default='FORWARD')     # Possible steps: FORWARD, BACKWARD_GRAD, BACKWARD_ERROR

args = parser.parse_args()

# Network parameters in_size
in_h = args.in_height
in_w = args.in_width
ch_in = args.ch_in
ch_out = args.ch_out
n_heads = args.n_heads
current_step = args.step
weight_init = args.weight
att_dim = args.att_dim
head_dim = (int) (att_dim / n_heads);

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
f.write('#define Tn_heads_l1 '+str(n_heads)+'\n')
f.write('#define Tatt_dim_l1 '+str(att_dim)+'\n')
f.write('#define Thead_dim_l1 '+str(head_dim)+'\n')


f.close()

class myNet(nn.Module):
  def __init__(self, in_h, in_w, n_heads, att_dim):
    super().__init__()
    self.mhsa = mhsa.MultiHeadedSelfAttention(dim=in_w, num_heads=n_heads, att_dim=att_dim)

  def forward(self, x, tgt_len):
    return self.mhsa(x=x, tgt_len=tgt_len)

net = myNet(in_h=in_h, in_w=in_w, n_heads=n_heads, att_dim=att_dim)
net.zero_grad()

def hook_fn1(m, i, o):

  cont = 0
  input_grad = []
  weight_grad = []
  output_grad = []
  f = open("mhsa-grads.h", "w")

  print("------------Output Grad------------")
  for grad in o:
    try:
      output_grad = grad
      f.write('#define G_OUTPUT_SIZE '+str(output_grad.numel())+'\n')
      print(output_grad)
      if current_step=='BACKWARD_GRAD' or current_step=='BACKWARD_ERROR':
          f.write('PI_L2 float OUTPUT_GRAD[G_OUTPUT_SIZE] = {'+dump.tensor_to_string(output_grad)+'};\n')
      else:
          f.write('PI_L2 float OUTPUT_GRAD[G_OUTPUT_SIZE] = {'+dump.tensor_to_string(output_grad)+'};\n')

    except AttributeError:
      print ("None found for Gradient (output)")

  f.close()


def hook_fn2(m, i, o):

     cont = 0
     input_grad = []
     weight_grad = []
     output_grad = []
     f = open("mhsa-output.h", "w")

     print("------------Output------------")
     for grad in o:
       try:
         if cont==0:
          output_grad = grad
          f.write('#define OUTPUT_SIZE '+str(output_grad.numel())+'\n')
          print(output_grad)
          f.write('PI_L2 float OUTPUT[OUTPUT_SIZE] = {'+dump.tensor_to_string(output_grad)+'};\n')
         cont+=1
       except AttributeError:
         print ("None found for Output")
     f.close()


gradsRnn = net.mhsa.register_full_backward_hook(hook_fn1)

inp = torch.div(torch.ones(ch_in, in_h, in_w), 1000)
for cin in range(ch_in):
  for hi in range(in_h):
    for wi in range(in_w):
      inp[cin, hi, wi] += (cin + hi - wi)*(cin + hi + wi) * 1/1e5

inp.requires_grad = True

label = torch.ones(in_h, in_w)

# Write input sequence
print("------------Input sequence------------")
f = open("input-sequence.h", "w")
f.write("#define INPUT_SIZE "+str(inp.numel())+'\n')
print(inp)
if current_step=='FORWARD':
  f.write('PI_L2 float INPUT[INPUT_SIZE] = {'+dump.tensor_to_string(inp)+'};\n')
else:
  f.write('PI_L2 float INPUT[INPUT_SIZE] = {'+dump.tensor_to_string(inp)+'};\n')
f.close()


# Prepare weight tensors for init
# Input weights
print("Shape input weights:")
print(net.mhsa.proj_in.weight.shape)
print(net.mhsa.proj_in.weight.data)
print("\n")
in_wgt_init_tensor = torch.zeros(att_dim * 3, in_w)
for hk in range(att_dim * 3):
    for wk in range(in_w):
        in_wgt_init_tensor[hk, wk] = (hk+wk)*weight_init
#Initialize input weights
with torch.no_grad():
    #net.conv.weight[:, :] = weight_init
    net.mhsa.proj_in.weight.data = deepcopy(in_wgt_init_tensor)
    #net.rnn.bias_ih_l0[:] = 0.0

in_wgt_init_tensor = torch.transpose(in_wgt_init_tensor, 0, 1)

# Print input weights to init file
f = open("init-defines.h", 'a')
f.write("\n\n// Input Projections Weigth Initialization\n")
f.write("#define INPUT_WGT_SIZE (3*Tatt_dim_l1*Tin_W_l1)\n")
f.write('PI_L2 float INPUT_WEIGHTS[INPUT_WGT_SIZE] = {'+dump.tensor_to_string(in_wgt_init_tensor)+'};\n')
f.close()


# Prepare weight tensors for output projection
# Output weights
print("Shape output projection weights:")
print(net.mhsa.proj_out.weight.data.shape)
print(net.mhsa.proj_out.weight.data)
print("\n")
output_proj_wgt_init_tensor = torch.zeros(in_w, att_dim)
for hk in range(in_w):
    for wk in range(att_dim):
        output_proj_wgt_init_tensor[hk, wk] = (hk+wk)*weight_init
#Initialize output weights
with torch.no_grad():
    net.mhsa.proj_out.weight.data = deepcopy(output_proj_wgt_init_tensor)


#output_proj_wgt_init_tensor = torch.transpose(output_proj_wgt_init_tensor, 0, 1)

# Print input weights to init file
f = open("init-defines.h", 'a')
f.write("\n\n")
f.write("#define OUTPUT_WGT_SIZE (Tatt_dim_l1*Tin_W_l1)\n")
f.write('PI_L2 float OUTPUT_WEIGHTS[OUTPUT_WGT_SIZE] = {'+dump.tensor_to_string(output_proj_wgt_init_tensor)+'};\n')
f.close()

criterion = nn.MSELoss()
out = net(x=inp, tgt_len=in_h)
print("out: ")
print(out.size())
print(label.size())
print(out)
loss = criterion(out, label)

f = open("mhsa-output.h", "w")
f.write('#define OUTPUT_SIZE '+str(out.numel())+'\n')
f.write('PI_L2 float OUTPUT[OUTPUT_SIZE] = {'+dump.tensor_to_string(out)+'};\n')
f.close()


net.zero_grad()
loss.backward()

input_wgt_grad = torch.transpose(net.mhsa.proj_in.weight.grad, 0, 1) 
output_wgt_grad = torch.transpose(net.mhsa.proj_out.weight.grad, 0, 1)
input_grad = inp.grad


f = open("mhsa-grads.h", 'a')
f.write('#define G_INPUT_WGT_SIZE '+str(input_wgt_grad.numel())+'\n')
f.write("PI_L2 float INPUT_WGT_GRAD[G_INPUT_WGT_SIZE] = {"+dump.tensor_to_string(input_wgt_grad)+"};\n")
f.write('#define G_OUTPUT_WGT_SIZE '+str(output_wgt_grad.numel())+'\n')
f.write("PI_L2 float OUTPUT_WGT_GRAD[G_OUTPUT_WGT_SIZE] = {"+dump.tensor_to_string(output_wgt_grad)+"};\n")
f.write("#define G_IN_SIZE "+str(input_grad.numel())+ '\n')
f.write("PI_L2 float INPUT_GRAD[G_IN_SIZE] = {"+dump.tensor_to_string(input_grad)+ "};\n")
f.close()

f = open("attention_scores.h", "w")
f.write('#define ATTENTION_S_LENGTH '+str(net.mhsa.scores.numel())+'\n')
f.write('PI_L2 float ATTENTION_SCORES[ATTENTION_S_LENGTH] = {'+dump.tensor_to_string(torch.transpose(net.mhsa.scores, 0, 1))+'};\n')
f.close()