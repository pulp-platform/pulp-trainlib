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

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import dump_utils as dump

parser = argparse.ArgumentParser("2D Convolution - Layer Test")
parser.add_argument( '--image_width', type=int, default=7)
parser.add_argument( '--image_height', type=int, default=7)
parser.add_argument( '--ker_width', type=int, default=3)
parser.add_argument( '--ker_height', type=int, default=3)
parser.add_argument( '--ch_in', type=int, default=8 )  
parser.add_argument( '--weight', type=float, default=0.1)
parser.add_argument( '--ch_out', type=int, default=8 )
parser.add_argument( '--step', default='FORWARD') # options: // FORWARD, BACKWARD_GRAD, BACKWARD_ERROR
parser.add_argument( '--bf16_format', type=int, default=1) # if == 1, data needs to be bfloat16 (no fp16 on that target)

args = parser.parse_args()

ker_h = args.ker_height
ker_w = args.ker_width
in_ch = args.ch_in
weight_init = args.weight
out_ch = args.ch_out
image_width = args.image_width
image_height = args.image_height
step = args.step
bf16_format = args.bf16_format

f = open("init-defines.h", "w")
f.write('#define Tker_H_l1 '+str(ker_h)+'\n')
f.write('#define Tker_W_l1 '+str(ker_w)+'\n')
f.write('#define Tin_C_l1 '+str(in_ch)+'\n')
f.write('#define weight_init '+str(weight_init)+'\n')
f.write('#define Tin_H_l1 '+str(image_height)+'\n')
f.write('#define Tin_W_l1 '+str(image_width)+'\n')
f.write('#define Tout_C_l1 '+str(out_ch)+'\n')
f.close()

f = open("step-check.h", "w")
f.write('#define '+args.step+'\n')
f.close()


class myNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(ker_h, ker_w))

  def forward(self, x):
    return self.conv(x)

if bf16_format == 1:
  net = myNet().bfloat16()
else: 
  net = myNet().half()
net.zero_grad()



def hook_fn1(m, i, o):

  cont = 0
  input_grad = []
  weight_grad = []
  output_grad = []
  f = open("conv2d-grads.h", "w")

  for grad in i:
    try:
      if cont==0:
        input_grad = grad

        f.write("#define G_IN_SIZE "+str(input_grad.numel())+ '\n')
        print("IN GRAD:")
        print(input_grad)
        f.write("PI_L2 fp16 INPUT_GRAD[G_IN_SIZE] = {"+dump.tensor_to_string(input_grad)+ "};\n")

      if cont==1:
        weight_grad = grad
        f.write('#define G_WGT_SIZE '+str(weight_grad.numel())+'\n')
        print(weight_grad)
        f.write('PI_L2 fp16 WEIGHT_GRAD[G_WGT_SIZE] = {'+dump.tensor_to_string(weight_grad)+'};\n')       

      cont += 1

    except AttributeError:
      print("None found for Gradient (input)")


  print("------------Output Grad------------")
  for grad in o:
    try:
      output_grad = grad
      f.write('#define G_OUTPUT_SIZE '+str(output_grad.numel())+'\n')
      print(output_grad)
      if step=='BACKWARD_GRAD' or step=='BACKWARD_ERROR':
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
     f = open("conv2d-output.h", "w")

     print("------------Output------------")
     for grad in o:
       try:
         output_grad = grad
         f.write('#define OUTPUT_SIZE '+str(output_grad.numel())+'\n')
         print(output_grad)
         f.write('PI_L2 fp16 OUTPUT[OUTPUT_SIZE] = {'+dump.tensor_to_string(output_grad)+'};\n')
       except AttributeError:
         print ("None found for Gradient")
     f.close()



gradsConv = net.conv.register_backward_hook(hook_fn1)
outConv = net.conv.register_forward_hook(hook_fn2)

if bf16_format == 1:
  inp = torch.div(torch.ones(1, in_ch, image_height, image_width), 1000).bfloat16()
  inp.requires_grad = True
  label = torch.ones(1, out_ch, image_height-ker_h+1, image_width-ker_w+1).bfloat16()
else:
  inp = torch.div(torch.ones(1, in_ch, image_height, image_width), 1000).half()
  inp.requires_grad = True
  label = torch.ones(1, out_ch, image_height-ker_h+1, image_width-ker_w+1).half()


# Write input image
print("------------Input image------------")
f = open("input-image.h", "w")
f.write("#define INPUT_SIZE "+str(inp.numel())+'\n')
print(inp)
if step=='FORWARD':
  f.write('PI_L2 fp16 INPUT[INPUT_SIZE] = {'+dump.tensor_to_string(inp)+'};\n')
else:
  f.write('PI_L2 fp16 INPUT[INPUT_SIZE] = {'+dump.tensor_to_string(inp)+'};\n')
f.close()

# Prepare weight tensors for init
print("Shape of conv2d kernel:")
print(net.conv.weight.data.shape)
print(net.conv.weight.data)
print("\n")

if bf16_format == 1:
  wgt_init_tensor = torch.zeros(out_ch, in_ch, ker_h, ker_w).bfloat16()
else:
  wgt_init_tensor = torch.zeros(out_ch, in_ch, ker_h, ker_w).half()
for o in range(out_ch):
  for i in range(in_ch):
    for hk in range(ker_h):
      for wk in range(ker_w):
        wgt_init_tensor[o, i, hk, wk] = (o+i+hk+wk)*weight_init

#print("!--- wgt_init_tensor ---!")
#print(wgt_init_tensor)
#print("!-----------------------!")

# Initialize weights
with torch.no_grad():
    #net.conv.weight[:, :] = weight_init
    net.conv.weight.data = deepcopy(wgt_init_tensor)
    net.conv.bias[:] = 0.0

#print("!--- Initialized weights ---!")
#print(net.conv.weight.data)
#print("!---------------------------!")

# Print weights to init file
f = open("init-defines.h", 'a')
f.write("\n\n// Weight initialization\n")
f.write("#define WGT_SIZE (Tout_C_l1*Tin_C_l1*Tker_H_l1*Tker_W_l1)\n")
f.write('PI_L2 float WEIGHTS[WGT_SIZE] = {'+dump.tensor_to_string(net.conv.weight.data)+'};\n')
f.close()

criterion = nn.MSELoss()
out = net(inp)
loss = criterion(out.float(), label.float())
net.zero_grad()

loss.backward()
