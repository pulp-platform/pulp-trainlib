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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import dump_utils as dump

parser = argparse.ArgumentParser("Depthwise Separable Convolution - Layer Test")
parser.add_argument( '--image_width', type=int, default=3)
parser.add_argument( '--image_height', type=int, default=3)
parser.add_argument( '--ker_width', type=int, default=3 )
parser.add_argument( '--ker_height', type=int, default=3)
parser.add_argument( '--ch_in_dw', type=int, default=128 )
parser.add_argument( '--weight', type=float, default=0.1)
parser.add_argument( '--ch_out_pw', type=int, default=8)
parser.add_argument( '--step', default='DW_FORWARD') # options: // DW_FORWARD, DW_BACKWARD_GRAD, DW_BACKWARD_ERROR, PW_FORWARD, PW_BACKWARD_GRAD, PW_BACKWARD_ERROR,
parser.add_argument( '--bypass_size_automation', type=int, default=0)
parser.add_argument( '--bf16_format', type=int, default=1) # if == 1, data needs to be bfloat16 (no fp16 on that target)

args = parser.parse_args()

bypass = args.bypass_size_automation
ker1 = 2
ker2_w = args.ker_width
ker2_h = args.ker_height
dw_channel = args.ch_in_dw
weight_init = args.weight
pw_channel = args.ch_out_pw
input_w = args.image_width+ker1-1
input_h = args.image_height+ker1-1
image_width = args.image_width
image_height = args.image_height
step = args.step
bf16_format = args.bf16_format


if (bypass == 0):
  # Select number of channels depending on DW or PW
  if step=='DW_FORWARD' or step=='DW_BACKWARD_ERROR' or step=='DW_BACKWARD_GRAD':
    dw_channel = 128
  elif step=='PW_FORWARD' or step=='PW_BACKWARD_ERROR' or step=='PW_BACKWARD_GRAD':
    dw_channel = 512
  else:
    print("Invalid step!")
    exit()

f = open("init-defines.h", "w")
f.write('#define Tker_H_l1 '+str(ker2_h)+'\n')
f.write('#define Tker_W_l1 '+str(ker2_w)+'\n')
f.write('#define Tin_C_l1 '+str(dw_channel)+'\n')
f.write('#define weight_init '+str(weight_init)+'\n')
f.write('#define Tin_W_l1 '+str(image_width)+'\n')
f.write('#define Tin_H_l1 '+str(image_height)+'\n')
f.write('#define Tout_C_l2 '+str(pw_channel)+'\n')
f.close()

f = open("step-check.h", "w")
f.write('#define '+args.step+'\n')
f.close()


class myNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.convDW0 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=ker1,  stride = 1, groups=dw_channel)
    self.convDW = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=(ker2_h, ker2_w),  stride = 1, groups=dw_channel)
    self.convPW = nn.Conv2d(dw_channel, pw_channel, 1, stride = 1)

  def forward(self, x):
    return self.convPW(self.convDW(self.convDW0(x)))

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
  f = open("dw-grads.h", "w")

  for grad in i:
    try:
      if cont==0:
          input_grad = grad

          f.write('#define IN_SIZE '+str(input_grad.numel())+'\n')
          print(weight_grad)
          f.write('PI_L2 fp16 INPUT_GRAD[IN_SIZE] = {'+dump.tensor_to_string(input_grad)+'};\n')

      if cont==1:
          weight_grad = grad
          f.write('#define WGT_SIZE '+str(weight_grad.numel())+'\n')
          print(weight_grad)
          f.write('PI_L2 fp16 WEIGHT_GRAD[WGT_SIZE] = {'+dump.tensor_to_string(weight_grad)+'};\n')

      cont += 1
    except AttributeError:
      print ("None found for Gradient")

  print("------------Output Grad------------")
  for grad in o:
    try:
      output_grad = grad
      f.write('#define G_OUTPUT_SIZE '+str(output_grad.numel())+'\n')
      print(output_grad)
      if step=='DW_BACKWARD_GRAD' or step=='DW_BACKWARD_ERROR':
          f.write('PI_L2 fp16 OUTPUT_GRAD[G_OUTPUT_SIZE] = {'+dump.tensor_to_string(output_grad)+'};\n')
      else:
          f.write('PI_L2 fp16 OUTPUT_GRAD[G_OUTPUT_SIZE] = {'+dump.tensor_to_string(output_grad)+'};\n')

    except AttributeError:
      print ("None found for Gradient")
  f.close()



def hook_fn2(m, i, o):

    cont = 0
    input_grad = []
    weight_grad = []
    output_grad = []
    f = open("input-image.h", "w")

    # NET PARAMETERS

    print("------------Output------------")
    for grad in o:
      try:
        output_grad = grad
        f.write('#define OUTPUT_SIZE '+str(output_grad.numel())+'\n')
        print(output_grad)
        if step=='DW_FORWARD' or step=='DW_BACKWARD_GRAD':
            f.write('PI_L2 fp16 OUTPUT[OUTPUT_SIZE] = {'+dump.tensor_to_string(output_grad)+'};\n')
        else:
            f.write('PI_L2 fp16 OUTPUT[OUTPUT_SIZE] = {'+dump.tensor_to_string(output_grad)+'};\n')
      except AttributeError:
        print ("None found for Gradient")
    f.close()



def hook_fn3(m, i, o):

  cont = 0
  input_grad = []
  weight_grad = []
  output_grad = []
  f = open("pw-grads.h", "w")

  for grad in i:
    try:
      if cont==0:
          input_grad = grad

          # NET PARAMETERS

          f.write('#define PW_IN_SIZE '+str(input_grad.numel())+'\n')
          print(weight_grad)
          f.write('PI_L2 fp16 PW_INPUT_GRAD[PW_IN_SIZE] = {'+dump.tensor_to_string(input_grad)+'};\n')

      if cont==1:
          weight_grad = grad
          f.write('#define PW_WGT_SIZE '+str(weight_grad.numel())+'\n')
          print(weight_grad)
          f.write('PI_L2 fp16 PW_WEIGHT_GRAD[PW_WGT_SIZE] = {'+dump.tensor_to_string(weight_grad)+'};\n')

      cont += 1
    except AttributeError:
      print ("None found for Gradient")

  print("------------Output Grad------------")
  for grad in o:
    try:
      output_grad = grad
      f.write('#define PW_OUTPUT_SIZE '+str(output_grad.numel())+'\n')
      print(output_grad)
      if step=='PW_BACKWARD_GRAD' or step=='PW_BACKWARD_ERROR':
          f.write('PI_L2 fp16 PW_OUTPUT_GRAD[PW_OUTPUT_SIZE] = {'+dump.tensor_to_string(output_grad)+'};\n')
      else:
          f.write('PI_L2 fp16 PW_OUTPUT_GRAD[PW_OUTPUT_SIZE] = {'+dump.tensor_to_string(output_grad)+'};\n')
    except AttributeError:
      print ("None found for Gradient")
  f.close()



def hook_fn4(m, i, o):

     cont = 0
     input_grad = []
     weight_grad = []
     output_grad = []
     f = open("dw-output.h", "w")

     print("------------Output------------")
     for grad in o:
       try:
         output_grad = grad
         f.write('#define DW_OUTPUT_SIZE '+str(output_grad.numel())+'\n')
         print(output_grad)
         if step=='PW_FORWARD' or step=='PW_BACKWARD_GRAD':
             f.write('PI_L2 fp16 DW_OUTPUT[DW_OUTPUT_SIZE] = {'+dump.tensor_to_string(output_grad)+'};\n')
         else:
             f.write('PI_L2 fp16 DW_OUTPUT[DW_OUTPUT_SIZE] = {'+dump.tensor_to_string(output_grad)+'};\n')
       except AttributeError:
         print ("None found for Gradient")
     f.close()



def hook_fn5(m, i, o):

    cont = 0
    input_grad = []
    weight_grad = []
    output_grad = []
    f = open("pw-output.h", "w")

    print("------------Output------------")
    for grad in o:
      try:
        output_grad = grad
        f.write('#define PW_OUTPUT_SIZE '+str(output_grad.numel())+'\n')
        print(output_grad)
        f.write('PI_L2 fp16 PW_OUTPUT[PW_OUTPUT_SIZE] = {'+dump.tensor_to_string(output_grad)+'};\n')
      except AttributeError:
        print ("None found for Gradient")
    f.close()


gradsDW = net.convDW0.register_forward_hook(hook_fn2)
gradsDW = net.convDW.register_backward_hook(hook_fn1)
gradsDW = net.convDW.register_forward_hook(hook_fn4)
gradsPW = net.convPW.register_backward_hook(hook_fn3)
gradsDW = net.convPW.register_forward_hook(hook_fn5)

if bf16_format == 1:
  inp = torch.ones(1, dw_channel, input_h, input_w).bfloat16()
  label = torch.ones(1, pw_channel, input_h-(ker1-1)-(ker2_h-1), input_w-(ker1-1)-(ker2_w-1)).bfloat16()
else: 
  inp = torch.ones(1, dw_channel, input_h, input_w).half()
  label = torch.ones(1, pw_channel, input_h-(ker1-1)-(ker2_h-1), input_w-(ker1-1)-(ker2_w-1)).half()  

with torch.no_grad():
    net.convDW0.weight[:, :] = 0.1
    net.convDW0.bias[:] = 0.0
    net.convDW.weight[:, :] = weight_init
    net.convDW.bias[:] = 0.0
    net.convPW.weight[:] = weight_init
    net.convPW.bias[:] = 0.0


criterion = nn.MSELoss()
out = net(inp)
loss = criterion(out.float(), label.float())
net.zero_grad()


loss.backward()