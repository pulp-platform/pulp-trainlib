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
Authors: Davide Nadalini, Leonardo Ravaglia
'''


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

args = parser.parse_args()

ker_h = args.ker_height
ker_w = args.ker_width
in_ch = args.ch_in
weight_init = args.weight
out_ch = args.ch_out
image_width = args.image_width
image_height = args.image_height
step = args.step

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

net = myNet()
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
        f.write("PI_L2 float INPUT_GRAD[G_IN_SIZE] = {"+dump.tensor_to_string(input_grad)+ "};\n")

      if cont==1:
        weight_grad = grad
        f.write('#define G_WGT_SIZE '+str(weight_grad.numel())+'\n')
        print(weight_grad)
        f.write('PI_L2 float WEIGHT_GRAD[G_WGT_SIZE] = {'+dump.tensor_to_string(weight_grad)+'};\n')       

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
     f = open("conv2d-output.h", "w")

     print("------------Output------------")
     for grad in o:
       try:
         output_grad = grad
         f.write('#define OUTPUT_SIZE '+str(output_grad.numel())+'\n')
         print(output_grad)
         f.write('PI_L2 float OUTPUT[OUTPUT_SIZE] = {'+dump.tensor_to_string(output_grad)+'};\n')
       except AttributeError:
         print ("None found for Gradient")
     f.close()



gradsConv = net.conv.register_backward_hook(hook_fn1)
outConv = net.conv.register_forward_hook(hook_fn2)


inp = torch.div(torch.ones(1, in_ch, image_height, image_width), 1000)
inp.requires_grad = True
label = torch.ones(1, out_ch, image_height-ker_h+1, image_width-ker_w+1)


# Write input image
print("------------Input image------------")
f = open("input-image.h", "w")
f.write("#define INPUT_SIZE "+str(inp.numel())+'\n')
print(inp)
if step=='FORWARD':
  f.write('PI_L2 float INPUT[INPUT_SIZE] = {'+dump.tensor_to_string(inp)+'};\n')
else:
  f.write('PI_L2 float INPUT[INPUT_SIZE] = {'+dump.tensor_to_string(inp)+'};\n')
f.close()


# Initialize weights
with torch.no_grad():
    net.conv.weight[:, :] = weight_init
    net.conv.bias[:] = 0.0


criterion = nn.MSELoss()
out = net(inp)
loss = criterion(out, label)
net.zero_grad()

loss.backward()
