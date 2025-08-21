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


from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import dump_utils as dump
import math

# Parse arguments
parser = argparse.ArgumentParser("Transposed 2D Convolution - Layer Test")
parser.add_argument( '--image_width', type=int, default=7)
parser.add_argument( '--image_height', type=int, default=7)
parser.add_argument( '--ker_width', type=int, default=3)
parser.add_argument( '--ker_height', type=int, default=3)
parser.add_argument( '--ch_in', type=int, default=8 )  
parser.add_argument( '--weight', type=float, default=0.01)
parser.add_argument( '--ch_out', type=int, default=8 )
parser.add_argument( '--step', default='FORWARD') # options: // FORWARD, BACKWARD_GRAD, BACKWARD_ERROR
parser.add_argument( '--h_pad', type=int, default=0)
parser.add_argument( '--w_pad', type=int, default=0)
parser.add_argument( '--h_str', type=int, default=1)
parser.add_argument( '--w_str', type=int, default=1)
parser.add_argument( '--HWC', type=int, default=0)
parser.add_argument( '--USE_BIASES', type=int, default=1)
parser.add_argument( '--bias', type=float, default=0.01)

args = parser.parse_args()

# Copy arguments
ker_h = args.ker_height
ker_w = args.ker_width
in_ch = args.ch_in
weight_init = args.weight
out_ch = args.ch_out
image_width = args.image_width
image_height = args.image_height
step = args.step
hpad = args.h_pad
wpad = args.w_pad
hstr = args.h_str
wstr = args.w_str
HWC_layout = args.HWC
use_biases = args.USE_BIASES
bias_init = args.bias

# Write init-defines.h
f = open("init-defines.h", "w")
f.write('#define Tker_H_l1 '+str(ker_h)+'\n')
f.write('#define Tker_W_l1 '+str(ker_w)+'\n')
f.write('#define Tin_C_l1 '+str(in_ch)+'\n')
f.write('#define weight_init '+str(weight_init)+'\n')
f.write('#define Tin_H_l1 '+str(image_height)+'\n')
f.write('#define Tin_W_l1 '+str(image_width)+'\n')
f.write('#define Tout_C_l1 '+str(out_ch)+'\n')
f.write('#define Tpad_H_l1 '+str(hpad)+'\n')
f.write('#define Tpad_W_l1 '+str(wpad)+'\n')
f.write('#define Tstr_H_l1 '+str(hstr)+'\n')
f.write('#define Tstr_W_l1 '+str(wstr)+'\n')
f.write('#define bias_init '+str(bias_init)+'\n')

f.close()

# Write step-check.h
f = open("step-check.h", "w")
f.write('#define '+args.step+'\n')
f.close()


# Output size 
out_size_h = int((image_height-1)*hstr-2*hpad+(ker_h-1)+1) #math.floor((image_height-ker_h+2*hpad+hstr)/hstr)
out_size_w = int((image_width -1)*wstr-2*wpad+(ker_w-1)+1) #math.floor((image_width-ker_w+2*wpad+wstr)/wstr)

#import pdb; pdb.set_trace()

class myNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.ConvTranspose2d(
      in_channels=in_ch,
      out_channels=out_ch,
      kernel_size=(ker_h, ker_w),
      padding=(hpad, wpad),
      stride=(hstr, wstr),
      bias=use_biases,
    )

  def forward(self, x):
    return self.conv(x)

net = myNet()
net.zero_grad()



def hook_fn1(m, i, o):
  """
  Backward hook that prints to conv2d-grads.h and to standard output details about layer dimensions.

  PyTorch documentation on the topic of backward hooks:
  https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook

  :param m: module
  :param i: grad_input
  :param o: grad_output
  :return: None
  """

  cont = 0
  input_grad = []
  weight_grad = []
  output_grad = []
  f = open("transpconv2d-grads.h", "w")

  for cont, grad in enumerate(i):
    try:
      if cont==0:
        input_grad = grad

        f.write("#define G_IN_SIZE "+str(input_grad.numel())+ '\n')
        print("\n>>>>> INPUT GRAD: <<<<<")
        print(input_grad)
        if HWC_layout == 0:
          f.write("PI_L2 float INPUT_GRAD[G_IN_SIZE] = {"+dump.tensor_to_string(input_grad)+ "};\n")
        elif HWC_layout == 1:
          ingrad = deepcopy(input_grad)
          ingrad = ingrad.permute(0,2,3,1)
          f.write("PI_L2 float INPUT_GRAD[G_IN_SIZE] = {"+dump.tensor_to_string(ingrad)+ "};\n")
        else:
          print("[utils/GM.py] Invalid data layout!!")
          exit()

      if cont==1:
        weight_grad = grad
        f.write('#define G_WGT_SIZE '+str(weight_grad.numel())+'\n')
        print(weight_grad)
        if HWC_layout == 0:
          f.write('PI_L2 float WEIGHT_GRAD[G_WGT_SIZE] = {'+dump.tensor_to_string(weight_grad)+'};\n')       
        elif HWC_layout == 1:
          wgt_grad = deepcopy(weight_grad)
          wgt_grad = weight_grad.permute(0,2,3,1)
          f.write('PI_L2 float WEIGHT_GRAD[G_WGT_SIZE] = {'+dump.tensor_to_string(wgt_grad)+'};\n')
        else:
          print("[utils/GM.py] Invalid data layout!!")
          exit()

      if cont==2 and use_biases == 1:
        bias_grad = grad

        if bias_grad is None:
          print("[utils/GM.py] Biases required but not found in PyTorch layer!")
          exit()

        f.write('#define G_BIAS_SIZE '+str(bias_grad.numel())+'\n')
        f.write('PI_L2 float BIAS_GRAD[G_BIAS_SIZE] = {'+dump.tensor_to_string(bias_grad)+'};\n')
        print(bias_grad)

    except AttributeError:
      print("None found for Gradient (input)")


  print("------------Output Grad------------")
  for grad in o:
    try:
      output_grad = grad
      f.write('#define G_OUTPUT_SIZE '+str(output_grad.numel())+'\n')
      print("\n>>>>> OUTPUT GRAD: <<<<<")
      print(output_grad)
      if step=='BACKWARD_GRAD' or step=='BACKWARD_ERROR':
          if HWC_layout == 0:
            f.write('PI_L2 float OUTPUT_GRAD[G_OUTPUT_SIZE] = {'+dump.tensor_to_string(output_grad)+'};\n')
          elif HWC_layout == 1:
            outgrad = deepcopy(output_grad)
            outgrad = outgrad.permute(0,2,3,1)
            f.write('PI_L2 float OUTPUT_GRAD[G_OUTPUT_SIZE] = {'+dump.tensor_to_string(outgrad)+'};\n')
          else:
            print("[utils/GM.py] Invalid data layout!!")
            exit()
      else:
          if HWC_layout == 0:
            f.write('PI_L2 float OUTPUT_GRAD[G_OUTPUT_SIZE] = {'+dump.tensor_to_string(output_grad)+'};\n')
          elif HWC_layout == 1:
            outgrad = deepcopy(output_grad)
            outgrad = outgrad.permute(0,2,3,1)
            f.write('PI_L2 float OUTPUT_GRAD[G_OUTPUT_SIZE] = {'+dump.tensor_to_string(outgrad)+'};\n')
          else:
            print("[utils/GM.py] Invalid data layout!!")
            exit()

    except AttributeError:
      print ("None found for Gradient (output)")

  f.close()



def hook_fn2(m, i, o):

      cont = 0
      input_grad = []
      weight_grad = []
      output_data = []
      f = open("transpconv2d-output.h", "w")

      for data in o:
        try:
          output_data = data
          f.write('#define OUTPUT_SIZE '+str(output_data.numel())+'\n')
          print("\n>>>>> OUTPUT DATA: <<<<<")
          print(output_data)
          if HWC_layout == 0:
            f.write('PI_L2 float OUTPUT[OUTPUT_SIZE] = {'+dump.tensor_to_string(output_data)+'};\n')
          elif HWC_layout == 1:
            outdata = output_data
            outdata = outdata.permute(1,2,0)
            f.write('PI_L2 float OUTPUT[OUTPUT_SIZE] = {'+dump.tensor_to_string(outdata)+'};\n')
          else:
            print("[utils/GM.py] Invalid data layout!!")
            exit()
        except AttributeError:
          print ("None found for Gradient")
      f.close()



gradsConv = net.conv.register_backward_hook(hook_fn1)
outConv = net.conv.register_forward_hook(hook_fn2)


inp = torch.div(torch.ones(1, in_ch, image_height, image_width), 1000)
for cin in range(in_ch):
  for hi in range(image_height):
    for wi in range(image_width):
      inp[0, cin, hi, wi] += (cin + hi - wi)*(cin + hi + wi) * 1/1e5

inp.requires_grad = True


label = torch.ones(1, out_ch, out_size_h, out_size_w)
for c in range(out_ch):
  for h in range(out_size_h):
    for w in range(out_size_w):
      label[0, c, h, w] += w*0.001


# Write input image
f = open("input-image.h", "w")
f.write("#define INPUT_SIZE "+str(inp.numel())+'\n')
print("\n>>>>>> INPUT DATA: <<<<<")
print(inp)
if step=='FORWARD':
  if HWC_layout == 0:
    f.write('PI_L2 float INPUT[INPUT_SIZE] = {'+dump.tensor_to_string(inp)+'};\n')
  elif HWC_layout == 1:
    indata = deepcopy(inp)
    indata = indata.permute(0,2,3,1)
    f.write('PI_L2 float INPUT[INPUT_SIZE] = {'+dump.tensor_to_string(indata)+'};\n')
  else:
    print("[utils/GM.py] Invalid data layout!!")
    exit()
else:
  if HWC_layout == 0:
    f.write('PI_L2 float INPUT[INPUT_SIZE] = {'+dump.tensor_to_string(inp)+'};\n')
  elif HWC_layout == 1:
    indata = deepcopy(inp)
    indata = indata.permute(0,2,3,1)
    f.write('PI_L2 float INPUT[INPUT_SIZE] = {'+dump.tensor_to_string(indata)+'};\n')
  else:
    print("[utils/GM.py] Invalid data layout!!")
    exit()
f.close()


# Prepare weight and bias tensors for init
print("Shape of conv2d kernel:")
print(net.conv.weight.data.shape)
print(net.conv.weight.data)
print("\n")

if use_biases == 1:
  print("Shape of conv2d bias array:")
  print(net.conv.bias.data.shape)
  print(net.conv.bias.data)
  print("\n")

wgt_init_tensor = torch.zeros(out_ch, in_ch, ker_h, ker_w)
for o in range(out_ch):
  for i in range(in_ch):
    for hk in range(ker_h):
      for wk in range(ker_w):
        wgt_init_tensor[o, i, hk, wk] = (o+i+hk+wk)*weight_init

bias_init_tensor = torch.zeros(out_ch)
if use_biases == 1:
  for co in range(out_ch):
    bias_init_tensor[co] = co * bias_init

#print("!--- wgt_init_tensor ---!")
#print(wgt_init_tensor)
#print("!-----------------------!")

#print("!--- bias_init_tensor ---!")
#print(bias_init_tensor)
#print("!-----------------------!")

# Initialize weights and biases
with torch.no_grad():
    #net.conv.weight[:, :] = weight_init
    net.conv.weight.data = deepcopy(wgt_init_tensor)
    if use_biases == 1:
      net.conv.bias.data = deepcopy(bias_init_tensor)

#print("!--- Initialized weights ---!")
#print(net.conv.weight.data)
#print("!---------------------------!")

#print("!--- Initialized bias ---!")
#print(net.conv.bias.data)
#print("!---------------------------!")

# Print weights to init file
f = open("init-defines.h", 'a')
f.write("\n\n// Weight and bias initialization\n")
f.write("#define WGT_SIZE (Tout_C_l1*Tin_C_l1*Tker_H_l1*Tker_W_l1)\n")
if use_biases == 1:
  f.write("#define BIAS_SIZE (Tout_C_l1)\n")
if HWC_layout == 0:
  f.write('PI_L2 float WEIGHTS[WGT_SIZE] = {'+dump.tensor_to_string(net.conv.weight.data)+'};\n')
elif HWC_layout == 1:
  weightdata = deepcopy(net.conv.weight.data)
  weightdata = net.conv.weight.data.permute(0,2,3,1)
  f.write('PI_L2 float WEIGHTS[WGT_SIZE] = {'+dump.tensor_to_string(weightdata)+'};\n')
else:
  print("[utils/GM.py] Invaid data layout!!")
  exit()
if use_biases == 1:
  f.write('PI_L2 float BIASES[BIAS_SIZE] = {'+dump.tensor_to_string(net.conv.bias.data)+'};\n')
f.close()

criterion = nn.MSELoss()
out = net(inp)
loss = criterion(out, label)
net.zero_grad()

loss.backward()

#import pdb; pdb.set_trace()

if HWC_layout == 0:
  print("\n\nCHW data layout:")
  print("Input Size: [{}, {}, {}] \t\t(GM CHW Data: {})".format(in_ch, image_height, image_width, inp.size()))
  print("Kernel Size: [{}, {}, {}, {}] \t(GM CHW Data: {})".format(out_ch, in_ch, ker_h, ker_w, net.conv.weight.data.size()))
  if use_biases == 1:
    print("Bias Size: [{}] \t(GM CHW Data: {})".format(out_ch, net.conv.bias.data.size()))
  print("Out Size: [{}, {}, {}] \t\t(GM CHW Data: {})\n\n".format(out_ch, out_size_h, out_size_w, out.size()))
elif HWC_layout == 1:
  print("\n\nHWC data layout:")
  print("Input Size: [{}, {}, {}] \t\t(GM HWC Data: {})".format(image_height, image_width, in_ch, inp.size()))
  print("Kernel Size: [{}, {}, {}, {}] \t(GM HWC Data: {})".format(out_ch, ker_h, ker_w, in_ch, net.conv.weight.data.size()))
  if use_biases == 1:
    print("Bias Size: [{}] \t(GM HWC Data: {})".format(out_ch, net.conv.bias.data.size()))
  print("Out Size: [{}, {}, {}] \t\t(GM HWC Data: {})\n\n".format(out_size_h, out_size_w, out_ch, out.size()))
else:
  print("[utils/GM.py] Invalid data layout!!")
  exit()
