import torch
from torch import nn
import torch.optim as optim
import numpy as np
import dump_utils as dump
import argparse
import random
import math


parser = argparse.ArgumentParser()
parser.add_argument("-CI", type=int, default=2)
parser.add_argument("-HI", type=int, default=3)
parser.add_argument("-WI", type=int, default=4)
parser.add_argument("-STEP", type=str, default='FORWARD')
parser.add_argument("-NUM_CORES", type=int, default=1)
parser.add_argument("-HWC", type=int, default=0)
parser.parse_args()
args = parser.parse_args()


#Parameters for the layers

CI = args.CI
HI = args.HI
WI = args.WI
 
HWC = args.HWC
STEP = args.STEP
NUM_CORES = args.NUM_CORES

test_data = 100*torch.rand(CI, HI, WI)
test_data.requires_grad = True
test_labels = torch.rand(CI, HI, WI)

# LAYER 1 SIZES
l1_in_ch = CI
l1_out_ch = CI
l1_hk = 1
l1_wk = 1
l1_hin = HI
l1_win = WI
l1_hstr = 1
l1_wstr = 1
l1_hpad = 0
l1_wpad = 0

f = open('init-defines.h', 'w')
f.write('// Layer1\n')
f.write('#define Tin_C_l1 '+str(l1_in_ch)+'\n')
f.write('#define Tout_C_l1 '+str(l1_out_ch)+'\n')
f.write('#define Tker_H_l1 '+str(l1_hk)+'\n')
f.write('#define Tker_W_l1 '+str(l1_wk)+'\n')
f.write('#define Tin_H_l1 '+str(l1_hin)+'\n')
f.write('#define Tin_W_l1 '+str(l1_win)+'\n')
f.write('#define Tout_H_l1 '+str(math.floor((l1_hin-l1_hk+2*l1_hpad+l1_hstr)/l1_hstr))+'\n')
f.write('#define Tout_W_l1 '+str(math.floor((l1_win-l1_wk+2*l1_wpad+l1_wstr)/l1_wstr))+'\n')
f.write('#define Tstr_H_l1 '+str(l1_hstr)+'\n')
f.write('#define Tstr_W_l1 '+str(l1_wstr)+'\n')
f.write('#define Tpad_H_l1 '+str(l1_hpad)+'\n')
f.write('#define Tpad_W_l1 '+str(l1_wpad)+'\n')
f.close()

f = open('init-defines.h', 'a')
f.write(f'#define {STEP}\n')
f.close()


# Simple input data 
inp = torch.torch.div(torch.randint(1000, [1, l1_in_ch, l1_hin, l1_win]), 1000)
inp.requires_grad = True

class DNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.l1= nn.InstanceNorm2d(num_features=CI, eps=1e-10, momentum=0, affine=True)

	def forward(self, x):
		x1 = self.l1(x)
		return x1

# Initialize network
net = DNN()
for p in net.parameters():
	nn.init.normal_(p, mean=0.0, std=1.0)
net.zero_grad()


# All-ones fake label 
output_test = net(inp)
label = torch.ones_like(output_test)
f = open('io_data.h', 'w')
f.write('// Init weights\n')
f.write(f'#define WGT_SIZE_L1  2*{l1_in_ch}\n')
f.write('PI_L2 float init_WGT_l1[WGT_SIZE_L1] = {'+dump.tensor_to_string(net.l1.weight.data)+dump.tensor_to_string(net.l1.bias.data)+'};\n')
f.close()

loss_fn = nn.MSELoss()

out = net(inp)
out.retain_grad()
loss = loss_fn(out, label)
loss.backward()
# Print data to golden model's file
f = open('io_data.h', 'a')
f.write('#define INSTN_WGT_G_SIZE 2*'+str(net.l1.weight.data.numel())+'\n')
f.write('PI_L2 float INSTN_WGT_GRAD[INSTN_WGT_G_SIZE] = {'+dump.tensor_to_string(net.l1.weight.grad)+dump.tensor_to_string(net.l1.bias.grad)+'};\n')
f.close()

f = open('io_data.h', 'a')
f.write('// Input and Output data\n')
f.write(f'#define IN_SIZE {CI*HI*WI}\n')
f.write('PI_L1 float INPUT[IN_SIZE] = {'+dump.tensor_to_string(inp)+'};\n')
f.write('#define INSTN_IN_G_SIZE '+str(inp.grad.numel())+'\n')
f.write('PI_L2 float INSTN_IN_GRAD[INSTN_IN_G_SIZE] = {'+dump.tensor_to_string(inp.grad)+'};\n')
f.write(f'#define OUT_SIZE {CI*HI*WI}\n')
f.write('PI_L2 float REFERENCE_OUTPUT[OUT_SIZE] = {'+dump.tensor_to_string(out)+'};\n')
f.write('PI_L1 float LABEL[OUT_SIZE] = {'+dump.tensor_to_string(label)+'};\n')
f.close()
