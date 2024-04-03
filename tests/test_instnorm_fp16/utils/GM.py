import torch
from torch import nn
import torch.optim as optim
import numpy as np
import dump_utils as dump
import argparse
import random
import math

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument("-CI", type=int, default=2)
parser.add_argument("-CO", type=int, default=2)
parser.add_argument("-HI", type=int, default=3)
parser.add_argument("-WI", type=int, default=4)
parser.add_argument("-DEBUG_INFO", type=int, default=0)
parser.add_argument("-STEP", type=str, default='FORWARD')
parser.add_argument("-NUM_CORES", type=int, default=1)
parser.add_argument("-HWC", type=int, default=0)
parser.add_argument("-EPOCHS", type=int, default=0)
parser.parse_args()
args = parser.parse_args()


#Parameters for the layers

CI = args.CI
HI = args.HI
WI = args.WI


CO = args.CO
 
HWC = args.HWC

STEP = args.STEP

NUM_CORES = args.NUM_CORES

test_data = 100*torch.rand(CI, HI, WI)
test_data.requires_grad = True
test_labels = torch.rand(CO, HI, WI)


# Define hyperparameters
learning_rate = 0.01
batch_size = 1
epochs = 0
if STEP=='BACKWARD_GRAD' or STEP=='BACKWARD_ERROR':
	epochs = 1

# LAYER 0 SIZES
l0_in_ch = CI
l0_out_ch = CI
l0_hk = 1
l0_wk = 1
l0_hin = HI
l0_win = WI
l0_hstr = 1
l0_wstr = 1
l0_hpad = 0
l0_wpad = 0
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
# LAYER 2 SIZES
l2_in_ch = CI
l2_out_ch = CO
l2_hk = 1
l2_wk = 1
l2_hin = HI
l2_win = WI
l2_hstr = 1
l2_wstr = 1
l2_hpad = 0
l2_wpad = 0

f = open('init-defines.h', 'w')
f.write('// Layer0\n')
f.write('#define Tin_C_l0 '+str(l0_in_ch)+'\n')
f.write('#define Tout_C_l0 '+str(l0_out_ch)+'\n')
f.write('#define Tker_H_l0 '+str(l0_hk)+'\n')
f.write('#define Tker_W_l0 '+str(l0_wk)+'\n')
f.write('#define Tin_H_l0 '+str(l0_hin)+'\n')
f.write('#define Tin_W_l0 '+str(l0_win)+'\n')
f.write('#define Tout_H_l0 '+str(math.floor((l0_hin-l0_hk+2*l0_hpad+l0_hstr)/l0_hstr))+'\n')
f.write('#define Tout_W_l0 '+str(math.floor((l0_win-l0_wk+2*l0_wpad+l0_wstr)/l0_wstr))+'\n')
f.write('#define Tstr_H_l0 '+str(l0_hstr)+'\n')
f.write('#define Tstr_W_l0 '+str(l0_wstr)+'\n')
f.write('#define Tpad_H_l0 '+str(l0_hpad)+'\n')
f.write('#define Tpad_W_l0 '+str(l0_wpad)+'\n')
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
f.write('// Layer2\n')
f.write('#define Tin_C_l2 '+str(l2_in_ch)+'\n')
f.write('#define Tout_C_l2 '+str(l2_out_ch)+'\n')
f.write('#define Tker_H_l2 '+str(l2_hk)+'\n')
f.write('#define Tker_W_l2 '+str(l2_wk)+'\n')
f.write('#define Tin_H_l2 '+str(l2_hin)+'\n')
f.write('#define Tin_W_l2 '+str(l2_win)+'\n')
f.write('#define Tout_H_l2 '+str(math.floor((l2_hin-l2_hk+2*l2_hpad+l2_hstr)/l2_hstr))+'\n')
f.write('#define Tout_W_l2 '+str(math.floor((l2_win-l2_wk+2*l2_wpad+l2_wstr)/l2_wstr))+'\n')
f.write('#define Tstr_H_l2 '+str(l2_hstr)+'\n')
f.write('#define Tstr_W_l2 '+str(l2_wstr)+'\n')
f.write('#define Tpad_H_l2 '+str(l2_hpad)+'\n')
f.write('#define Tpad_W_l2 '+str(l2_wpad)+'\n')
f.close()

f = open('init-defines.h', 'a')
f.write('\n// HYPERPARAMETERS\n')
f.write('#define LEARNING_RATE '+str(learning_rate)+'\n')
f.write('#define EPOCHS '+str(epochs)+'\n')
f.write('#define BATCH_SIZE '+str(batch_size)+'\n')
f.write(f'#define {STEP}\n')
f.close()


# Simple input data 
inp = torch.torch.div(torch.randint(1000, [batch_size, l0_in_ch, l0_hin, l0_win]), 1000).half().to(device)

def hook_fn(m, i, o):
	print(m)
	print("------------Input Grad------------")

	for grad in i:
		try:
			print(grad.shape)
			f = open('io_data.h', 'a')
			f.write('#define INSTN_IN_G_SIZE '+str(grad.numel())+'\n')
			f.write('PI_L2 fp16 INSTN_IN_GRAD[INSTN_IN_G_SIZE] = {'+dump.tensor_to_string(grad)+'};\n')
			f.close()

		except AttributeError: 
			print ("None found for Gradient")

	print("------------Output Grad------------")
	for grad in o:  
		try:
			print(grad.shape)
		except AttributeError: 
			print("None found for Gradient")
		print("\n")


class Sumnode():
	def __init__(self, ls):
		self.MySkipNode = ls

class Skipnode():
	def __init__(self):
		self.data = 0

	def __call__(self, x):
		self.data = x
		return self.data

class DNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.l0 = nn.Conv2d(in_channels=l0_in_ch, out_channels=l0_out_ch, kernel_size=1, stride=1, bias=False)
		self.l1= nn.InstanceNorm2d(num_features=CI, eps=1e-10, momentum=0, affine=True)
		self.l2 = nn.Conv2d(in_channels=l2_in_ch, out_channels=l2_out_ch, kernel_size=1, stride=1, bias=False)

	def forward(self, x):
		x = self.l0(x)
		x = self.l1(x)
		x = self.l2(x) #.float()
		return x

# Initialize network
net = DNN().half().to(device)
for p in net.parameters():
	nn.init.normal_(p, mean=0.0, std=1.0)
net.zero_grad()

net.l1.register_backward_hook(hook_fn)


# All-ones fake label 
output_test = net(inp)
label = torch.ones_like(output_test)
f = open('io_data.h', 'w')
f.write('// Init weights\n')
f.write('#define WGT_SIZE_L0 '+str(l0_in_ch*l0_out_ch*l0_hk*l0_wk)+'\n')
f.write('PI_L2 fp16 init_WGT_l0[WGT_SIZE_L0] = {'+dump.tensor_to_string(net.l0.weight.data)+'};\n')
f.write(f'#define WGT_SIZE_L1  2*{l1_in_ch}\n')
f.write('PI_L2 fp16 init_WGT_l1[WGT_SIZE_L1] = {'+dump.tensor_to_string(net.l1.weight.data)+dump.tensor_to_string(net.l1.bias.data)+'};\n')
f.write('#define WGT_SIZE_L2 '+str(l2_in_ch*l2_out_ch*l2_hk*l2_wk)+'\n')
f.write('PI_L2 fp16 init_WGT_l2[WGT_SIZE_L2] = {'+dump.tensor_to_string(net.l2.weight.data)+'};\n')
f.close()

optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0)
loss_fn = nn.MSELoss()

# Train the DNN
for batch in range(epochs):
	optimizer.zero_grad()
	out = net(inp)
	loss = loss_fn(out, label)
	loss.backward()
	# Print data to golden model's file
	f = open('io_data.h', 'a')
	f.write('#define INSTN_WGT_G_SIZE 2*'+str(net.l1.weight.data.numel())+'\n')
	f.write('PI_L2 fp16 INSTN_WGT_GRAD[INSTN_WGT_G_SIZE] = {'+dump.tensor_to_string(net.l1.weight.grad)+dump.tensor_to_string(net.l1.bias.grad)+'};\n')
	f.close()
	optimizer.step()

# Inference once after training
out = net(inp)

f = open('io_data.h', 'a')
f.write('// Input and Output data\n')
f.write(f'#define IN_SIZE {CI*HI*WI}\n')
f.write('PI_L1 fp16 INPUT[IN_SIZE] = {'+dump.tensor_to_string(inp)+'};\n')
out_size = (int(math.floor(l2_hin-l2_hk+2*l2_hpad+l2_hstr)/l2_hstr)) * (int(math.floor(l2_win-l2_wk+2*l2_wpad+l2_wstr)/l2_wstr)) * l2_out_ch
f.write('#define OUT_SIZE '+str(out_size)+'\n')
f.write('PI_L2 fp16 REFERENCE_OUTPUT[OUT_SIZE] = {'+dump.tensor_to_string(out)+'};\n')
f.write('PI_L1 fp16 LABEL[OUT_SIZE] = {'+dump.tensor_to_string(label)+'};\n')
f.close()