import torch
import torch.nn as nn
import torch.optim as optim
import dump_utils as dump
import math

# Set device
if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')
# Define hyperparameters
learning_rate = 0.1 # 0.001
batch_size = 1
epochs = 5

# LAYER 0 SIZES
l0_in_ch = 2
l0_out_ch = 4
l0_hk = 3
l0_wk = 3
l0_hin = 8
l0_win = 8
l0_hstr = 1
l0_wstr = 1
l0_hpad = 0
l0_wpad = 0
# LAYER 1 SIZES
l1_in_ch = 4
l1_out_ch = 4
l1_hk = 1
l1_wk = 1
l1_hin = 6
l1_win = 6
l1_hstr = 1
l1_wstr = 1
l1_hpad = 0
l1_wpad = 0
# LAYER 2 SIZES
l2_in_ch = 144
l2_out_ch = 2
l2_hk = 1
l2_wk = 1
l2_hin = 1
l2_win = 1
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
f.close()


# Simple input data 
inp = torch.torch.div(torch.rand(batch_size, l0_in_ch, l0_hin, l0_win), 1e6).to(device)

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
		self.l0 = nn.Conv2d(in_channels=l0_in_ch, out_channels=l0_out_ch, kernel_size=(l0_hk, l0_wk), padding=(l0_hpad, l0_wpad), stride=(l0_hstr, l0_wstr), bias=0)
		self.l1 = nn.ReLU()
		self.l2 = nn.Linear(in_features=l2_in_ch, out_features=l2_out_ch, bias=1)

	def forward(self, x):
		x = self.l0(x)
		x = self.l1(x.float())
		x = torch.reshape(x, (-1,))
		x = self.l2(x).float()
		return x

# Initialize network
net = DNN().to(device)
for p in net.parameters():
	nn.init.normal_(p, mean=0.0, std=1.0)
net.zero_grad()

# Freeze weights for sparse update


# All-ones fake label 
output_test = net(inp).to(device)
label = torch.ones_like(output_test).to(device)
f = open('io_data.h', 'w')
f.write('// Init weights\n')
f.write('#define WGT_SIZE_L0 '+str(l0_in_ch*l0_out_ch*l0_hk*l0_wk)+'\n')
f.write('PI_L2 float init_WGT_l0[WGT_SIZE_L0] = {'+dump.tensor_to_string(net.l0.weight.data)+'};\n')
f.write('#define WGT_SIZE_L1 '+str(l1_in_ch*l1_out_ch*l1_hk*l1_wk)+'\n')
f.write('PI_L2 float init_WGT_l1[WGT_SIZE_L1];\n')
f.write('#define WGT_SIZE_L2 '+str(l2_in_ch*l2_out_ch*l2_hk*l2_wk)+'\n')
f.write('#define BIAS_SIZE_L2 '+str(l2_out_ch)+'\n')
f.write('PI_L2 float init_WGT_l2[WGT_SIZE_L2] = {'+dump.tensor_to_string(net.l2.weight.data)+'};\n')
f.write('PI_L2 float init_BIAS_l2[BIAS_SIZE_L2] = {'+dump.tensor_to_string(net.l2.bias.data)+'};\n')
f.close()

optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0)
loss_fn = nn.MSELoss()

train_loss_list = []
# Train the DNN
for batch in range(epochs):
	optimizer.zero_grad()
	out = net(inp)
	loss = loss_fn(out, label)
	train_loss_list.append(loss)
	loss.backward()
	optimizer.step()

train_loss = torch.tensor(train_loss_list)

# Inference once after training
out = net(inp)

f = open('io_data.h', 'a')
f.write('// Input and Output data\n')
f.write('#define IN_SIZE 128\n')
f.write('PI_L1 float INPUT[IN_SIZE] = {'+dump.tensor_to_string(inp)+'};\n')
out_size = (int(math.floor(l2_hin-l2_hk+2*l2_hpad+l2_hstr)/l2_hstr)) * (int(math.floor(l2_win-l2_wk+2*l2_wpad+l2_wstr)/l2_wstr)) * l2_out_ch
f.write('#define OUT_SIZE '+str(out_size)+'\n')
f.write('PI_L2 float REFERENCE_OUTPUT[OUT_SIZE] = {'+dump.tensor_to_string(out)+'};\n')
f.write('PI_L1 float LABEL[OUT_SIZE] = {'+dump.tensor_to_string(label)+'};\n')
f.write('PI_L2 float TRAIN_LOSS['+str(epochs)+'] = {'+dump.tensor_to_string(train_loss)+'};\n')
f.close()
