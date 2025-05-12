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

#Visualize data with more precision
torch.set_printoptions(precision=10, sci_mode=False)

parser = argparse.ArgumentParser("FCN Layer Test")
parser.add_argument( '--in_size', type=int, default=1024 )
parser.add_argument( '--out_size', type=int, default=8 )
parser.add_argument( '--file_name', type=str, default='linear-data.h')
parser.add_argument( '--step', type=str, default='FORWARD')     # Possible steps: FORWARD, BACKWARD_GRAD, BACKWARD_ERROR
parser.add_argument( '--use_bias', type=int, default=0)
args = parser.parse_args()

# Network parametersin_size
in_size = args.in_size
out_size = args.out_size
simple_kernel = False
current_step = args.step
use_bias = args.use_bias

# Net step
f_step = open('step-check.h', 'w')
f_step.write('#define ' + str(current_step) + '\n')
f_step.close()

# Data file
f = open(args.file_name, "w") 

f.write('#define Tin_l0 ' + str(in_size) + '\n')
f.write('#define Tout_l0 ' + str(out_size) + '\n\n')

f.write("#define L0_IN_CH     (Tin_l0)\n")
f.write("#define L0_OUT_CH    (Tout_l0)\n")
f.write("#define L0_WEIGHTS   (L0_IN_CH*L0_OUT_CH)\n")

# Sample linear layer
class LinLayer (nn.Module):

    def __init__(self):
        super(LinLayer, self).__init__()
        self.lin = nn.Linear(in_features=in_size, out_features=out_size, bias=bool(use_bias))

    def forward(self, x):
        out = self.lin(x)
        return out


# Training hyperparameters
lr = 1
initial_weights = torch.zeros(out_size, in_size) 

temp_value = 0.01
if simple_kernel:
    initial_weights[0:out_size] = 0.01
else:
    for i in range(out_size):
        for j in range(in_size):
            initial_weights[i][j] = temp_value
            temp_value = temp_value + 0.01

if use_bias == True:
    temp_value_bias = 0.5
    initial_bias = torch.zeros(out_size)
    for i in range(out_size):
        initial_bias[i] = temp_value_bias
        temp_value_bias = temp_value_bias + 0.5

indata = torch.div(torch.ones(in_size), 100000)
indata.requires_grad = True
print("\nInput data is: ", indata, indata.shape, indata.dtype)
f.write('PI_L2 float INPUT_VECTOR[L0_IN_CH] = {'+dump.tensor_to_string(indata)+'};\n')

label = torch.ones(out_size)

# Define and initialize net
net = LinLayer()
print("\nInitializing net parameters to {}.\nParameters are: ".format(initial_weights))


net.lin.weight = nn.Parameter(initial_weights)
if use_bias == True:
    net.lin.bias = nn.Parameter(initial_bias)
for name, parameter in net.named_parameters():
    print(name, parameter, parameter.shape)


f.write('PI_L2 float L0_WEIGHTS_params[L0_WEIGHTS] = {'+dump.tensor_to_string(net.lin.weight)+'};\n')
if use_bias == True:
    f.write('PI_L2 float L0_BIAS_params[L0_OUT_CH] = {'+dump.tensor_to_string(net.lin.bias)+'};\n')
else: 
    zero_biases = torch.zeros(out_size)
    f.write('PI_L2 float L0_BIAS_params[L0_OUT_CH] = {'+dump.tensor_to_string(zero_biases)+'};\n')


# Optimizer and criterion
criterion = nn.MSELoss()

for i in range(1):
    # Do a forward computation
    net.zero_grad()
    output = net(indata)
    print("\nNet output is: ", output, output.shape, output.dtype)
    f.write('PI_L2 float L0_OUT_FW [L0_OUT_CH] = {'+dump.tensor_to_string(output)+'};\n')

    loss = criterion(output, label)
    print("\nLoss is: ", loss, loss.shape, loss.dtype)
    f.write('PI_L2 float L0_LOSS = '+str(loss.item())+';\n')

    # Manually compute outdiff
    loss_meanval = 1/out_size
    output_diff = loss_meanval * 2.0 * (output - label)
    print("\nOutput loss is: ", output_diff, output_diff.shape, output_diff.dtype)
    f.write('PI_L2 float L0_OUT_GRAD [L0_OUT_CH] = {'+dump.tensor_to_string(output_diff)+'};\n')

    # Backward and show gradients
    loss.backward()
    print("\nNetwork gradients are: ")
    for name, parameter in net.named_parameters():
        print(name, parameter.grad, parameter.grad.shape, parameter.grad.dtype)
        if name == 'lin.weight':
            f.write('PI_L2 float L0_WEIGHT_GRAD [L0_WEIGHTS] = {'+dump.tensor_to_string(parameter.grad)+'};\n')
        elif name == 'lin.bias' and use_bias == True: 
            f.write('PI_L2 float L0_BIAS_GRAD [L0_OUT_CH] = {'+dump.tensor_to_string(parameter.grad)+'};\n')
    if use_bias == False:
        zero_biases = torch.zeros(out_size)
        f.write('PI_L2 float L0_BIAS_GRAD [L0_OUT_CH] = {'+dump.tensor_to_string(zero_biases)+'};\n')
    '''
    for name, parameter in net.named_parameters():
        print(name, parameter.grad, parameter.grad.shape, parameter.grad.dtype)
    f.write('PI_L2 float L0_WEIGHT_GRAD [L0_WEIGHTS] = {'+dump.tensor_to_string(parameter.grad)+'};\n')
    '''
    print("\nInput grad is: ", indata.grad)
    f.write('PI_L2 float L0_IN_GRAD [L0_IN_CH] = {'+dump.tensor_to_string(indata.grad)+'};\n')

    f.write('\n\n')

f.close()
