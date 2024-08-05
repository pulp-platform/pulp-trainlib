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


"""
    This script generates matrices for matrix multiply tests
"""

import torch
import torch.nn as nn
import argparse
import dump_utils as dump
import losses

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser("Loss Functions test")
parser.add_argument( '--out_size', type=int, default=16 )
parser.add_argument( '--value', type=float, default=0.5 )
parser.add_argument( '--loss_fn', type=int, default=0)
parser.add_argument( '--format', type=int, default=0)

device = 'cpu'

args = parser.parse_args()

out_size = args.out_size
value = args.value
loss_type = args.loss_fn
data_type = args.format

if loss_type == 0:
    loss_type = 'L1Loss'
elif loss_type == 1:
    loss_type = 'MSE'
elif loss_type == 2:
    loss_type = 'CrossEntropy'
elif loss_type == 3:
    loss_type = 'berHuLoss'

if data_type == 0:
    data_type = 'FP16'
elif data_type == 1:
    data_type = 'bfloat16'

print(f"Evaluating {loss_type}..")

# Fake output tensor
if loss_type == 'L1Loss':
    output = torch.ones(out_size)
    with torch.no_grad():
        for i in range(out_size):
            output[i] += i*value + 0.25
    # Fake label
    label = torch.ones(out_size)

if loss_type == 'MSE':
    output = torch.ones(out_size)
    with torch.no_grad():
        for i in range(out_size):
            output[i] += i*value
    # Fake label
    label = torch.ones(out_size)

elif loss_type == 'CrossEntropy':
    output = torch.ones(1, out_size)
    with torch.no_grad():
        for i in range(out_size):
            output[0][i] += i*value
    # Fake label
    label = torch.ones(1, out_size)
    
elif loss_type == 'berHuLoss':
    output = torch.ones(1, out_size)
    with torch.no_grad():
        for i in range(out_size):
            output[0][i] += i*value
    # Fake label
    label = torch.ones(1, out_size)

if data_type == 'FP16':
    output = output.half().to(device)
    label = label.half().to(device)
elif data_type == 'bfloat16':
    output = output.bfloat16().to(device)
    label = label.bfloat16().to(device) 
else:
    print("Invalid FP16_FORMAT!!")
    exit()

output.requires_grad = True

# Loss function
if loss_type == 'L1Loss':
    if data_type == 'FP16':
        loss_fn = nn.L1Loss().half().to(device)
    elif data_type == 'bfloat16':
        loss_fn = nn.L1Loss().bfloat16().to(device)
elif loss_type == 'MSE':
    if data_type == 'FP16':
        loss_fn = nn.MSELoss().half().to(device)
    elif data_type == 'bfloat16':
        loss_fn = nn.MSELoss().bfloat16().to(device)
elif loss_type == 'CrossEntropy':
    if data_type == 'FP16':
        loss_fn = nn.CrossEntropyLoss().half().to(device)
    elif data_type == 'bfloat16':
        loss_fn = nn.CrossEntropyLoss().bfloat16().to(device)
elif loss_type == 'berHuLoss':
    if data_type == 'FP16':
        loss_fn = losses.berHuLoss_fp16().half().to(device)
    elif data_type == 'bfloat16':
        loss_fn = losses.berHuLoss_bfloat16().bfloat16().to(device)

loss = loss_fn(output, label)
loss.backward()

print("Output is:")
print(output)
print("Output grad is:")
print(output.grad)
print("Label is:")
print(label)

f = open("loss_values.h", "w")

f.write("#define OUT_SIZE "+str(out_size)+"\n")
f.write("PI_L1 fp16 LOSS = {"+str(loss.data.item())+"};\n")
f.write("PI_L1 fp16 OUTPUT[OUT_SIZE] = {"+dump.tensor_to_string(output)+"};\n")
f.write("PI_L1 fp16 OUTPUT_GRAD[OUT_SIZE] = {"+dump.tensor_to_string(output.grad)+"};\n")
f.write("PI_L1 fp16 LABEL[OUT_SIZE] = {"+dump.tensor_to_string(label)+"};\n")

f.close()
