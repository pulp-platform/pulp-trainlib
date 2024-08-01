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
Authors: Alberto Dequino
'''


"""
    This script generates mask & data for dropout test
"""

import torch
import torch.nn as nn
import argparse
import dump_utils as dump

parser = argparse.ArgumentParser("Dropout Test")
# Dropout arguments
parser.add_argument( '--in_size', type=int, default=10000)
parser.add_argument( '--prob', type=float, default=0.1)
# General arguments
parser.add_argument( '--file_name', type=str, default='dropout_data.h')
parser.add_argument( '--type', type=str, default='float')       # float, fp16 to select the desired format
args = parser.parse_args()

# Network parametersin_size
in_size = args.in_size
data_type = args.type
prob = args.prob

"""
Generate input & mask for Dropout
"""

if  data_type == 'float':
    A = torch.rand(in_size, dtype=torch.float)
elif data_type == 'fp16':
    A = torch.rand(in_size, dtype=torch.bfloat16)
drop = nn.Dropout(p=prob)
B = drop(A)
if  data_type == 'float':
    C = (B==0).type(torch.float)*(1/(1-prob))
elif data_type == 'fp16':
    C = (B==0).type(torch.bfloat16)*(1/(1-prob))

# Print data and create data header file
f = open('net_args.h', "w") 

# Setup the compilation parameter for the data type
if data_type == 'float':
    f.write('// Float32 Dropout\n#define FLOAT32\n\n')
elif data_type == 'fp16':
    f.write('// Float16 Dropout\n#define FLOAT16\n\n')
else: 
    print("Invalid data type selection!!")

# Write sizes in header file
f.write('#define IN_SIZE ' + str(in_size) + '\n')
f.write('\n')

f.close()

# Write data to file
f = open(args.file_name, "w")

print("\nInput Data: ")
print("\nA is: ", A, A.shape, A.dtype)
f.write('PI_L1 ' + data_type + ' input[IN_SIZE] = {'+dump.tensor_to_string(A)+'};\n')
f.write('PI_L1 ' + data_type + ' mask[IN_SIZE] = {'+dump.tensor_to_string(C)+'};\n')

print("\n\n")

f.close()
