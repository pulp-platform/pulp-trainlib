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
Authors: Davide Nadalini
'''


"""
    This script generates mask & data for dropout test
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import dump_utils as dump

parser = argparse.ArgumentParser("Interpolation Test")
# Interpolation arguments
parser.add_argument( '--in_ch', type=int, default=8)
parser.add_argument( '--in_height', type=int, default=16)
parser.add_argument( '--in_width', type=int, default=16)
parser.add_argument( '--out_height', type=int, default=8)
parser.add_argument( '--out_width', type=int, default=8)
# General arguments
parser.add_argument( '--file_name', type=str, default='intp_data.h')
parser.add_argument( '--type', type=str, default='float')       # float, bfloat16 to select the desired format
args = parser.parse_args()

# Function parameters
Ch = args.in_ch
H_in = args.in_height
W_in = args.in_width
H_out = args.out_height
W_out = args.out_width
filename = args.file_name
data_type = args.type

torch.set_printoptions(threshold=10000)

"""
Generate input & mask for Dropout
"""

# Determine data type for data generation on PULP
if  data_type == 'float':
    wr_data_type = 'float'
elif data_type == 'bfloat16':
    wr_data_type = 'fp16'

if  data_type == 'float':
    indata = torch.zeros(1, Ch, H_in, W_in, dtype=torch.float)
    for c in range(Ch):
        for h in range(H_in):
            for w in range(W_in):
                indata[0][c][h][w] = 0.1 + 0.1*c + 0.1*h + 0.1*w
elif data_type == 'bfloat16':
    indata = torch.zeros(1, Ch, H_in, W_in, dtype=torch.bfloat16)
    for c in range(Ch):
        for h in range(H_in):
            for w in range(W_in):
                indata[0][c][h][w] = 0.1 + 0.1*c*h*w

#import pdb; pdb.set_trace()

# Nearest neighbour
near_oudata = F.interpolate(indata, size=[H_out, W_out], mode='nearest')
# Bilinear
bil_outdata = F.interpolate(indata, size=[H_out, W_out], mode='bilinear')

# Print data and create data header file
f = open('net_args.h', "w") 

# Ifdef guard
f.write('#ifndef GENERAL_DEFINES\n')
f.write('#define GENERAL_DEFINES\n\n')

# Setup the compilation parameter for the data type
if data_type == 'float':
    f.write('// Float32 Interpolation\n#define FLOAT32\n\n')
elif data_type == 'bfloat16':
    f.write('// Float16 Interpolation\n#define BFLOAT16\n\n')
else: 
    print("Invalid data type selection!!")

# Write sizes in header file
f.write('#define IN_CH ' + str(Ch) + '\n')
f.write('#define IN_H ' + str(H_in) + '\n')
f.write('#define IN_W ' + str(W_in) + '\n')
f.write('#define OUT_H ' + str(H_out) + '\n')
f.write('#define OUT_W ' + str(W_out) + '\n')
f.write('\n')

# end of ifdef guard
f.write('\n#endif\n')

f.close()

# Write data to file
f = open(args.file_name, "w")

print("\nInput data: ")
print("\nInput is: ", indata, indata.shape, indata.dtype)
print("\nNearest output is: ", near_oudata, near_oudata.shape, near_oudata.dtype)
print("\nBilinear output is: ", bil_outdata, bil_outdata.shape, bil_outdata.dtype)
f.write('PI_L1 ' + wr_data_type + ' INDATA[IN_CH*IN_H*IN_W] = {'+dump.tensor_to_string(indata)+'};\n')
f.write('PI_L2 ' + wr_data_type + ' OUTDATA_NEAR[IN_CH*OUT_H*OUT_W] = {'+dump.tensor_to_string(near_oudata)+'};\n')
f.write('PI_L2 ' + wr_data_type + ' OUTDATA_BIL[IN_CH*OUT_H*OUT_W] = {'+dump.tensor_to_string(bil_outdata)+'};\n')

print("\n\n")

f.close()
