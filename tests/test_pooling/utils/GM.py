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


parser = argparse.ArgumentParser("Pooling tests")
parser.add_argument( '--in_h', type=int, default=16 )
parser.add_argument( '--in_w', type=int, default=16 )
parser.add_argument( '--in_c', type=int, default=8 )
parser.add_argument( '--ker_h', type=int, default=2 )
parser.add_argument( '--ker_w', type=int, default=2 )
parser.add_argument( '--stride_h', type=int, default=1 )
parser.add_argument( '--stride_w', type=int, default=1 )
parser.add_argument( '--value', type=float, default=0.5 )

args = parser.parse_args()

in_h = args.in_h
in_w = args.in_w
in_c = args.in_c
ker_h = args.ker_h
ker_w = args.ker_w
stride_h = args.stride_h
stride_w = args.stride_w
value = args.value

# Fake output tensor
maxinput = torch.ones(in_c, in_h, in_w)
avginput = torch.ones(in_c, in_h, in_w)
with torch.no_grad():
    for k in range(in_c):
        for i in range(in_w):
            for j in range(in_h):
                maxinput[k, i, j] += (i+j+k)*value
                avginput[k, i, j] += (i+j+k)*value
# Fake label
maxlabel = torch.ones(in_c, int((in_h-ker_h+stride_h)/stride_h), int((in_w-ker_w+stride_w)/stride_w))
avglabel = torch.ones(in_c, int((in_h-ker_h+stride_h)/stride_h), int((in_w-ker_w+stride_w)/stride_w))

print("maxlabel:")
print(maxlabel.size())
print("avglabel:")
print(avglabel.size())

maxinput.requires_grad = True
avginput.requires_grad = True

# Loss function
loss_fn = nn.MSELoss()

# Pooling functions
class MaxPool (nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()
        self.maxpool = nn.MaxPool2d((ker_h, ker_w), (stride_h, stride_w))
    def forward(self, x):
        out = self.maxpool(x)
        return out

class AvgPool (nn.Module):
    def __init__(self):
        super(AvgPool, self).__init__()
        self.avgpool = nn.AvgPool2d((ker_h, ker_w), (stride_h, stride_w))
    def forward(self, x):
        out = self.avgpool(x)
        return out


maxpool = MaxPool()
avgpool = AvgPool()

# Compute the output and the backward of both
maxout = maxpool(maxinput)
avgout = avgpool(avginput)

maxout.retain_grad()
avgout.retain_grad()

print("Maxout: ")
print(maxout.size())
print("Avgout: ")
print(avgout.size())

maxloss = loss_fn(maxout, maxlabel)
avgloss = loss_fn(avgout, avglabel)

maxloss.backward()
avgloss.backward()

# print("\n*** MAXPOOL DATA ***")
# print("MaxPool out is:")
# print(maxout)
# print("MaxPool out grad is:")
# print(maxout.grad)
# print("MaxPool in grad is:")
# print(maxinput.grad)

# print("\n*** AVGPOOL DATA ***")
# print("AvgPool out is:")
# print(avgout)
# print("AvgPool out grad is:")
# print(avgout.grad)
# print("AvgPool in grad is:")
# print(avginput.grad)

# Write setup to file
f = open("init_defines.h", "w")

f.write("// Layer sizes\n")
f.write("#define Tin_C "+str(in_c)+"\n")
f.write("#define Tin_H "+str(in_h)+"\n")
f.write("#define Tin_W "+str(in_w)+"\n")
f.write("#define Tker_H "+str(ker_h)+"\n")
f.write("#define Tker_W "+str(ker_w)+"\n")
f.write("#define H_STR "+str(stride_h)+"\n")
f.write("#define W_STR "+str(stride_w)+"\n")
f.write("#define Tout_H ((Tin_H-Tker_H+H_STR)/H_STR)\n")
f.write("#define Tout_W ((Tin_W-Tker_W+W_STR)/W_STR)\n")
f.write("#define Tout_C Tin_C\n")

f.close()

# Write data to file
f = open("pool_data.h", "w")

f.write("#define IN_SIZE "+str(in_c*in_h*in_w)+"\n")
f.write("#define OUT_SIZE "+str(in_c*int((in_h-ker_h+stride_h)/stride_h)*int((in_w-ker_w+stride_w)/stride_w))+"\n")

f.write("PI_L2 float MAXLOSS = {"+str(maxloss.data.item())+"};\n")
f.write("PI_L2 float MAXOUTPUT[OUT_SIZE] = {"+dump.tensor_to_string(maxout)+"};\n")
f.write("PI_L2 float MAXOUTPUT_GRAD[OUT_SIZE] = {"+dump.tensor_to_string(maxout.grad)+"};\n")
f.write("PI_L1 float MAXIN[IN_SIZE] = {"+dump.tensor_to_string(maxinput)+"};\n")
f.write("PI_L2 float MAXIN_GRAD[IN_SIZE] = {"+dump.tensor_to_string(maxinput.grad)+"};\n")
f.write("PI_L1 float MAXLABEL[OUT_SIZE] = {"+dump.tensor_to_string(maxlabel)+"};\n")

f.write("PI_L2 float AVGLOSS = {"+str(avgloss.data.item())+"};\n")
f.write("PI_L2 float AVGOUTPUT[OUT_SIZE] = {"+dump.tensor_to_string(avgout)+"};\n")
f.write("PI_L2 float AVGOUTPUT_GRAD[OUT_SIZE] = {"+dump.tensor_to_string(avgout.grad)+"};\n")
f.write("PI_L1 float AVGIN[IN_SIZE] = {"+dump.tensor_to_string(avginput)+"};\n")
f.write("PI_L2 float AVGIN_GRAD[IN_SIZE] = {"+dump.tensor_to_string(avginput.grad)+"};\n")
f.write("PI_L1 float AVGLABEL[OUT_SIZE] = {"+dump.tensor_to_string(avglabel)+"};\n")

f.close()
