
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

"""
    This script generates matrices for matrix multiply tests
"""

import torch
import torch.nn as nn
import argparse
import dump_utils as dump


parser = argparse.ArgumentParser("Loss Functions test")
parser.add_argument( '--out_size', type=int, default=16 )
parser.add_argument( '--value', type=float, default=0.5 )

args = parser.parse_args()

out_size = args.out_size
value = args.value

# Fake output tensor
output = torch.ones(out_size)
with torch.no_grad():
    for i in range(out_size):
        output[i] += i*value
# Fake label
label = torch.ones(out_size)

output.requires_grad = True

# Loss function
loss_fn = nn.MSELoss()

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
f.write("PI_L1 float LOSS = {"+str(loss.data.item())+"};\n")
f.write("PI_L1 float OUTPUT[OUT_SIZE] = {"+dump.tensor_to_string(output)+"};\n")
f.write("PI_L1 float OUTPUT_GRAD[OUT_SIZE] = {"+dump.tensor_to_string(output.grad)+"};\n")
f.write("PI_L1 float LABEL[OUT_SIZE] = {"+dump.tensor_to_string(label)+"};\n")

f.close()
