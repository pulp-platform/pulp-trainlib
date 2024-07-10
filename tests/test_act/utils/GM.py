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
import torch.nn.functional as F
import argparse
import dump_utils as dump


parser = argparse.ArgumentParser("Activations tests")
parser.add_argument( '--in_h', type=int, default=16 )
parser.add_argument( '--in_w', type=int, default=16 )
parser.add_argument( '--in_c', type=int, default=8 )
parser.add_argument( '--value', type=float, default=0.5 )
parser.add_argument( '--data_type', type=str, default='FP32')

args = parser.parse_args()

in_h = args.in_h
in_w = args.in_w
in_c = args.in_c
value = args.value
data_type = args.data_type



"""
CASE 1: FP32 DATA
"""

if data_type == 'FP32':

    # Fake output tensor
    reluinput = torch.ones(in_c, in_h, in_w)
    leakyreluinput = torch.ones(in_c, in_h, in_w)
    softminput = torch.ones(in_c, in_h, in_w)
    sigmoidinput = torch.ones(in_c, in_h, in_w)
    with torch.no_grad():
        for k in range(in_c):
            for i in range(in_w):
                for j in range(in_h):
                    reluinput[k, i, j] += (i+j+k)*value
                    leakyreluinput[k, i, j] += (i+j+k)*value
                    softminput[k, i, j] += (i+j+k)*value
                    sigmoidinput[k, i, j] += (i+j+k)*value
    # Fake label
    relulabel = torch.ones(in_c, int((in_h)), int((in_w)))
    leakyrelulabel = torch.ones(in_c, int((in_h)), int((in_w)))
    softmlabel = torch.flatten(torch.ones(in_c, int((in_h)), int((in_w))))
    sigmoidlabel = torch.ones(in_c, int((in_h)), int((in_w)))

    print("relulabel:")
    print(relulabel.size())
    print("leakyrelulabel:")
    print(leakyrelulabel.size())
    print("softmlabel:")
    print(softmlabel.size())
    print("sigmoidlabel:")
    print(sigmoidlabel.size())

    reluinput.requires_grad = True
    leakyreluinput.requires_grad = True
    softminput.requires_grad = True
    sigmoidinput.requires_grad = True

    # Loss function
    loss_fn = nn.MSELoss()

    # Activation functions
    class ReLU (nn.Module):
        def __init__(self):
            super(ReLU, self).__init__()
            self.relu = nn.ReLU()
        def forward(self, x):
            out = self.relu(x)
            return out
        
    class LeakyReLU (nn.Module):
        def __init__(self):
            super(LeakyReLU, self).__init__()
            self.leakyrelu = nn.LeakyReLU()
        def forward(self, x):
            out = self.leakyrelu(x)
            return out

    class SoftMax (nn.Module):
        def __init__(self):
            super(SoftMax, self).__init__()
            self.softmax = nn.Softmax(dim=0)
        def forward(self, x):
            x = torch.flatten(x, start_dim=0, end_dim=-1)
            out = self.softmax(x)
            #out = F.softmax(x, dim=0)
            return out
        
    class Sigmoid (nn.Module):
        def __init__(self):
            super(Sigmoid, self).__init__()
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            out = self.sigmoid(x)
            return out


    relu = ReLU()
    leakyrelu = LeakyReLU()
    softmax = SoftMax()
    sigmoid = Sigmoid()

    # Compute the output and the backward of both
    reluout = relu(reluinput)
    leakyreluout = leakyrelu(leakyreluinput)
    softmout = softmax(softminput)
    sigmoidout = sigmoid(sigmoidinput)

    reluout.retain_grad()
    leakyreluout.retain_grad()
    softmout.retain_grad()
    sigmoidout.retain_grad()

    print("reluout: ")
    print(reluout.size())
    print("leakyreluout: ")
    print(leakyreluout.size())
    print("softmout: ")
    print(softmout.size())
    print("sigmoidout: ")
    print(sigmoidout.size())

    reluloss = loss_fn(reluout, relulabel)
    leakyreluloss = loss_fn(leakyreluout, leakyrelulabel)
    softmloss = loss_fn(softmout, softmlabel)
    sigmoidloss = loss_fn(sigmoidout, sigmoidlabel)

    reluloss.backward()
    leakyreluloss.backward()
    softmloss.backward()
    sigmoidloss.backward()

    print("\n*** RELU DATA ***")
    print("ReLU out is:")
    print(reluout)
    print("ReLU out grad is:")
    print(reluout.grad)
    print("ReLU in grad is:")
    print(reluinput.grad)

    print("\n*** LEAKY RELU DATA ***")
    print("LeakyReLU out is:")
    print(leakyreluout)
    print("LeakyReLU out grad is:")
    print(leakyreluout.grad)
    print("LeakyReLU in grad is:")
    print(leakyreluinput.grad)

    print("\n*** SOFTMAX DATA ***")
    print("SoftMax out is:")
    print(softmout)
    print("SoftMax out grad is:")
    print(softmout.grad)
    print("SoftMax in grad is:")
    print(softminput.grad)

    print("\n*** SIGMOID DATA ***")
    print("Sigmoid out is:")
    print(sigmoidout)
    print("Sigmoid out grad is:")
    print(sigmoidout.grad)
    print("Sigmoid in grad is:")
    print(sigmoidinput.grad)

    # Write setup to file
    f = open("init_defines.h", "w")

    f.write("// Layer sizes\n")
    f.write("#define Tin_C "+str(in_c)+"\n")
    f.write("#define Tin_H "+str(in_h)+"\n")
    f.write("#define Tin_W "+str(in_w)+"\n")
    f.write("#define Tout_H Tin_H\n")
    f.write("#define Tout_W Tin_W\n")
    f.write("#define Tout_C Tin_C\n")

    f.close()

    # Write data to file
    f = open("act_data.h", "w")

    f.write("#define IN_SIZE "+str(in_c*in_h*in_w)+"\n")
    f.write("#define OUT_SIZE "+str(in_c*int(in_h)*int(in_w))+"\n")

    f.write("PI_L2 float RELULOSS = {"+str(reluloss.data.item())+"};\n")
    f.write("PI_L2 float RELUOUTPUT[OUT_SIZE] = {"+dump.tensor_to_string(reluout)+"};\n")
    f.write("PI_L2 float RELUOUTPUT_GRAD[OUT_SIZE] = {"+dump.tensor_to_string(reluout.grad)+"};\n")
    f.write("PI_L1 float RELUIN[IN_SIZE] = {"+dump.tensor_to_string(reluinput)+"};\n")
    f.write("PI_L2 float RELUIN_GRAD[IN_SIZE] = {"+dump.tensor_to_string(reluinput.grad)+"};\n")
    f.write("PI_L1 float RELULABEL[OUT_SIZE] = {"+dump.tensor_to_string(relulabel)+"};\n")

    f.write("PI_L2 float LEAKYRELULOSS = {"+str(leakyreluloss.data.item())+"};\n")
    f.write("PI_L2 float LEAKYRELUOUTPUT[OUT_SIZE] = {"+dump.tensor_to_string(leakyreluout)+"};\n")
    f.write("PI_L2 float LEAKYRELUOUTPUT_GRAD[OUT_SIZE] = {"+dump.tensor_to_string(leakyreluout.grad)+"};\n")
    f.write("PI_L1 float LEAKYRELUIN[IN_SIZE] = {"+dump.tensor_to_string(leakyreluinput)+"};\n")
    f.write("PI_L2 float LEAKYRELUIN_GRAD[IN_SIZE] = {"+dump.tensor_to_string(leakyreluinput.grad)+"};\n")
    f.write("PI_L1 float LEAKYRELULABEL[OUT_SIZE] = {"+dump.tensor_to_string(leakyrelulabel)+"};\n")

    f.write("PI_L2 float SOFTMLOSS = {"+str(softmloss.data.item())+"};\n")
    f.write("PI_L2 float SOFTMOUTPUT[OUT_SIZE] = {"+dump.tensor_to_string(softmout)+"};\n")
    f.write("PI_L2 float SOFTMOUTPUT_GRAD[OUT_SIZE] = {"+dump.tensor_to_string(softmout.grad)+"};\n")
    f.write("PI_L1 float SOFTMIN[IN_SIZE] = {"+dump.tensor_to_string(softminput)+"};\n")
    f.write("PI_L2 float SOFTMIN_GRAD[IN_SIZE] = {"+dump.tensor_to_string(softminput.grad)+"};\n")
    f.write("PI_L1 float SOFTMLABEL[OUT_SIZE] = {"+dump.tensor_to_string(softmlabel)+"};\n")

    f.write("PI_L2 float SIGMOIDLOSS = {"+str(sigmoidloss.data.item())+"};\n")
    f.write("PI_L2 float SIGMOIDOUTPUT[OUT_SIZE] = {"+dump.tensor_to_string(sigmoidout)+"};\n")
    f.write("PI_L2 float SIGMOIDOUTPUT_GRAD[OUT_SIZE] = {"+dump.tensor_to_string(sigmoidout.grad)+"};\n")
    f.write("PI_L1 float SIGMOIDIN[IN_SIZE] = {"+dump.tensor_to_string(sigmoidinput)+"};\n")
    f.write("PI_L2 float SIGMOIDIN_GRAD[IN_SIZE] = {"+dump.tensor_to_string(sigmoidinput.grad)+"};\n")
    f.write("PI_L1 float SIGMOIDLABEL[OUT_SIZE] = {"+dump.tensor_to_string(sigmoidlabel)+"};\n")

    f.close()



"""
CASE 2: FP16 DATA
"""

if data_type == 'FP16':

    # Fake output tensor
    reluinput = torch.ones(in_c, in_h, in_w)
    leakyreluinput = torch.ones(in_c, in_h, in_w)
    softminput = torch.ones(in_c, in_h, in_w)
    sigmoidinput = torch.ones(in_c, in_h, in_w)
    with torch.no_grad():
        for k in range(in_c):
            for i in range(in_w):
                for j in range(in_h):
                    reluinput[k, i, j] += (i+j+k)*value
                    leakyreluinput[k, i, j] += (i+j+k)*value
                    softminput[k, i, j] += (i+j+k)*value
                    sigmoidinput[k, i, j] += (i+j+k)*value
    # Fake label
    relulabel = torch.ones(in_c, int((in_h)), int((in_w)))
    leakyrelulabel = torch.ones(in_c, int((in_h)), int((in_w)))
    softmlabel = torch.flatten(torch.ones(in_c, int((in_h)), int((in_w))))
    sigmoidlabel = torch.ones(in_c, int((in_h)), int((in_w)))

    print("relulabel:")
    print(relulabel.size())
    print("leakyrelulabel:")
    print(leakyrelulabel.size())
    print("softmlabel:")
    print(softmlabel.size())
    print("sigmoidlabel:")
    print(sigmoidlabel.size())

    reluinput.requires_grad = True
    leakyreluinput.requires_grad = True
    softminput.requires_grad = True
    sigmoidinput.requires_grad = True

    # Loss function
    loss_fn = nn.MSELoss()

    # Activation functions
    class ReLU (nn.Module):
        def __init__(self):
            super(ReLU, self).__init__()
            self.relu = nn.ReLU()
        def forward(self, x):
            out = self.relu(x)
            return out
    
    class LeakyReLU (nn.Module):
        def __init__(self):
            super(LeakyReLU, self).__init__()
            self.leakyrelu = nn.LeakyReLU()
        def forward(self, x):
            out = self.leakyrelu(x)
            return out

    class SoftMax (nn.Module):
        def __init__(self):
            super(SoftMax, self).__init__()
            self.softmax = nn.Softmax(dim=0)
        def forward(self, x):
            x = torch.flatten(x, start_dim=0, end_dim=-1)
            out = self.softmax(x)
            #out = F.softmax(x, dim=0)
            return out
        
    class Sigmoid (nn.Module):
        def __init__(self):
            super(Sigmoid, self).__init__()
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            out = self.sigmoid(x)
            return out


    relu = ReLU()
    leakyrelu = LeakyReLU()
    softmax = SoftMax()
    sigmoid = Sigmoid()

    # Compute the output and the backward of both
    reluout = relu(reluinput)
    leakyreluout = leakyrelu(leakyreluinput)
    softmout = softmax(softminput)
    sigmoidout = sigmoid(sigmoidinput)

    reluout.retain_grad()
    leakyreluout.retain_grad()
    softmout.retain_grad()
    sigmoidout.retain_grad()

    print("reluout: ")
    print(reluout.size())
    print("leakyreluout: ")
    print(leakyreluout.size())
    print("softmout: ")
    print(softmout.size())
    print("sigmoidout: ")
    print(sigmoidout.size())

    reluloss = loss_fn(reluout, relulabel)
    leakyreluloss = loss_fn(leakyreluout, leakyrelulabel)
    softmloss = loss_fn(softmout, softmlabel)
    sigmoidloss = loss_fn(sigmoidout, sigmoidlabel)

    reluloss.backward()
    leakyreluloss.backward()
    softmloss.backward()
    sigmoidloss.backward()

    print("\n*** RELU DATA ***")
    print("ReLU out is:")
    print(reluout)
    print("ReLU out grad is:")
    print(reluout.grad)
    print("ReLU in grad is:")
    print(reluinput.grad)

    print("\n*** LEAKY RELU DATA ***")
    print("LeakyReLU out is:")
    print(leakyreluout)
    print("LeakyReLU out grad is:")
    print(leakyreluout.grad)
    print("LeakyReLU in grad is:")
    print(reluinput.grad)

    print("\n*** SOFTMAX DATA ***")
    print("SoftMax out is:")
    print(softmout)
    print("SoftMax out grad is:")
    print(softmout.grad)
    print("SoftMax in grad is:")
    print(softminput.grad)

    print("\n*** SIGMOID DATA ***")
    print("Sigmoid out is:")
    print(sigmoidout)
    print("Sigmoid out grad is:")
    print(sigmoidout.grad)
    print("Sigmoid in grad is:")
    print(sigmoidinput.grad)

    # Write setup to file
    f = open("init_defines.h", "w")

    f.write("// Layer sizes\n")
    f.write("#define Tin_C "+str(in_c)+"\n")
    f.write("#define Tin_H "+str(in_h)+"\n")
    f.write("#define Tin_W "+str(in_w)+"\n")
    f.write("#define Tout_H Tin_H\n")
    f.write("#define Tout_W Tin_W\n")
    f.write("#define Tout_C Tin_C\n")

    f.close()

    # Write data to file
    f = open("act_data.h", "w")

    f.write("#define IN_SIZE "+str(in_c*in_h*in_w)+"\n")
    f.write("#define OUT_SIZE "+str(in_c*int(in_h)*int(in_w))+"\n")

    f.write("PI_L2 fp16 RELULOSS = {"+str(reluloss.data.item())+"};\n")
    f.write("PI_L2 fp16 RELUOUTPUT[OUT_SIZE] = {"+dump.tensor_to_string(reluout.half())+"};\n")
    f.write("PI_L2 fp16 RELUOUTPUT_GRAD[OUT_SIZE] = {"+dump.tensor_to_string(reluout.grad.half())+"};\n")
    f.write("PI_L1 fp16 RELUIN[IN_SIZE] = {"+dump.tensor_to_string(reluinput.half())+"};\n")
    f.write("PI_L2 fp16 RELUIN_GRAD[IN_SIZE] = {"+dump.tensor_to_string(reluinput.grad.half())+"};\n")
    f.write("PI_L1 fp16 RELULABEL[OUT_SIZE] = {"+dump.tensor_to_string(relulabel.half())+"};\n")

    f.write("PI_L2 fp16 LEAKYRELULOSS = {"+str(leakyreluloss.data.item())+"};\n")
    f.write("PI_L2 fp16 LEAKYRELUOUTPUT[OUT_SIZE] = {"+dump.tensor_to_string(leakyreluout.half())+"};\n")
    f.write("PI_L2 fp16 LEAKYRELUOUTPUT_GRAD[OUT_SIZE] = {"+dump.tensor_to_string(leakyreluout.grad.half())+"};\n")
    f.write("PI_L1 fp16 LEAKYRELUIN[IN_SIZE] = {"+dump.tensor_to_string(leakyreluinput.half())+"};\n")
    f.write("PI_L2 fp16 LEAKYRELUIN_GRAD[IN_SIZE] = {"+dump.tensor_to_string(leakyreluinput.grad.half())+"};\n")
    f.write("PI_L1 fp16 LEAKYRELULABEL[OUT_SIZE] = {"+dump.tensor_to_string(leakyrelulabel.half())+"};\n")

    f.write("PI_L2 fp16 SOFTMLOSS = {"+str(softmloss.data.item())+"};\n")
    f.write("PI_L2 fp16 SOFTMOUTPUT[OUT_SIZE] = {"+dump.tensor_to_string(softmout.half())+"};\n")
    f.write("PI_L2 fp16 SOFTMOUTPUT_GRAD[OUT_SIZE] = {"+dump.tensor_to_string(softmout.grad.half())+"};\n")
    f.write("PI_L1 fp16 SOFTMIN[IN_SIZE] = {"+dump.tensor_to_string(softminput.half())+"};\n")
    f.write("PI_L2 fp16 SOFTMIN_GRAD[IN_SIZE] = {"+dump.tensor_to_string(softminput.grad.half())+"};\n")
    f.write("PI_L1 fp16 SOFTMLABEL[OUT_SIZE] = {"+dump.tensor_to_string(softmlabel.half())+"};\n")

    f.write("PI_L2 fp16 SIGMOIDLOSS = {"+str(sigmoidloss.data.item())+"};\n")
    f.write("PI_L2 fp16 SIGMOIDOUTPUT[OUT_SIZE] = {"+dump.tensor_to_string(sigmoidout.half())+"};\n")
    f.write("PI_L2 fp16 SIGMOIDOUTPUT_GRAD[OUT_SIZE] = {"+dump.tensor_to_string(sigmoidout.grad.half())+"};\n")
    f.write("PI_L1 fp16 SIGMOIDIN[IN_SIZE] = {"+dump.tensor_to_string(sigmoidinput.half())+"};\n")
    f.write("PI_L2 fp16 SIGMOIDIN_GRAD[IN_SIZE] = {"+dump.tensor_to_string(sigmoidinput.grad.half())+"};\n")
    f.write("PI_L1 fp16 SIGMOIDLABEL[OUT_SIZE] = {"+dump.tensor_to_string(sigmoidlabel.half())+"};\n")

    f.close()
