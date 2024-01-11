import torch
from torch import nn
import numpy as np
import dump_utils
from dump_utils import TensorToArray
from dump_utils import WriteArray
import argparse
import random


parser = argparse.ArgumentParser()
parser.add_argument("-CI", type=int, default=2)
parser.add_argument("-HI", type=int, default=3)
parser.add_argument("-WI", type=int, default=4)
parser.add_argument("-KER", type=int, default=1)
parser.add_argument("-DEBUG_INFO", type=int, default=0)
parser.add_argument("-STEP", type=str, default='FORWARD')
parser.add_argument("-NUM_CORES", type=int, default=1)
parser.add_argument("-HWC", type=int, default=0)
parser.add_argument("-DATA_TYPE", type=str, default='FLOAT32')
parser.add_argument("-USE_IM2COL", type=int, default=1)
parser.add_argument("-USE_DMA", type=int, default=0)
parser.add_argument("-MATMUL_TYPE", type=int, default=0)
parser.parse_args()
args = parser.parse_args()


#Parameters for the layers

CI = args.CI
HI = args.HI
WI = args.WI

KER = args.KER 
PAD = int((KER-1)/2)
print(PAD)


CO = CI
WO = WI - KER + 1 + 2*PAD
HO = HI - KER + 1 + 2*PAD
 
HWC = args.HWC

FORMAT = args.DATA_TYPE
STEP = args.STEP
DEBUG = args.DEBUG_INFO


USE_IM2COL = args.USE_IM2COL
USE_DMA = args.USE_DMA

MATMUL_TYPE=args.MATMUL_TYPE

data_type = str()
if(FORMAT == 'FLOAT32'):
    data_type = 'float'
else:
    data_type = 'fp16'

test_data = 100*torch.rand(CI, HI, WI)
test_data.requires_grad = True
test_labels = torch.rand(CO, HO, WO)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(CI, CO, KER, stride=1, padding=PAD, bias = False)
        self.conv2 = nn.Conv2d(CI, CO, KER, stride=1, padding=PAD, bias = False)
        self.relu = nn.ReLU()
        self.loss = nn.MSELoss()
        self.L = 0


    def forward(self, X):
        print("INPUT:\n")
        print(X)
        conv_out1 = self.conv1(X)
        if(conv_out1.size() != X.size()):
            print(f"Dimension mismatch: Conv Out: {conv_out1.size()},  Input: {X.size()}\n")
        
        
        print("CONV1 OUT:\n")
        print(conv_out1)
        res_out = conv_out1 + X
        print("RES OUT:\n")
        print(res_out)
        relu_out = self.relu(res_out)
        print("RELU OUT:\n")
        print(relu_out)
        return relu_out

    def Loss(self, out, label):
        
        self.L = self.loss(out, label)
        print(f"Loss: {self.L}")
        print("--------------------------------\n")
    def backward(self):
        self.L.backward()


net = Net()

out = net(test_data)

net.Loss(out, test_labels)

net.backward()


#GENERATE .h FILES 
f = open("init_defines.h", "w")

f.write("#ifndef INIT_DEFINES \n#define INIT_DEFINES\n")
f.write("#define CI "+str(CI)+"\n")
f.write("#define HI "+str(HI)+"\n")
f.write("#define WI "+str(WI)+"\n")
f.write("#define CO "+str(CO)+"\n")
f.write("#define HO "+str(HO)+"\n")
f.write("#define WO "+str(WO)+"\n")
f.write("#define KER_SIZE "+str(KER)+"\n")
f.write("#define KER_DIM "+str(CI*CO*KER*KER)+"\n")
f.write("#define PAD_SIZE "+str(PAD)+"\n")
f.write(f"#define MATMUL_TYPE {MATMUL_TYPE} \n")
f.write("#define " + FORMAT + "\n")
f.write("#define " + STEP + "\n")
if DEBUG==1:
    f.write("#define DEBUG\n")
if(HWC):
    f.write("int HWC = 1;\n")
else:
    f.write("int HWC = 0;\n")
if(USE_IM2COL):
    f.write("int USE_IM2COL = 1;\n")
else:
    f.write("int USE_IM2COL = 0;\n") 
if(USE_DMA):
    f.write("int USE_DMA = 1;\n")
else:
    f.write("int USE_DMA = 0;\n") 
f.write("#endif\n")
f.close()



f = open("data.h", "w")

f.write("#ifndef DATA \n#define DATA\n")

in_data_array = TensorToArray(test_data, HWC)
WriteArray(in_data_array, "input_data", f,"PI_L1 " +  data_type)

labels_array = TensorToArray(test_labels, HWC)
WriteArray(labels_array, "labels", f, data_type)

in_grad_array = TensorToArray(test_data.grad, HWC)
WriteArray(in_grad_array, "expected_input_diff", f, data_type)

f.write(f"\n{data_type} expected_loss = {net.L};\n")

conv_filters_array = []
for n in range(net.conv1.weight.size(0)):
    for c in range(net.conv1.weight.size(1)):
        for h in range(net.conv1.weight.size(2)):
            for w in range(net.conv1.weight.size(3)):
                conv_filters_array.append(net.conv1.weight[n][c][h][w])

WriteArray(conv_filters_array, "coeff_conv1_data", f, "PI_L1 " + data_type)

f.write("#endif\n")

f.close()

